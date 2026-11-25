"""Run separate low-order runway-roll and repeatable airborne wind screens."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import calibrate_endurance_task_baseline, evaluate_endurance_task, load_endurance_profile
from 低阶地面滑跑模型 import GroundRollAssumptions, simulate_ground_roll
from 飞推控闭环物理模型 import LongitudinalDragDisturbance
from 全任务剖面定义 import preliminary_hybrid_aircraft
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from VSPAERO气动代理模型 import VSPAeroSurrogate


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "四项统一集成_停机动态修正重跑v2_20261002"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "后续工作_地面段与风扰动验证_20261002"
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE23点.json"
DESIGN_LABELS = (
    "wing_area_m2",
    "aspect_ratio",
    "taper_ratio",
    "sweep_deg",
    "tip_twist_deg",
    "fuselage_length_m",
    "fuselage_diameter_m",
    "propeller_diameter_m",
    "payload_mass_kg",
    "hybrid_system_mass_kg",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = [float(value) for value in candidate["decision"]]
    design = AirframeDesign(**dict(zip(DESIGN_LABELS, decision[:10])))
    calibration = calibrate_endurance_task_baseline()
    surrogate = VSPAeroSurrogate.from_json(SURROGATE_PATH)
    aero = surrogate.corrected_aero(design, calibration.dimension_calibration)
    aircraft = preliminary_hybrid_aircraft(
        design,
        calibration.dimension_calibration,
        calibration.assumptions,
        aero_override=aero,
    )
    profile = load_endurance_profile()
    speeds = [float(segment["speed_target_mps"]) for segment in profile["segments"]]
    speeds[2] = decision[10]
    engine_caps = decision[11:17]
    motor_assist = decision[17:23]

    wind_cases = (
        ("still_air", ()),
        (
            "headwind_step_8mps",
            (
                LongitudinalDragDisturbance(
                    phase_name="endurance_loiter",
                    start_time_s=3600.0,
                    duration_s=60.0,
                    additional_drag_n=0.0,
                    response_step_s=2.0,
                    recovery_window_s=900.0,
                    recovery_dwell_s=30.0,
                    headwind_step_mps=8.0,
                ),
            ),
        ),
        (
            "sinusoidal_gust_8mps_10s",
            (
                LongitudinalDragDisturbance(
                    phase_name="endurance_loiter",
                    start_time_s=3600.0,
                    duration_s=60.0,
                    additional_drag_n=0.0,
                    response_step_s=0.5,
                    recovery_window_s=900.0,
                    recovery_dwell_s=30.0,
                    headwind_gust_amplitude_mps=8.0,
                    headwind_gust_period_s=10.0,
                ),
            ),
        ),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wind_rows: list[dict[str, object]] = []
    baseline_evaluation = None
    for label, disturbances in wind_cases:
        evaluation = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            calibration=calibration.dimension_calibration,
            assumptions=calibration.assumptions,
            aero_override=aero,
            dispatch_policy=MPCEnergyManagementCoordinator(),
            powertrain=HybridSystemPowertrain(config=HybridSystemAdapterConfig()),
            disturbances=disturbances,
        )
        if label == "still_air":
            baseline_evaluation = evaluation
        frame = evaluation.simulation.profile
        frame.to_csv(output_dir / f"{label}_完整时序.csv", index=False)
        wind_rows.append(
            {
                "case": label,
                "task_feasible": evaluation.result.feasible,
                "total_violation": evaluation.result.total_violation,
                "termination_reason": evaluation.simulation.termination_reason,
                "endurance_h": evaluation.achieved_duration_h,
                "max_headwind_mps": float(frame["equivalent_headwind_mps"].max()),
                "min_headwind_mps": float(frame["equivalent_headwind_mps"].min()),
                "minimum_speed_mps": float(frame["v_mps"].min()),
                "maximum_speed_mps": float(frame["v_mps"].max()),
                "maximum_altitude_deviation_m": float(frame["e_h_m"].abs().max()),
                "maximum_speed_deviation_mps": float(frame["e_v_mps"].abs().max()),
                "recovery_metrics": evaluation.simulation.response_metrics,
            }
        )

    ground_assumption_cases = (
        ("baseline", GroundRollAssumptions()),
        ("headwind_5mps", GroundRollAssumptions(headwind_mps=5.0)),
        ("tailwind_3mps", GroundRollAssumptions(headwind_mps=-3.0)),
        ("uphill_2deg", GroundRollAssumptions(runway_slope_deg=2.0)),
        ("downhill_2deg", GroundRollAssumptions(runway_slope_deg=-2.0)),
        ("rough_runway_high_braking", GroundRollAssumptions(
            rolling_friction_coefficient=0.06,
            braking_deceleration_mps2=3.0,
        )),
        ("low_braking", GroundRollAssumptions(braking_deceleration_mps2=1.0)),
    )
    if baseline_evaluation is None:
        raise RuntimeError("Still-air baseline evaluation was not produced.")
    flare = baseline_evaluation.simulation.profile.loc[
        baseline_evaluation.simulation.profile["phase"] == "flare"
    ].iloc[-1]
    available_shaft_kw = (
        aircraft.engine_electrical_power_max_kw + aircraft.motor_power_max_kw
    )
    ground_rows: list[dict[str, object]] = []
    for label, assumptions in ground_assumption_cases:
        for phase, power_kw, initial_speed, mass in (
            ("takeoff", available_shaft_kw, None, aircraft.initial_mass_kg),
            ("landing", 0.0, float(flare["v_mps"]), float(flare["mass_kg"])),
        ):
            result = simulate_ground_roll(
                aircraft,
                phase,
                power_kw,
                assumptions=assumptions,
                initial_speed_mps=initial_speed,
                mass_kg=mass,
            )
            ground_rows.append(
                {
                    "case": label,
                    **result.__dict__,
                    "flare_terminal_speed_mps": float(flare["v_mps"])
                    if phase == "landing"
                    else None,
                    "flare_terminal_mass_kg": float(flare["mass_kg"])
                    if phase == "landing"
                    else None,
                }
            )
    pd.DataFrame(ground_rows).to_csv(output_dir / "地面滑跑参数扫描.csv", index=False)
    pd.DataFrame(wind_rows).to_csv(output_dir / "空中风阶跃与阵风对比.csv", index=False)
    metadata = {
        "candidate": str(args.candidate.resolve()),
        "ground_model": "separate low-order longitudinal runway model; not integrated into point-mass airborne equations",
        "ground_assumption_cases": {
            label: asdict(assumptions)
            for label, assumptions in ground_assumption_cases
        },
        "wind_model": (
            "quasi-steady relative-airflow speed input changes dynamic pressure, lift, drag, and propeller thrust; "
            "not a resolved atmospheric wind field or ground-track model"
        ),
        "wind_cases": [
            {
                "case": "headwind_step_8mps",
                "step_mps": 8.0,
                "duration_s": 60.0,
            },
            {
                "case": "sinusoidal_gust_8mps_10s",
                "peak_amplitude_mps": 8.0,
                "period_s": 10.0,
                "duration_s": 60.0,
            },
        ],
        "limits": [
            "Runway friction, braking, lift coefficient, and slope are screening assumptions.",
            "Wind is injected into the low-order longitudinal relative-airflow model; it does not resolve wind shear, turbulence spectra, or vertical gusts.",
            "The point-mass air model and runway-roll model are validated separately only at software level.",
        ],
    }
    (output_dir / "地面段与风扰动验证说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(wind_rows)[
        ["case", "task_feasible", "endurance_h", "max_headwind_mps", "maximum_speed_deviation_mps"]
    ].to_string(index=False))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
