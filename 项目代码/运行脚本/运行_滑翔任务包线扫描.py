"""Search adjusted glide speed/height/duration schedules against physical screens."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import (
    build_endurance_mission,
    calibrate_endurance_task_baseline,
    evaluate_endurance_task,
    load_endurance_profile,
)
from 飞推控闭环物理模型 import FlightControlParameters, G, isa_density
from 全任务剖面定义 import preliminary_hybrid_aircraft
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from VSPAERO气动代理模型 import load_vsp_surrogate


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "H6_三项风险闭环复验"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "后续工作_滑翔任务包线扫描_20261002"
SURROGATE_PATH = (
    ROOT
    / "气动代理模型数据"
    / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
)
MIN_INSTANTANEOUS_SINK_RATE_MPS = -0.35
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
CASES = (
    ("source_unpowered", 42.0, 800.0, 3000.0, 6.0, False),
    ("short_glide_slow", 35.0, 2400.0, 600.0, 6.0, False),
    ("short_glide_mid", 40.0, 2400.0, 600.0, 6.0, False),
    ("short_glide_fast", 45.0, 2400.0, 600.0, 6.0, False),
    ("mid_glide_slow", 35.0, 1800.0, 1200.0, 8.0, False),
    ("mid_glide_mid", 40.0, 1800.0, 1200.0, 8.0, False),
    ("mid_glide_fast", 45.0, 1800.0, 1200.0, 8.0, False),
    ("long_glide_slow", 35.0, 1200.0, 1800.0, 10.0, False),
    ("long_glide_mid", 40.0, 1200.0, 1800.0, 10.0, False),
    ("long_glide_fast", 45.0, 1200.0, 1800.0, 10.0, False),
    ("short_glide_35mps_300s", 35.0, 2400.0, 300.0, 6.0, False),
    ("short_glide_35mps_330s", 35.0, 2400.0, 330.0, 6.0, False),
    ("short_glide_35mps_360s", 35.0, 2400.0, 360.0, 6.0, False),
    ("short_glide_35mps_390s", 35.0, 2400.0, 390.0, 6.0, False),
    ("mid_glide_35mps_600s", 35.0, 1800.0, 600.0, 8.0, False),
    ("mid_glide_35mps_650s", 35.0, 1800.0, 650.0, 8.0, False),
    ("mid_glide_35mps_700s", 35.0, 1800.0, 700.0, 8.0, False),
    ("mid_glide_35mps_750s", 35.0, 1800.0, 750.0, 8.0, False),
    ("long_glide_35mps_900s", 35.0, 1200.0, 900.0, 10.0, False),
    ("long_glide_35mps_1000s", 35.0, 1200.0, 1000.0, 10.0, False),
    ("long_glide_35mps_1100s", 35.0, 1200.0, 1100.0, 10.0, False),
    ("short_glide_40mps_270s", 40.0, 2400.0, 270.0, 6.0, False),
    ("short_glide_40mps_300s", 40.0, 2400.0, 300.0, 6.0, False),
    ("short_glide_40mps_330s", 40.0, 2400.0, 330.0, 6.0, False),
    ("mid_glide_40mps_500s", 40.0, 1800.0, 500.0, 8.0, False),
    ("mid_glide_40mps_540s", 40.0, 1800.0, 540.0, 8.0, False),
    ("mid_glide_40mps_580s", 40.0, 1800.0, 580.0, 8.0, False),
    ("long_glide_40mps_760s", 40.0, 1200.0, 760.0, 10.0, False),
    ("long_glide_40mps_820s", 40.0, 1200.0, 820.0, 10.0, False),
    ("long_glide_40mps_880s", 40.0, 1200.0, 880.0, 10.0, False),
    ("unpowered_high_speed_50", 50.0, 800.0, 3000.0, 6.0, False),
    ("unpowered_high_speed_55", 55.0, 800.0, 3000.0, 6.0, False),
    ("unpowered_high_speed_60", 60.0, 800.0, 3000.0, 6.0, False),
    ("engine_only_descent_slow", 35.0, 800.0, 3000.0, 6.0, True),
    ("engine_only_descent_nominal", 42.0, 800.0, 3000.0, 6.0, True),
    ("engine_only_descent_fast", 50.0, 800.0, 3000.0, 6.0, True),
    ("engine_only_descent_upper_bound", 55.0, 800.0, 3000.0, 6.0, True),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stall-margin", type=float, default=1.10)
    parser.add_argument("--altitude-target-tolerance-m", type=float, default=200.0)
    parser.add_argument(
        "--glide-speed-path-angle-gain",
        type=float,
        default=0.02,
        help="Preliminary glide speed feedback gain in rad/(m/s).",
    )
    parser.add_argument(
        "--glide-entry-acceleration-s",
        type=float,
        default=90.0,
        help="Powered speed build-up before the rate-limited engine shutdown.",
    )
    args = parser.parse_args()
    if (
        args.stall_margin < 1.0
        or args.altitude_target_tolerance_m <= 0.0
        or args.glide_speed_path_angle_gain < 0.0
        or args.glide_entry_acceleration_s < 0.0
    ):
        raise SystemExit(
            "stall margin must be >=1, altitude tolerance positive, and glide gain nonnegative."
        )

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = [float(value) for value in candidate["decision"]]
    design = AirframeDesign(**dict(zip(DESIGN_LABELS, decision[:10])))
    calibration = calibrate_endurance_task_baseline()
    surrogate = load_vsp_surrogate(SURROGATE_PATH)
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
    flight_controller = FlightControlParameters(
        glide_speed_path_angle_gain_rad_per_mps=args.glide_speed_path_angle_gain,
        glide_entry_acceleration_duration_s=args.glide_entry_acceleration_s,
    )
    baseline_mission = build_endurance_mission(profile)
    base_durations = {phase.name: phase.duration_s for phase in baseline_mission.phases}
    base_glide_duration = base_durations["long_descent"]
    base_loiter_duration = base_durations["endurance_loiter"]
    rows: list[dict[str, object]] = []
    evaluations: list[tuple[dict[str, object], object]] = []

    for (
        label,
        target_speed,
        target_altitude,
        glide_duration,
        gamma_limit,
        engine_available,
    ) in CASES:
        entry_acceleration_s = 0.0 if engine_available else args.glide_entry_acceleration_s
        shutdown_s = (
            aircraft.engine_electrical_power_max_kw
            / max(aircraft.engine_power_rate_limit_kwps, 1e-9)
            + 1.0
            if not engine_available
            else 0.0
        )
        loiter_duration = base_loiter_duration + base_glide_duration - glide_duration
        loiter_duration -= entry_acceleration_s + shutdown_s
        phase_overrides = {
            "endurance_loiter": {"duration_s": loiter_duration},
            "long_descent": {
                "duration_s": glide_duration,
                "altitude_target_m": target_altitude,
                "speed_target_mps": target_speed,
                "max_flight_path_angle_deg": gamma_limit,
                "time_step_s": 5.0,
            },
        }
        evaluation = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=(
                [0.0 if index == 3 else value for index, value in enumerate(motor_assist)]
                if engine_available
                else motor_assist
            ),
            speeds_mps=speeds,
            calibration=calibration.dimension_calibration,
            assumptions=calibration.assumptions,
            aero_override=aero,
            dispatch_policy=MPCEnergyManagementCoordinator(),
            powertrain=HybridSystemPowertrain(
                config=HybridSystemAdapterConfig()
            ),
            unpowered_glide_phases=() if engine_available else ("long_descent",),
            phase_overrides=phase_overrides,
            flight_controller=flight_controller,
            enforce_phase_engine_requirements=engine_available,
        )
        glide = evaluation.simulation.profile.loc[
            evaluation.simulation.profile["phase"] == "long_descent"
        ].copy()
        if glide.empty:
            row = {
                "case": label,
                "descent_mode": (
                    "engine_only_motor_off" if engine_available else "unpowered_glide"
                ),
                "target_speed_mps": target_speed,
                "target_altitude_m": target_altitude,
                "duration_s": glide_duration,
                "max_flight_path_angle_deg": gamma_limit,
                "task_level_feasible": evaluation.result.feasible,
                "termination_reason": evaluation.simulation.termination_reason,
                "achieved_task_duration_h": evaluation.achieved_duration_h,
                "physical_envelope_pass": False,
                "failure_reason": "task_terminated_before_descent_phase",
                "maximum_lift_shortfall_fraction_of_weight": float("inf"),
                "minimum_stall_speed_margin": 0.0,
            }
            rows.append(row)
            print(f"{label}: pass=False, reason=task_terminated_before_descent_phase")
            continue
        mass = glide["mass_kg"].to_numpy(dtype=float)
        speed = glide["v_mps"].to_numpy(dtype=float)
        altitude = glide["h_m"].to_numpy(dtype=float)
        rho = np.asarray([isa_density(float(value)) for value in altitude])
        stall_speed = np.sqrt(
            2.0 * mass * G
            / np.maximum(
                rho
                * design.wing_area_m2
                * min(
                    aircraft.cl_max,
                    aircraft.cl_alpha_per_rad
                    * flight_controller.alpha_max_rad,
                ),
                1e-9,
            )
        )
        sink_rate = -speed * np.sin(glide["gamma_rad"].to_numpy(dtype=float))
        lift_shortfall_ratio = float(
            np.max(glide["lift_shortfall_n"].to_numpy(dtype=float) / np.maximum(mass * G, 1.0))
        )
        min_stall_margin = float(np.min(speed / stall_speed))
        max_alpha_deg = float(np.rad2deg(glide["alpha_rad"].max()))
        terminal_altitude = float(glide["h_m"].iloc[-1])
        target_error = abs(terminal_altitude - target_altitude)
        mean_sink = float(np.mean(sink_rate))
        max_sink = float(np.max(sink_rate))
        min_sink = float(np.min(sink_rate))
        minimum_engine_power = float(glide["p_engine_kw"].min())
        maximum_abs_motor_power = float(glide["p_motor_kw"].abs().max())
        physical_checks = {
            "task_constraints": evaluation.result.feasible,
            "engine_mode": (
                minimum_engine_power >= 1.0
                if engine_available
                else float(glide["p_engine_kw"].abs().max()) <= 1e-6
            ),
            "motor_off": maximum_abs_motor_power <= 1e-6,
            "stall_margin": min_stall_margin >= args.stall_margin,
            "lift_shortfall": lift_shortfall_ratio <= 0.01,
            "angle_of_attack": max_alpha_deg <= 12.0,
            "airborne_at_glide_exit": terminal_altitude > 50.0,
            "altitude_target": target_error <= args.altitude_target_tolerance_m,
            "descent_rate": (
                min_sink >= MIN_INSTANTANEOUS_SINK_RATE_MPS
                and 0.2 <= mean_sink <= 8.0
                and max_sink <= 10.0
            ),
        }
        row = {
            "case": label,
            "descent_mode": "engine_only_motor_off" if engine_available else "unpowered_glide",
            "target_speed_mps": target_speed,
            "target_altitude_m": target_altitude,
            "duration_s": glide_duration,
            "max_flight_path_angle_deg": gamma_limit,
            "task_level_feasible": evaluation.result.feasible,
            "termination_reason": evaluation.simulation.termination_reason,
            "achieved_task_duration_h": evaluation.achieved_duration_h,
            "glide_start_altitude_m": float(glide["h_m"].iloc[0]),
            "glide_start_speed_mps": float(speed[0]),
            "glide_start_stall_margin": float(speed[0] / stall_speed[0]),
            "glide_end_altitude_m": terminal_altitude,
            "altitude_target_abs_error_m": target_error,
            "minimum_speed_mps": float(speed.min()),
            "minimum_stall_speed_margin": min_stall_margin,
            "maximum_alpha_deg": max_alpha_deg,
            "maximum_lift_shortfall_n": float(glide["lift_shortfall_n"].max()),
            "maximum_lift_shortfall_fraction_of_weight": lift_shortfall_ratio,
            "mean_sink_rate_mps": mean_sink,
            "minimum_sink_rate_mps": min_sink,
            "maximum_sink_rate_mps": max_sink,
            "engine_power_max_kw": float(glide["p_engine_kw"].max()),
            "engine_power_min_kw": minimum_engine_power,
            "motor_power_max_kw": float(glide["p_motor_kw"].max()),
            "motor_power_abs_max_kw": maximum_abs_motor_power,
            "physical_envelope_pass": all(physical_checks.values()),
            **{f"check_{name}": value for name, value in physical_checks.items()},
        }
        rows.append(row)
        evaluations.append((row, evaluation))
        print(
            f"{label}: pass={row['physical_envelope_pass']}, "
            f"V/Vstall={min_stall_margin:.3f}, h={terminal_altitude:.1f} m, "
            f"sink={mean_sink:.2f} m/s"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(rows).sort_values(
        ["physical_envelope_pass", "maximum_lift_shortfall_fraction_of_weight"],
        ascending=[False, True],
    )
    summary_frame.to_csv(output_dir / "滑翔任务组合包线扫描.csv", index=False)
    passing = [pair for pair in evaluations if pair[0]["physical_envelope_pass"]]
    if not evaluations:
        raise RuntimeError(
            "No glide cases reached the target phase; see the CSV for task termination details."
        )
    selected = passing[0] if passing else min(
        evaluations,
        key=lambda pair: pair[0]["maximum_lift_shortfall_fraction_of_weight"]
        + max(0.0, args.stall_margin - pair[0]["minimum_stall_speed_margin"]),
    )
    selected[1].simulation.profile.to_csv(
        output_dir / "代表性滑翔组合完整时序.csv",
        index=False,
    )
    report = {
        "candidate": str(args.candidate.resolve()),
        "case_count": len(rows),
        "passing_case_count": len(passing),
        "physical_envelope_criteria": {
            "minimum_speed_to_stall_margin": args.stall_margin,
            "maximum_lift_shortfall_fraction_of_weight": 0.01,
            "maximum_alpha_deg": 12.0,
            "minimum_glide_exit_altitude_m": 50.0,
            "altitude_target_tolerance_m": args.altitude_target_tolerance_m,
            "sink_rate_screen_mps": {
                "minimum_mean": 0.2,
                "maximum_mean": 8.0,
                "maximum_instantaneous": 10.0,
                "minimum_instantaneous": MIN_INSTANTANEOUS_SINK_RATE_MPS,
            },
        },
        "glide_speed_path_angle_gain_rad_per_mps": args.glide_speed_path_angle_gain,
        "glide_entry_acceleration_duration_s": args.glide_entry_acceleration_s,
        "stall_speed_calculation": (
            "Uses min(CLmax, CL_alpha * sin(alpha_max)) to match the alpha-limited plant."
        ),
        "selected_case": selected[0]["case"],
        "selected_case_pass": selected[0]["physical_envelope_pass"],
        "task_adjustment": "Glide duration is reallocated against endurance_loiter so total target duration remains 80 h; powered-descent cases require at least 1 kW actual engine output and constrain actual motor power to zero. Pure-glide cases bypass only this powered-task requirement for diagnostic comparison.",
        "interpretation": (
            "If no case passes, the current longitudinal low-order model did not establish a glide-envelope-feasible schedule; "
            "the task profile should remain explicitly unpassed pending mission adjustment or higher-fidelity aerodynamics."
        ),
        "limitations": [
            "The pass criteria are preliminary screening criteria, not certified aircraft performance limits.",
            "The plant uses a point-mass longitudinal model and estimated aerodynamic/powertrain parameters.",
        ],
    }
    (output_dir / "滑翔任务包线判定.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"通过组合：{len(passing)}/{len(rows)}；结果目录：{output_dir}")


if __name__ == "__main__":
    main()
