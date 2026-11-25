"""Run ground-roll, glide-envelope, wind-step, and battery diagnostic checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import calibrate_endurance_task_baseline, evaluate_endurance_task, load_endurance_profile
from 低阶地面滑跑模型 import GroundRollAssumptions, simulate_ground_roll
from 全任务剖面定义 import preliminary_hybrid_aircraft
from 飞推控闭环物理模型 import LongitudinalDragDisturbance
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from VSPAERO气动代理模型 import VSPAeroSurrogate


DEFAULT_CANDIDATE = (
    ROOT / "运行结果" / "四项统一集成_前四项推进_交付_20261002" / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "四项补充验证_20261002"
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型.json"
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = candidate["decision"]
    design = AirframeDesign(**dict(zip(DESIGN_LABELS, decision[:10])))
    calibration = calibrate_endurance_task_baseline()
    surrogate = VSPAeroSurrogate.from_json(SURROGATE_PATH)
    aero = surrogate.corrected_aero(design, calibration.dimension_calibration)
    aircraft = preliminary_hybrid_aircraft(
        design, calibration.dimension_calibration, calibration.assumptions, aero_override=aero
    )
    ground_assumptions = GroundRollAssumptions()
    replay_path = args.candidate.parent / "NSGA-II与MOEA-D择优候选_MPC时序.csv"
    replay = pd.read_csv(replay_path)
    flare_terminal = replay.loc[replay["phase"] == "flare"].iloc[-1]
    ground_results = [
        simulate_ground_roll(
            aircraft,
            "takeoff",
            aircraft.engine_electrical_power_max_kw + aircraft.motor_power_max_kw,
            assumptions=ground_assumptions,
        ),
        simulate_ground_roll(
            aircraft,
            "landing",
            0.0,
            assumptions=ground_assumptions,
            initial_speed_mps=float(flare_terminal["v_mps"]),
            mass_kg=float(flare_terminal["mass_kg"]),
        ),
    ]
    ground_payload = [result.__dict__ for result in ground_results]
    ground_payload[0]["input_state_source"] = "candidate MTOW and sized propulsion ratings"
    ground_payload[1]["input_state_source"] = str(replay_path.resolve())
    ground_payload[1]["flare_terminal_speed_mps"] = float(flare_terminal["v_mps"])
    ground_payload[1]["flare_terminal_mass_kg"] = float(flare_terminal["mass_kg"])
    (out / "起飞着陆地面滑跑筛查.json").write_text(
        json.dumps(ground_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    profile = load_endurance_profile()
    speeds = [float(segment["speed_target_mps"]) for segment in profile["segments"]]
    speeds[2] = float(candidate["isr_cruise_speed_mps"])
    engine_caps = candidate["engine_caps_kw"]
    motor_assist = candidate["motor_assist_kw"]
    glide_rows: list[dict[str, object]] = []
    for glide_step_s in (20.0, 5.0, 1.0):
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
            unpowered_glide_phases=("long_descent",),
            phase_overrides={
                "long_descent": {
                    "altitude_target_m": 0.0,
                    "speed_target_mps": 50.0,
                    "time_step_s": glide_step_s,
                }
            },
        )
        frame = evaluation.simulation.profile
        glide = frame.loc[frame["phase"] == "long_descent"]
        frame.to_csv(out / f"无动力滑翔_{glide_step_s:.0f}秒步长_时序.csv", index=False)
        glide_rows.append(
            {
                "glide_speed_target_mps": 50.0,
                "glide_time_step_s": glide_step_s,
                "feasible": evaluation.result.feasible,
                "total_violation": evaluation.result.total_violation,
                "constraint_violations": {
                    name: value
                    for name, value in evaluation.result.constraints.items()
                    if value > 0.0
                },
                "termination_reason": evaluation.simulation.termination_reason,
                "endurance_h": evaluation.achieved_duration_h,
                "glide_end_altitude_m": float(glide["h_m"].iloc[-1]),
                "glide_min_speed_mps": float(glide["v_mps"].min()),
                "glide_max_alpha_deg": float(glide["alpha_rad"].max() * 180.0 / 3.141592653589793),
                "glide_max_lift_shortfall_n": float(glide["lift_shortfall_n"].max()),
                "engine_power_max_in_glide_kw": float(glide["p_engine_kw"].max()),
                "motor_power_max_in_glide_kw": float(glide["p_motor_kw"].max()),
            }
        )
    pd.DataFrame(glide_rows).to_csv(out / "无动力滑翔速度包线扫描.csv", index=False)

    event = LongitudinalDragDisturbance(
        phase_name="endurance_loiter",
        start_time_s=3600.0,
        duration_s=60.0,
        additional_drag_n=0.0,
        response_step_s=2.0,
        recovery_window_s=900.0,
        recovery_dwell_s=30.0,
        headwind_step_mps=8.0,
    )
    wind_evaluation = evaluate_endurance_task(
        design,
        engine_caps_kw=engine_caps,
        motor_assist_kw=motor_assist,
        speeds_mps=speeds,
        calibration=calibration.dimension_calibration,
        assumptions=calibration.assumptions,
        aero_override=aero,
        dispatch_policy=MPCEnergyManagementCoordinator(),
        powertrain=HybridSystemPowertrain(config=HybridSystemAdapterConfig()),
        disturbances=(event,),
    )
    wind_frame = wind_evaluation.simulation.profile
    wind_frame.to_csv(out / "8mps等效迎风阶跃_MPC时序.csv", index=False)
    summary = {
        "model_status": "preliminary low-order screening; not hardware or airworthiness validation",
        "candidate_source": str(args.candidate.resolve()),
        "ground_roll": ground_payload,
        "glide_sweep": glide_rows,
        "wind_step": {
            "event": {
                "phase": event.phase_name,
                "start_after_phase_entry_s": event.start_time_s,
                "duration_s": event.duration_s,
                "equivalent_headwind_step_mps": event.headwind_step_mps,
                "force_interpretation": "parasite-drag increment only; not a resolved wind field",
            },
            "mission_feasible": wind_evaluation.result.feasible,
            "termination_reason": wind_evaluation.simulation.termination_reason,
            "battery_diagnostics": {
                key: value
                for key, value in wind_evaluation.result.diagnostics.items()
                if key.startswith("battery_")
            },
            "max_equivalent_headwind_mps": float(wind_frame["equivalent_headwind_mps"].max()),
        },
        "boundary": [
            "Ground-roll aerodynamic coefficients and friction/braking assumptions are provisional.",
            "Glide sweep uses the reconstructed longitudinal point-mass model and declared alpha/speed limits.",
            "Equivalent headwind changes parasite-drag load only and does not resolve ground track or vertical gust response.",
            "Battery energy-throughput metrics are diagnostic and do not replace the PPT three-objective system.",
        ],
    }
    (out / "前四项补充验证摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(glide_rows).to_string(index=False))
    print(f"\nSupplemental evidence: {out}")


if __name__ == "__main__":
    main()
