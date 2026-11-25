"""Run full-task baseline and compound-fault supervisory replay."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import (
    calibrate_endurance_task_baseline,
    evaluate_endurance_task,
    load_endurance_profile,
)
from 飞推控闭环物理模型 import MissionFaultEvent
from 全任务剖面定义 import preliminary_hybrid_aircraft
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import (
    FaultTolerantFuzzyCoordinator,
    FuzzyAdaptiveCoordinator,
    HealthState,
    ScheduledFaultCoordinator,
)
from VSPAERO气动代理模型 import VSPAeroSurrogate


DEFAULT_CANDIDATE = (
    ROOT / "运行结果" / "动力下降模式正式重优化_20261003"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE23点.json"
DESIGN_LABELS = (
    "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
    "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
    "payload_mass_kg", "hybrid_system_mass_kg",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "任务级组合故障验证_20261004",
    )
    parser.add_argument("--engine-availability", type=float, default=0.60)
    parser.add_argument("--motor-availability", type=float, default=0.50)
    parser.add_argument("--battery-energy-fraction", type=float, default=0.50)
    parser.add_argument("--sensor-valid", action="store_true")
    args = parser.parse_args()
    for name in ("engine_availability", "motor_availability", "battery_energy_fraction"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1].")

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = [float(value) for value in candidate["decision"]]
    design = AirframeDesign(**dict(zip(DESIGN_LABELS, decision[:10])))
    calibration = calibrate_endurance_task_baseline()
    aero = VSPAeroSurrogate.from_json(SURROGATE_PATH).corrected_aero(
        design, calibration.dimension_calibration
    )
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
    powertrain_factory = lambda: HybridSystemPowertrain(
        config=HybridSystemAdapterConfig()
    )

    timed_fault = MissionFaultEvent(
        phase_name="endurance_loiter",
        start_phase_elapsed_s=3600.0,
        duration_s=600.0,
        engine_available_fraction=args.engine_availability,
        motor_available_fraction=args.motor_availability,
        battery_available_energy_fraction=args.battery_energy_fraction,
        sensor_data_valid=args.sensor_valid,
    )
    scenarios = (
        (
            "nominal",
            FaultTolerantFuzzyCoordinator(
                FuzzyAdaptiveCoordinator(), HealthState()
            ),
            (),
        ),
        (
            "timed_compound_fault",
            ScheduledFaultCoordinator(FuzzyAdaptiveCoordinator(), (timed_fault,)),
            (timed_fault,),
        ),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label, policy, fault_events in scenarios:
        evaluation = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            calibration=calibration.dimension_calibration,
            assumptions=calibration.assumptions,
            aero_override=aero,
            dispatch_policy=policy,
            powertrain=powertrain_factory(),
            fault_events=fault_events,
        )
        frame = evaluation.simulation.profile
        frame.to_csv(output_dir / f"{label}_完整任务时序.csv", index=False)
        active_fault = frame["fault_active"].astype(bool)
        phase_frame = frame[frame["phase"] == "endurance_loiter"]
        active_times = phase_frame.loc[phase_frame["fault_active"].astype(bool), "phase_elapsed_s"]
        fallback_modes = frame["power_mode"].astype(str).str.contains("sensor_fallback")
        active_fault_rows = frame.loc[active_fault]
        post_fault_rows = (
            frame.loc[frame["time_s"] > float(active_fault_rows["time_s"].max())]
            if not active_fault_rows.empty
            else frame.iloc[0:0]
        )
        sample_dt_s = frame["time_s"].diff().fillna(0.0)
        rows.append(
            {
                "scenario": label,
                "task_feasible": evaluation.result.feasible,
                "total_violation": evaluation.result.total_violation,
                "termination_reason": evaluation.simulation.termination_reason,
                "duration_h": evaluation.achieved_duration_h,
                "fuel_used_kg": max(
                    0.0,
                    aircraft.initial_mass_kg - float(frame["mass_kg"].iloc[-1]),
                ),
                "terminal_fuel_kg": max(
                    0.0,
                    float(frame["mass_kg"].iloc[-1]) - aircraft.empty_mass_kg,
                ),
                "terminal_soc": float(frame["soc"].iloc[-1]),
                "minimum_soc": float(frame["soc"].min()),
                "minimum_engine_power_kw": float(frame["p_engine_kw"].min()),
                "maximum_engine_power_kw": float(frame["p_engine_kw"].max()),
                "maximum_motor_power_kw": float(frame["p_motor_kw"].max()),
                "minimum_speed_mps": float(frame["v_mps"].min()),
                "maximum_altitude_error_m": float(frame["e_h_m"].abs().max()),
                "fault_active_duration_s": (
                    float(sample_dt_s.loc[active_fault].sum())
                    if not active_fault_rows.empty
                    else 0.0
                ),
                "sensor_fallback_sample_count": int(fallback_modes.sum()),
                "post_fault_nominal_dispatch_resumed": (
                    None
                    if active_fault_rows.empty
                    else bool(
                        not post_fault_rows.empty
                        and not post_fault_rows["power_mode"]
                        .astype(str).str.contains("sensor_fallback").any()
                    )
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "任务级组合故障汇总.csv", index=False)
    metadata = {
        "candidate": str(args.candidate.resolve()),
        "fault_health_state": asdict(
            HealthState(
                engine_available_fraction=args.engine_availability,
                motor_available_fraction=args.motor_availability,
                battery_available_energy_fraction=args.battery_energy_fraction,
                sensor_data_valid=args.sensor_valid,
            )
        ),
        "injection": {
            "phase": timed_fault.phase_name,
            "phase_elapsed_start_s": timed_fault.start_phase_elapsed_s,
            "duration_s": timed_fault.duration_s,
            "health": asdict(
                HealthState(
                    engine_available_fraction=args.engine_availability,
                    motor_available_fraction=args.motor_availability,
                    battery_available_energy_fraction=args.battery_energy_fraction,
                    sensor_data_valid=args.sensor_valid,
                )
            ),
        },
        "battery_semantics": (
            "battery_available_energy_fraction currently modifies supervisory SOC reserve only; "
            "it does not reduce physical capacity in the battery plant"
        ),
        "model_status": "estimated scenario for software-level task replay; not calibrated or hardware validated",
        "limitations": [
            "Fault onset is scheduled in phase-local time and clears after its configured duration.",
            "Motor availability and engine availability derate supervisory dispatch bounds; they are not component damage models.",
            "Reduced battery available-energy fraction changes reserve logic only, not the battery's physical capacity.",
            "Results use the current estimated component and aerodynamic assumptions.",
        ],
    }
    (output_dir / "任务级组合故障说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
