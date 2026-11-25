"""Run matched JSBSim full-profile controller cases with scheduled supervisory faults."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 规则混动协同控制 import RuleBasedHybridCoordinator
from 智能能量管理与容错调度 import (
    FuzzyAdaptiveCoordinator,
    HealthState,
    MPCParameters,
    MPCEnergyManagementCoordinator,
    NEURAL_PHASES,
    NeuralImitationCoordinator,
    ScheduledFaultCoordinator,
)

DEFAULT_NEURAL_MODEL = (
    ROOT / "运行结果" / "神经网络MPC模仿策略训练测试_20261011"
    / "神经网络MPC模仿策略.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "PTDL_JSBSim六自由度全阶段故障容错闭环_20261013"
FAULT_START_S = 30.0
FAULT_DURATION_S = 120.0
PHASE_POWER_CAPS_KW = {
    "main_climb_to_leveloff": (120.0, 30.0),
    "short_leveloff_to_isr_altitude": (120.0, 20.0),
    "endurance_loiter": (100.0, 8.0),
    "long_descent": (70.0, 0.0),
    "approach_descent": (80.0, 15.0),
    "flare": (70.0, 15.0),
}


def _fault_cases() -> dict[str, HealthState | None]:
    return {
        "battery_isolation": HealthState(battery_available_energy_fraction=0.60),
        "engine_derating": HealthState(engine_available_fraction=0.60),
        "motor_derating": HealthState(motor_available_fraction=0.50),
        "sensor_failure": HealthState(sensor_data_valid=False),
        "compound_fault": HealthState(
            engine_available_fraction=0.75,
            motor_available_fraction=0.75,
            battery_available_energy_fraction=0.60,
            sensor_data_valid=False,
        ),
    }


def _policy_factories(neural_model: Path):
    return {
        "rule": RuleBasedHybridCoordinator,
        "fuzzy": FuzzyAdaptiveCoordinator,
        "mpc": lambda: MPCEnergyManagementCoordinator(MPCParameters()),
        "neural_mpc_imitation": lambda: NeuralImitationCoordinator(neural_model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neural-model", type=Path, default=DEFAULT_NEURAL_MODEL)
    parser.add_argument("--mission-hours", type=float, default=4.0)
    parser.add_argument("--integration-step-s", type=float, default=0.05)
    parser.add_argument("--control-period-s", type=float, default=2.0)
    parser.add_argument("--terminal-control-period-s", type=float, default=0.5)
    parser.add_argument("--sample-period-s", type=float, default=30.0)
    parser.add_argument(
        "--fault-cases", nargs="+", choices=list(_fault_cases()),
        default=["engine_derating", "motor_derating", "sensor_failure", "compound_fault"],
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=NEURAL_PHASES,
        default=list(NEURAL_PHASES),
        help="Mission phases receiving injected faults; nominal baselines are always run.",
    )
    args = parser.parse_args()
    if args.mission_hours <= 3.5:
        parser.error("--mission-hours must exceed 3.5 h to leave positive loiter time.")
    if not args.neural_model.is_file():
        parser.error(f"Neural MPC-imitation model not found: {args.neural_model}")

    runner_dir = ROOT / "运行脚本"
    if str(runner_dir) not in sys.path:
        sys.path.insert(0, str(runner_dir))
    from 运行_PTDL估计六自由度全任务闭环 import run
    from PTDL六自由度参考模型 import PTDL6DoFParameters

    factories = _policy_factories(args.neural_model)
    cases = {name: _fault_cases()[name] for name in args.fault_cases}
    aircraft = PTDL6DoFParameters()
    estimated_stall_speed_mps = math.sqrt(
        2.0
        * aircraft.mtow_kg
        * 9.80665
        / (1.225 * aircraft.reference_wing_area_m2 * 1.42)
    )
    minimum_speed_screen_mps = 1.10 * estimated_stall_speed_mps
    minimum_fuel_reserve_kg = 0.05 * aircraft.fuel_mass_kg
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    scenarios = [(None, "nominal", None)]
    scenarios.extend(
        (phase, fault_name, health)
        for phase in args.phases
        for fault_name, health in cases.items()
    )
    for fault_phase, fault_name, health in scenarios:
        fault_start_s = 150.0 if fault_phase == "flare" else FAULT_START_S
        events = ()
        if health is not None:
            from 飞推控闭环物理模型 import MissionFaultEvent

            events = (
                MissionFaultEvent(
                    phase_name=str(fault_phase),
                    start_phase_elapsed_s=fault_start_s,
                    duration_s=FAULT_DURATION_S,
                    engine_available_fraction=health.engine_available_fraction,
                    motor_available_fraction=health.motor_available_fraction,
                    battery_available_energy_fraction=health.battery_available_energy_fraction,
                    sensor_data_valid=health.sensor_data_valid,
                ),
            )
        for policy_name, factory in factories.items():
            case_dir = output_dir / (fault_phase or "nominal") / fault_name / policy_name
            started = time.perf_counter()
            try:
                summary = run(
                    output_dir=case_dir,
                    integration_step_s=args.integration_step_s,
                    control_period_s=args.control_period_s,
                    terminal_control_period_s=args.terminal_control_period_s,
                    sample_period_s=args.sample_period_s,
                    mission_hours=args.mission_hours,
                    mission_mode="full",
                    pitch_kp=4.0,
                    pitch_kd=4.0,
                    dispatch_policy=ScheduledFaultCoordinator(factory(), events),
                )
                elapsed = time.perf_counter() - started
                profile = pd.read_csv(
                    case_dir / "PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv"
                )
                active = profile.loc[
                    (profile["phase"] == fault_phase)
                    & (profile["phase_elapsed_s"] >= fault_start_s)
                    & (
                        profile["phase_elapsed_s"]
                        < fault_start_s + FAULT_DURATION_S
                    )
                ]
                fallback_samples = int(
                    active["dispatch_mode"]
                    .astype(str)
                    .str.contains("sensor_fallback")
                    .sum()
                )
                engine_violations = 0
                motor_violations = 0
                if not active.empty:
                    engine_violations = int(
                        (
                            active["engine_power_kw"]
                            > active["engine_power_cap_kw"] + 1e-6
                        ).sum()
                    )
                    motor_violations = int(
                        (
                            (
                                active["motor_bus_power_kw"]
                                < active["motor_power_min_kw"] - 1e-6
                            )
                            | (
                                active["motor_bus_power_kw"]
                                > active["motor_power_max_kw"] + 1e-6
                            )
                        ).sum()
                    )
                fault_verified = fault_name == "nominal"
                fault_applicable = fault_name != "nominal"
                phase_engine_cap, phase_motor_cap = PHASE_POWER_CAPS_KW.get(
                    str(fault_phase), (0.0, 0.0)
                )
                if fault_name == "engine_derating" and not active.empty:
                    fault_verified = bool(
                        (
                            active["engine_power_cap_kw"]
                            <= phase_engine_cap * health.engine_available_fraction + 1e-6
                        ).all()
                    )
                elif fault_name == "motor_derating" and not active.empty:
                    fault_applicable = phase_motor_cap > 0.0
                    fault_verified = (
                        fault_applicable
                        and bool(
                            (
                                active["motor_power_max_kw"]
                                <= phase_motor_cap * health.motor_available_fraction + 1e-6
                            ).all()
                        )
                    )
                elif fault_name == "sensor_failure" and not active.empty:
                    fault_verified = fallback_samples > 0
                elif fault_name == "battery_isolation" and not active.empty:
                    initial_capacity = float(profile.iloc[0]["battery_effective_capacity_ah"])
                    after_window = profile.loc[
                        profile["time_s"] > float(active.iloc[-1]["time_s"])
                    ]
                    fault_verified = bool(
                        (abs(active["battery_available_fraction"] - 0.60) < 1e-9).all()
                        and (
                            abs(active["battery_effective_capacity_ah"] - 0.60 * initial_capacity)
                            < 1e-8
                        ).all()
                        and (after_window["battery_reconnection_count"] > 0).any()
                    )
                elif fault_name == "compound_fault" and not active.empty:
                    fault_verified = bool(
                        fallback_samples > 0
                        and (
                            active["engine_power_cap_kw"]
                            <= phase_engine_cap * health.engine_available_fraction + 1e-6
                        ).all()
                        and (
                            active["motor_power_max_kw"]
                            <= phase_motor_cap * health.motor_available_fraction + 1e-6
                        ).all()
                        and (active["soc_target"] > 0.50).all()
                        and (abs(active["battery_available_fraction"] - 0.60) < 1e-9).all()
                    )
                speed = float(summary["minimum_speed_mps_sampled"])
                fuel_margin = float(summary["fuel_margin_kg_estimated"])
                max_pitch = float(summary["maximum_abs_pitch_deg_sampled"])
                max_bank = float(summary["maximum_abs_bank_deg_sampled"])
                row = {
                    **{
                        "battery_" + key: value
                        for key, value in summary["battery_energy_accounting"].items()
                        if isinstance(value, (int, float)) or value is None
                    },
                    "fault_case": fault_name,
                    "fault_phase": fault_phase or "",
                    "policy": policy_name,
                    "mission_hours_requested": args.mission_hours,
                    "mission_completed": bool(summary["mission_completed"]),
                    "touchdown_detected": bool(summary["touchdown_detected"]),
                    "simulation_duration_h": float(summary["simulation_duration_s"]) / 3600.0,
                    "minimum_speed_mps": speed,
                    "minimum_speed_screen_mps": minimum_speed_screen_mps,
                    "minimum_soc": float(summary["minimum_soc"]),
                    "fuel_margin_kg": fuel_margin,
                    "minimum_fuel_reserve_kg": minimum_fuel_reserve_kg,
                    "maximum_abs_pitch_deg": max_pitch,
                    "maximum_abs_bank_deg": max_bank,
                    "speed_screen_pass": speed >= minimum_speed_screen_mps,
                    "fuel_reserve_screen_pass": fuel_margin >= minimum_fuel_reserve_kg,
                    "attitude_screen_pass": max_pitch <= 25.0 and max_bank <= 35.0,
                    "fault_window_sample_count": int(len(active)),
                    "battery_reconnection_count": int(profile["battery_reconnection_count"].max()),
                    "fault_effect_applicable": fault_applicable,
                    "sensor_fallback_sample_count": fallback_samples,
                    "fault_effect_verified_in_profile": fault_verified,
                    "fault_window_engine_bound_violations": engine_violations,
                    "fault_window_motor_bound_violations": motor_violations,
                    "case_screen_pass": (
                        bool(summary["mission_completed"])
                        and (fault_verified or not fault_applicable)
                        and engine_violations == 0
                        and motor_violations == 0
                        and speed >= minimum_speed_screen_mps
                        and fuel_margin >= minimum_fuel_reserve_kg
                        and max_pitch <= 25.0
                        and max_bank <= 35.0
                    ),
                    "wall_runtime_s": elapsed,
                    "run_summary": str(case_dir / "PTDL_JSBSim全任务闭环摘要.json"),
                }
            except Exception as exc:
                elapsed = time.perf_counter() - started
                row = {
                    "fault_case": fault_name,
                    "fault_phase": fault_phase or "",
                    "policy": policy_name,
                    "mission_hours_requested": args.mission_hours,
                    "mission_completed": False,
                    "touchdown_detected": False,
                    "case_screen_pass": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "wall_runtime_s": elapsed,
                    "run_summary": str(case_dir / "PTDL_JSBSim全任务闭环摘要.json"),
                }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / "六自由度故障容错闭环比较.csv", index=False)
            print(
                f"{fault_name}/{policy_name}: "
                f"pass={row['case_screen_pass']}, touchdown={row['touchdown_detected']}, "
                f"wall={elapsed:.1f}s"
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "六自由度故障容错闭环比较.csv", index=False)
    summary = {
        "study": "matched JSBSim six-phase full-profile runs with timed dispatch-health events",
        "model_status": "estimated aircraft and propulsion assumptions; software simulation, not calibrated or flight validated",
        "mission_mode": "full; mission_hours is the total profile duration including descent/approach/flare",
        "mission_hours": args.mission_hours,
        "integration_step_s": args.integration_step_s,
        "control_period_s": args.control_period_s,
        "fault_phases": list(args.phases),
        "fault_start_phase_elapsed_s": {
            phase: 150.0 if phase == "flare" else FAULT_START_S
            for phase in args.phases
        },
        "fault_duration_s": FAULT_DURATION_S,
        "policies": list(factories),
        "fault_cases": {
            "nominal": None,
            **{
                name: {
                "engine_available_fraction": health.engine_available_fraction,
                "motor_available_fraction": health.motor_available_fraction,
                "battery_available_energy_fraction": health.battery_available_energy_fraction,
                "sensor_data_valid": health.sensor_data_valid,
            }
                for name, health in cases.items()
            },
        },
        "phase_coverage": list(args.phases),
        "screening_limits": {
            "minimum_speed_mps": minimum_speed_screen_mps,
            "minimum_fuel_reserve_kg": minimum_fuel_reserve_kg,
            "maximum_abs_pitch_deg": 25.0,
            "maximum_abs_bank_deg": 35.0,
            "basis": "estimated reference-aircraft software screening, not certification limits",
        },
        "case_count": len(frame),
        "passed_cases": int(frame["case_screen_pass"].fillna(False).sum()),
        "fault_cases_per_phase": len(cases) * len(factories),
        "comparison_csv": str(output_dir / "六自由度故障容错闭环比较.csv"),
        "limitations": [
            "Faults are injected separately into each selected mission phase; nominal baselines are run once per invocation.",
            "A four-hour full profile does not demonstrate 80-hour faulted endurance.",
            "Battery availability now scales equivalent parallel-string capacity and resistance, retains offline charge, and reconnects by charge conservation; no cell aging, equalization-current or thermal transient model.",
            "The motor is already disabled in powered long-descent, so a motor-only derating event in that phase is not an effective stressor and is marked not applicable.",
            "The neural policy remains an MPC imitation model.",
            "Results are model-internal and do not establish component fault physics, hardware performance, or flight safety.",
        ],
    }
    (output_dir / "六自由度故障容错闭环说明.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
