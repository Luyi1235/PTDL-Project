"""Compare health-supervised energy managers under phase-local injected faults."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 飞推控闭环物理模型 import DispatchObservation, MissionFaultEvent
from 规则混动协同控制 import RuleBasedHybridCoordinator
from 智能能量管理与容错调度 import (
    FuzzyAdaptiveCoordinator,
    HealthState,
    MPCEnergyManagementCoordinator,
    NEURAL_PHASES,
    NeuralImitationCoordinator,
    ScheduledFaultCoordinator,
)

DEFAULT_NEURAL_MODEL = (
    ROOT / "运行结果" / "神经网络MPC模仿策略训练测试_20261011"
    / "神经网络MPC模仿策略.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "阶段化容错调度比较_20261012"
FAULT_CASES = {
    "engine_derating": HealthState(engine_available_fraction=0.60),
    "motor_derating": HealthState(motor_available_fraction=0.50),
    "battery_energy_derating": HealthState(battery_available_energy_fraction=0.50),
    "sensor_failure": HealthState(sensor_data_valid=False),
    "compound_fault": HealthState(
        engine_available_fraction=0.60,
        motor_available_fraction=0.50,
        battery_available_energy_fraction=0.50,
        sensor_data_valid=False,
    ),
}
PHASE_DEMAND_SCALE = {
    "main_climb_to_leveloff": 0.95,
    "short_leveloff_to_isr_altitude": 0.80,
    "endurance_loiter": 0.55,
    "long_descent": 0.35,
    "approach_descent": 0.45,
    "flare": 0.30,
}


def _policy_factories(neural_model: Path):
    return {
        "rule": RuleBasedHybridCoordinator,
        "fuzzy": FuzzyAdaptiveCoordinator,
        "mpc": MPCEnergyManagementCoordinator,
        "neural_mpc_imitation": lambda: NeuralImitationCoordinator(neural_model),
    }


def _observation(phase: str, elapsed_s: float) -> DispatchObservation:
    phase_position = elapsed_s / 78.0
    demand = 120.0 * PHASE_DEMAND_SCALE[phase] * (
        0.78 + 0.12 * math.sin(2.0 * math.pi * phase_position)
    )
    soc = 0.58 + 0.12 * math.sin(2.0 * math.pi * phase_position + 0.4)
    speed_error = 1.2 * math.sin(2.0 * math.pi * phase_position + 1.0)
    altitude_error = 180.0 * math.sin(2.0 * math.pi * phase_position)
    return DispatchObservation(
        phase_name=phase,
        time_step_s=2.0,
        soc=float(soc),
        altitude_error_m=float(altitude_error),
        speed_error_mps=float(speed_error),
        required_propulsion_kw=float(demand),
        engine_power_cap_kw=120.0,
        motor_power_min_kw=-30.0,
        motor_power_max_kw=80.0,
        battery_capacity_mj=360.0,
        battery_soc_min=0.15,
        battery_soc_max=1.0,
        battery_soc_reserve=0.20,
        engine_power_rate_limit_kwps=10.0,
        phase_elapsed_s=elapsed_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-model", type=Path, default=DEFAULT_NEURAL_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase-duration-s", type=float, default=80.0)
    parser.add_argument("--sample-period-s", type=float, default=2.0)
    args = parser.parse_args()
    if not args.neural_model.is_file():
        parser.error(f"Neural MPC-imitation model not found: {args.neural_model}")
    if args.phase_duration_s <= 50.0 or args.sample_period_s <= 0.0:
        parser.error("Phase duration must exceed the 20 s fault window; sample period must be positive.")

    factories = _policy_factories(args.neural_model)
    rows: list[dict[str, object]] = []
    phase_starts = 30.0
    fault_duration = 20.0
    step_count = int(math.ceil(args.phase_duration_s / args.sample_period_s))
    for phase in NEURAL_PHASES:
        for fault_name, health in FAULT_CASES.items():
            event = MissionFaultEvent(
                phase_name=phase,
                start_phase_elapsed_s=phase_starts,
                duration_s=fault_duration,
                engine_available_fraction=health.engine_available_fraction,
                motor_available_fraction=health.motor_available_fraction,
                battery_available_energy_fraction=health.battery_available_energy_fraction,
                sensor_data_valid=health.sensor_data_valid,
            )
            for policy_name, factory in factories.items():
                base = factory()
                base.reset()
                policy = ScheduledFaultCoordinator(base, (event,))
                for sample_index in range(step_count):
                    elapsed_s = sample_index * args.sample_period_s
                    observation = _observation(phase, elapsed_s)
                    dispatch = policy.dispatch(observation)
                    active = phase_starts <= elapsed_s < phase_starts + fault_duration
                    active_health = health if active else HealthState()
                    engine_cap = (
                        observation.engine_power_cap_kw
                        * active_health.engine_available_fraction
                    )
                    motor_min = (
                        observation.motor_power_min_kw
                        * active_health.motor_available_fraction
                    )
                    motor_max = (
                        observation.motor_power_max_kw
                        * active_health.motor_available_fraction
                    )
                    motor_balance = (
                        observation.required_propulsion_kw - dispatch.engine_target_kw
                    )
                    rows.append(
                        {
                            "phase": phase,
                            "fault": fault_name,
                            "policy": policy_name,
                            "elapsed_s": elapsed_s,
                            "fault_active": active,
                            "mode": dispatch.mode,
                            "required_propulsion_kw": observation.required_propulsion_kw,
                            "engine_target_kw": dispatch.engine_target_kw,
                            "engine_cap_kw": engine_cap,
                            "motor_balance_kw": motor_balance,
                            "motor_min_kw": motor_min,
                            "motor_max_kw": motor_max,
                            "engine_bound_violation": (
                                dispatch.engine_target_kw < -1e-8
                                or dispatch.engine_target_kw > engine_cap + 1e-8
                            ),
                            "motor_bound_violation": (
                                motor_balance < motor_min - 1e-8
                                or motor_balance > motor_max + 1e-8
                            ),
                            "power_balance_feasible": (
                                motor_min - 1e-8
                                <= motor_balance
                                <= motor_max + 1e-8
                            ),
                            "sensor_fallback": dispatch.mode == "sensor_fallback",
                        }
                    )

    frame = pd.DataFrame(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "阶段化容错调度逐步比较.csv", index=False)
    summary = (
        frame.groupby(["phase", "fault", "policy"], as_index=False)
        .agg(
            samples=("elapsed_s", "size"),
            fault_active_samples=("fault_active", "sum"),
            sensor_fallback_samples=("sensor_fallback", "sum"),
            engine_bound_violations=("engine_bound_violation", "sum"),
            motor_bound_violations=("motor_bound_violation", "sum"),
            power_balance_feasible_ratio=("power_balance_feasible", "mean"),
        )
    )
    summary.to_csv(output_dir / "阶段化容错调度汇总.csv", index=False)
    controller_summary = (
        frame.groupby(["fault", "policy"], as_index=False)
        .agg(
            phase_cases=("phase", "nunique"),
            engine_bound_violations=("engine_bound_violation", "sum"),
            motor_bound_violations=("motor_bound_violation", "sum"),
            power_balance_feasible_ratio=("power_balance_feasible", "mean"),
            sensor_fallback_samples=("sensor_fallback", "sum"),
        )
    )
    controller_summary.to_csv(
        output_dir / "阶段化容错策略总体比较.csv", index=False
    )
    metadata = {
        "study": "phase-stratified, deterministic supervisory fault-injection comparison",
        "model_status": "software-level dispatch study with estimated ratings; not component damage or flight validation",
        "policies": list(factories),
        "phases": list(NEURAL_PHASES),
        "fault_cases": {
            name: {
                "engine_available_fraction": health.engine_available_fraction,
                "motor_available_fraction": health.motor_available_fraction,
                "battery_available_energy_fraction": health.battery_available_energy_fraction,
                "sensor_data_valid": health.sensor_data_valid,
            }
            for name, health in FAULT_CASES.items()
        },
        "fault_window": {
            "phase_elapsed_start_s": phase_starts,
            "duration_s": fault_duration,
        },
        "phase_duration_s": args.phase_duration_s,
        "sample_period_s": args.sample_period_s,
        "case_count": len(NEURAL_PHASES) * len(FAULT_CASES) * len(factories),
        "dispatch_samples": int(len(frame)),
        "safety_projection_summary": {
            "engine_bound_violations": int(frame["engine_bound_violation"].sum()),
            "motor_bound_violations": int(frame["motor_bound_violation"].sum()),
            "power_balance_feasible_ratio": float(
                frame["power_balance_feasible"].mean()
            ),
        },
        "battery_derating_semantics": (
            "The current health supervisor raises SOC reserve targets; battery plant capacity "
            "is not physically reduced by this availability fraction."
        ),
        "interpretation": (
            "Power-balance infeasibility denotes a demand outside available engine/motor limits; "
            "it is reported separately from command-bound violations."
        ),
        "outputs": {
            "stepwise_csv": str(output_dir / "阶段化容错调度逐步比较.csv"),
            "summary_csv": str(output_dir / "阶段化容错调度汇总.csv"),
            "controller_summary_csv": str(
                output_dir / "阶段化容错策略总体比较.csv"
            ),
        },
    }
    (output_dir / "阶段化容错调度比较说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(controller_summary.to_string(index=False))
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
