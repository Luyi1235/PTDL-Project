"""Compare PTDL energy-management policies in identical JSBSim six-phase cases."""

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
    MPCParameters,
    MPCEnergyManagementCoordinator,
    NeuralImitationCoordinator,
)
from PTDL六自由度参考模型 import PTDL6DoFParameters

DEFAULT_NEURAL_POLICY = (
    ROOT
    / "运行结果"
    / "神经网络MPC模仿策略训练测试_20261011"
    / "神经网络MPC模仿策略.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "PTDL_JSBSim六自由度典型工况控制策略对比_20261012",
    )
    parser.add_argument("--mission-hours", type=float, default=6.0)
    parser.add_argument("--integration-step-s", type=float, default=0.05)
    parser.add_argument("--control-period-s", type=float, default=2.0)
    parser.add_argument("--terminal-control-period-s", type=float, default=0.5)
    parser.add_argument("--sample-period-s", type=float, default=60.0)
    parser.add_argument("--neural-policy-model", type=Path, default=DEFAULT_NEURAL_POLICY)
    args = parser.parse_args()
    if args.mission_hours < 3.5:
        parser.error("--mission-hours must be at least 3.5 with the configured descent profile.")
    if not args.neural_policy_model.is_file():
        parser.error(f"Neural imitation policy file does not exist: {args.neural_policy_model}")

    runner_dir = ROOT / "运行脚本"
    if str(runner_dir) not in sys.path:
        sys.path.insert(0, str(runner_dir))
    from 运行_PTDL估计六自由度全任务闭环 import run

    policy_factories = {
        "rule": RuleBasedHybridCoordinator,
        "fuzzy": FuzzyAdaptiveCoordinator,
        "mpc": lambda: MPCEnergyManagementCoordinator(MPCParameters()),
        "neural-imitation": lambda: NeuralImitationCoordinator(args.neural_policy_model),
    }
    aircraft_reference = PTDL6DoFParameters()
    estimated_stall_speed_mps = math.sqrt(
        2.0
        * aircraft_reference.mtow_kg
        * 9.80665
        / (
            1.225
            * aircraft_reference.reference_wing_area_m2
            * 1.42
        )
    )
    minimum_speed_screen_mps = 1.10 * estimated_stall_speed_mps
    minimum_fuel_reserve_kg = 0.05 * aircraft_reference.fuel_mass_kg
    scenarios = {
        "nominal": {},
        "slow_approach": {
            "approach_descent": {"speed_mps": 30.0},
            "flare": {
                "speed_mps": max(28.5, minimum_speed_screen_mps + 1.5)
            },
        },
        "unpowered_descent": {
            "long_descent": {"engine_cap_kw": 0.0, "motor_cap_kw": 0.0},
        },
        "short_power_loss_recovery": {
            "long_descent": {"power_off_duration_s": 300.0},
        },
        "extended_power_loss_recovery": {
            "long_descent": {"power_off_duration_s": 900.0},
        },
        "long_power_loss_recovery": {
            "long_descent": {"power_off_duration_s": 1200.0},
        },
        "glide_to_low_altitude_recovery": {
            "long_descent": {"power_off_duration_s": 1300.0},
        },
        "glide_to_low_altitude_controlled_recovery": {
            "long_descent": {
                "power_off_duration_s": 1300.0,
                "engine_cap_kw": 25.0,
                "speed_mps": 30.0,
            },
        },
        "engine_degraded": {
            "main_climb_to_leveloff": {"engine_cap_kw": 90.0},
            "short_leveloff_to_isr_altitude": {"engine_cap_kw": 90.0},
            "endurance_loiter": {"engine_cap_kw": 75.0},
            "long_descent": {"engine_cap_kw": 50.0},
            "approach_descent": {"engine_cap_kw": 60.0},
            "flare": {"engine_cap_kw": 55.0},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scenario_name, phase_overrides in scenarios.items():
        for policy_name, policy_factory in policy_factories.items():
            run_dir = args.output_dir / scenario_name / policy_name
            started = time.perf_counter()
            elapsed = time.perf_counter() - started
            try:
                summary = run(
                    output_dir=run_dir,
                    integration_step_s=args.integration_step_s,
                    control_period_s=args.control_period_s,
                    terminal_control_period_s=args.terminal_control_period_s,
                    sample_period_s=args.sample_period_s,
                    mission_hours=args.mission_hours,
                    mission_mode="full",
                    pitch_kp=4.0,
                    pitch_kd=4.0,
                    phase_overrides=phase_overrides,
                    dispatch_policy=policy_factory(),
                )
                elapsed = time.perf_counter() - started
            except Exception as exc:
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "scenario": scenario_name,
                        "policy": policy_name,
                        "mission_hours_requested": args.mission_hours,
                        "mission_completed": False,
                        "touchdown_detected": False,
                        "case_screen_pass": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "wall_runtime_s": elapsed,
                        "run_summary": str(run_dir / "PTDL_JSBSim全任务闭环摘要.json"),
                    }
                )
                print(
                    f"{scenario_name}/{policy_name}: failed={type(exc).__name__}, "
                    f"wall={elapsed:.1f}s"
                )
                continue
            row: dict[str, object] = {
                "scenario": scenario_name,
                "policy": policy_name,
                "mission_hours_requested": args.mission_hours,
                "mission_completed": bool(summary["mission_completed"]),
                "touchdown_detected": bool(summary["touchdown_detected"]),
                "simulation_duration_h": float(summary["simulation_duration_s"]) / 3600.0,
                "minimum_speed_mps": summary["minimum_speed_mps_sampled"],
                "estimated_stall_speed_mps": estimated_stall_speed_mps,
                "minimum_speed_screen_mps": minimum_speed_screen_mps,
                "minimum_speed_margin_mps": (
                    float(summary["minimum_speed_mps_sampled"])
                    - minimum_speed_screen_mps
                ),
                "minimum_soc": summary["minimum_soc"],
                "fuel_used_kg": summary["fuel_used_kg_estimated"],
                "fuel_margin_kg": summary["fuel_margin_kg_estimated"],
                "minimum_fuel_reserve_kg": minimum_fuel_reserve_kg,
                "maximum_abs_pitch_deg": summary["maximum_abs_pitch_deg_sampled"],
                "maximum_abs_bank_deg": summary["maximum_abs_bank_deg_sampled"],
                "minimum_speed_screen_pass": (
                    float(summary["minimum_speed_mps_sampled"])
                    >= minimum_speed_screen_mps
                ),
                "fuel_reserve_screen_pass": (
                    float(summary["fuel_margin_kg_estimated"])
                    >= minimum_fuel_reserve_kg
                ),
                "attitude_screen_pass": (
                    float(summary["maximum_abs_pitch_deg_sampled"]) <= 25.0
                    and float(summary["maximum_abs_bank_deg_sampled"]) <= 35.0
                ),
                "case_screen_pass": (
                    bool(summary["mission_completed"])
                    and float(summary["minimum_speed_mps_sampled"])
                    >= minimum_speed_screen_mps
                    and float(summary["fuel_margin_kg_estimated"])
                    >= minimum_fuel_reserve_kg
                    and float(summary["maximum_abs_pitch_deg_sampled"]) <= 25.0
                    and float(summary["maximum_abs_bank_deg_sampled"]) <= 35.0
                ),
                "wall_runtime_s": elapsed,
                "run_summary": str(run_dir / "PTDL_JSBSim全任务闭环摘要.json"),
            }
            for phase_name, metrics in summary["phase_metrics_sampled"].items():
                for metric_name, value in metrics.items():
                    row[f"{phase_name}_{metric_name}"] = value
            rows.append(row)
            print(
                f"{scenario_name}/{policy_name}: "
                f"touchdown={summary['touchdown_detected']}, "
                f"fuel={summary['fuel_used_kg_estimated']:.1f} kg, "
                f"SOCmin={summary['minimum_soc']:.3f}, wall={elapsed:.1f}s"
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "六自由度典型工况控制器比较.csv", index=False)
    metadata = {
        "status": "estimated-parameter JSBSim software comparison; not calibrated or flight validated",
        "mission_hours": args.mission_hours,
        "integration_step_s": args.integration_step_s,
        "cruise_control_period_s": args.control_period_s,
        "terminal_control_period_s": args.terminal_control_period_s,
        "terminal_landing_grace_s": 7200.0,
        "screening_limits": {
            "basis": "estimated reference-aircraft screening, not certification limits",
            "stall_speed_estimate_mps": estimated_stall_speed_mps,
            "minimum_speed_factor": 1.10,
            "minimum_speed_screen_mps": minimum_speed_screen_mps,
            "fuel_reserve_fraction": 0.05,
            "minimum_fuel_reserve_kg": minimum_fuel_reserve_kg,
            "maximum_abs_pitch_deg": 25.0,
            "maximum_abs_bank_deg": 35.0,
            "minimum_speed_uses_sampled_trajectory": True,
        },
        "scenarios": scenarios,
        "policies": list(policy_factories),
        "phase_coverage": [
            "main_climb_to_leveloff",
            "short_leveloff_to_isr_altitude",
            "endurance_loiter",
            "long_descent_powered",
            "approach_descent",
            "flare",
        ],
        "exclusions": [
            "Takeoff and landing ground roll are candidate-level screens coupled by takeoff speed, fuel/SOC use, touchdown speed, and remaining mass; they remain separate low-order ground dynamics.",
            "Unpowered descent is represented by zero engine and motor caps during the long-descent phase; it is not asserted feasible unless the case passes the shared screens.",
            "Short power-loss recovery applies zero propulsion caps for the first 300 s of long descent, then restores the phase caps; this is a bounded recovery test, not a full unpowered glide.",
            "Extended power-loss recovery applies zero propulsion caps for the first 900 s of long descent, then restores the phase caps; this is a longer bounded glide/recovery test, not a full unpowered glide.",
            "Long power-loss recovery applies zero propulsion caps for the first 1200 s of long descent, then restores the phase caps; this tests a 20 min bounded glide segment followed by powered recovery.",
            "Glide-to-low-altitude recovery applies zero propulsion caps for the first 1300 s, then restores power near the 800 m long-descent target; approach and landing remain powered.",
            "Aircraft, propulsion, and aerodynamic values remain estimated assumptions.",
        ],
        "comparison_csv": str(args.output_dir / "六自由度典型工况控制器比较.csv"),
    }
    (args.output_dir / "六自由度典型工况控制器比较摘要.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
