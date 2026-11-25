"""Compare five energy-management policies in matched 80 h closed-loop runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import (  # noqa: E402
    calibrate_endurance_task_baseline,
    evaluate_endurance_task,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402
from ptdl_hybrid_system_adapter import (  # noqa: E402
    HybridSystemAdapterConfig,
    HybridSystemPowertrain,
)
from 响应指标 import (  # noqa: E402
    RESPONSE_ALTITUDE_TOLERANCE_FRACTION,
    paired_response_metrics,
    project_response_stress_scenario,
    zero_amplitude_disturbances,
)
from 规则混动协同控制 import RuleBasedHybridCoordinator  # noqa: E402
from 智能能量管理与容错调度 import (  # noqa: E402
    FuzzyAdaptiveCoordinator,
    MPCEnergyManagementCoordinator,
    NeuralImitationCoordinator,
)
from VSPAERO气动代理模型 import VSPAeroSurrogate  # noqa: E402


DEFAULT_CANDIDATE = (
    ROOT / "运行结果" / "H6_三项风险闭环复验" / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_NEURAL_MODEL = (
    ROOT
    / "运行结果"
    / "神经网络MPC模仿策略训练测试_20261011"
    / "神经网络MPC模仿策略.json"
)
DEFAULT_SURROGATE = ROOT / "气动代理模型数据" / "气动代理模型.json"
DEFAULT_OUTPUT = ROOT / "运行结果" / "神经网络闭环策略对比_20261012"
PHASES = (
    "main_climb_to_leveloff",
    "short_leveloff_to_isr_altitude",
    "endurance_loiter",
    "long_descent",
    "approach_descent",
    "flare",
)


def _design(payload: dict[str, object]) -> AirframeDesign:
    fields = (
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
    values = payload["design"]
    return AirframeDesign(**{name: float(values[name]) for name in fields})


def _policy_factories(model_path: Path):
    return {
        "stage_fixed": lambda: None,
        "rule_soc_feedback": RuleBasedHybridCoordinator,
        "fuzzy_adaptive": FuzzyAdaptiveCoordinator,
        "mpc": MPCEnergyManagementCoordinator,
        "neural_mpc_imitation": lambda: NeuralImitationCoordinator(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--neural-model", type=Path, default=DEFAULT_NEURAL_MODEL)
    parser.add_argument("--surrogate", type=Path, default=DEFAULT_SURROGATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mission-hours", type=float, default=80.0)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    design = _design(candidate)
    engine_caps = [float(value) for value in candidate["engine_caps_kw"]]
    motor_assist = [float(value) for value in candidate["motor_assist_kw"]]
    speeds = [
        52.0,
        36.01,
        float(candidate["isr_cruise_speed_mps"]),
        42.0,
        30.0,
        21.0,
    ]
    calibration = calibrate_endurance_task_baseline()
    aero = VSPAeroSurrogate.from_json(args.surrogate).corrected_aero(
        design, calibration.dimension_calibration
    )
    disturbance = project_response_stress_scenario(8.0)[0]
    factories = _policy_factories(args.neural_model)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nominal_runs: dict[str, object] = {}
    disturbed_runs: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for policy_name, factory in factories.items():
        for condition, events in (
            ("nominal", zero_amplitude_disturbances((disturbance,))),
            ("8mps_headwind_step", (disturbance,)),
        ):
            policy = factory()
            evaluation = evaluate_endurance_task(
                design,
                engine_caps_kw=engine_caps,
                motor_assist_kw=motor_assist,
                speeds_mps=speeds,
                target_total_duration_h=args.mission_hours,
                calibration=calibration.dimension_calibration,
                assumptions=calibration.assumptions,
                aero_override=aero,
                dispatch_policy=policy,
                powertrain=HybridSystemPowertrain(
                    config=HybridSystemAdapterConfig()
                ),
                disturbances=events,
            )
            evaluation.simulation.profile.to_csv(
                output_dir / f"{policy_name}_{condition}_time_history.csv",
                index=False,
            )
            if condition == "nominal":
                nominal_runs[policy_name] = evaluation
            else:
                disturbed_runs[policy_name] = evaluation

            profile = evaluation.simulation.profile
            fuel_used = float(
                np.trapezoid(
                    profile["fuel_flow_kgps"].to_numpy(dtype=float),
                    profile["time_s"].to_numpy(dtype=float),
                )
            )
            rows.append(
                {
                    "policy": policy_name,
                    "condition": condition,
                    "feasible": bool(evaluation.result.feasible),
                    "termination_reason": evaluation.simulation.termination_reason,
                    "achieved_duration_h": evaluation.achieved_duration_h,
                    "fuel_used_kg": fuel_used,
                    "terminal_fuel_kg": evaluation.simulation.terminal_fuel_kg,
                    "minimum_soc": float(profile["soc"].min()),
                    "terminal_soc": float(profile["soc"].iloc[-1]),
                    "J_energy_mj": float(
                        evaluation.result.objectives["J_energy_mj"]
                    ),
                    "J_stability": float(
                        evaluation.result.objectives["J_stability"]
                    ),
                    "J_response_s": float(
                        evaluation.result.objectives["J_response_s"]
                    ),
                }
            )

    response_rows: list[dict[str, object]] = []
    for policy_name in factories:
        nominal = nominal_runs[policy_name]
        disturbed = disturbed_runs[policy_name]
        phase = next(
            item
            for item in disturbed.mission.phases
            if item.name == disturbance.phase_name
        )
        metrics = paired_response_metrics(
            disturbed.simulation.profile,
            nominal.simulation.profile,
            disturbance.phase_name,
            disturbance.start_time_s,
            disturbance.duration_s,
            disturbance.recovery_window_s,
            disturbance.recovery_dwell_s,
            max(
                10.0,
                RESPONSE_ALTITUDE_TOLERANCE_FRACTION
                * max(abs(phase.altitude_target_m), 100.0),
            ),
        )
        response_rows.append({"policy": policy_name, **metrics})

    summary = pd.DataFrame(rows)
    responses = pd.DataFrame(response_rows)
    summary.to_csv(output_dir / "五策略80h闭环比较.csv", index=False)
    responses.to_csv(output_dir / "迎风阶跃配对响应.csv", index=False)
    metadata = {
        "study": "matched 80 h low-order closed-loop energy-management comparison",
        "candidate": str(args.candidate.resolve()),
        "neural_model": str(args.neural_model.resolve()),
        "aerodynamic_model": "same corrected VSPAERO surrogate for every policy",
        "powertrain_model": "same Hybrid_System adapter and component assumptions",
        "mission_hours": args.mission_hours,
        "conditions": [
            {
                "type": "paired nominal baseline",
                "disturbance_amplitude_mps": 0.0,
                "integration_refinement": (
                    "retains the event timing and 2 s response-window integration "
                    "used by the disturbed run"
                ),
            },
            {
                "type": "equivalent headwind step",
                "phase": disturbance.phase_name,
                "start_after_phase_entry_s": disturbance.start_time_s,
                "duration_s": disturbance.duration_s,
                "headwind_step_mps": disturbance.headwind_step_mps,
                "response_window_s": disturbance.recovery_window_s,
                "recovery_dwell_s": disturbance.recovery_dwell_s,
            },
        ],
        "neural_policy_scope": (
            "safety-projected MLP imitation of one estimated-model MPC configuration; "
            "not an independently optimized or flight-validated policy"
        ),
        "limitations": [
            "All propulsion and aircraft inputs remain estimated assumptions.",
            "The equivalent headwind affects the longitudinal model and is not a resolved wind field.",
            "A successful 80 h simulation is software evidence, not a calibrated endurance claim.",
        ],
    }
    (output_dir / "神经网络闭环策略对比说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(responses.to_string(index=False))


if __name__ == "__main__":
    main()
