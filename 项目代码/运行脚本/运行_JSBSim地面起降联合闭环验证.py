"""Run one saved design through ground roll, JSBSim flight, and landing roll."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PTDL联合优化高保真接口 import PTDL6DoFFullMissionOptimizerValidator
from ptdl_标定缓存 import load_dimension_calibration
from 机体参数化设计与气动阻力 import AirframeDesign, aero_from_design
from 智能能量管理与容错调度 import (
    FuzzyAdaptiveCoordinator,
    MPCEnergyManagementCoordinator,
)
from 规则混动协同控制 import RuleBasedHybridCoordinator


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "JSBSim80h先续航后着陆_NS新口径烟雾_20261012"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
PHASES = (
    "main_climb_to_leveloff",
    "short_leveloff_to_isr_altitude",
    "endurance_loiter",
    "long_descent",
    "approach_descent",
    "flare",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "JSBSim地面起降联合闭环_20261012",
    )
    parser.add_argument("--mission-hours", type=float, default=80.0)
    parser.add_argument(
        "--duration-basis",
        choices=("endurance-before-landing", "total-including-landing"),
        default="endurance-before-landing",
    )
    parser.add_argument("--integration-step-s", type=float, default=0.05)
    parser.add_argument("--control-period-s", type=float, default=2.0)
    parser.add_argument("--runway-length-m", type=float)
    parser.add_argument(
        "--controller", choices=("rule", "fuzzy", "mpc"), default="mpc"
    )
    args = parser.parse_args()
    if not args.candidate.is_file():
        parser.error(f"Candidate file not found: {args.candidate}")

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    design_data = candidate.get("design")
    if not design_data:
        labels = candidate["decision_labels"]
        values = candidate["decision"]
        mapping = dict(zip(labels, values))
        design_data = {
            name: mapping[name]
            for name in AirframeDesign.__dataclass_fields__
        }
    design = AirframeDesign(**design_data)
    calibration = load_dimension_calibration()
    aero = aero_from_design(design, calibration)

    engine_values = candidate.get("engine_caps_kw", [])
    motor_values = candidate.get("motor_assist_kw", [])
    if not engine_values or not motor_values:
        mapping = dict(zip(candidate["decision_labels"], candidate["decision"]))
        engine_values = [
            mapping[f"engine_cap_{phase}_kw"] for phase in PHASES
        ]
        motor_values = [
            mapping[f"motor_assist_{phase}_kw"] for phase in PHASES
        ]
    engine_caps = dict(zip(PHASES, map(float, engine_values)))
    motor_assist = dict(zip(PHASES, map(float, motor_values)))
    cruise_speed = float(
        candidate.get(
            "isr_cruise_speed_mps",
            candidate["decision"][
                candidate["decision_labels"].index("isr_cruise_speed_mps")
            ],
        )
    )
    controller_factory = {
        "rule": RuleBasedHybridCoordinator,
        "fuzzy": FuzzyAdaptiveCoordinator,
        "mpc": MPCEnergyManagementCoordinator,
    }[args.controller]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validator = PTDL6DoFFullMissionOptimizerValidator(
        output_dir=args.output_dir / "JSBSim候选",
        mission_hours=args.mission_hours,
        mission_duration_basis=args.duration_basis,
        integration_step_s=args.integration_step_s,
        control_period_s=args.control_period_s,
        sample_period_s=600.0,
        dispatch_policy_factory=controller_factory,
        runway_length_m=args.runway_length_m,
    )
    started = perf_counter()
    result = validator(
        design,
        aero,
        calibration,
        cruise_speed,
        float(candidate.get("cruise_altitude_m", 2400.0)),
        max(engine_caps.values()),
        engine_caps_kw=engine_caps,
        motor_assist_kw=motor_assist,
        initial_soc=float(candidate.get("initial_soc", 0.95)),
    )
    elapsed = perf_counter() - started
    report = {
        "status": "completed",
        "candidate": str(args.candidate.resolve()),
        "controller": args.controller,
        "mission_hours": args.mission_hours,
        "duration_basis": args.duration_basis,
        "wall_runtime_s": elapsed,
        "feasible": all(value <= 0.0 for value in result["constraints"].values()),
        "constraints": dict(result["constraints"]),
        "diagnostics": dict(result["diagnostics"]),
        "records": validator.records,
        "model_scope": (
            "estimated point-mass runway roll linked by speed, fuel, SOC, "
            "and mass handoff to candidate-specific JSBSim flight; JSBSim "
            "touchdown state is then passed to the estimated landing-roll model"
        ),
        "limitations": [
            "Runway roll uses a separate longitudinal point-mass model, not JSBSim gear/tire dynamics.",
            "Runway length is only constrained when --runway-length-m is provided.",
            "Ground-roll energy uses estimated component efficiencies and a power-share approximation.",
            "All aircraft, aerodynamic, propulsion, and controller values remain estimated.",
        ],
    }
    output = args.output_dir / "地面起降JSBSim候选闭环摘要.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
