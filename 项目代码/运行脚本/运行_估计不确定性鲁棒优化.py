"""Worst-case multi-scenario PTDL joint optimization with replaceable estimates."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from MOEAD多目标联合优化 import OBJECTIVE_LABELS, run_ptdl_80h_joint_moead
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from VSPAERO气动代理模型 import FEATURE_LABELS, VSPAeroSurrogate
from 响应指标 import project_response_stress_scenario


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "动力下降模式正式重优化_20261003"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE23点.json"
DOMAIN_DATA_PATH = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_23点.csv"

ESTIMATE_SCENARIOS = (
    {
        "label": "nominal_estimate",
        "engine_rating_factor": 1.0,
        "motor_rating_factor": 1.0,
        "battery_energy_factor": 1.0,
        "engine_efficiency_factor": 1.0,
    },
    {
        "label": "conservative_minus10pct",
        "engine_rating_factor": 0.90,
        "motor_rating_factor": 0.90,
        "battery_energy_factor": 0.90,
        "engine_efficiency_factor": 0.95,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "估计不确定性鲁棒优化_多种子加预算_20261004",
    )
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261004, 20261005, 20261006])
    args = parser.parse_args()
    if args.resolution < 1 or args.generations < 1:
        raise SystemExit("resolution and generations must be positive.")

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = np.asarray(candidate["decision"], dtype=float)
    surrogate = VSPAeroSurrogate.from_json(SURROGATE_PATH)
    domain = pd.read_csv(DOMAIN_DATA_PATH)
    missing = set(FEATURE_LABELS).difference(domain.columns)
    if missing:
        raise ValueError(f"Aerodynamic DOE is missing features: {sorted(missing)}")
    bounds = {
        label: (float(domain[label].min()), float(domain[label].max()))
        for label in FEATURE_LABELS
    }
    aero_provider = lambda design, calibration: surrogate.corrected_aero(design, calibration)
    powertrain_factory = partial(
        HybridSystemPowertrain,
        config=HybridSystemAdapterConfig(),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    algorithms = (
        ("NSGA-II", "NSGA-II"),
        ("MOEA/D", "MOEA-D"),
        ("Hybrid NSGA-II/MOPSO", "Hybrid-NSGA-II-MOPSO"),
    )
    for seed in args.seeds:
        for algorithm, folder in algorithms:
            run_dir = output_dir / f"seed_{seed}" / folder
            result = run_ptdl_80h_joint_moead(
                run_dir,
                resolution=args.resolution,
                generations=args.generations,
                seed=seed,
                aero_provider=aero_provider,
                aero_model_label="VSPAERO corrected surrogate",
                powertrain_factory=powertrain_factory,
                dispatch_policy_factory=MPCEnergyManagementCoordinator,
                seed_decisions=(decision,),
                algorithm=algorithm,
                design_feature_bounds=bounds,
                disturbances=project_response_stress_scenario(),
                uncertainty_scenarios=ESTIMATE_SCENARIOS,
            )
            optimizer_run = result["run"]
            evaluations = optimizer_run.evaluations
            feasible = [item for item in evaluations if item.feasible]
            rows.append(
                {
                    "seed": seed,
                    "algorithm": algorithm,
                    "population_size": len(evaluations),
                    "generations": args.generations,
                    "final_feasible_count": len(feasible),
                    "final_feasible_ratio": len(feasible) / max(len(evaluations), 1),
                    "best_feasible_fuel_objective": (
                        min(item.objectives[0] for item in feasible) if feasible else float("nan")
                    ),
                    "best_feasible_stability_objective": (
                        min(item.objectives[1] for item in feasible) if feasible else float("nan")
                    ),
                    "best_feasible_response_objective": (
                        min(item.objectives[2] for item in feasible) if feasible else float("nan")
                    ),
                    "archive_feasible_count": sum(
                        item.feasible for item in result["feasible_archive"].evaluations
                    ),
                }
            )
            print(
                f"完成 seed={seed} {algorithm}: "
                f"final feasible={len(feasible)}/{len(evaluations)}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "鲁棒优化算法汇总.csv", index=False)
    grouped = summary.groupby("algorithm").agg(
        seed_count=("seed", "nunique"),
        mean_final_feasible_ratio=("final_feasible_ratio", "mean"),
        std_final_feasible_ratio=("final_feasible_ratio", "std"),
        mean_archive_feasible_count=("archive_feasible_count", "mean"),
        std_archive_feasible_count=("archive_feasible_count", "std"),
    )
    grouped.to_csv(output_dir / "多种子算法统计.csv")
    metadata = {
        "candidate_anchor": str(args.candidate.resolve()),
        "algorithms": ["NSGA-II", "MOEA/D", "Hybrid NSGA-II/MOPSO"],
        "objectives": list(OBJECTIVE_LABELS),
        "aggregation": "for every candidate, retain the maximum objective and constraint value across all scenarios",
        "scenarios": ESTIMATE_SCENARIOS,
        "resolution": args.resolution,
        "population_size": int((args.resolution + 1) * (args.resolution + 2) / 2),
        "generations": args.generations,
        "seeds": args.seeds,
        "run_count": len(rows),
        "aero_surrogate_training_samples": surrogate.training_samples,
        "aero_featurewise_bounds_only": bounds,
        "limitations": [
            "Scenario ranges are analyst-selected estimates, not statistical confidence intervals.",
            "This is a low-budget robust search and does not establish global optimality.",
            "The VSPAERO surrogate and low-order propulsion model remain provisional.",
        ],
    }
    (output_dir / "鲁棒优化情景与边界.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n逐种子结果：")
    print(summary.to_string(index=False))
    print("\n多种子汇总：")
    print(grouped.to_string())
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
