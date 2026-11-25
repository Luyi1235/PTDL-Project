"""Paired multi-seed NSGA-II/MOEA-D comparison on the component-backed PTDL case."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from MOEAD多目标联合优化 import OBJECTIVE_LABELS, run_ptdl_80h_joint_moead
from 八十小时续航任务模型 import calibrate_endurance_task_baseline
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 响应指标 import project_response_stress_scenario


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "H6_三项风险闭环复验"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "H6_非标定扩展验证" / "多随机种子算法对比"
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
DEFAULT_AERO_DOMAIN_DATA = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_131点_Ew_物理选点_网格修复_自适应补样_20261011.csv"
DEFAULT_SEEDS = (20260929, 20260930, 20261001)


def _hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    points = points[np.all(points < reference, axis=1)]
    if not len(points):
        return 0.0
    y_breaks = sorted(set(points[:, 0].tolist()))
    area = 0.0
    for index, y in enumerate(y_breaks):
        next_y = y_breaks[index + 1] if index + 1 < len(y_breaks) else reference[0]
        active = points[points[:, 0] <= y]
        best_z = float(np.min(active[:, 1]))
        area += max(0.0, next_y - y) * max(0.0, reference[1] - best_z)
    return float(area)


def _hypervolume_3d(points: np.ndarray, reference: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0.0
    points = np.unique(points[np.all(points < reference, axis=1)], axis=0)
    if not len(points):
        return 0.0
    x_breaks = sorted(set(points[:, 0].tolist()))
    volume = 0.0
    for index, x in enumerate(x_breaks):
        next_x = x_breaks[index + 1] if index + 1 < len(x_breaks) else reference[0]
        active = points[points[:, 0] <= x, 1:]
        volume += max(0.0, next_x - x) * _hypervolume_2d(active, reference[1:])
    return float(volume)


def _normalise(points: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    span = np.maximum(nadir - ideal, 1e-12)
    return np.clip((points - ideal) / span, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在相同模型、同种群规模和相同种子下重复比较 NSGA-II 与 MOEA/D。"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surrogate", type=Path, default=SURROGATE_PATH)
    parser.add_argument("--aero-domain-data", type=Path, default=DEFAULT_AERO_DOMAIN_DATA)
    parser.add_argument("--resolution", type=int, default=3)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--headwind-step-mps",
        type=float,
        default=8.0,
        help="用于响应目标的配对迎风阶跃压力强度。",
    )
    args = parser.parse_args()
    if args.resolution < 1 or args.generations < 1 or not args.seeds:
        raise SystemExit("resolution、generations 和 seeds 必须为正数/非空。")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = np.asarray(candidate["decision"], dtype=float)
    calibration = calibrate_endurance_task_baseline()
    surrogate = load_vsp_surrogate(args.surrogate)
    domain_frame = pd.read_csv(args.aero_domain_data)
    missing_features = set(FEATURE_LABELS).difference(domain_frame.columns)
    if missing_features:
        raise ValueError(f"Aerodynamic DOE is missing features: {sorted(missing_features)}")
    design_feature_bounds = {
        label: (float(domain_frame[label].min()), float(domain_frame[label].max()))
        for label in FEATURE_LABELS
    }
    aero_provider = lambda design, active_calibration: surrogate.corrected_aero(
        design, active_calibration
    )
    powertrain_factory = partial(
        HybridSystemPowertrain, config=HybridSystemAdapterConfig()
    )
    algorithms = (
        ("NSGA-II", "NSGA-II"),
        ("MOEA/D", "MOEA-D"),
        ("MOPSO", "MOPSO"),
        ("Hybrid NSGA-II/MOPSO", "Hybrid-NSGA-II-MOPSO"),
    )
    run_rows: list[dict[str, object]] = []
    convergence_frames: list[pd.DataFrame] = []
    generation_frames: list[pd.DataFrame] = []

    for algorithm, folder in algorithms:
        for seed in args.seeds:
            run_dir = output_dir / folder / f"seed_{seed}"
            result = run_ptdl_80h_joint_moead(
                run_dir,
                resolution=args.resolution,
                generations=args.generations,
                seed=seed,
                aero_provider=aero_provider,
                aero_model_label="VSPAERO-corrected PTDL surrogate",
                powertrain_factory=powertrain_factory,
                dispatch_policy_factory=MPCEnergyManagementCoordinator,
                seed_decisions=[decision],
                algorithm=algorithm,
                design_feature_bounds=design_feature_bounds,
                disturbances=project_response_stress_scenario(
                    args.headwind_step_mps
                ),
            )
            prefix = {
                "NSGA-II": "ptdl_80h_joint_nsga_ii",
                "MOEA/D": "ptdl_80h_joint_moead",
                "MOPSO": "ptdl_80h_joint_mopso",
                "Hybrid NSGA-II/MOPSO": "ptdl_80h_joint_hybrid_nsga_ii_mopso",
            }[algorithm]
            metadata = json.loads(
                (run_dir / f"{prefix}_metadata.json").read_text(encoding="utf-8")
            )
            archive_frame = pd.read_csv(
                run_dir / f"{prefix}_generation_feasible_pareto.csv"
            ).copy()
            archive_frame["algorithm"] = algorithm
            archive_frame["seed"] = seed
            generation_frames.append(archive_frame)
            history = pd.read_csv(run_dir / f"{prefix}_convergence.csv")
            history["algorithm"] = algorithm
            history["seed"] = seed
            convergence_frames.append(history)
            run = result["run"]
            final_feasible = [item for item in run.evaluations if item.feasible]
            run_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "resolution": args.resolution,
                    "population_size": int(
                        (args.resolution + 1) * (args.resolution + 2) / 2
                    ),
                    "generations": args.generations,
                    "final_feasible_count": len(final_feasible),
                    "final_feasible_ratio": len(final_feasible) / len(run.evaluations),
                    "final_population_pareto_count": metadata["final_population_pareto_count"],
                    "historical_feasible_pareto_count": metadata["feasible_archive_count"],
                    "unique_evaluated_candidate_count": metadata[
                        "evaluated_candidate_count"
                    ],
                    "objective_evaluation_calls": len(
                        run.evaluated_archive or run.evaluations
                    ),
                    "evaluations_per_nominal_budget": (
                        len(run.evaluated_archive or run.evaluations)
                        / (
                            (args.generations + 1)
                            * (args.resolution + 1)
                            * (args.resolution + 2)
                            / 2
                        )
                    ),
                    "runtime_s": metadata["optimizer"]["runtime_s"],
                    **{
                        f"historical_pareto_min_{label}": float(
                            pd.read_csv(run_dir / f"{prefix}_feasible_pareto.csv")[label].min()
                        )
                        for label in OBJECTIVE_LABELS
                    },
                }
            )
            pd.DataFrame(run_rows).to_csv(
                output_dir / "多随机种子单次运行结果.csv", index=False
            )
            print(f"完成 {algorithm} seed={seed}")

    runs_frame = pd.DataFrame(run_rows)
    convergence = pd.concat(convergence_frames, ignore_index=True)
    generations = pd.concat(generation_frames, ignore_index=True)
    all_objectives = generations[list(OBJECTIVE_LABELS)].to_numpy(float)
    ideal = np.min(all_objectives, axis=0)
    nadir = np.max(all_objectives, axis=0)
    reference = np.full(len(OBJECTIVE_LABELS), 1.05)
    convergence_rows: list[dict[str, object]] = []
    for (algorithm, seed), group in generations.groupby(["algorithm", "seed"]):
        for generation, front in group.groupby("generation"):
            points = _normalise(front[list(OBJECTIVE_LABELS)].to_numpy(float), ideal, nadir)
            convergence_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "generation": int(generation),
                    "feasible_pareto_count": len(front),
                    "shared_normalized_hypervolume": _hypervolume_3d(points, reference),
                }
            )
    generation_metrics = pd.DataFrame(convergence_rows)
    statistics = (
        runs_frame.groupby("algorithm")
        .agg(
            seed_count=("seed", "count"),
            final_feasible_ratio_mean=("final_feasible_ratio", "mean"),
            final_feasible_ratio_std=("final_feasible_ratio", "std"),
            historical_pareto_count_mean=("historical_feasible_pareto_count", "mean"),
            historical_pareto_count_std=("historical_feasible_pareto_count", "std"),
            runtime_s_mean=("runtime_s", "mean"),
            runtime_s_std=("runtime_s", "std"),
        )
        .reset_index()
    )
    hv_summary = generation_metrics.loc[
        generation_metrics["generation"] == args.generations
    ].groupby("algorithm")["shared_normalized_hypervolume"].agg(["mean", "std"]).reset_index()
    hv_summary.rename(
        columns={
            "mean": "final_shared_hypervolume_mean",
            "std": "final_shared_hypervolume_std",
        },
        inplace=True,
    )
    statistics = statistics.merge(hv_summary, on="algorithm", how="left")
    runs_frame.to_csv(output_dir / "多随机种子单次运行结果.csv", index=False)
    statistics.to_csv(output_dir / "算法多随机种子统计.csv", index=False)
    convergence.to_csv(output_dir / "多随机种子逐代收敛原始值.csv", index=False)
    generation_metrics.to_csv(output_dir / "多随机种子逐代Pareto超体积.csv", index=False)
    metadata = {
        "study": "paired multi-seed algorithm benchmark on the component-backed PTDL H=6 problem with paired-response wind-step objective",
        "candidate_anchor": str(args.candidate.resolve()),
        "aero_model": str(args.surrogate.resolve()),
        "aero_training_samples": surrogate.training_samples,
        "aero_featurewise_design_bounds": design_feature_bounds,
        "aero_domain_caveat": "featurewise range limits do not establish convex-hull interpolation",
        "algorithms": [algorithm for algorithm, _ in algorithms],
        "seeds": args.seeds,
        "resolution": args.resolution,
        "population_size": int((args.resolution + 1) * (args.resolution + 2) / 2),
        "generations": args.generations,
        "same_seed_initial_population": True,
        "budget_basis": (
            "same population size and generation count; each implementation performs "
            "one population-sized initialization plus one population-sized evaluation "
            "batch per generation; objective_evaluation_calls records actual calls"
        ),
        "response_stress_scenario": {
            "phase": "endurance_loiter",
            "start_after_phase_entry_s": 3600.0,
            "duration_s": 60.0,
            "headwind_step_mps": args.headwind_step_mps,
            "response_step_s": 2.0,
            "recovery_window_s": 900.0,
            "recovery_dwell_s": 30.0,
            "status": "low-order stress-test assumption, not measured wind data",
        },
        "shared_hypervolume_normalization": {
            "ideal": ideal.tolist(),
            "nadir": nadir.tolist(),
            "reference_point": reference.tolist(),
        },
        "limitations": [
            "Exploratory repeated-run comparison at a smaller budget than the formal H=6 delivery search.",
            "A single prescribed feasible candidate is injected into every initial population.",
            "All component ratings and efficiencies remain provisional; this benchmark does not validate hardware.",
            "MOPSO is a project prototype and has not been benchmarked against a validated optimization library.",
        ],
    }
    (output_dir / "多随机种子算法对比说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(statistics.to_string(index=False))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
