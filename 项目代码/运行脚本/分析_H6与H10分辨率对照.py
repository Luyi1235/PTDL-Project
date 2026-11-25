"""Recompute H=6/H=10 multiobjective results on one shared objective scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = ("J_energy_mj", "J_stability", "J_response_s")
REFERENCE_POINT = np.array([1.05, 1.05, 1.05], dtype=float)
DEFAULT_H6 = ROOT / "运行结果" / "H6四算法同预算_20代_8种子_131GP_20261011"
DEFAULT_H10 = ROOT / "运行结果" / "H10四算法同预算_20代_3种子_131GP_20261011"
DEFAULT_OUTPUT = ROOT / "运行结果" / "H6_H10统一尺度分辨率对照_20261012"


def hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    points = points[
        np.all(np.isfinite(points), axis=1)
        & np.all(points < reference, axis=1)
    ]
    if not len(points):
        return 0.0
    points = points[np.argsort(points[:, 0])]
    area = 0.0
    best_y = float(reference[1])
    index = 0
    while index < len(points):
        x = float(points[index, 0])
        next_index = index
        while next_index < len(points) and points[next_index, 0] == x:
            best_y = min(best_y, float(points[next_index, 1]))
            next_index += 1
        next_x = (
            float(points[next_index, 0])
            if next_index < len(points)
            else float(reference[0])
        )
        area += max(next_x - x, 0.0) * max(float(reference[1]) - best_y, 0.0)
        index = next_index
    return area


def hypervolume_3d(points: np.ndarray, reference: np.ndarray) -> float:
    points = points[
        np.all(np.isfinite(points), axis=1)
        & np.all(points < reference, axis=1)
    ]
    if not len(points):
        return 0.0
    points = points[np.argsort(points[:, 0])]
    volume = 0.0
    active: list[np.ndarray] = []
    index = 0
    while index < len(points):
        x = float(points[index, 0])
        next_index = index
        while next_index < len(points) and points[next_index, 0] == x:
            active.append(points[next_index, 1:])
            next_index += 1
        next_x = (
            float(points[next_index, 0])
            if next_index < len(points)
            else float(reference[0])
        )
        slice_area = hypervolume_2d(np.asarray(active), reference[1:])
        volume += max(next_x - x, 0.0) * slice_area
        index = next_index
    return volume


def read_runs(root: Path, resolution: int) -> list[dict[str, object]]:
    summary_path = root / "多随机种子单次运行结果.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summaries = pd.read_csv(summary_path)
    records: list[dict[str, object]] = []
    for front_path in sorted(root.rglob("*_final_population_feasible_pareto.csv")):
        frame = pd.read_csv(front_path)
        if not set(OBJECTIVES).issubset(frame.columns) or frame.empty:
            continue
        algorithm = str(frame["algorithm"].iloc[0])
        seed_dir = front_path.parent.name
        seed = int(seed_dir.removeprefix("seed_"))
        run_match = summaries.loc[
            (summaries["algorithm"] == algorithm) & (summaries["seed"] == seed)
        ]
        if len(run_match) != 1:
            raise ValueError(
                f"Expected one run summary for {algorithm}, seed {seed} in {root}."
            )
        objectives = frame.loc[:, OBJECTIVES].to_numpy(dtype=float)
        records.append(
            {
                "resolution": resolution,
                "algorithm": algorithm,
                "seed": seed,
                "objectives": objectives,
                "pareto_count": len(objectives),
                "feasible_ratio": float(run_match.iloc[0]["final_feasible_ratio"]),
                "runtime_s": float(run_match.iloc[0]["runtime_s"]),
                "evaluation_calls": int(
                    run_match.iloc[0]["objective_evaluation_calls"]
                ),
            }
        )
    if not records:
        raise ValueError(f"No final feasible Pareto fronts found under {root}.")
    return records


def summarize(records: list[dict[str, object]], ideal: np.ndarray, nadir: np.ndarray) -> pd.DataFrame:
    span = np.maximum(nadir - ideal, 1e-12)
    for record in records:
        normalized = (record["objectives"] - ideal) / span
        record["common_scale_hypervolume"] = hypervolume_3d(
            normalized, REFERENCE_POINT
        )
    rows: list[dict[str, object]] = []
    frame = pd.DataFrame(
        [
            {
                "resolution": record["resolution"],
                "algorithm": record["algorithm"],
                "seed": record["seed"],
                "pareto_count": record["pareto_count"],
                "feasible_ratio": record["feasible_ratio"],
                "runtime_s": record["runtime_s"],
                "evaluation_calls": record["evaluation_calls"],
                "common_scale_hypervolume": record["common_scale_hypervolume"],
            }
            for record in records
        ]
    )
    for (resolution, algorithm), group in frame.groupby(
        ["resolution", "algorithm"], sort=True
    ):
        rows.append(
            {
                "resolution": int(resolution),
                "algorithm": algorithm,
                "seed_count": int(len(group)),
                "mean_final_feasible_ratio": float(group["feasible_ratio"].mean()),
                "std_final_feasible_ratio": float(group["feasible_ratio"].std(ddof=0)),
                "mean_final_pareto_count": float(group["pareto_count"].mean()),
                "std_final_pareto_count": float(group["pareto_count"].std(ddof=0)),
                "mean_runtime_s": float(group["runtime_s"].mean()),
                "std_runtime_s": float(group["runtime_s"].std(ddof=0)),
                "mean_objective_evaluation_calls": float(
                    group["evaluation_calls"].mean()
                ),
                "mean_common_scale_hypervolume": float(
                    group["common_scale_hypervolume"].mean()
                ),
                "std_common_scale_hypervolume": float(
                    group["common_scale_hypervolume"].std(ddof=0)
                ),
                "min_common_scale_hypervolume": float(
                    group["common_scale_hypervolume"].min()
                ),
                "max_common_scale_hypervolume": float(
                    group["common_scale_hypervolume"].max()
                ),
            }
        )
    return pd.DataFrame(rows), frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h6-dir", type=Path, default=DEFAULT_H6)
    parser.add_argument("--h10-dir", type=Path, default=DEFAULT_H10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = read_runs(args.h6_dir, 6) + read_runs(args.h10_dir, 10)
    all_objectives = np.vstack([record["objectives"] for record in records])
    ideal = np.min(all_objectives, axis=0)
    nadir = np.max(all_objectives, axis=0)
    summary, per_seed = summarize(records, ideal, nadir)
    h6_settings = json.loads(
        (args.h6_dir / "多随机种子算法对比说明.json").read_text(encoding="utf-8")
    )
    h10_settings = json.loads(
        (args.h10_dir / "多随机种子算法对比说明.json").read_text(encoding="utf-8")
    )
    h6_nominal_budget = int(h6_settings["population_size"]) * (
        int(h6_settings["generations"]) + 1
    )
    h10_nominal_budget = int(h10_settings["population_size"]) * (
        int(h10_settings["generations"]) + 1
    )
    budget_ratio = h10_nominal_budget / max(h6_nominal_budget, 1)
    if abs(budget_ratio - 1.0) <= 0.03:
        budget_note = (
            "The nominal evaluation budgets are matched within 3%. The comparison "
            "therefore emphasizes resolution and its convergence behavior under "
            "the same evaluation budget."
        )
    else:
        budget_note = (
            f"H=10 uses about {budget_ratio:.2f} times the nominal evaluation "
            "budget of H=6, so this compares a resolution-plus-budget package, "
            "not resolution alone."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "H6_H10统一尺度算法汇总.csv", index=False)
    per_seed.to_csv(args.output_dir / "H6_H10统一尺度逐种子结果.csv", index=False)
    report = {
        "study": "H6 versus H10 paired four-algorithm resolution comparison",
        "objective_order": list(OBJECTIVES),
        "shared_ideal": ideal.tolist(),
        "shared_nadir": nadir.tolist(),
        "shared_hypervolume_reference": REFERENCE_POINT.tolist(),
        "h6_run_directory": str(args.h6_dir.resolve()),
        "h10_run_directory": str(args.h10_dir.resolve()),
        "run_settings": {
            "h6": h6_settings,
            "h10": h10_settings,
            "nominal_evaluations_per_seed": {
                "h6": h6_nominal_budget,
                "h10": h10_nominal_budget,
                "h10_to_h6_ratio": budget_ratio,
            },
        },
        "normalization_note": (
            "All final feasible Pareto points from both resolutions define one "
            "common ideal/nadir. Hypervolume values are therefore comparable "
            "across H=6 and H=10, unlike the per-run normalization in the original "
            "benchmark summaries."
        ),
        "budget_note": budget_note,
        "limitations": [
            "The H=6 and H=10 runs use different random-seed sets and seed counts.",
            "All Pareto sets are conditional on the same injected feasible seed candidate.",
            "Model and propulsion parameters are provisional estimates.",
        ],
        "summary_csv": str((args.output_dir / "H6_H10统一尺度算法汇总.csv").resolve()),
        "per_seed_csv": str((args.output_dir / "H6_H10统一尺度逐种子结果.csv").resolve()),
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "H6_H10统一尺度分辨率对照.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
