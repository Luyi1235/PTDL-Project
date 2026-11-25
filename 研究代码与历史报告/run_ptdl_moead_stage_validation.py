"""Reproducible MOEA/D validation matrix for the PTDL 80 h joint problem."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptdl_2_2_2_optimizer import run_ptdl_80h_joint_moead


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_moead_stage_validation_results"
RESOLUTIONS = (6, 8, 12)
SEEDS = (20260950, 20260951, 20260952)
GENERATIONS = 60
OBJECTIVES = ("J_energy_mj", "J_stability", "J_response_s")


def pareto_indices(frame: pd.DataFrame) -> list[int]:
    feasible = frame.loc[frame["feasible"].astype(bool)]
    values = feasible[list(OBJECTIVES)].to_numpy(dtype=float)
    return [
        int(feasible.index[index])
        for index in range(len(feasible))
        if not np.any(np.all(values <= values[index], axis=1) & np.any(values < values[index], axis=1))
    ]


def weights_for_resolution(resolution: int) -> int:
    return (resolution + 1) * (resolution + 2) // 2


def write_convergence_figures(history: pd.DataFrame) -> None:
    colors = {6: "#1F4E79", 8: "#548235", 12: "#C55A11"}
    grouped = history.groupby(["resolution", "generation"], as_index=False)
    feasibility = grouped["feasible_ratio"].agg(["mean", "min", "max"]).reset_index()

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for resolution in RESOLUTIONS:
        rows = feasibility.loc[feasibility["resolution"] == resolution]
        ax.plot(rows["generation"], rows["mean"], color=colors[resolution], lw=1.8, label=f"H={resolution}")
        ax.fill_between(rows["generation"], rows["min"], rows["max"], color=colors[resolution], alpha=0.14)
    ax.set(xlabel="Generation", ylabel="Feasible ratio", ylim=(-0.02, 1.04), title="Feasibility convergence across repeated runs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "validation_feasibility_convergence.png", dpi=240)
    plt.close(fig)

    labels = ("Energy objective (MJ)", "Stability objective", "Response objective (s)")
    columns = ("best_J_energy_mj", "best_J_stability", "best_J_response_s")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), constrained_layout=True)
    for axis, column, label in zip(axes, columns, labels):
        summary = grouped[column].agg(["mean", "min", "max"]).reset_index()
        for resolution in RESOLUTIONS:
            rows = summary.loc[summary["resolution"] == resolution]
            axis.plot(rows["generation"], rows["mean"], color=colors[resolution], lw=1.5, label=f"H={resolution}")
            axis.fill_between(rows["generation"], rows["min"], rows["max"], color=colors[resolution], alpha=0.12)
        axis.set(xlabel="Generation", ylabel=label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Best feasible objective histories: mean with seed envelope")
    fig.savefig(OUTPUT / "validation_objective_convergence.png", dpi=240)
    plt.close(fig)


def write_pareto_figure(pareto: pd.DataFrame) -> None:
    colors = {6: "#1F4E79", 8: "#548235", 12: "#C55A11"}
    fig = plt.figure(figsize=(8.0, 6.1), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for resolution in RESOLUTIONS:
        rows = pareto.loc[pareto["resolution"] == resolution]
        ax.scatter(
            rows["J_energy_mj"], rows["J_stability"], rows["J_response_s"],
            color=colors[resolution], s=19, depthshade=False, label=f"H={resolution}",
        )
    ax.set(
        xlabel="J_energy (MJ)", ylabel="J_stability", zlabel="J_response (s)",
        title="Pooled non-dominated solutions from validation matrix",
    )
    ax.legend(frameon=True, loc="upper left")
    fig.savefig(OUTPUT / "validation_pooled_pareto_3d.png", dpi=240)
    plt.close(fig)


def build_summary(population: pd.DataFrame, run_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for resolution in RESOLUTIONS:
        subset = population.loc[population["resolution"] == resolution]
        pareto = subset.loc[pareto_indices(subset)]
        runs = run_summary.loc[run_summary["resolution"] == resolution]
        row: dict[str, float | int] = {
            "resolution_H": resolution,
            "weights_per_run": weights_for_resolution(resolution),
            "runs": len(runs),
            "final_samples": len(subset),
            "final_feasible_samples": int(subset["feasible"].astype(bool).sum()),
            "final_feasible_ratio": float(subset["feasible"].astype(bool).mean()),
            "pooled_pareto_count": len(pareto),
            "mean_run_pareto_count": float(runs["run_pareto_count"].mean()),
            "std_run_pareto_count": float(runs["run_pareto_count"].std(ddof=0)),
        }
        for objective in OBJECTIVES:
            row[f"{objective}_pareto_min"] = float(pareto[objective].min())
            row[f"{objective}_pareto_median"] = float(pareto[objective].median())
            row[f"{objective}_pareto_max"] = float(pareto[objective].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    populations: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    run_rows: list[dict[str, float | int]] = []

    for resolution in RESOLUTIONS:
        for seed in SEEDS:
            run_dir = OUTPUT / f"h{resolution}_seed_{seed}"
            run_ptdl_80h_joint_moead(run_dir, resolution=resolution, generations=GENERATIONS, seed=seed)
            population = pd.read_csv(run_dir / "ptdl_80h_joint_moead_final_population.csv")
            population["resolution"] = resolution
            population["seed"] = seed
            populations.append(population)
            history = pd.read_csv(run_dir / "ptdl_80h_joint_moead_convergence.csv")
            history["resolution"] = resolution
            history["seed"] = seed
            histories.append(history)
            local_pareto = population.loc[pareto_indices(population)]
            run_rows.append(
                {
                    "resolution": resolution,
                    "seed": seed,
                    "weights": weights_for_resolution(resolution),
                    "final_samples": len(population),
                    "final_feasible_ratio": float(population["feasible"].astype(bool).mean()),
                    "run_pareto_count": len(local_pareto),
                    **{f"{objective}_min": float(local_pareto[objective].min()) for objective in OBJECTIVES},
                }
            )

    merged_population = pd.concat(populations, ignore_index=True)
    merged_history = pd.concat(histories, ignore_index=True)
    run_summary = pd.DataFrame(run_rows)
    pooled_pareto = merged_population.loc[pareto_indices(merged_population)].copy()
    summary = build_summary(merged_population, run_summary)

    merged_population.to_csv(OUTPUT / "validation_final_population.csv", index=False)
    merged_history.to_csv(OUTPUT / "validation_convergence_history.csv", index=False)
    run_summary.to_csv(OUTPUT / "validation_run_summary.csv", index=False)
    pooled_pareto.to_csv(OUTPUT / "validation_pooled_pareto.csv", index=False)
    summary.to_csv(OUTPUT / "validation_resolution_summary.csv", index=False)
    write_convergence_figures(merged_history)
    write_pareto_figure(pooled_pareto)

    metadata = {
        "purpose": "Stage-report-grade MOEA/D algorithm validation on the preliminary PTDL 80 h joint model.",
        "hard_constraint": "T_endurance >= 80 h",
        "resolutions_H": list(RESOLUTIONS),
        "weights_per_run": {str(resolution): weights_for_resolution(resolution) for resolution in RESOLUTIONS},
        "generations": GENERATIONS,
        "seeds": list(SEEDS),
        "independent_runs": len(RESOLUTIONS) * len(SEEDS),
        "decision_count": 23,
        "objectives": list(OBJECTIVES),
        "model_boundary": "Preliminary low-order joint dimension-mass-control model; engine map and hybrid component parameters remain provisional pending supplied data.",
        "final_samples": int(len(merged_population)),
        "final_feasible_samples": int(merged_population["feasible"].astype(bool).sum()),
        "pooled_pareto_count": int(len(pooled_pareto)),
    }
    (OUTPUT / "validation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
