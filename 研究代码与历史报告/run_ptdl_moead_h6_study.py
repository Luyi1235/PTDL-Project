"""Repeated H=6 MOEA/D study and three-objective Pareto visualization for PTDL."""

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
OUTPUT = ROOT / "ptdl_moead_h6_study_results"
SEEDS = (20260920, 20260921, 20260922)
H = 6
GENERATIONS = 24


def pareto_indices(frame: pd.DataFrame) -> list[int]:
    feasible = frame.loc[frame["feasible"].astype(bool)]
    values = feasible[["J_energy_mj", "J_stability", "J_response_s"]].to_numpy(dtype=float)
    selected: list[int] = []
    for i in range(len(feasible)):
        is_dominated = np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))
        if not is_dominated:
            selected.append(int(feasible.index[i]))
    return selected


def write_plots(history: pd.DataFrame, pareto: pd.DataFrame) -> None:
    aggregate = history.groupby("generation", as_index=False)["feasible_ratio"].agg(["mean", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(aggregate["generation"], aggregate["mean"], marker="o", ms=3, lw=1.6, color="#1F4E79", label="Mean")
    ax.fill_between(aggregate["generation"], aggregate["min"], aggregate["max"], color="#9DC3E6", alpha=0.5, label="Seed range")
    ax.set(xlabel="Generation", ylabel="Feasible ratio", ylim=(-0.03, 1.03), title="H=6 MOEA/D feasibility convergence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "h6_moead_feasibility_convergence.png", dpi=240)
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    seed_codes = pd.Categorical(pareto["seed"], categories=sorted(pareto["seed"].unique()), ordered=True).codes
    points = ax.scatter(
        pareto["J_energy_mj"], pareto["J_stability"], pareto["J_response_s"],
        c=seed_codes, cmap="viridis", s=46, depthshade=True,
    )
    ax.set(xlabel="J_energy (MJ)", ylabel="J_stability", zlabel="J_response (s)", title="H=6 merged feasible Pareto front")
    colorbar = fig.colorbar(points, ax=ax, pad=0.08, shrink=0.72)
    colorbar.set_ticks([0, 1, 2])
    colorbar.set_ticklabels(["Seed 1", "Seed 2", "Seed 3"])
    colorbar.set_label("Random seed")
    fig.savefig(OUTPUT / "h6_moead_pareto_front_3d.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    points = ax.scatter(
        pareto["J_energy_mj"], pareto["J_stability"], c=pareto["J_response_s"], cmap="plasma", s=52
    )
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("J_response (s)")
    ax.set(xlabel="J_energy (MJ)", ylabel="J_stability", title="H=6 Pareto front projection")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "h6_moead_pareto_energy_stability.png", dpi=240)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    populations: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    for seed in SEEDS:
        run_dir = OUTPUT / f"seed_{seed}"
        run_ptdl_80h_joint_moead(run_dir, resolution=H, generations=GENERATIONS, seed=seed)
        population = pd.read_csv(run_dir / "ptdl_80h_joint_moead_final_population.csv")
        population["seed"] = seed
        populations.append(population)
        history = pd.read_csv(run_dir / "ptdl_80h_joint_moead_convergence.csv")
        history["seed"] = seed
        histories.append(history)

    merged = pd.concat(populations, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    pareto = merged.loc[pareto_indices(merged)].copy()
    merged.to_csv(OUTPUT / "h6_merged_final_population.csv", index=False)
    history.to_csv(OUTPUT / "h6_merged_convergence.csv", index=False)
    pareto.to_csv(OUTPUT / "h6_merged_feasible_pareto.csv", index=False)
    write_plots(history, pareto)
    metadata = {
        "hard_constraint": "T_endurance >= 80 h",
        "moead": {"resolution_H": H, "weights_per_seed": 28, "generations": GENERATIONS, "seeds": list(SEEDS)},
        "population_records": int(len(merged)),
        "feasible_records": int(merged["feasible"].astype(bool).sum()),
        "merged_pareto_count": int(len(pareto)),
        "response_metric": "preliminary power-reserve and command-switch response proxy",
    }
    (OUTPUT / "h6_study_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
