"""Dense H=8 MOEA/D study and interpolated Pareto-surface visualization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd

from ptdl_2_2_2_optimizer import run_ptdl_80h_joint_moead


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_moead_h8_surface_results"
SEEDS = (20260930, 20260931, 20260932)
H = 8
GENERATIONS = 28


def pareto_indices(frame: pd.DataFrame) -> list[int]:
    feasible = frame.loc[frame["feasible"].astype(bool)]
    values = feasible[["J_energy_mj", "J_stability", "J_response_s"]].to_numpy(dtype=float)
    return [
        int(feasible.index[i])
        for i in range(len(feasible))
        if not np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))
    ]


def write_surface_figure(pareto: pd.DataFrame) -> None:
    # J_response is visualized as z=f(J_energy, J_stability) over sampled
    # non-dominated points. The triangular faces only interpolate adjacent
    # samples; they are not additional physical simulations.
    x = pareto["J_energy_mj"].to_numpy(dtype=float)
    y = pareto["J_stability"].to_numpy(dtype=float)
    z = pareto["J_response_s"].to_numpy(dtype=float)
    tri = Triangulation(x, y)
    fig = plt.figure(figsize=(8.4, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_trisurf(tri, z, cmap="viridis", alpha=0.76, linewidth=0.18, edgecolor="#41627F")
    ax.scatter(x, y, z, color="#A61C00", s=14, depthshade=False, label="Simulated non-dominated samples")
    ax.set(
        xlabel="J_energy (MJ)", ylabel="J_stability", zlabel="J_response (s)",
        title="H=8 Pareto surface approximation",
    )
    ax.legend(loc="upper left", frameon=True)
    colorbar = fig.colorbar(surface, ax=ax, pad=0.08, shrink=0.72)
    colorbar.set_label("Interpolated J_response (s)")
    fig.savefig(OUTPUT / "h8_pareto_surface_3d.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    contours = ax.tricontourf(tri, z, levels=14, cmap="viridis")
    ax.scatter(x, y, c="#A61C00", edgecolor="white", linewidth=0.35, s=26, label="Simulated samples")
    colorbar = fig.colorbar(contours, ax=ax)
    colorbar.set_label("Interpolated J_response (s)")
    ax.set(xlabel="J_energy (MJ)", ylabel="J_stability", title="H=8 Pareto surface projection")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "h8_pareto_surface_projection.png", dpi=240)
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
    merged.to_csv(OUTPUT / "h8_merged_final_population.csv", index=False)
    history.to_csv(OUTPUT / "h8_merged_convergence.csv", index=False)
    pareto.to_csv(OUTPUT / "h8_merged_feasible_pareto.csv", index=False)
    write_surface_figure(pareto)
    metadata = {
        "hard_constraint": "T_endurance >= 80 h",
        "moead": {"resolution_H": H, "weights_per_seed": 45, "generations": GENERATIONS, "seeds": list(SEEDS)},
        "population_records": int(len(merged)),
        "feasible_records": int(merged["feasible"].astype(bool).sum()),
        "merged_pareto_count": int(len(pareto)),
        "surface_note": "Triangulated visualization of simulated non-dominated points, not additional simulation data.",
    }
    (OUTPUT / "h8_surface_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
