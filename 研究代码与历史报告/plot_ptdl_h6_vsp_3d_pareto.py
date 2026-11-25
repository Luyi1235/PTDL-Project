"""Plot the three-objective H=6 VSPAERO-corrected PTDL Pareto set."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "ptdl_moead_h6_vsp_coupled_results"
INPUT = RESULTS / "h6_vsp_feasible_pareto_with_aero.csv"
OUTPUT = RESULTS / "h6_vsp_pareto_3d.png"


def main() -> None:
    pareto = pd.read_csv(INPUT)
    figure = plt.figure(figsize=(8.6, 6.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    points = axis.scatter(
        pareto["J_energy_mj"],
        pareto["J_stability"],
        pareto["J_response_s"],
        c=pareto["J_response_s"],
        cmap="viridis",
        s=48,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
    )
    colorbar = figure.colorbar(points, ax=axis, pad=0.10, shrink=0.72)
    colorbar.set_label("J_response (s)")

    representatives = (
        ("Min. energy", pareto.loc[pareto["J_energy_mj"].idxmin()], "^", "#D55E00"),
        ("Min. stability", pareto.loc[pareto["J_stability"].idxmin()], "s", "#0072B2"),
        ("Min. response", pareto.loc[pareto["J_response_s"].idxmin()], "D", "#009E73"),
    )
    for label, row, marker, color in representatives:
        axis.scatter(
            row["J_energy_mj"], row["J_stability"], row["J_response_s"],
            marker=marker, s=90, color=color, edgecolors="black", linewidths=0.6, label=label,
        )

    axis.set(
        xlabel="J_energy (MJ)",
        ylabel="J_stability",
        zlabel="J_response (s)",
        title="H=6 VSPAERO-corrected feasible Pareto set",
    )
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    axis.zaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", frameon=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
