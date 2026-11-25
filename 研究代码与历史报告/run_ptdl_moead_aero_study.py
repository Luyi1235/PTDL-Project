"""Reproducible MOEA/D convergence and OpenVSP/VSPAERO audit study for PTDL.

This is an algorithm-validation study, not an engineering sizing release.  It
keeps the provisional 80 h model assumptions explicit while expanding the
initial H=2, 6-generation smoke test into repeated MOEA/D runs, a merged
non-dominated set, and serial VSPAERO checks of representative geometries.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptdl_2_2_2_optimizer import run_ptdl_80h_joint_moead
from ptdl_airframe_design import AirframeDesign, design_metrics
from ptdl_vspaero_adapter import run_vspaero_polar


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "ptdl_moead_aero_study_results"
SEEDS = (20260910, 20260911, 20260912)
RESOLUTION = 4
GENERATIONS = 24


def feasible_pareto_indices(frame: pd.DataFrame) -> list[int]:
    """Return non-dominated indices among records already passing all constraints."""
    feasible = frame.loc[frame["feasible"].astype(bool)].copy()
    values = feasible[["J_energy_mj", "J_stability", "J_response_s"]].to_numpy(dtype=float)
    keep: list[int] = []
    for i in range(len(feasible)):
        dominated = np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))
        if not dominated:
            keep.append(int(feasible.index[i]))
    return keep


def select_representatives(pareto: pd.DataFrame) -> pd.DataFrame:
    """Select energy, stability, and balanced representatives without duplicates."""
    if pareto.empty:
        raise RuntimeError("No feasible candidate was available for aerodynamic verification.")
    objective_columns = ["J_energy_mj", "J_stability", "J_response_s"]
    low = pareto[objective_columns].min()
    high = pareto[objective_columns].max()
    span = (high - low).replace(0.0, 1.0)
    scored = pareto.copy()
    scored["balanced_score"] = ((scored[objective_columns] - low) / span).sum(axis=1)
    ordered = [
        ("minimum_energy", scored["J_energy_mj"].idxmin()),
        ("minimum_stability", scored["J_stability"].idxmin()),
        ("balanced", scored["balanced_score"].idxmin()),
    ]
    rows = []
    used: set[int] = set()
    for label, index in ordered:
        if int(index) in used:
            continue
        used.add(int(index))
        row = scored.loc[index].copy()
        row["representative"] = label
        rows.append(row)
    if len(rows) == 1 and len(scored) > 1:
        remaining = scored.drop(index=next(iter(used)))
        row = remaining.loc[remaining["J_response_s"].idxmin()].copy()
        row["representative"] = "minimum_response"
        rows.append(row)
    return pd.DataFrame(rows)


def design_from_record(row: pd.Series) -> AirframeDesign:
    return AirframeDesign(
        wing_area_m2=float(row["wing_area_m2"]),
        aspect_ratio=float(row["aspect_ratio"]),
        taper_ratio=float(row["taper_ratio"]),
        sweep_deg=float(row["sweep_deg"]),
        tip_twist_deg=float(row["tip_twist_deg"]),
        fuselage_length_m=float(row["fuselage_length_m"]),
        fuselage_diameter_m=float(row["fuselage_diameter_m"]),
        propeller_diameter_m=float(row["propeller_diameter_m"]),
        payload_mass_kg=float(row["payload_mass_kg"]),
        hybrid_system_mass_kg=float(row["hybrid_system_mass_kg"]),
    )


def create_plots(histories: pd.DataFrame, pareto: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)
    all_seed = histories.groupby("generation", as_index=False)["feasible_ratio"].agg(["mean", "min", "max"]).reset_index()
    ax.plot(all_seed["generation"], all_seed["mean"], color="#1F4E79", marker="o", ms=3, label="Mean")
    ax.fill_between(all_seed["generation"], all_seed["min"], all_seed["max"], color="#9DC3E6", alpha=0.5, label="Seed range")
    ax.set(xlabel="Generation", ylabel="Feasible ratio", ylim=(-0.03, 1.03), title="MOEA/D feasibility convergence")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "moead_feasibility_convergence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)
    if not pareto.empty:
        points = ax.scatter(pareto["J_energy_mj"], pareto["J_stability"], c=pareto["J_response_s"], cmap="viridis", s=48)
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label("J_response (s)")
    ax.set(xlabel="J_energy (MJ)", ylabel="J_stability", title="Merged feasible Pareto set")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT_DIR / "merged_pareto_energy_stability.png", dpi=220)
    plt.close(fig)


def run_postprocess() -> dict[str, object]:
    """Create merged Pareto, sensitivity and VSPAERO artifacts from completed runs."""
    merged_path = OUTPUT_DIR / "merged_final_population.csv"
    history_path = OUTPUT_DIR / "merged_convergence.csv"
    if not merged_path.exists() or not history_path.exists():
        raise FileNotFoundError("Merged optimization outputs are required before post-processing.")
    merged = pd.read_csv(merged_path)
    history = pd.read_csv(history_path)
    pareto = merged.loc[feasible_pareto_indices(merged)].copy()
    pareto.to_csv(OUTPUT_DIR / "merged_feasible_pareto.csv", index=False)
    create_plots(history, pareto)

    representatives = select_representatives(pareto)
    aero_rows: list[dict[str, float | str]] = []
    for _, row in representatives.iterrows():
        label = str(row["representative"])
        design = design_from_record(row)
        metrics = design_metrics(design)
        polar = run_vspaero_polar(design, OUTPUT_DIR / "vspaero" / label)
        aero_rows.append(
            {
                "representative": label,
                "seed": int(row["seed"]),
                "J_energy_mj": float(row["J_energy_mj"]),
                "J_stability": float(row["J_stability"]),
                "J_response_s": float(row["J_response_s"]),
                "wing_area_m2": design.wing_area_m2,
                "aspect_ratio": design.aspect_ratio,
                "wing_span_m": metrics.wing_span_m,
                "propeller_diameter_m": design.propeller_diameter_m,
                "vspaero_cl_alpha_per_rad": polar.cl_alpha_per_rad,
                "vspaero_oswald_efficiency": polar.oswald_efficiency,
                "polar_points": int(len(polar.data)),
                "geometry_file": str(polar.geometry_path.relative_to(ROOT)),
            }
        )
    aero = pd.DataFrame(aero_rows)
    aero.to_csv(OUTPUT_DIR / "representative_vspaero_summary.csv", index=False)
    representatives.to_csv(OUTPUT_DIR / "representative_pareto_candidates.csv", index=False)

    numeric = merged.select_dtypes(include=[np.number])
    controls = [
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m", "payload_mass_kg",
        "hybrid_system_mass_kg", "isr_cruise_speed_mps",
    ]
    sensitivity = numeric[controls + ["J_energy_mj", "J_stability", "J_response_s"]].corr(method="spearman").loc[
        controls, ["J_energy_mj", "J_stability", "J_response_s"]
    ]
    sensitivity.to_csv(OUTPUT_DIR / "spearman_sensitivity_screen.csv")

    summary = {
        "study_type": "repeated preliminary 80 h joint design-control MOEA/D with serial VSPAERO VLM verification",
        "model_boundary": "low-order flight-propulsion-control model; engine and hybrid system data remain provisional",
        "hard_constraint": "T_endurance >= 80 h",
        "moead": {"resolution_H": RESOLUTION, "weights_per_seed": 15, "generations": GENERATIONS, "seeds": list(SEEDS)},
        "population_records": int(len(merged)),
        "feasible_records": int(merged["feasible"].astype(bool).sum()),
        "merged_pareto_count": int(len(pareto)),
        "representative_vspaero_count": int(len(aero)),
        "vspaero_note": "VSPAERO is inviscid VLM and validates lift slope and induced-drag trend only; it does not replace viscous drag or CFD verification.",
    }
    (OUTPUT_DIR / "study_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_optimization() -> None:
    """Run the repeated MOEA/D experiments and write merged raw outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_frames: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    for seed in SEEDS:
        run_dir = OUTPUT_DIR / f"seed_{seed}"
        run_ptdl_80h_joint_moead(
            run_dir,
            resolution=RESOLUTION,
            generations=GENERATIONS,
            seed=seed,
        )
        final = pd.read_csv(run_dir / "ptdl_80h_joint_moead_final_population.csv")
        final["seed"] = seed
        final_frames.append(final)
        history = pd.read_csv(run_dir / "ptdl_80h_joint_moead_convergence.csv")
        history["seed"] = seed
        histories.append(history)

    merged = pd.concat(final_frames, ignore_index=True)
    merged.to_csv(OUTPUT_DIR / "merged_final_population.csv", index=False)
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(OUTPUT_DIR / "merged_convergence.csv", index=False)


def main() -> None:
    run_optimization()
    summary = run_postprocess()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true", help="Use existing merged optimizer outputs and only perform analysis plus VSPAERO.")
    parser.add_argument("--optimize-only", action="store_true", help="Run only the repeated MOEA/D experiments without plotting or VSPAERO.")
    args = parser.parse_args()
    if args.postprocess_only and args.optimize_only:
        parser.error("Choose at most one of --postprocess-only and --optimize-only.")
    if args.postprocess_only:
        print(json.dumps(run_postprocess(), ensure_ascii=False, indent=2))
    elif args.optimize_only:
        run_optimization()
        print("Repeated MOEA/D optimization complete.")
    else:
        main()
