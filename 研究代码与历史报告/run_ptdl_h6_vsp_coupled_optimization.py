"""Run one H=6 PTDL optimization with VSPAERO-corrected flight aerodynamics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptdl_2_2_2_optimizer import run_ptdl_80h_joint_moead
from ptdl_80h_endurance_profile import calibrate_endurance_task_baseline
from ptdl_airframe_design import AirframeDesign
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
SURROGATE_PATH = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
OUTPUT = ROOT / "ptdl_moead_h6_vsp_coupled_results"
H = 6
GENERATIONS = 60
SEED = 20260950
OBJECTIVES = ("J_energy_mj", "J_stability", "J_response_s")


def pareto_indices(frame: pd.DataFrame) -> list[int]:
    feasible = frame.loc[frame["feasible"].astype(bool)]
    values = feasible[list(OBJECTIVES)].to_numpy(dtype=float)
    return [
        int(feasible.index[index])
        for index in range(len(feasible))
        if not np.any(np.all(values <= values[index], axis=1) & np.any(values < values[index], axis=1))
    ]


def design_from_row(row: pd.Series) -> AirframeDesign:
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


def write_figures(population: pd.DataFrame, pareto: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    points = ax.scatter(
        pareto["J_energy_mj"], pareto["J_stability"], c=pareto["J_response_s"],
        cmap="viridis", edgecolor="white", linewidth=0.35, s=42,
    )
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("J_response (s)")
    ax.set(xlabel="J_energy (MJ)", ylabel="J_stability", title="H=6 VSPAERO-corrected feasible Pareto set")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "h6_vsp_pareto_energy_stability.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(population["J_energy_mj"], population["vsp_corrected_oswald_efficiency"], "o", color="#1F4E79", ms=4)
    ax.set(xlabel="J_energy (MJ)", ylabel="VSPAERO-corrected Oswald efficiency", title="VSPAERO correction active in evaluated candidates")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "h6_vsp_oswald_vs_energy.png", dpi=240)
    plt.close(fig)


def main() -> None:
    if not SURROGATE_PATH.exists():
        raise FileNotFoundError("Run build_ptdl_vsp_aero_surrogate.py before the VSPAERO-coupled optimization.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    surrogate = VSPAeroSurrogate.from_json(SURROGATE_PATH)
    provider = lambda design, calibration: surrogate.corrected_aero(design, calibration)
    run_ptdl_80h_joint_moead(
        OUTPUT,
        resolution=H,
        generations=GENERATIONS,
        seed=SEED,
        aero_provider=provider,
        aero_model_label="VSPAERO_DOE_surrogate_relative_correction",
    )
    population = pd.read_csv(OUTPUT / "ptdl_80h_joint_moead_final_population.csv")
    calibration = calibrate_endurance_task_baseline().dimension_calibration
    corrections: list[dict[str, float]] = []
    for _, row in population.iterrows():
        design = design_from_row(row)
        cl_alpha, oswald = surrogate.predict(design)
        corrected = surrogate.corrected_aero(design, calibration)
        corrections.append(
            {
                "vsp_surrogate_cl_alpha_per_rad": cl_alpha,
                "vsp_surrogate_oswald_efficiency": oswald,
                "vsp_corrected_cl_alpha_per_rad": corrected.cl_alpha_per_rad,
                "vsp_corrected_oswald_efficiency": corrected.oswald_efficiency,
            }
        )
    population = pd.concat((population, pd.DataFrame(corrections)), axis=1)
    pareto = population.loc[pareto_indices(population)].copy()
    population.to_csv(OUTPUT / "h6_vsp_final_population_with_aero.csv", index=False)
    pareto.to_csv(OUTPUT / "h6_vsp_feasible_pareto_with_aero.csv", index=False)
    write_figures(population, pareto)
    metadata = {
        "study_type": "H=6 MOEA/D with VSPAERO DOE surrogate in the flight-aerodynamic evaluation path",
        "hard_constraint": "T_endurance >= 80 h",
        "moead": {"resolution_H": H, "weights": 28, "generations": GENERATIONS, "seed": SEED},
        "vspaero_doe_samples": surrogate.training_samples,
        "aero_coupling": "VSPAERO surrogate corrects CL_alpha and Oswald efficiency relative to the baseline; calibrated component-build-up CD0 remains the parasite-drag component.",
        "final_samples": len(population),
        "final_feasible_samples": int(population["feasible"].astype(bool).sum()),
        "pareto_samples": len(pareto),
    }
    (OUTPUT / "h6_vsp_coupled_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
