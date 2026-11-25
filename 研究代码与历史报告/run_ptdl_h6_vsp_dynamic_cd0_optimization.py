"""Run H=6 MOEA/D on the closed-loop PTDL model with dynamic component CD0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ptdl_2_2_2_optimizer import run_ptdl_80h_joint_moead
from ptdl_80h_endurance_profile import calibrate_endurance_task_baseline
from ptdl_airframe_design import AirframeDesign, aircraft_from_design
from ptdl_flight_propulsion_control import isa_density, parasite_drag_coefficient
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
SURROGATE_PATH = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
OUTPUT = ROOT / "ptdl_moead_h6_vsp_dynamic_cd0_results"
H = 6
GENERATIONS = 60
SEED = 20260960
OBJECTIVES = ("J_energy_mj", "J_stability", "J_response_s")


def design_from_row(row: pd.Series) -> AirframeDesign:
    labels = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
        "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{label: float(row[label]) for label in labels})


def pareto_indices(frame: pd.DataFrame) -> list[int]:
    feasible = frame.loc[frame["feasible"].astype(bool)]
    values = feasible[list(OBJECTIVES)].to_numpy(dtype=float)
    return [
        int(index)
        for local_index, index in enumerate(feasible.index)
        if not np.any(np.all(values <= values[local_index], axis=1) & np.any(values < values[local_index], axis=1))
    ]


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
        aero_model_label="VSPAERO_CLalpha_e_surrogate_plus_dynamic_component_CD0",
    )

    population = pd.read_csv(OUTPUT / "ptdl_80h_joint_moead_final_population.csv")
    calibration = calibrate_endurance_task_baseline().dimension_calibration
    aero_records: list[dict[str, float]] = []
    loiter_altitude_m = 6000.0
    for _, row in population.iterrows():
        design = design_from_row(row)
        cl_alpha, oswald = surrogate.predict(design)
        corrected = surrogate.corrected_aero(design, calibration)
        aircraft = aircraft_from_design(design, calibration, aero_override=corrected)
        loiter_cd0 = parasite_drag_coefficient(
            aircraft,
            isa_density(loiter_altitude_m),
            float(row["isr_cruise_speed_mps"]),
        )
        aero_records.append(
            {
                "vsp_surrogate_cl_alpha_per_rad": cl_alpha,
                "vsp_surrogate_oswald_efficiency": oswald,
                "vsp_corrected_cl_alpha_per_rad": corrected.cl_alpha_per_rad,
                "vsp_corrected_oswald_efficiency": corrected.oswald_efficiency,
                "cd0_reference": corrected.cd0,
                "cd0_loiter_dynamic": loiter_cd0,
            }
        )
    population = pd.concat((population, pd.DataFrame(aero_records)), axis=1)
    pareto = population.loc[pareto_indices(population)].copy()
    population.to_csv(OUTPUT / "h6_dynamic_cd0_final_population_with_aero.csv", index=False)
    pareto.to_csv(OUTPUT / "h6_dynamic_cd0_feasible_pareto_with_aero.csv", index=False)

    metadata = {
        "study_type": "H=6 MOEA/D with VSPAERO correction and dynamic component-build-up CD0",
        "hard_constraint": "T_endurance >= 80 h",
        "moead": {"resolution_H": H, "weights": 28, "generations": GENERATIONS, "seed": SEED},
        "vspaero_doe_samples": surrogate.training_samples,
        "aero_coupling": "VSPAERO surrogate corrects CL_alpha and e; CD0 is a ULTRA-calibrated wing-fuselage-tail-installation build-up updated at each flight step by Reynolds number.",
        "final_samples": len(population),
        "final_feasible_samples": int(population["feasible"].astype(bool).sum()),
        "pareto_samples": len(pareto),
    }
    (OUTPUT / "h6_dynamic_cd0_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
