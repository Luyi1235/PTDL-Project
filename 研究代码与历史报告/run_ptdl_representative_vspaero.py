"""Run serial VSPAERO polars for representative completed MOEA/D candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ptdl_airframe_design import AirframeDesign, design_metrics
from ptdl_vspaero_adapter import run_vspaero_polar


ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "ptdl_moead_aero_study_results"


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


def main() -> None:
    candidates = pd.read_csv(STUDY / "representative_pareto_candidates.csv")
    records: list[dict[str, float | int | str]] = []
    for _, row in candidates.iterrows():
        label = str(row["representative"])
        print(f"Running VSPAERO: {label}", flush=True)
        design = design_from_record(row)
        metrics = design_metrics(design)
        polar = run_vspaero_polar(design, STUDY / "vspaero" / label)
        records.append(
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
    pd.DataFrame(records).to_csv(STUDY / "representative_vspaero_summary.csv", index=False)
    print("VSPAERO representatives complete", flush=True)


if __name__ == "__main__":
    main()
