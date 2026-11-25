"""Recompute representative H=6 Pareto geometries with raw OpenVSP/VSPAERO."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ptdl_airframe_design import AirframeDesign
from ptdl_vspaero_adapter import run_vspaero_polar


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "ptdl_moead_h6_vsp_coupled_results"
PARETO_PATH = RESULTS / "h6_vsp_feasible_pareto_with_aero.csv"
RAW_OUTPUT = RESULTS / "raw_vspaero_recheck"


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


def select_representatives(pareto: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "minimum_energy": pareto.loc[pareto["J_energy_mj"].idxmin()],
        "minimum_stability": pareto.loc[pareto["J_stability"].idxmin()],
        "minimum_response": pareto.loc[pareto["J_response_s"].idxmin()],
    }


def main() -> None:
    pareto = pd.read_csv(PARETO_PATH)
    RAW_OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | str]] = []
    for label, row in select_representatives(pareto).items():
        polar = run_vspaero_polar(design_from_row(row), RAW_OUTPUT / label)
        records.append(
            {
                "representative": label,
                "candidate_index": int(row["candidate_index"]),
                "J_energy_mj": float(row["J_energy_mj"]),
                "J_stability": float(row["J_stability"]),
                "J_response_s": float(row["J_response_s"]),
                "wing_area_m2": float(row["wing_area_m2"]),
                "aspect_ratio": float(row["aspect_ratio"]),
                "sweep_deg": float(row["sweep_deg"]),
                "surrogate_cl_alpha_per_rad": float(row["vsp_surrogate_cl_alpha_per_rad"]),
                "raw_vspaero_cl_alpha_per_rad": polar.cl_alpha_per_rad,
                "cl_alpha_relative_error_pct": 100.0 * (float(row["vsp_surrogate_cl_alpha_per_rad"]) / polar.cl_alpha_per_rad - 1.0),
                "surrogate_oswald_efficiency": float(row["vsp_surrogate_oswald_efficiency"]),
                "raw_vspaero_oswald_efficiency": polar.oswald_efficiency,
                "oswald_relative_error_pct": 100.0 * (float(row["vsp_surrogate_oswald_efficiency"]) / polar.oswald_efficiency - 1.0),
            }
        )
    result = pd.DataFrame(records)
    path = RESULTS / "h6_vsp_representative_raw_vspaero.csv"
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(f"Saved raw VSPAERO verification: {path}")


if __name__ == "__main__":
    main()
