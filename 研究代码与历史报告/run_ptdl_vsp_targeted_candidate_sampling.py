"""Enrich the VSPAERO DOE around robust PTDL Pareto candidates.

All added samples use the original surrogate reference condition (3000 m,
36.01 m/s) so they can be appended to the 15-point training set.  The study
does not overwrite the original DOE or production surrogate; it writes an
augmented candidate model for comparison and review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from ptdl_airframe_design import AirframeDesign, design_metrics
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from ptdl_vspaero_adapter import run_vspaero_polar


ROOT = Path(__file__).resolve().parent
SOURCE_TRAINING = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_doe_training.csv"
SOURCE_SURROGATE = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
PARETO = ROOT / "ptdl_moead_h6_robust_vsp_results" / "稳健H6_MOEAD可行Pareto候选.csv"
OUTPUT = ROOT / "ptdl_vsp_targeted_candidate_results"
REFERENCE_SPEED_MPS = 36.01
REFERENCE_ALTITUDE_M = 3000.0


def _design_from_row(row: pd.Series) -> AirframeDesign:
    labels = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
        "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{label: float(row[label]) for label in labels})


def _targeted_designs() -> list[tuple[str, str, AirframeDesign]]:
    candidates = pd.read_csv(PARETO).sort_values("candidate_index").reset_index(drop=True)
    if len(candidates) < 2:
        raise RuntimeError("The robust Pareto result must contain two unique candidates before targeted VSPAERO sampling.")
    a, b = (_design_from_row(candidates.iloc[index]) for index in range(2))
    return [
        ("robust_a_center", "robust candidate A", a),
        ("robust_a_area_minus", "candidate A wing area -5%", replace(a, wing_area_m2=0.95 * a.wing_area_m2)),
        ("robust_a_area_plus", "candidate A wing area +5%", replace(a, wing_area_m2=1.05 * a.wing_area_m2)),
        ("robust_a_ar_minus", "candidate A aspect ratio -1.5", replace(a, aspect_ratio=a.aspect_ratio - 1.5)),
        ("robust_a_taper_plus", "candidate A taper ratio +0.08", replace(a, taper_ratio=min(0.70, a.taper_ratio + 0.08))),
        ("robust_b_center", "robust candidate B", b),
        ("robust_b_area_plus", "candidate B wing area +5%", replace(b, wing_area_m2=1.05 * b.wing_area_m2)),
        ("robust_b_sweep_plus", "candidate B sweep +2 deg", replace(b, sweep_deg=min(8.0, b.sweep_deg + 2.0))),
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(SOURCE_TRAINING)
    prior_surrogate = VSPAeroSurrogate.from_json(SOURCE_SURROGATE)
    records: list[dict[str, object]] = []
    targeted = _targeted_designs()
    for number, (label, description, design) in enumerate(targeted, start=1):
        folder = OUTPUT / "polars" / label
        polar_csv = folder / "vspaero_polar.csv"
        if polar_csv.exists() and (folder / "airframe.vsp3").exists():
            # Re-run is intentionally avoided only when the complete polar is
            # present.  The parser remains independent of OpenVSP state.
            from ptdl_vspaero_adapter import load_vspaero_polar
            polar_file = next(folder.glob("*.polar"), None)
            if polar_file is None:
                raise RuntimeError(f"Missing .polar file for cached targeted sample {label}.")
            polar = load_vspaero_polar(polar_file, design, folder / "airframe.vsp3", polar_csv)
            print(f"[{number}/{len(targeted)}] Reusing VSPAERO {label}", flush=True)
        else:
            print(f"[{number}/{len(targeted)}] Running VSPAERO {label}", flush=True)
            polar = run_vspaero_polar(design, folder, speed_mps=REFERENCE_SPEED_MPS, altitude_m=REFERENCE_ALTITUDE_M)
        predicted_cl, predicted_e = prior_surrogate.predict(design)
        metrics = design_metrics(design)
        records.append(
            {
                "sample": label,
                "description": description,
                **asdict(design),
                "wing_span_m": metrics.wing_span_m,
                "vspaero_cl_alpha_per_rad": polar.cl_alpha_per_rad,
                "vspaero_oswald_efficiency": polar.oswald_efficiency,
                "prior_surrogate_cl_alpha_per_rad": predicted_cl,
                "prior_surrogate_oswald_efficiency": predicted_e,
                "prior_cl_alpha_error_per_rad": predicted_cl - polar.cl_alpha_per_rad,
                "prior_oswald_error": predicted_e - polar.oswald_efficiency,
                "polar_points": len(polar.data),
                "geometry_file": str(polar.geometry_path.relative_to(ROOT)),
            }
        )
    targeted_frame = pd.DataFrame(records)
    targeted_frame.to_csv(OUTPUT / "稳健候选邻域_VSPAERO极线与预测误差.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "study_type": "targeted VSPAERO DOE enrichment around two unique robust Pareto candidates",
        "reference_condition": {"altitude_m": REFERENCE_ALTITUDE_M, "speed_mps": REFERENCE_SPEED_MPS},
        "original_doe_samples": len(original),
        "targeted_samples": len(targeted_frame),
        "next_step": "Run run_ptdl_vsp_targeted_surrogate_fit.py to train and validate the augmented surrogate in a process independent of OpenVSP.",
    }
    (OUTPUT / "运行说明.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote targeted VSPAERO polar summaries to: {OUTPUT}")


if __name__ == "__main__":
    main()
