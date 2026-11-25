"""Generate a serial VSPAERO DOE and fit the PTDL aerodynamic correction model."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ptdl_airframe_design import BASELINE, AirframeDesign
from ptdl_vsp_aero_surrogate import FEATURE_LABELS, fit_vsp_surrogate
from ptdl_vspaero_adapter import load_vspaero_polar, run_vspaero_polar


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_vsp_aero_surrogate_results"
SEED = 20260909
DOE_SIZE = 15
BOUNDS = {
    "wing_area_m2": (22.0, 40.0),
    "aspect_ratio": (18.0, 31.0),
    "taper_ratio": (0.30, 0.70),
    "sweep_deg": (0.0, 8.0),
    "tip_twist_deg": (-3.0, 1.5),
    "fuselage_length_m": (7.5, 10.0),
    "fuselage_diameter_m": (0.65, 0.95),
}


def latin_hypercube_designs() -> list[AirframeDesign]:
    if DOE_SIZE < 11:
        raise ValueError("DOE_SIZE must leave at least ten samples for surrogate fitting.")
    rng = np.random.default_rng(SEED)
    count = DOE_SIZE - 1
    dimensions = len(FEATURE_LABELS)
    unit = np.empty((count, dimensions))
    for column in range(dimensions):
        unit[:, column] = (rng.permutation(count) + rng.uniform(size=count)) / count
    designs = [BASELINE]
    for row in unit:
        values = {
            label: lower + coordinate * (upper - lower)
            for coordinate, label, (lower, upper) in zip(row, FEATURE_LABELS, (BOUNDS[label] for label in FEATURE_LABELS))
        }
        designs.append(
            AirframeDesign(
                **values,
                propeller_diameter_m=BASELINE.propeller_diameter_m,
                payload_mass_kg=BASELINE.payload_mass_kg,
                hybrid_system_mass_kg=BASELINE.hybrid_system_mass_kg,
            )
        )
    return designs


def run_polar_with_retry(design: AirframeDesign, output_dir: Path):
    error: Exception | None = None
    for attempt in range(2):
        try:
            return run_vspaero_polar(design, output_dir)
        except Exception as caught:  # VSPAERO may leave a short-lived sidecar process on Windows.
            error = caught
            if attempt == 0:
                time.sleep(2.0)
    raise RuntimeError(f"VSPAERO failed for {output_dir.name}: {error}") from error


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    designs = latin_hypercube_designs()
    records: list[dict[str, float | int | str]] = []
    cl_targets: list[float] = []
    e_targets: list[float] = []
    for index, design in enumerate(designs):
        label = "baseline" if index == 0 else f"doe_{index:02d}"
        polar_dir = OUTPUT / "polars" / label
        polar_path = polar_dir / "airframe_DegenGeom.polar"
        geometry_path = polar_dir / "airframe.vsp3"
        if polar_path.exists() and geometry_path.exists():
            print(f"[{index + 1}/{len(designs)}] Reusing VSPAERO {label}", flush=True)
            polar = load_vspaero_polar(polar_path, design, geometry_path, polar_dir / "vspaero_polar.csv")
        else:
            print(f"[{index + 1}/{len(designs)}] VSPAERO {label}", flush=True)
            polar = run_polar_with_retry(design, polar_dir)
        cl_targets.append(polar.cl_alpha_per_rad)
        e_targets.append(polar.oswald_efficiency)
        record: dict[str, float | int | str] = {
            "sample": label,
            **asdict(design),
            "vspaero_cl_alpha_per_rad": polar.cl_alpha_per_rad,
            "vspaero_oswald_efficiency": polar.oswald_efficiency,
            "polar_points": len(polar.data),
            "geometry_file": str(polar.geometry_path.relative_to(ROOT)),
        }
        records.append(record)

    print("Fitting VSPAERO surrogate", flush=True)
    surrogate = fit_vsp_surrogate(designs, cl_targets, e_targets)
    print("Writing VSPAERO surrogate artifacts", flush=True)
    training = pd.DataFrame(records)
    predictions = np.asarray([surrogate.predict(design) for design in designs])
    training["surrogate_cl_alpha_per_rad"] = predictions[:, 0]
    training["surrogate_oswald_efficiency"] = predictions[:, 1]
    training["cl_alpha_error_per_rad"] = training["surrogate_cl_alpha_per_rad"] - training["vspaero_cl_alpha_per_rad"]
    training["oswald_error"] = training["surrogate_oswald_efficiency"] - training["vspaero_oswald_efficiency"]
    training.to_csv(OUTPUT / "vsp_aero_doe_training.csv", index=False)
    surrogate.to_json(OUTPUT / "vsp_aero_surrogate.json")
    summary = {
        "source": "OpenVSP VSPAERO serial VLM DOE",
        "doe_size": len(designs),
        "seed": SEED,
        "features": list(FEATURE_LABELS),
        "ridge_lambda": surrogate.ridge_lambda,
        "baseline_vspaero_cl_alpha_per_rad": surrogate.baseline_cl_alpha_per_rad,
        "baseline_vspaero_oswald_efficiency": surrogate.baseline_oswald_efficiency,
        "training_cl_alpha_rmse_per_rad": float(np.sqrt(np.mean(training["cl_alpha_error_per_rad"] ** 2))),
        "training_oswald_rmse": float(np.sqrt(np.mean(training["oswald_error"] ** 2))),
        "model_note": "The surrogate corrects VSPAERO lift-slope and induced-drag trends relative to the baseline. Calibrated component-build-up parasite drag remains in the flight model.",
    }
    (OUTPUT / "vsp_aero_surrogate_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
