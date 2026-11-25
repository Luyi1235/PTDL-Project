"""Fit and validate the enriched PTDL VSPAERO surrogate without OpenVSP.

This intentionally consumes cached polar summaries only.  Keeping this stage
separate from OpenVSP makes the DOE-to-surrogate workflow reproducible even
when a VSPAERO child process does not return cleanly on Windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ptdl_airframe_design import AirframeDesign
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate, fit_vsp_surrogate


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_doe_training.csv"
TARGETED = ROOT / "ptdl_vsp_targeted_candidate_results" / "稳健候选邻域_VSPAERO极线与预测误差.csv"
OUTPUT = ROOT / "ptdl_vsp_targeted_candidate_results"


def design_from_row(row: pd.Series) -> AirframeDesign:
    labels = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
        "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{label: float(row[label]) for label in labels})


def loo_statistics(designs: list[AirframeDesign], cl_values: np.ndarray, e_values: np.ndarray) -> pd.DataFrame:
    predictions = np.empty((len(designs), 2))
    for withheld in range(len(designs)):
        keep = [index for index in range(len(designs)) if index != withheld]
        surrogate = fit_vsp_surrogate(
            [designs[index] for index in keep], cl_values[keep], e_values[keep], require_baseline=False
        )
        predictions[withheld] = surrogate.predict(designs[withheld])
    rows = []
    for name, actual, predicted in (
        ("CL_alpha_per_rad", cl_values, predictions[:, 0]),
        ("oswald_efficiency", e_values, predictions[:, 1]),
    ):
        error = predicted - actual
        total = float(np.sum((actual - np.mean(actual)) ** 2))
        rows.append({
            "quantity": name,
            "sample_count": len(actual),
            "loo_mae": float(np.mean(np.abs(error))),
            "loo_rmse": float(np.sqrt(np.mean(error**2))),
            "loo_max_abs_error": float(np.max(np.abs(error))),
            "loo_r2": 1.0 - float(np.sum(error**2)) / total if total > 1e-12 else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    original = pd.read_csv(SOURCE)
    targeted = pd.read_csv(TARGETED)
    drop_columns = (
        "description", "wing_span_m", "prior_surrogate_cl_alpha_per_rad", "prior_surrogate_oswald_efficiency",
        "prior_cl_alpha_error_per_rad", "prior_oswald_error",
    )
    augmented = pd.concat((original, targeted.drop(columns=list(drop_columns))), ignore_index=True, sort=False)
    designs = [design_from_row(row) for _, row in augmented.iterrows()]
    cl_values = augmented["vspaero_cl_alpha_per_rad"].to_numpy(dtype=float)
    e_values = augmented["vspaero_oswald_efficiency"].to_numpy(dtype=float)
    surrogate = fit_vsp_surrogate(designs, cl_values, e_values)
    loo = loo_statistics(designs, cl_values, e_values)

    augmented.to_csv(OUTPUT / "VSPAERO扩充训练集_23点.csv", index=False, encoding="utf-8-sig")
    surrogate.to_json(OUTPUT / "VSPAERO扩充代理模型.json")
    loo.to_csv(OUTPUT / "VSPAERO扩充代理模型_留一验证统计.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "运行说明.json").write_text(json.dumps({
        "study_type": "targeted VSPAERO DOE enrichment around two unique robust Pareto candidates",
        "original_doe_samples": len(original), "targeted_samples": len(targeted), "augmented_samples": len(augmented),
        "training_model_note": "The augmented surrogate remains a review model until candidate recheck and validation are complete.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(loo.to_string(index=False))


if __name__ == "__main__":
    main()
