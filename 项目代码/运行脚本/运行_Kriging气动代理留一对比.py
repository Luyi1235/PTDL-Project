"""Compare polynomial and Kriging VSPAERO surrogates using nested LOO selection."""

from __future__ import annotations

import json
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (
    FEATURE_LABELS,
    KrigingVSPAeroSurrogate,
    VSPAeroSurrogate,
    fit_kriging_vsp_surrogate,
    fit_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign


DEFAULT_DATA = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_23点.csv"
DEFAULT_OUTPUT = ROOT / "运行结果" / "Kriging气动代理留一比较_20261004"
TARGET_COLUMNS = {
    "CL_alpha_per_rad": "vspaero_cl_alpha_per_rad",
    "oswald_efficiency": "vspaero_oswald_efficiency",
}
OUTPUT_PATHS = {
    "CL_alpha_per_rad": 0,
    "oswald_efficiency": 1,
}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / total) if total > 1e-12 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--oswald-bounds",
        type=float,
        nargs=2,
        default=(0.55, 0.95),
        metavar=("LOW", "HIGH"),
        help="Prediction clipping bounds for both surrogate families.",
    )
    args = parser.parse_args()
    if args.oswald_bounds[0] >= args.oswald_bounds[1]:
        raise ValueError("--oswald-bounds must be an increasing pair.")
    data = pd.read_csv(args.data)
    required = {*FEATURE_LABELS, *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"DOE file is missing columns: {missing}")
    designs = [
        AirframeDesign(
            **{label: float(row[label]) for label in FEATURE_LABELS}
        )
        for _, row in data.iterrows()
    ]
    targets = {
        label: data[column].to_numpy(dtype=float)
        for label, column in TARGET_COLUMNS.items()
    }
    predictions = {
        model: {
            label: np.empty(len(data), dtype=float)
            for label in TARGET_COLUMNS
        }
        for model in ("ridge_quadratic", "kriging")
    }
    uncertainties = {
        label: np.empty(len(data), dtype=float)
        for label in TARGET_COLUMNS
    }
    detail_rows: list[dict[str, object]] = []
    for held_out in range(len(data)):
        training = [index for index in range(len(data)) if index != held_out]
        training_designs = [designs[index] for index in training]
        ridge = fit_vsp_surrogate(
            training_designs,
            targets["CL_alpha_per_rad"][training],
            targets["oswald_efficiency"][training],
        )
        ridge = replace(
            ridge, oswald_efficiency_bounds=tuple(args.oswald_bounds)
        )
        kriging = fit_kriging_vsp_surrogate(
            training_designs,
            targets["CL_alpha_per_rad"][training],
            targets["oswald_efficiency"][training],
        )
        kriging = replace(
            kriging, oswald_efficiency_bounds=tuple(args.oswald_bounds)
        )
        ridge_predictions = ridge.predict(designs[held_out])
        kriging_predictions = kriging.predict_with_uncertainty(designs[held_out])
        for label, index in OUTPUT_PATHS.items():
            predictions["ridge_quadratic"][label][held_out] = ridge_predictions[index]
            predictions["kriging"][label][held_out] = kriging_predictions[index]
            uncertainties[label][held_out] = kriging_predictions[index + 2]
            detail_rows.append(
                {
                    "held_out_index": held_out,
                    "quantity": label,
                    "actual": float(targets[label][held_out]),
                    "ridge_quadratic_loo": float(ridge_predictions[index]),
                    "kriging_loo": float(kriging_predictions[index]),
                    "kriging_predictive_std": float(kriging_predictions[index + 2]),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for label, actual in targets.items():
        for model_name in ("ridge_quadratic", "kriging"):
            metrics = _metrics(actual, predictions[model_name][label])
            coverage = float(
                np.mean(
                    np.abs(actual - predictions[model_name][label])
                    <= 1.645 * uncertainties[label]
                )
            ) if model_name == "kriging" else float("nan")
            summary_rows.append(
                {
                    "quantity": label,
                    "model": model_name,
                    **metrics,
                    "kriging_90pct_interval_coverage": coverage,
                }
            )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(output / "Kriging留一预测明细.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "代理模型留一误差对比.csv", index=False)
    full_model = fit_kriging_vsp_surrogate(
        designs,
        targets["CL_alpha_per_rad"],
        targets["oswald_efficiency"],
    )
    full_model = replace(
        full_model, oswald_efficiency_bounds=tuple(args.oswald_bounds)
    )
    full_model.to_json(output / "VSPAERO_Kriging模型.json")
    metadata = {
        "training_data": str(args.data.resolve()),
        "sample_count": len(data),
        "oswald_prediction_bounds": list(args.oswald_bounds),
        "cross_validation": "outer leave-one-out; Kriging scale/nugget selected by inner exact LOO on each training fold",
        "selected_full_data_length_scales": full_model.length_scales.tolist(),
        "selected_full_data_nuggets": full_model.nuggets.tolist(),
        "limitations": [
            "Small DOE; predictive uncertainty is conditional on the chosen isotropic Gaussian kernel and is not a calibrated confidence interval.",
            "Feature-wise range does not prove candidate interpolation inside the DOE convex hull.",
            "VSPAERO VLM remains an inviscid source model; parasite drag remains supplied by the separate low-order buildup.",
        ],
    }
    (output / "Kriging交叉验证说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\n结果目录：{output.resolve()}")


if __name__ == "__main__":
    main()
