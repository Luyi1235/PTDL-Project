"""Compare leave-one-out surrogate errors before/after adding solver-produced DOE labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (
    FEATURE_LABELS,
    fit_kriging_vsp_surrogate,
    fit_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign


def _scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / total) if total > 1e-12 else float("nan"),
    }


def _evaluate(frame: pd.DataFrame, dataset_name: str) -> list[dict[str, object]]:
    designs = [
        AirframeDesign(**{name: float(row[name]) for name in FEATURE_LABELS})
        for _, row in frame.iterrows()
    ]
    targets = {
        "cl_alpha_per_rad": frame["vspaero_cl_alpha_per_rad"].to_numpy(dtype=float),
        "oswald_efficiency": frame["vspaero_oswald_efficiency"].to_numpy(dtype=float),
    }
    predictions = {
        model_name: {target: np.empty(len(frame), dtype=float) for target in targets}
        for model_name in ("ridge_quadratic", "kriging")
    }
    for held_out in range(len(frame)):
        training = [index for index in range(len(frame)) if index != held_out]
        train_designs = [designs[index] for index in training]
        ridge = fit_vsp_surrogate(
            train_designs,
            targets["cl_alpha_per_rad"][training],
            targets["oswald_efficiency"][training],
        )
        kriging = fit_kriging_vsp_surrogate(
            train_designs,
            targets["cl_alpha_per_rad"][training],
            targets["oswald_efficiency"][training],
        )
        ridge_prediction = ridge.predict(designs[held_out])
        kriging_prediction = kriging.predict(designs[held_out])
        for target, column in (("cl_alpha_per_rad", 0), ("oswald_efficiency", 1)):
            predictions["ridge_quadratic"][target][held_out] = ridge_prediction[column]
            predictions["kriging"][target][held_out] = kriging_prediction[column]

    records: list[dict[str, object]] = []
    for target, actual in targets.items():
        for model_name in predictions:
            records.append(
                {
                    "dataset": dataset_name,
                    "sample_count": len(frame),
                    "target": target,
                    "model": model_name,
                    **_scores(actual, predictions[model_name][target]),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-data",
        type=Path,
        default=ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_23点.csv",
    )
    parser.add_argument(
        "--new-labels",
        type=Path,
        default=ROOT / "运行结果" / "VSPAERO新增DOE实算_20261004" / "VSPAERO新增DOE实算标签.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "VSPAERO新增样本留一影响_20261004",
    )
    parser.add_argument("--openvsp-version", default="3.53.1")
    parser.add_argument("--vspaero-version", default="7.2.2")
    args = parser.parse_args()

    base = pd.read_csv(args.base_data)
    raw_new = pd.read_csv(args.new_labels)
    required_features = {*FEATURE_LABELS, "cl_alpha_per_rad", "oswald_efficiency", "polar_points"}
    missing = required_features.difference(raw_new.columns)
    if missing:
        raise ValueError(f"new label file missing required fields: {sorted(missing)}")
    usable = raw_new.loc[
        (raw_new["status"] == "solver_completed")
        & (raw_new["polar_points"] >= 3)
        & np.isfinite(raw_new["cl_alpha_per_rad"])
        & np.isfinite(raw_new["oswald_efficiency"])
    ].copy()
    new_data = usable.loc[:, FEATURE_LABELS].copy()
    new_data.insert(0, "sample", usable["sample"].astype(str).to_numpy())
    new_data["vspaero_cl_alpha_per_rad"] = usable["cl_alpha_per_rad"].to_numpy(dtype=float)
    new_data["vspaero_oswald_efficiency"] = usable["oswald_efficiency"].to_numpy(dtype=float)
    new_data["polar_points"] = usable["polar_points"].to_numpy(dtype=int)
    new_data["source_status"] = "solver_completed_requires_geometry_review"

    base_for_fit = base.loc[:, ["sample", *FEATURE_LABELS, "vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]].copy()
    new_for_fit = new_data.loc[:, ["sample", *FEATURE_LABELS, "vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]]
    combined = pd.concat([base_for_fit, new_for_fit], ignore_index=True)
    if not np.isfinite(
        combined[["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Surrogate candidate dataset contains non-finite labels.")

    records = _evaluate(base_for_fit, "original_23")
    records.extend(_evaluate(combined, "original_23_plus_new_labels"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    new_data.to_csv(output / "新增求解标签_待复核.csv", index=False)
    combined.to_csv(output / "代理留一对照候选数据_未替换正式集.csv", index=False)
    summary = pd.DataFrame(records)
    summary.to_csv(output / "原集与扩展候选留一误差.csv", index=False)
    metadata = {
        "base_dataset": str(args.base_data.resolve()),
        "new_labels": str(args.new_labels.resolve()),
        "base_sample_count": len(base),
        "solver_success_count": len(raw_new),
        "usable_new_label_count": len(new_data),
        "expanded_candidate_count": len(combined),
        "formal_training_dataset_modified": False,
        "openvsp_version": args.openvsp_version,
        "vspaero_version": args.vspaero_version,
        "mesh_warning_observed": "PGFace Invalid" in (
            ROOT.parents[1] / "tools" / "PTDL_VSPAERO_DOE.log"
        ).read_text(encoding="utf-8", errors="replace"),
        "quality_boundary": [
            "Candidates retain VSPAERO labels but require review of PGFace Invalid mesh warnings.",
            "Leave-one-out is an internal consistency check, not an independent validation dataset.",
            "Do not replace the production surrogate from this comparison alone.",
        ],
    }
    (output / "评估边界.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\n正式 23 点数据和代理模型未改动：{output}")


if __name__ == "__main__":
    main()
