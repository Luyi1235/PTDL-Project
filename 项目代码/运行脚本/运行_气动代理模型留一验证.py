"""Reproduce leave-one-out error and feature-range audit for the VSPAERO surrogate."""

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

from VSPAERO气动代理模型 import FEATURE_LABELS, VSPAeroSurrogate, fit_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign


DEFAULT_DATA = ROOT / "气动代理模型数据" / "VSPAERO_DOE原始样本_15点.csv"
DEFAULT_MODEL = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE23点.json"
DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "四项统一集成_停机动态修正重跑v2_20261002"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "后续工作_气动代理留一验证_20261002"
TARGET_COLUMNS = {
    "CL_alpha_per_rad": "vspaero_cl_alpha_per_rad",
    "oswald_efficiency": "vspaero_oswald_efficiency",
}


def _designs(frame: pd.DataFrame) -> list[AirframeDesign]:
    return [
        AirframeDesign(
            **{
                label: float(row[label])
                for label in FEATURE_LABELS
            }
        )
        for _, row in frame.iterrows()
    ]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    ss_total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "mean_bias": float(np.mean(errors)),
        "r2": float(1.0 - np.sum(errors**2) / ss_total) if ss_total > 1e-12 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fit-output", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {*FEATURE_LABELS, *TARGET_COLUMNS.values()}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"DOE dataset is missing columns: {missing}")
    if len(data) < 11:
        raise ValueError("At least eleven DOE rows are required for leave-one-out refits.")

    designs = _designs(data)
    predictions = {
        label: np.empty(len(data), dtype=float)
        for label in TARGET_COLUMNS
    }
    for held_out in range(len(data)):
        training_indices = [index for index in range(len(data)) if index != held_out]
        model = fit_vsp_surrogate(
            [designs[index] for index in training_indices],
            [float(data.iloc[index][TARGET_COLUMNS["CL_alpha_per_rad"]]) for index in training_indices],
            [float(data.iloc[index][TARGET_COLUMNS["oswald_efficiency"]]) for index in training_indices],
        )
        predicted_cl, predicted_oswald = model.predict(designs[held_out])
        predictions["CL_alpha_per_rad"][held_out] = predicted_cl
        predictions["oswald_efficiency"][held_out] = predicted_oswald

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for label, target_column in TARGET_COLUMNS.items():
        actual = data[target_column].to_numpy(dtype=float)
        predicted = predictions[label]
        for index, row in data.iterrows():
            rows.append(
                {
                    "sample": row.get("sample", f"sample_{index:02d}"),
                    "quantity": label,
                    "actual": float(actual[index]),
                    "loo_predicted": float(predicted[index]),
                    "loo_error": float(predicted[index] - actual[index]),
                    "absolute_error": float(abs(predicted[index] - actual[index])),
                }
            )
        summary.append(
            {
                "quantity": label,
                "sample_count": len(data),
                **_metrics(actual, predicted),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "气动代理留一预测明细.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "气动代理留一误差统计.csv", index=False)

    if args.fit_output is not None:
        model = fit_vsp_surrogate(
            designs,
            data[TARGET_COLUMNS["CL_alpha_per_rad"]].to_numpy(dtype=float),
            data[TARGET_COLUMNS["oswald_efficiency"]].to_numpy(dtype=float),
        )
        model.to_json(args.fit_output)
    else:
        model = VSPAeroSurrogate.from_json(args.model)
    bounds = {
        label: {
            "minimum": float(data[label].min()),
            "maximum": float(data[label].max()),
        }
        for label in FEATURE_LABELS
    }
    candidate_payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    candidate_values = np.asarray(candidate_payload["decision"][:10], dtype=float)
    candidate_design = AirframeDesign(
        **{
            label: float(value)
            for label, value in zip(
                (
                    "wing_area_m2",
                    "aspect_ratio",
                    "taper_ratio",
                    "sweep_deg",
                    "tip_twist_deg",
                    "fuselage_length_m",
                    "fuselage_diameter_m",
                ),
                candidate_values[:7],
            )
        }
    )
    candidate_features = {
        label: float(getattr(candidate_design, label))
        for label in FEATURE_LABELS
    }
    feature_audit = {
        label: {
            **bounds[label],
            "candidate_value": candidate_features[label],
            "in_training_range": bounds[label]["minimum"]
            <= candidate_features[label]
            <= bounds[label]["maximum"],
        }
        for label in FEATURE_LABELS
    }
    in_range_count = sum(item["in_training_range"] for item in feature_audit.values())
    applicability = {
        "model_training_samples": model.training_samples,
        "leave_one_out_sample_count": len(data),
        "feature_bounds_are_empirical_doe_min_max": True,
        "candidate_feature_range_pass": in_range_count == len(feature_audit),
        "candidate_features_within_range_count": in_range_count,
        "candidate_feature_count": len(feature_audit),
        "feature_audit": feature_audit,
        "limitations": [
            "Feature-wise min/max checks do not prove interpolation inside the DOE convex hull.",
            "LOO metrics characterize this small VSPAERO DOE and are not a guarantee for every candidate.",
            "VSPAERO VLM does not provide viscous parasite drag; the existing calibrated drag buildup remains separate.",
        ],
    }
    (output_dir / "气动代理适用范围审计.json").write_text(
        json.dumps(applicability, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps(applicability, ensure_ascii=False, indent=2))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
