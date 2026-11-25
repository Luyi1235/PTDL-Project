"""Refit versioned Ew surrogates with prediction limits matched to the new labels."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (  # noqa: E402
    FEATURE_LABELS,
    OSWALD_EFFICIENCY_DEFINITION,
    fit_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402


DATA_DIR = ROOT / "气动代理模型数据"
DATA_VERSION = "Ew_物理选点_网格修复_20261006"
MODEL_VERSION = f"{DATA_VERSION}_边界适配"
OUTPUTS = {
    40: (
        DATA_DIR / f"VSPAERO_DOE基准样本_40点_{DATA_VERSION}.csv",
        DATA_DIR / f"气动代理模型_基准DOE40点_{MODEL_VERSION}.json",
    ),
    100: (
        DATA_DIR / f"VSPAERO_DOE扩充样本_100点_{DATA_VERSION}.csv",
        DATA_DIR / f"气动代理模型_扩充DOE100点_{MODEL_VERSION}.json",
    ),
}
VALIDATION = DATA_DIR / f"VSPAERO_DOE代理模型留一验证_{MODEL_VERSION}.json"
OUTPUT_BOUNDS = (0.45, 1.00)
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def _fit(frame: pd.DataFrame):
    designs = [
        AirframeDesign(**{name: float(row[name]) for name in FEATURE_LABELS})
        for _, row in frame.iterrows()
    ]
    return replace(
        fit_vsp_surrogate(
            designs,
            frame[TARGETS[0]].to_numpy(float),
            frame[TARGETS[1]].to_numpy(float),
        ),
        oswald_efficiency_bounds=OUTPUT_BOUNDS,
    )


def _loo(frame: pd.DataFrame) -> dict:
    actual = frame.loc[:, TARGETS].to_numpy(float)
    predictions = np.empty_like(actual)
    for index in range(len(frame)):
        training = frame.drop(frame.index[index])
        model = _fit(training)
        design = AirframeDesign(
            **{name: float(frame.iloc[index][name]) for name in FEATURE_LABELS}
        )
        predictions[index] = model.predict(design)
    result = {}
    for index, target in enumerate(TARGETS):
        error = predictions[:, index] - actual[:, index]
        total_variance = float(np.sum((actual[:, index] - actual[:, index].mean()) ** 2))
        result[target] = {
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error**2))),
            "R2": float(1.0 - np.sum(error**2) / total_variance)
            if total_variance > 1e-15
            else None,
        }
    return result


def main() -> None:
    existing = [model for _, model in OUTPUTS.values() if model.exists()]
    if VALIDATION.exists():
        existing.append(VALIDATION)
    if existing:
        raise FileExistsError(
            "Refusing to overwrite versioned model or validation outputs: "
            + ", ".join(str(path) for path in existing)
        )
    datasets = {}
    validation = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_version": DATA_VERSION,
        "model_version": MODEL_VERSION,
        "target_definition": OSWALD_EFFICIENCY_DEFINITION,
        "model_family": "quadratic-basis ridge surrogate",
        "oswald_prediction_bounds": list(OUTPUT_BOUNDS),
        "datasets": {},
        "caveat": "VSPAERO VLM leave-one-out only; not independent wind-tunnel or flight validation.",
    }
    for count, (dataset_path, model_path) in OUTPUTS.items():
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Completed DOE dataset not found: {dataset_path}")
        frame = pd.read_csv(dataset_path)
        if len(frame) != count or frame["sample"].duplicated().any():
            raise ValueError(f"{dataset_path} must contain {count} unique samples.")
        if set(frame["oswald_efficiency_definition"].dropna()) != {
            OSWALD_EFFICIENCY_DEFINITION
        }:
            raise ValueError(f"{dataset_path} contains a mixed or unexpected Ew definition.")
        if frame.loc[:, FEATURE_LABELS].isna().any().any():
            raise ValueError(f"{dataset_path} has missing design features.")
        model = _fit(frame)
        model.to_json(model_path)
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        payload["dataset_version"] = DATA_VERSION
        payload["oswald_efficiency_definition"] = OSWALD_EFFICIENCY_DEFINITION
        payload["mesh_policy"] = (
            "thin lifting surfaces; thick fuselage; reject face-mesh warnings"
        )
        model_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        targets = frame[TARGETS[1]].to_numpy(float)
        validation["datasets"][str(count)] = {
            "sample_count": count,
            "source_file": str(dataset_path),
            "model_file": str(model_path),
            "ew_min": float(targets.min()),
            "ew_median": float(np.median(targets)),
            "ew_max": float(targets.max()),
            "ew_below_legacy_0_55_clip": int(np.sum(targets < 0.55)),
            "ew_above_legacy_0_95_clip": int(np.sum(targets > 0.95)),
            "leave_one_out": _loo(frame),
        }
        datasets[count] = frame
    VALIDATION.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
