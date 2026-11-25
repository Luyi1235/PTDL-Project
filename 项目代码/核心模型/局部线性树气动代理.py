"""Geometry-only predictors for regularized and asymmetric tree experiments."""

from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np

from 气动几何特征 import interference_features
from 局部几何气动代理 import wake_path_features
from 尺度约束气动代理 import geometry_transform
from 分目标气动高斯过程 import FEATURES
from 机体参数化设计与气动阻力 import aero_from_design


def tree_features(x, mode):
    normalized = geometry_transform(x, "canonical-area")
    if mode == "shape":
        return normalized[:, 1:]
    if mode == "interference":
        return interference_features(normalized)
    if mode == "wake_path":
        return wake_path_features(normalized)
    raise ValueError("Unknown geometry feature map")


def fitted_prediction(component, x):
    config = component["config"]
    if config["family"] == "baseline":
        values = interference_features(geometry_transform(x, config["transform"]))
    else:
        values = tree_features(x, config["features"])
        values = (values[:, component["columns"]] - component["center"]) / component["scale"]
    return np.asarray(component["estimator"].predict(values), float)


@dataclass(frozen=True)
class LocalLinearTreeAero:
    payload: dict

    @property
    def training_samples(self):
        return int(self.payload["training_samples"])

    def predict_many(self, x):
        x = geometry_transform(x, "identity")
        result = np.column_stack([
            sum(c["weight"] * fitted_prediction(c, x) for c in target["components"])
            for target in self.payload["targets"]
        ])
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite aerodynamic prediction")
        return result

    def predict(self, design):
        return tuple(self.predict_many([[getattr(design, key) for key in FEATURES]])[0])

    def predict_with_uncertainty(self, design):
        return (*self.predict(design), *[float(t["interval_std"]) for t in self.payload["targets"]])

    def corrected_aero(self, design, calibration):
        low = aero_from_design(design, calibration)
        lift, wake = self.predict(design)
        lift0, wake0 = self.payload["baseline_targets"]
        return replace(
            low,
            cl_alpha_per_rad=float(np.clip(low.cl_alpha_per_rad * lift / lift0, 3.5, 8.5)),
            oswald_efficiency=float(np.clip(low.oswald_efficiency * wake / wake0, .45, 1.)),
        )

    @classmethod
    def load(cls, path):
        # Only load trusted, locally generated training artifacts.
        payload = joblib.load(Path(path))
        if payload.get("schema") != "ptdl-local-linear-tree-aero-v1":
            raise ValueError("Unknown local linear tree surrogate")
        if len(payload["targets"]) != 2 or payload["training_samples"] <= 0:
            raise ValueError("Invalid surrogate dimensions")
        for target in payload["targets"]:
            weights = np.array([c["weight"] for c in target["components"]], float)
            if (not len(weights) or np.any(weights < 0) or not np.isfinite(weights).all()
                    or not np.isclose(weights.sum(), 1.) or target["interval_std"] <= 0):
                raise ValueError("Invalid surrogate components")
        return cls(payload)
