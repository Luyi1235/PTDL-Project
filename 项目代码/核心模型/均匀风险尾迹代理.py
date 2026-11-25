"""Trusted local tree artifacts with an unchanged lift predictor."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from 尺度约束气动代理 import geometry_transform
from 气动几何特征 import interference_features
from 精细气动代理 import FineAeroSurrogate
from 尾迹射线网格特征 import FEATURE_NAMES


def features(x, mesh, config):
    values = interference_features(geometry_transform(x, config["transform"]))
    if config["mesh"]:
        if mesh is None or np.shape(mesh) != (len(values), len(FEATURE_NAMES)):
            raise ValueError("Wake-ray geometry descriptors are required")
        values = np.column_stack((values, mesh))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictor input")
    return values


def component_prediction(component, x, mesh):
    if component["kind"] == "reference":
        return FineAeroSurrogate(component["payload"]).predict_many(x)[:, 1]
    return component["estimator"].predict(features(x, mesh, component["config"]))


@dataclass(frozen=True)
class UniformRiskWakeAero:
    payload: dict

    def predict_many(self, x, mesh=None):
        x = geometry_transform(x, "identity")
        p = FineAeroSurrogate(self.payload["baseline"]).predict_many(x)
        p[:, 1] = sum(c["weight"]*component_prediction(c, x, mesh)
                      for c in self.payload["wake_components"])
        if not np.isfinite(p).all():
            raise RuntimeError("Nonfinite aerodynamic prediction")
        return p

    @property
    def requires_mesh(self):
        return any(c["kind"] != "reference" and c["config"]["mesh"]
                   for c in self.payload["wake_components"])

    @classmethod
    def load(cls, path):
        # Joblib artifacts are trusted local research output.
        payload = joblib.load(Path(path))
        if payload.get("schema") != "ptdl-uniform-risk-wake-v1":
            raise ValueError("Unknown wake surrogate format")
        weights = np.array([c["weight"] for c in payload["wake_components"]])
        if (not len(weights) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
            raise ValueError("Invalid wake components")
        return cls(payload)
