"""Suppress numerical-scale descriptors consistently during fit and prediction."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from 均匀风险尾迹代理 import features as raw_features, UniformRiskWakeAero
from 精细气动代理 import FineAeroSurrogate


def features(x, mesh, config):
    values = raw_features(x, mesh, config).copy()
    tolerance = float(config["zero_tolerance"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite descriptor tolerance required")
    values[abs(values) < tolerance] = 0.
    return values


@dataclass(frozen=True)
class StableMeshWakeAero:
    payload: dict

    def predict_many(self, x, mesh=None):
        x = np.atleast_2d(np.asarray(x, float))
        result = FineAeroSurrogate(self.payload["baseline"]).predict_many(x)
        wake = np.zeros(len(x))
        for component in self.payload["wake_components"]:
            if component["kind"] == "fine":
                p = FineAeroSurrogate(component["payload"]).predict_many(x)[:, 1]
            elif component["kind"] == "prior":
                p = UniformRiskWakeAero(component["payload"]).predict_many(x, mesh)[:, 1]
            else:
                p = component["estimator"].predict(
                    features(x, mesh, component["config"]), thread_count=2)
            wake += component["weight"]*p
        result[:, 1] = wake
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite stable-descriptor prediction")
        return result

    @property
    def requires_mesh(self):
        return any(c["kind"] == "tree" or (
            c["kind"] == "prior" and UniformRiskWakeAero(c["payload"]).requires_mesh)
            for c in self.payload["wake_components"])

    @classmethod
    def load(cls, path):
        payload = joblib.load(Path(path))
        if payload.get("schema") != "ptdl-stable-mesh-wake-v1":
            raise ValueError("Unsupported stable-descriptor model")
        weights = np.array([c["weight"] for c in payload["wake_components"]])
        if (not len(weights) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
            raise ValueError("Invalid stable-descriptor mixture")
        return cls(payload)
