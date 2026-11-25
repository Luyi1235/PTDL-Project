"""Geometry-normalized wake predictors with training-only population weights."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from 局部几何气动代理 import numeric_tree_predict, validate_numeric_model
from 分目标气动高斯过程 import FEATURES, target_mean, target_prediction
from 机体参数化设计与气动阻力 import aero_from_design


def geometry_transform(x, mode):
    x = np.atleast_2d(np.asarray(x, float))
    if (x.shape[1] != 7 or not np.isfinite(x).all()
            or np.any(x[:, [0, 1, 2, 5, 6]] <= 0)):
        raise ValueError("Expected seven finite geometry variables with positive dimensions")
    if mode == "identity":
        return x.copy()
    if mode != "canonical-area":
        raise ValueError("Unknown geometry normalization")
    result = x.copy()
    factor = np.sqrt(30. / x[:, 0])
    result[:, 0] = 30.
    result[:, 5:7] *= factor[:, None]
    return result


@dataclass(frozen=True)
class ScalePopulationAero:
    payload: dict

    @property
    def training_samples(self):
        return int(self.payload["training_samples"])

    def predict_many(self, x):
        x = geometry_transform(x, "identity")
        result = np.zeros((len(x), 2))
        result[:, 0] = [target_mean(self.payload["baseline"]["targets"][0], row[None]) for row in x]
        for component in self.payload["wake_components"]:
            inputs = geometry_transform(x, component["transform"])
            model = component["model"]
            values = (numeric_tree_predict(model, inputs) if component["kind"] == "tree"
                      else np.array([target_mean(model, row[None]) for row in inputs]))
            result[:, 1] += component["weight"] * values
        return result

    def predict(self, design):
        return tuple(self.predict_many([[getattr(design, key) for key in FEATURES]])[0])

    def predict_with_uncertainty(self, design):
        x = np.array([[getattr(design, key) for key in FEATURES]])
        values = self.predict(design)
        _, lift_std = target_prediction(self.payload["baseline"]["targets"][0], x)
        return (*values, lift_std, float(self.payload["wake_interval_std"]))

    def corrected_aero(self, design, calibration):
        low = aero_from_design(design, calibration)
        lift, wake = self.predict(design)
        lift0, wake0 = self.payload["baseline"]["baseline_targets"]
        return replace(
            low, cl_alpha_per_rad=float(np.clip(low.cl_alpha_per_rad * lift/lift0, 3.5, 8.5)),
            oswald_efficiency=float(np.clip(low.oswald_efficiency * wake/wake0, .45, 1.)))

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf8")

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf8"))
        if payload.get("schema") != "ptdl-scale-population-aero-v1":
            raise ValueError("Unknown scale-population surrogate schema")
        components = payload["wake_components"]
        weights = np.array([c["weight"] for c in components])
        if (not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.) or payload["wake_interval_std"] <= 0
                or not np.isfinite(payload["wake_interval_std"])):
            raise ValueError("Invalid scale-population ensemble")
        for component in components:
            geometry_transform([[30., 24., .5, 3., -1., 8., .8]], component["transform"])
            if component["kind"] == "tree":
                validate_numeric_model(component["model"])
            elif component["kind"] != "target":
                raise ValueError("Unknown scale-population component")
        return cls(payload)
