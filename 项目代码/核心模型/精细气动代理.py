"""Portable, target-wise tree and kernel aerodynamic regression."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from 尺度约束气动代理 import geometry_transform
from 局部几何气动代理 import numeric_tree_predict, validate_numeric_model
from 分目标气动高斯过程 import FEATURES, target_mean
from 机体参数化设计与气动阻力 import aero_from_design


def shape_features(x):
    x = geometry_transform(x, "canonical-area")
    return x[:, 1:]


def kernel_matrix(x, z, kind, gamma):
    x, z = np.asarray(x, float), np.asarray(z, float)
    d2 = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(z*z, axis=1)[None, :]
                    - 2*x @ z.T, 0.)
    if kind == "rbf":
        return np.exp(-gamma*d2)
    if kind == "matern32":
        r = np.sqrt(3*gamma*d2)
        return (1+r)*np.exp(-r)
    raise ValueError("Unknown kernel")


def component_prediction(component, x):
    if component["kind"] == "tree":
        return numeric_tree_predict(component["model"],
                                    geometry_transform(x, component["transform"]))
    if component["kind"] == "target":
        return np.array([target_mean(component["model"], row[None]) for row in x])
    if component["kind"] != "kernel":
        raise ValueError("Unknown fine surrogate component")
    f = (shape_features(x)-np.asarray(component["center"]))/np.asarray(component["scale"])
    k = kernel_matrix(f, component["support"], component["kernel"], component["gamma"])
    return (k @ np.asarray(component["dual"]) + component["intercept"])*component["y_scale"] \
        + component["y_center"]


@dataclass(frozen=True)
class FineAeroSurrogate:
    payload: dict

    @property
    def training_samples(self):
        return int(self.payload["training_samples"])

    def predict_many(self, x):
        x = geometry_transform(x, "identity")
        return np.column_stack([sum(c["weight"]*component_prediction(c, x)
                                    for c in target["components"])
                                for target in self.payload["targets"]])

    def predict(self, design):
        return tuple(self.predict_many([[getattr(design, key) for key in FEATURES]])[0])

    def predict_with_uncertainty(self, design):
        return (*self.predict(design), *[float(t["interval_std"]) for t in self.payload["targets"]])

    def corrected_aero(self, design, calibration):
        low = aero_from_design(design, calibration)
        lift, wake = self.predict(design)
        lift0, wake0 = self.payload["baseline_targets"]
        return replace(low, cl_alpha_per_rad=float(np.clip(low.cl_alpha_per_rad*lift/lift0, 3.5, 8.5)),
                       oswald_efficiency=float(np.clip(low.oswald_efficiency*wake/wake0, .45, 1.)))

    def to_json(self, path):
        self.validate()
        Path(path).write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf8")

    def validate(self):
        if self.payload.get("schema") != "ptdl-fine-aero-v1" or len(self.payload["targets"]) != 2:
            raise ValueError("Invalid fine aerodynamic surrogate schema")
        if (len(self.payload["baseline_targets"]) != 2
                or not np.isfinite(self.payload["baseline_targets"]).all()
                or min(self.payload["baseline_targets"]) <= 0):
            raise ValueError("Invalid aerodynamic reference")
        for target in self.payload["targets"]:
            components = target["components"]
            weights = np.array([c["weight"] for c in components], float)
            if (not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0)
                    or not np.isclose(weights.sum(), 1.)
                    or not np.isfinite(target["interval_std"]) or target["interval_std"] <= 0):
                raise ValueError("Invalid component weights or prediction error scale")
            for c in components:
                if c["kind"] == "tree":
                    geometry_transform([[30., 24., .5, 3., -1., 8., .8]], c["transform"])
                    validate_numeric_model(c["model"])
                elif c["kind"] == "kernel":
                    center, scale, support, dual = map(np.asarray,
                        (c["center"], c["scale"], c["support"], c["dual"]))
                    if (center.shape != (6,) or scale.shape != (6,) or np.any(scale <= 0)
                            or support.ndim != 2 or support.shape[1] != 6
                            or dual.shape != (len(support),) or not len(support)
                            or c["gamma"] <= 0 or c["y_scale"] <= 0
                            or c["kernel"] not in ("rbf", "matern32")
                            or not all(np.isfinite(a).all() for a in (center, scale, support, dual))
                            or not np.isfinite([c["gamma"], c["y_scale"], c["y_center"],
                                               c["intercept"]]).all()):
                        raise ValueError("Invalid portable kernel")
                elif c["kind"] != "target":
                    raise ValueError("Invalid model component")

    @classmethod
    def from_json(cls, path):
        obj = cls(json.loads(Path(path).read_text(encoding="utf8")))
        obj.validate()
        return obj
