"""Portable geometry surrogate using constant or linear numerical tree leaves."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from 局部几何气动代理 import LocalGeometryAero
from 气动几何特征 import interference_features
from 分目标气动高斯过程 import FEATURES


def features(x, mode):
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError("Expected seven finite geometry inputs")
    if mode == "raw":
        return x
    if mode == "interference":
        return interference_features(x)
    raise ValueError("Unknown linear-leaf feature map")


def _evaluate(node, x, indices, output):
    if "leaf_value" in node:
        if "leaf_const" in node:
            columns = node["leaf_features"]
            output[indices] = node["leaf_const"] + x[indices][:, columns] @ node["leaf_coeff"]
        else:
            output[indices] = node["leaf_value"]
        return
    if node["decision_type"] != "<=":
        raise ValueError("Only numerical splits are supported")
    left = x[indices, node["split_feature"]] <= node["threshold"]
    if left.any():
        _evaluate(node["left_child"], x, indices[left], output)
    if (~left).any():
        _evaluate(node["right_child"], x, indices[~left], output)


def predict_trees(model, x):
    f = features(x, model["features"])
    z = (f - model["center"]) / model["scale"]
    result = np.zeros(len(z))
    for tree in model["trees"]:
        output = np.empty(len(z))
        _evaluate(tree, z, np.arange(len(z)), output)
        result += output
    return result


@dataclass(frozen=True)
class LinearLeafAero:
    payload: dict

    @property
    def training_samples(self):
        return self.payload["training_samples"]

    def predict_many(self, x):
        result = np.zeros((len(x), 2))
        result[:, 0] = LocalGeometryAero(self.payload["lift_reference"]).predict_many(x)[:, 0]
        for component in self.payload["wake_components"]:
            values = (predict_trees(component["model"], x) if component["kind"] == "linear-tree"
                      else LocalGeometryAero(component["model"]).predict_many(x)[:, 1])
            result[:, 1] += component["weight"] * values
        return result

    def predict(self, design):
        x = np.asarray([[getattr(design, key) for key in FEATURES]])
        return tuple(self.predict_many(x)[0])

    def predict_with_uncertainty(self, design):
        prediction = self.predict(design)
        lift_std = LocalGeometryAero(self.payload["lift_reference"]).predict_with_uncertainty(design)[2]
        return (*prediction, lift_std, float(self.payload["wake_interval_std"]))

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf8")

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf8"))
        if payload.get("schema") != "ptdl-linear-leaf-aero-v1":
            raise ValueError("Unknown linear-leaf schema")
        weights = np.array([c["weight"] for c in payload["wake_components"]])
        if (not len(weights) or np.any(weights < 0) or not np.isfinite(weights).all()
                or not np.isclose(weights.sum(), 1.) or payload["wake_interval_std"] <= 0
                or not np.isfinite(payload["wake_interval_std"])):
            raise ValueError("Invalid linear-leaf ensemble")
        for component in payload["wake_components"]:
            if component["kind"] == "linear-tree":
                model = component["model"]
                count = features([[30., 24., .5, 3., -1., 8., .8]], model["features"]).shape[1]
                if (len(model["center"]) != count or len(model["scale"]) != count
                        or not np.isfinite(model["center"]).all()
                        or not np.isfinite(model["scale"]).all()
                        or np.any(np.asarray(model["scale"]) <= 0)):
                    raise ValueError("Invalid linear-leaf normalization")
                values = predict_trees(model, np.array([[30., 24., .5, 3., -1., 8., .8]]))
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite tree prediction")
            elif component["kind"] != "reference":
                raise ValueError("Unknown linear-leaf component")
        return cls(payload)
