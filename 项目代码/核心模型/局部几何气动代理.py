"""Geometry-local wake surrogate with a portable numerical-tree evaluator."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from 气动几何特征 import interference_features
from 分目标气动高斯过程 import FEATURES, target_mean, target_prediction
from 机体参数化设计与气动阻力 import aero_from_design


def wake_path_features(x):
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError("Expected seven finite geometry inputs")
    area, ar, taper, sweep, twist, length, diameter = x.T
    if np.any(np.column_stack((area, ar, taper, length, diameter)) <= 0):
        raise ValueError("Geometry dimensions must be positive")
    span = np.sqrt(area * ar)
    root = 2 * area / (span * (1 + taper))
    mac = 2 / 3 * root * (1 + taper + taper**2) / (1 + taper)
    tail_area = .65 * area * mac / (.52 * length)
    tail_span = np.sqrt(4 * tail_area)
    tail_root = 2 * tail_area / (1.5 * tail_span)
    extra = []
    # Straight-ray offsets are geometry descriptors, not calculated downwash.
    for fraction in (0., .5, 1.):
        y = fraction * tail_span / 2
        wing_chord = root * (1 - (1 - taper) * 2 * y / span)
        tail_chord = tail_root * (1 - .5 * fraction)
        dx = (.48 * length + y * np.tan(np.deg2rad(8.))
              - y * np.tan(np.deg2rad(sweep)))
        dz = .2 * diameter - y * np.tan(np.deg2rad(3.))
        extra.extend((dx / mac, dz / mac, wing_chord / mac,
                      tail_chord / mac, y / span))
        for alpha in (-3., -1., 1., 3., 5.):
            extra.append((dz - dx * np.tan(np.deg2rad(alpha))) / mac)
    extra.extend((np.cos(np.deg2rad(sweep)), np.sin(np.deg2rad(sweep)),
                  np.deg2rad(twist) * tail_span / span, tail_area / area / ar))
    return np.column_stack((interference_features(x), *extra))


def model_features(x, mode):
    if mode == "wake_path":
        return wake_path_features(x)
    if mode == "interference":
        return interference_features(x)
    raise ValueError("Unknown local geometry feature map")


def _tree_prediction(node, values, indices, output):
    if "value" in node:
        output[indices] = node["value"]
        return
    split = node["split"]
    goes_right = values[indices, split["float_feature_index"]] > split["border"]
    if np.any(~goes_right):
        _tree_prediction(node["left"], values, indices[~goes_right], output)
    if np.any(goes_right):
        _tree_prediction(node["right"], values, indices[goes_right], output)


def numeric_tree_predict(model, x):
    values = model_features(x, model["features"]).astype(np.float32).astype(float)
    result = np.zeros(len(values))
    if "oblivious_trees" in model:
        for tree in model["oblivious_trees"]:
            indices = np.zeros(len(values), dtype=int)
            for bit, split in enumerate(tree.get("splits") or []):
                indices += ((values[:, split["float_feature_index"]] > split["border"])
                            .astype(int) << bit)
            result += np.asarray(tree["leaf_values"])[indices]
    else:
        indices = np.arange(len(values))
        for tree in model["trees"]:
            output = np.empty(len(values))
            _tree_prediction(tree, values, indices, output)
            result += output
    return model["scale"] * result + model["bias"]


def validate_numeric_model(model):
    if model["features"] not in ("interference", "wake_path"):
        raise ValueError("Unsupported feature map")
    nfeatures = model_features([[30., 24., .5, 3., -1., 8., .8]], model["features"]).shape[1]

    def split_valid(split):
        if (split["split_type"] != "FloatFeature"
                or not 0 <= split["float_feature_index"] < nfeatures
                or not np.isfinite(split["border"])):
            raise ValueError("Invalid numerical split")

    def tree_valid(node):
        if "value" in node:
            if not np.isfinite(node["value"]):
                raise ValueError("Invalid leaf value")
        else:
            split_valid(node["split"])
            tree_valid(node["left"])
            tree_valid(node["right"])

    if not np.isfinite([model["scale"], model["bias"]]).all():
        raise ValueError("Invalid model scale")
    if "oblivious_trees" in model:
        for tree in model["oblivious_trees"]:
            splits = tree.get("splits") or []
            for split in splits:
                split_valid(split)
            if (len(tree["leaf_values"]) != 2**len(splits)
                    or not np.isfinite(tree["leaf_values"]).all()):
                raise ValueError("Invalid symmetric leaves")
    elif "trees" in model:
        for tree in model["trees"]:
            tree_valid(tree)
    else:
        raise ValueError("Missing numerical trees")


@dataclass(frozen=True)
class LocalGeometryAero:
    payload: dict

    @property
    def training_samples(self):
        return int(self.payload["training_samples"])

    def predict_many(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        lift = np.array([target_mean(self.payload["baseline"]["targets"][0], row[None])
                         for row in x])
        wake = np.zeros(len(x))
        for component in self.payload["wake_components"]:
            model = component["model"]
            prediction = (numeric_tree_predict(model, x) if component["kind"] == "local-tree"
                          else np.array([target_mean(model, row[None]) for row in x]))
            wake += component["weight"] * prediction
        return np.column_stack((lift, wake))

    def predict(self, design):
        x = [[getattr(design, key) for key in FEATURES]]
        return tuple(self.predict_many(x)[0])

    def predict_with_uncertainty(self, design):
        means = self.predict(design)
        x = np.array([[getattr(design, key) for key in FEATURES]])
        _, lift_std = target_prediction(self.payload["baseline"]["targets"][0], x)
        return (*means, lift_std, float(self.payload["wake_interval_std"]))

    def corrected_aero(self, design, calibration):
        low = aero_from_design(design, calibration)
        lift, wake = self.predict(design)
        lift0, wake0 = self.payload["baseline"]["baseline_targets"]
        return replace(
            low, cl_alpha_per_rad=float(np.clip(low.cl_alpha_per_rad * lift / lift0, 3.5, 8.5)),
            oswald_efficiency=float(np.clip(low.oswald_efficiency * wake / wake0, .45, 1.)))

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf8")

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf8"))
        if payload.get("schema") != "ptdl-local-geometry-aero-v1":
            raise ValueError("Unknown local geometry surrogate schema")
        weights = np.array([c["weight"] for c in payload["wake_components"]])
        if (not len(weights) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)
                or not np.isfinite(payload["wake_interval_std"])
                or payload["wake_interval_std"] <= 0):
            raise ValueError("Invalid wake ensemble")
        for component in payload["wake_components"]:
            if component["kind"] == "local-tree":
                validate_numeric_model(component["model"])
            elif component["kind"] != "reference":
                raise ValueError("Unknown local model component")
        return cls(payload)
