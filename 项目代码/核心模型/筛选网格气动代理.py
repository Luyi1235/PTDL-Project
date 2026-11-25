"""Fold-local geometry feature selection with an unchanged aerodynamic target."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from 稳定网格尾迹代理 import features
from 精细气动代理 import FineAeroSurrogate


def select_columns(values, target, geometry_columns, keep_mesh, threads=2):
    values, target = np.asarray(values, float), np.asarray(target, float)
    if values.ndim != 2 or target.shape != (len(values),) or not np.isfinite(values).all():
        raise ValueError("Finite two-dimensional training features required")
    if not np.isfinite(target).all() or not 0 < geometry_columns <= values.shape[1]:
        raise ValueError("Invalid training targets or geometry column count")
    if keep_mesh is None:
        return np.arange(values.shape[1], dtype=int)
    if not isinstance(keep_mesh, int) or keep_mesh <= 0:
        raise ValueError("Positive mesh feature budget required")
    candidates = np.flatnonzero(np.std(values[:, geometry_columns:], axis=0) > 1e-10)
    candidates += geometry_columns
    if len(candidates) <= keep_mesh:
        return np.r_[np.arange(geometry_columns), candidates].astype(int)
    selector = ExtraTreesRegressor(
        n_estimators=96, max_depth=10, min_samples_leaf=8, max_features=.7,
        random_state=2026102001, n_jobs=threads,
    ).fit(values, target)
    order = np.argsort(-selector.feature_importances_[candidates], kind="stable")
    return np.r_[np.arange(geometry_columns),
                 np.sort(candidates[order[:keep_mesh]])].astype(int)


def predict_component(component, x, mesh):
    matrix = features(x, mesh, component["config"])
    return component["estimator"].predict(
        matrix[:, component["columns"]], thread_count=2)


@dataclass(frozen=True)
class SelectedMeshAero:
    payload: dict

    def predict_many(self, x, mesh=None):
        x = np.atleast_2d(np.asarray(x, float))
        baseline = FineAeroSurrogate(self.payload["baseline"]).predict_many(x)
        wake = np.zeros(len(x))
        for component in self.payload["wake_components"]:
            prediction = (baseline[:, 1] if component["kind"] == "reference"
                          else predict_component(component, x, mesh))
            wake += component["weight"] * prediction
        baseline[:, 1] = wake
        if not np.isfinite(baseline).all():
            raise ValueError("Nonfinite selected-mesh aerodynamic prediction")
        return baseline

    @property
    def requires_mesh(self):
        return any(c["kind"] != "reference" for c in self.payload["wake_components"])

    @classmethod
    def load(cls, path):
        payload = joblib.load(Path(path))
        if payload.get("schema") != "ptdl-selected-mesh-aero-v1":
            raise ValueError("Unsupported selected-mesh model schema")
        weights = np.array([c["weight"] for c in payload["wake_components"]])
        if (not len(weights) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
            raise ValueError("Invalid selected-mesh mixture weights")
        return cls(payload)
