"""Joint force-profile and efficiency regression with geometry-only inputs."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from 均匀风险尾迹代理 import features
from 精细气动代理 import FineAeroSurrogate

ANGLES = np.array([-3., -1., 1., 3., 5.])
BASIS = np.column_stack((np.ones(5), (ANGLES-1.)/4., ((ANGLES-1.)/4.)**2))
PINVERSE = np.linalg.pinv(BASIS)


def force_coefficients(polar, aspect_ratio):
    values = np.asarray(polar, float)
    if values.shape != (len(aspect_ratio), 5, 3) or not np.isfinite(values).all():
        raise ValueError("Five finite CL, CLiw and CDiw observations required")
    values = values.copy()
    values[:, :, 2] *= np.pi*np.asarray(aspect_ratio)[:, None]
    return np.einsum("ka,nac->nck", PINVERSE, values).reshape(len(values), 9)


def reconstruct(coefficients, fallback):
    coefficients = np.asarray(coefficients, float)
    if coefficients.shape != (len(fallback), 9) or not np.isfinite(coefficients).all():
        raise ValueError("Finite quadratic force coefficients required")
    curves = np.einsum("ak,nck->nac", BASIS, coefficients.reshape(-1, 3, 3))
    cl, clw, scaled_drag = (curves[:, :, i] for i in range(3))
    eligible = (abs(clw) >= .15) & (scaled_drag > 0)
    efficiencies = np.divide(cl*cl, scaled_drag, out=np.zeros_like(cl),
                             where=scaled_drag > 0)
    result = np.asarray(fallback, float).copy()
    for index in range(len(result)):
        selected = efficiencies[index, eligible[index]]
        if len(selected):
            result[index] = np.median(selected)
    return result


def model_input(x, mesh, fitted):
    values = features(x, mesh, fitted["config"])
    return values[:, fitted["columns"]]


def fitted_prediction(fitted, x, mesh, mode):
    values = fitted["estimator"].predict(model_input(x, mesh, fitted), thread_count=2)
    values = values*fitted["scale"]+fitted["center"]
    direct = values[:, 9]
    if mode == "direct":
        return direct
    profile = reconstruct(values[:, :9], direct)
    if mode == "profile":
        return profile
    if mode == "half":
        return .5*direct+.5*profile
    raise ValueError("Unknown profile reconstruction mode")


@dataclass(frozen=True)
class PolarStructureAero:
    payload: dict

    def predict_many(self, x, mesh=None):
        x = np.atleast_2d(np.asarray(x, float))
        p = FineAeroSurrogate(self.payload["baseline"]).predict_many(x)
        result = np.zeros(len(x))
        for component in self.payload["wake_components"]:
            if component["kind"] == "reference":
                wake = FineAeroSurrogate(component["payload"]).predict_many(x)[:, 1]
            elif component["kind"] == "wake_reference":
                from 均匀风险尾迹代理 import UniformRiskWakeAero
                wake = UniformRiskWakeAero(component["payload"]).predict_many(x, mesh)[:, 1]
            else:
                wake = fitted_prediction(component["fitted"], x, mesh, component["mode"])
            result += component["weight"]*wake
        p[:, 1] = result
        if not np.isfinite(p).all():
            raise ValueError("Nonfinite force-profile prediction")
        return p

    @property
    def requires_mesh(self):
        from 均匀风险尾迹代理 import UniformRiskWakeAero
        return any(
            (c["kind"] == "wake_reference"
             and UniformRiskWakeAero(c["payload"]).requires_mesh)
            or (c["kind"] == "profile" and c["fitted"]["config"]["mesh"])
            for c in self.payload["wake_components"])

    @classmethod
    def load(cls, path):
        payload = joblib.load(Path(path))
        if payload.get("schema") != "ptdl-polar-structure-v1":
            raise ValueError("Unsupported force-profile surrogate")
        weights = np.array([c["weight"] for c in payload["wake_components"]])
        if (not len(weights) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
            raise ValueError("Invalid force-profile mixture")
        return cls(payload)
