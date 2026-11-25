"""Portable target-wise GP with dimensionless geometry and OOF intervals."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from 机体参数化设计与气动阻力 import AirframeDesign, aero_from_design

FEATURES = (
    "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg",
    "tip_twist_deg", "fuselage_length_m", "fuselage_diameter_m",
)


def feature_map(x, mode):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 7 or not np.isfinite(x).all() or np.any(x[:, 0] <= 0):
        raise ValueError("Expected finite seven-column geometry with positive area")
    if mode == "raw":
        return x
    if mode == "interference":
        from 气动几何特征 import interference_features
        return interference_features(x)
    if mode == "physics":
        area, ar, taper, sweep, twist, length, diameter = x.T
        span = np.sqrt(area * ar)
        root = 2 * area / (span * (1 + taper))
        mac = 2 / 3 * root * (1 + taper + taper**2) / (1 + taper)
        return np.column_stack((ar, taper, sweep, twist, length / span, diameter / mac,
                                diameter / span, mac / length, diameter / length))
    if mode != "dimensionless":
        raise ValueError(f"Unknown feature map: {mode}")
    return np.column_stack((x[:, 1:5], x[:, 5:7] / np.sqrt(x[:, :1])))


def kernel_cross(left, right, length, amplitude, kind):
    r2 = np.sum(((left[:, None, :] - right[None, :, :]) / length) ** 2, axis=2)
    if kind == "rbf":
        return amplitude * np.exp(-0.5 * r2)
    radius = np.sqrt(np.maximum(r2, 0))
    if kind == "matern32":
        a = np.sqrt(3) * radius
        return amplitude * (1 + a) * np.exp(-a)
    if kind == "matern52":
        a = np.sqrt(5) * radius
        return amplitude * (1 + a + a*a/3) * np.exp(-a)
    raise ValueError(f"Unknown kernel: {kind}")


def scalar_prediction(model, x):
    z = (feature_map(x, model["features"]) - model["feature_mean"]) / model["feature_scale"]
    cross = kernel_cross(
        z, np.asarray(model["train"]), np.asarray(model["length"]),
        model["amplitude"], model["kernel"],
    )[0]
    mean = float(model["mean"] + model["scale"] * (cross @ model["dual"]))
    variance = max(0., model["amplitude"] + model["noise"]
                   - cross @ np.asarray(model["inverse"]) @ cross)
    return mean, float(model["scale"] * np.sqrt(variance) * model["interval_scale"])


def scalar_mean(model, x):
    z = (feature_map(x, model["features"])-model["feature_mean"])/model["feature_scale"]
    cross = kernel_cross(z, np.asarray(model["train"]), np.asarray(model["length"]),
                         model["amplitude"], model["kernel"])[0]
    return float(model["mean"]+model["scale"]*(cross @ model["dual"]))


def boosting_mean(model, x):
    row = feature_map(x, model["features"])[0]
    if "feature_mean" in model:
        row = (row-model["feature_mean"])/model["feature_scale"]
    result = model["baseline"]
    for tree in model["trees"]:
        node = 0
        while not tree["is_leaf"][node]:
            node = (tree["left"][node] if row[tree["feature_idx"][node]]
                    <= tree["num_threshold"][node] else tree["right"][node])
        result += tree["value"][node]
    return float(result)


def oblivious_mean(model, x):
    # CatBoost compares numerical inputs after conversion to single precision.
    row = feature_map(x, model["features"])[0].astype(np.float32)
    total = 0.
    for tree in model["trees"]:
        index = sum((1 << i) for i, (column, threshold) in enumerate(
            zip(tree["columns"], tree["thresholds"])) if float(row[column]) > threshold)
        total += tree["leaves"][index]
    value = model["bias"]+model["scale"]*total
    if model["transform"] == "log":
        return float(np.exp(value))
    if model["transform"] == "inverse":
        if value <= 0:
            raise ValueError("Nonpositive inverse aerodynamic prediction")
        return float(1/value)
    return float(value)


def trend_mean(model, x):
    if model["trend"] == "finite-wing":
        return float(2*np.pi/(1+2/x[0, 1]))
    if model["trend"] == "zero":
        return 0.
    z = (feature_map(x, model["trend_features"])-model["trend_mean"])/model["trend_scale"]
    powers = np.asarray(model["powers"], int)
    basis = np.prod(z[0, None, :]**powers, axis=1)
    return float(model["intercept"]+basis @ model["coefficients"])


def target_mean(model, x):
    kind = model.get("kind")
    if kind == "crossfit-residual":
        return (target_mean(model["base"], x)
                +model["shrinkage"]*target_mean(model["correction"], x))
    if kind == "oblivious-boosting":
        return oblivious_mean(model, x)
    if kind == "angle-regime-ensemble":
        gate = model["gate"]
        z = (feature_map(x, gate["features"])-gate["feature_mean"])/gate["feature_scale"]
        cross = kernel_cross(z, np.asarray(gate["train"]), np.asarray(gate["length"]),
                             gate["amplitude"], gate["kernel"])
        values = np.asarray(gate["mean"])+np.asarray(gate["scale"])*(cross @ gate["dual"])[0]
        regime = str(int((abs(values) >= .15).astype(int) @ (2**np.arange(5))))
        fallback = target_mean(model["fallback"], x)
        if regime not in model["experts"]:
            return fallback
        return (1-model["blend"])*fallback+model["blend"]*target_mean(model["experts"][regime], x)
    if kind == "force-polar-ensemble":
        values = np.array([target_mean(component, x) for component in model["components"]])
        if model["reconstruction"] == "polynomial":
            values = np.array([np.polyval(row, [-3., -1., 1., 3., 5.])
                               for row in values.reshape(3, 3)])
        else:
            values = values.reshape(3, 5)
        cl, clw, cdw = values
        eligible = (abs(clw) >= .15) & (cdw > 0)
        if eligible.sum() < 3:
            return target_mean(model["fallback"], x)
        return float(np.median(cl[eligible]**2/(np.pi*x[0, 1]*cdw[eligible])))
    if kind == "autoregressive-gp":
        from 多精度气动高斯过程 import autoregressive_prediction
        return autoregressive_prediction(model, x, uncertainty=False)
    if kind == "weighted-ensemble":
        return float(sum(w*target_mean(m, x) for w, m in zip(model["weights"], model["models"])))
    if kind == "hist-boosting":
        return boosting_mean(model, x)
    if kind == "trend-residual":
        value = trend_mean(model, x)+scalar_mean(model["residual"], x)
        if model["transform"] == "log":
            return float(np.exp(value))
        if model["transform"] == "inverse":
            return float(1/max(value, 1e-6))
        return value
    if kind == "working-angle-gated":
        gates = [scalar_mean(component, x) for component in model["gates"]]
        regime = str(sum((1 << i) for i, value in enumerate(gates) if abs(value) >= .15))
        return target_mean(model["experts"].get(regime, model["fallback"]), x)
    if kind == "polar":
        mean = reconstructed_ew([scalar_mean(m, x) for m in model["components"]], x[0, 1])
        return mean if np.isfinite(mean) else scalar_mean(model["fallback"], x)
    return scalar_mean(model, x)


def target_prediction(model, x):
    kind = model.get("kind")
    if kind == "autoregressive-gp":
        from 多精度气动高斯过程 import autoregressive_prediction
        return autoregressive_prediction(model, x)
    if kind in ("weighted-ensemble", "hist-boosting", "angle-regime-ensemble",
                "force-polar-ensemble", "oblivious-boosting", "crossfit-residual"):
        return target_mean(model, x), float(model["interval_std"])
    if kind == "trend-residual":
        _, std = scalar_prediction(model["residual"], x)
        value = target_mean(model, x)
        factor = value if model["transform"] == "log" else value**2 if model["transform"] == "inverse" else 1.
        return value, std*factor
    if kind == "working-angle-gated":
        return gated_prediction(model, x)
    if kind == "polar":
        mean = reconstructed_ew([scalar_mean(m, x) for m in model["components"]], x[0, 1])
        std = float(model["interval_std"])
        if not np.isfinite(mean):
            mean, fallback_std = scalar_prediction(model["fallback"], x)
            std = max(std, fallback_std)
        return mean, std
    return scalar_prediction(model, x)


def gated_prediction(model, x):
    gates = [scalar_prediction(component, x)[0] for component in model["gates"]]
    regime = str(sum((1 << i) for i, value in enumerate(gates) if abs(value) >= .15))
    expert = model["experts"].get(regime, model["fallback"])
    return scalar_prediction(expert, x)


def reconstructed_ew(coefficients, aspect_ratio):
    coefficients=np.asarray(coefficients).reshape(3,3)
    angles=np.asarray([-3.,-1.,1.,3.,5.])
    cl,clw,cdw=[np.polyval(c,angles) for c in coefficients]
    working=abs(clw)>=.15
    if working.sum()<3 or np.any(cdw[working]<=0):
        return float("nan")
    # VSPAERO Ew uses total CL in its numerator, and wake CD in its denominator.
    return float(np.median(cl[working]**2/(np.pi*aspect_ratio*cdw[working])))


@dataclass(frozen=True)
class TargetwiseAeroGP:
    payload: dict

    @property
    def training_samples(self):
        return int(self.payload["training_samples"])

    def predict_with_uncertainty(self, design):
        x = np.asarray([[getattr(design, k) for k in FEATURES]])
        means, stds = [], []
        for model in self.payload["targets"]:
            mean, std = target_prediction(model, x)
            means.append(mean)
            stds.append(std)
        return (*means, *stds)

    def predict(self, design):
        x = np.asarray([[getattr(design, k) for k in FEATURES]])
        return tuple(target_mean(model, x) for model in self.payload["targets"])

    def corrected_aero(self, design, calibration):
        low = aero_from_design(design, calibration)
        cl, ew = self.predict(design)
        cl0, ew0 = self.payload["baseline_targets"]
        return replace(
            low,
            cl_alpha_per_rad=float(np.clip(low.cl_alpha_per_rad * cl / cl0, 3.5, 8.5)),
            oswald_efficiency=float(np.clip(low.oswald_efficiency * ew / ew0, .45, 1.)),
        )

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "ptdl_targetwise_gp_v1" or tuple(payload["feature_labels"]) != FEATURES:
            raise ValueError("Unsupported target-wise GP schema")
        if len(payload["targets"]) != 2:
            raise ValueError("Two aerodynamic targets required")
        baseline=np.asarray(payload["baseline_targets"],float)
        if baseline.shape!=(2,) or not np.isfinite(baseline).all() or np.any(baseline<=0):
            raise ValueError("Invalid aerodynamic baseline")
        models=[]
        def collect(target, depth=0):
            if depth > 12:
                raise ValueError("Nested aerodynamic model too deep")
            kind = target.get("kind")
            if kind == "crossfit-residual":
                if (not np.isfinite([target["shrinkage"], target["interval_std"]]).all()
                        or not 0 < target["shrinkage"] <= 1 or target["interval_std"] <= 0):
                    raise ValueError("Invalid cross-fitted residual correction")
                collect(target["base"], depth+1)
                collect(target["correction"], depth+1)
                return
            if kind == "oblivious-boosting":
                p = {"raw": 7, "dimensionless": 6, "physics": 9, "interference": 28}.get(target["features"])
                if (p is None or target["transform"] not in ("identity", "log", "inverse")
                        or target.get("float32_input") is not True or not target["trees"]
                        or not np.isfinite([target["scale"], target["bias"], target["interval_std"]]).all()
                        or target["interval_std"] <= 0):
                    raise ValueError("Invalid oblivious boosting model")
                for tree in target["trees"]:
                    depth = len(tree["columns"])
                    if (depth > 12 or len(tree["thresholds"]) != depth
                            or len(tree["leaves"]) != 2**depth
                            or not np.isfinite(tree["thresholds"]).all()
                            or not np.isfinite(tree["leaves"]).all()
                            or any(type(c) is not int or c not in range(p) for c in tree["columns"])):
                        raise ValueError("Invalid oblivious boosting tree")
                return
            if kind in ("angle-regime-ensemble", "force-polar-ensemble"):
                if not np.isfinite(target["interval_std"]) or target["interval_std"] <= 0:
                    raise ValueError("Invalid decomposed surrogate error band")
                collect(target["fallback"], depth+1)
                if kind == "force-polar-ensemble":
                    expected = {"polynomial": 9, "pointwise": 15}.get(target["reconstruction"])
                    if expected is None or len(target["components"]) != expected:
                        raise ValueError("Invalid force reconstruction components")
                    for component in target["components"]:
                        collect(component, depth+1)
                else:
                    if (not np.isfinite(target["blend"]) or not 0 <= target["blend"] <= 1
                            or set(target["experts"])-{str(i) for i in range(32)}):
                        raise ValueError("Invalid angle regime mixture")
                    gate = target["gate"]
                    p = {"raw": 7, "dimensionless": 6, "physics": 9}.get(gate["features"])
                    n = len(gate["train"])
                    if p is None or n < 1 or gate["kernel"] not in ("rbf", "matern32", "matern52"):
                        raise ValueError("Invalid angle gate features")
                    for key, shape in (("train", (n, p)), ("dual", (n, 5)), ("mean", (5,)),
                                       ("scale", (5,)), ("length", (p,)),
                                       ("feature_mean", (p,)), ("feature_scale", (p,))):
                        value = np.asarray(gate[key], float)
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError(f"Invalid angle gate array: {key}")
                    if (not np.isfinite(gate["amplitude"]) or gate["amplitude"] <= 0
                            or min(gate["scale"]) <= 0 or min(gate["length"]) <= 0
                            or min(gate["feature_scale"]) <= 0):
                        raise ValueError("Invalid angle gate scale")
                    for expert in target["experts"].values():
                        collect(expert, depth+1)
                return
            if kind == "autoregressive-gp":
                collect(target["low_gp"], depth+1)
                low = target["low_gp"]
                if low.get("kind") is not None:
                    raise ValueError("Co-Kriging requires a scalar low-fidelity GP")
                n, p = len(target["high_x"]), len(low["length"])
                if n < 4 or target["kernel"] not in ("rbf", "matern32", "matern52"):
                    raise ValueError("Invalid high-fidelity covariance")
                for key, shape in (("high_x", (n, 7)), ("high_z", (n, p)),
                                   ("low_cross_high", (n, len(low["train"]))),
                                   ("length", (p,)), ("dual", (n,)), ("inverse", (n, n)),
                                   ("combined_low_dual", (len(low["train"]),))):
                    value = np.asarray(target[key], float)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(f"Invalid co-Kriging array: {key}")
                if (not np.isfinite([target[k] for k in
                                     ("rho", "bias", "amplitude", "noise", "interval_scale")]).all()
                        or min(target["rho"], target["amplitude"], target["interval_scale"]) <= 0
                        or target["noise"] < 0 or min(target["length"]) <= 0):
                    raise ValueError("Invalid co-Kriging parameter")
                return
            if kind == "weighted-ensemble":
                weights = np.asarray(target["weights"], float)
                if (weights.shape != (len(target["models"]),) or not len(weights)
                        or not np.isfinite(weights).all() or np.any(weights <= 0)
                        or abs(weights.sum()-1) > 1e-9
                        or not np.isfinite(target["interval_std"]) or target["interval_std"] <= 0):
                    raise ValueError("Invalid aerodynamic ensemble")
                for component in target["models"]:
                    collect(component, depth+1)
                return
            if kind == "hist-boosting":
                if target["features"] not in ("raw", "dimensionless", "physics", "interference"):
                    raise ValueError("Invalid boosting features")
                if not np.isfinite(target["baseline"]) or not target["trees"]:
                    raise ValueError("Invalid boosting model")
                if depth == 0 and (not np.isfinite(target.get("interval_std", np.nan))
                                   or target["interval_std"] <= 0):
                    raise ValueError("Standalone boosting requires an empirical error band")
                p = {"raw": 7, "dimensionless": 6, "physics": 9, "interference": 28}[target["features"]]
                if "feature_mean" in target or "feature_scale" in target:
                    for key in ("feature_mean", "feature_scale"):
                        values = np.asarray(target.get(key, []), float)
                        if values.shape != (p,) or not np.isfinite(values).all():
                            raise ValueError("Invalid boosting feature scaling")
                    if min(target["feature_scale"]) <= 0:
                        raise ValueError("Invalid boosting feature scale")
                for tree in target["trees"]:
                    n = len(tree["value"])
                    if n < 1 or any(len(tree[k]) != n for k in
                                    ("feature_idx", "num_threshold", "left", "right", "is_leaf")):
                        raise ValueError("Invalid boosting tree shape")
                    if not np.isfinite(tree["value"]).all() or not np.isfinite(tree["num_threshold"]).all():
                        raise ValueError("Nonfinite boosting tree")
                    for i in range(n):
                        if tree["is_leaf"][i] not in (0, 1):
                            raise ValueError("Invalid boosting leaf flag")
                        if not tree["is_leaf"][i]:
                            if (tree["feature_idx"][i] not in range(p)
                                    or tree["left"][i] not in range(i+1, n)
                                    or tree["right"][i] not in range(i+1, n)):
                                raise ValueError("Invalid boosting tree branch")
                return
            if kind == "trend-residual":
                if target["trend"] not in ("zero", "finite-wing", "polynomial1", "polynomial2"):
                    raise ValueError("Invalid aerodynamic trend")
                if target["transform"] not in ("identity", "log", "inverse"):
                    raise ValueError("Invalid aerodynamic target transform")
                if target["trend"].startswith("polynomial"):
                    powers = np.asarray(target["powers"], float)
                    coefficients = np.asarray(target["coefficients"], float)
                    if (target["trend_features"] != "dimensionless"
                            or powers.shape != (len(coefficients), 6)
                            or not np.isfinite(powers).all() or np.any(powers < 0)
                            or np.any(powers != np.floor(powers))
                            or np.any(powers.sum(1) > int(target["trend"][-1]))
                            or not np.isfinite(coefficients).all()
                            or not np.isfinite(target["intercept"])):
                        raise ValueError("Invalid polynomial trend")
                    for key in ("trend_mean", "trend_scale"):
                        values = np.asarray(target[key], float)
                        if values.shape != (6,) or not np.isfinite(values).all():
                            raise ValueError("Invalid polynomial scaling")
                    if min(target["trend_scale"]) <= 0:
                        raise ValueError("Invalid polynomial scale")
                collect(target["residual"], depth+1)
                return
            if target.get("kind") == "working-angle-gated":
                if len(target["gates"]) != 2 or not target["experts"]:
                    raise ValueError("Invalid working-angle gate")
                if set(target["experts"]) - {"0", "1", "2", "3"}:
                    raise ValueError("Unsupported working-angle regime")
                models.extend(target["gates"])
                models.extend(target["experts"].values())
                models.append(target["fallback"])
            elif target.get("kind")=="polar":
                if len(target["components"])!=9 or not np.isfinite(target["interval_std"]) or target["interval_std"]<=0:
                    raise ValueError("Invalid polar GP")
                models.extend(target["components"])
                models.append(target["fallback"])
            else:
                if kind is not None:
                    raise ValueError("Unsupported aerodynamic target kind")
                models.append(target)
        for target in payload["targets"]:
            collect(target)
        for model in models:
            if model["features"] not in ("raw","dimensionless","physics") or model["kernel"] not in ("rbf","matern32","matern52"):
                raise ValueError("Unsupported GP feature or kernel")
            if not np.isfinite([model[k] for k in ("mean","scale","amplitude","noise","interval_scale")]).all():
                raise ValueError("Nonfinite GP parameter")
            n = len(model["train"])
            p = {"raw": 7, "dimensionless": 6, "physics": 9}[model["features"]]
            for key, shape in (("train", (n,p)), ("inverse", (n,n)), ("dual", (n,)),
                               ("length", (p,)), ("feature_mean", (p,)), ("feature_scale", (p,))):
                value = np.asarray(model[key], float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid GP array: {key}")
            if min(model["scale"], model["amplitude"], model["interval_scale"]) <= 0:
                raise ValueError("Invalid GP scale")
            if min(model["length"]) <= 0 or min(model["feature_scale"]) <= 0 or model["noise"] < 0:
                raise ValueError("Invalid GP length/noise")
        return cls(payload)
