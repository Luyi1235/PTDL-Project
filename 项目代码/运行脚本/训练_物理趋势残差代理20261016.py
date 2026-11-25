"""Compare physical means, polynomial trends and target transforms on development data."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map, scalar_prediction
from VSPAERO气动代理模型 import FEATURE_LABELS
from 运行_分目标气动代理攻关 import fit_target
from 训练_主动增量混合代理20261016 import metric

PRIOR = ROOT / "运行结果" / "主动增量代理_20261015"
OUT = ROOT / "运行结果" / "物理趋势残差代理_20261016"
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def trend_mean(model, x):
    if model["trend"] == "finite-wing":
        return 2*np.pi/(1+2/x[:, 1])
    if model["trend"] == "zero":
        return np.zeros(len(x))
    z = (feature_map(x, model["trend_features"])-model["trend_mean"])/model["trend_scale"]
    powers = np.asarray(model["powers"], int)
    basis = np.prod(z[:, None, :]**powers[None, :, :], axis=2)
    return model["intercept"]+basis@np.array(model["coefficients"])


def fit_trend_residual(x, y, trend, transform, kernel):
    converted = np.log(y) if transform == "log" else 1/y if transform == "inverse" else y
    model = dict(kind="trend-residual", trend=trend, transform=transform)
    if trend.startswith("polynomial"):
        degree = int(trend[-1])
        features = feature_map(x, "dimensionless")
        mean, scale = features.mean(0), np.maximum(features.std(0), 1e-9)
        polynomial = PolynomialFeatures(degree=degree, include_bias=False)
        basis = polynomial.fit_transform((features-mean)/scale)
        ridge = RidgeCV(alphas=[.01, .1, 1., 10., 100.])
        ridge.fit(basis, converted)
        model.update(trend_features="dimensionless", trend_mean=mean.tolist(),
                     trend_scale=scale.tolist(), powers=polynomial.powers_.tolist(),
                     coefficients=ridge.coef_.tolist(), intercept=float(ridge.intercept_),
                     ridge_alpha=float(ridge.alpha_))
    residual = converted-trend_mean(model, x)
    model["residual"] = fit_target(x, residual, "physics", kernel)[1]
    return model


def trend_prediction(model, x):
    outputs = trend_mean(model, x)+np.array([
        scalar_prediction(model["residual"], row[None, :])[0] for row in x])
    if model["transform"] == "log":
        return np.exp(outputs)
    if model["transform"] == "inverse":
        return 1/np.maximum(outputs, 1e-6)
    return outputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PRIOR / "增量训练样本.csv")
    payload = json.loads((PRIOR / "增量构型气动代理.json").read_text(encoding="utf8"))
    records, selected_records = [], []
    for t, target in enumerate(TARGETS):
        mask = np.isfinite(frame[target].to_numpy(float))
        if t:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        subset = frame.loc[mask]
        x, y = subset[list(FEATURE_LABELS)].to_numpy(float), subset[target].to_numpy(float)
        folds = list(KFold(5, shuffle=True, random_state=2026101506).split(x))
        configs = ([("finite-wing", "identity", k) for k in ("rbf", "matern32", "matern52")]
                   if t == 0 else [("zero", a, k) for a in ("log", "inverse")
                                   for k in ("matern32", "matern52")])
        configs += [(trend, transform, "matern32")
                    for trend in ("polynomial1", "polynomial2")
                    for transform in (("identity",) if t == 0 else ("identity", "log"))]

        def cv(config):
            prediction = np.zeros(len(y))
            for train, test in folds:
                model = fit_trend_residual(x[train], y[train], *config)
                prediction[test] = trend_prediction(model, x[test])
            record = dict(target=t, name="_".join(config), **metric(y, prediction))
            print(json.dumps(record), flush=True)
            return prediction, config, record

        with ThreadPoolExecutor(max_workers=6) as pool:
            candidates = list(pool.map(cv, configs))
        prior_predictions = pd.read_csv(PRIOR / f"选定模型折外预测_{t}.csv").prediction.to_numpy()
        prior_record = dict(target=t, name="prior-mixture", **metric(y, prior_predictions))
        candidates.append((prior_predictions, None, prior_record))
        records.extend(item[2] for item in candidates)
        blends = []
        for prediction, config, record in candidates[:-1]:
            for weight in (.25, .5, .75):
                mixed = weight*prediction+(1-weight)*prior_predictions
                mixed_record = dict(target=t, name=f"{weight}*{record['name']}+{1-weight}*prior",
                                    **metric(y, mixed))
                records.append(mixed_record)
                blends.append((mixed, (weight, config), mixed_record))
        winner = min(candidates+blends, key=lambda item: item[2]["rmse"])
        prediction, config, record = winner
        if config is not None:
            if len(config) == 2:
                weight, settings = config
                model = fit_trend_residual(x, y, *settings)
                payload["targets"][t] = dict(
                    kind="weighted-ensemble", models=[model, payload["targets"][t]],
                    weights=[weight, 1-weight],
                    interval_std=max(1e-9, float(np.quantile(abs(prediction-y), .9, method="higher")/1.645)))
            else:
                model = fit_trend_residual(x, y, *config)
                payload["targets"][t] = dict(
                    kind="weighted-ensemble", models=[model], weights=[1.],
                    interval_std=max(1e-9, float(np.quantile(abs(prediction-y), .9, method="higher")/1.645)))
        selected_records.append(record)
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            OUT / f"选定模型折外预测_{t}.csv", index=False)
        pd.DataFrame(records).to_csv(OUT / "训练集五折模型比较.csv", index=False)
    payload["selection"] = "Training-fold comparison of physical/polynomial means, target transforms and GP residuals"
    (OUT / "物理趋势候选代理.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf8")
    (OUT / "训练集模型选择.json").write_text(json.dumps(dict(
        selected=selected_records, training_samples=len(frame),
        all_heldout_labels_excluded_from_fitting=True), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
