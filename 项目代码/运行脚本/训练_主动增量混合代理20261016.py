"""Select GP/boosting mixtures on training folds before opening fresh labels."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map, scalar_prediction, gated_prediction
from VSPAERO气动代理模型 import FEATURE_LABELS
from 运行_分目标气动代理攻关 import fit_target
from 训练_尾迹分区趋势代理20261015 import fit_trend

TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
PRIOR = ROOT / "运行结果" / "增量构型气动代理_20261015"


def boosting_payload(estimator, mode):
    trees = []
    for predictors in estimator._predictors:
        nodes = predictors[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("Only continuous aerodynamic features supported")
        trees.append({name: nodes[name].tolist() for name in
                      ("value", "feature_idx", "num_threshold", "left", "right", "is_leaf")})
    return dict(kind="hist-boosting", features=mode,
                baseline=float(estimator._baseline_prediction.item()), trees=trees)


def predict_boosting(model, x):
    features = feature_map(x, model["features"])
    outputs = np.full(len(features), model["baseline"])
    for tree in model["trees"]:
        for i, row in enumerate(features):
            node = 0
            while not tree["is_leaf"][node]:
                node = (tree["left"][node] if row[tree["feature_idx"][node]]
                        <= tree["num_threshold"][node] else tree["right"][node])
            outputs[i] += tree["value"][node]
    return outputs


def portable_target_prediction(model, x):
    if model.get("kind") == "weighted-ensemble":
        return sum(weight*portable_target_prediction(component, x)
                   for weight, component in zip(model["weights"], model["models"]))
    if model.get("kind") == "hist-boosting":
        return predict_boosting(model, x)
    if model.get("kind") == "trend-residual":
        from 训练_物理趋势残差代理20261016 import trend_prediction
        return trend_prediction(model, x)
    function = gated_prediction if model.get("kind") == "working-angle-gated" else scalar_prediction
    return np.array([function(model, row[None, :])[0] for row in x])


def metric(y, prediction):
    error = prediction-y
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))),
                r2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PRIOR / "增量训练样本.csv")
    if args.extra_csv:
        extra = pd.read_csv(args.extra_csv)
        frame = pd.concat([frame, extra[frame.columns.drop("development_source")].assign(
            development_source="active_solver_20261016")], ignore_index=True)
    if frame.duplicated(list(FEATURE_LABELS)).any():
        raise ValueError("Repeated training geometry")
    frame.to_csv(out / "增量训练样本.csv", index=False)
    payload = json.loads((PRIOR / "增量构型气动代理.json").read_text(encoding="utf8"))
    records, chosen_records = [], []

    for t, target in enumerate(TARGETS):
        mask = np.isfinite(frame[target].to_numpy(float))
        if t:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        subset = frame.loc[mask]
        x, y = subset[list(FEATURE_LABELS)].to_numpy(float), subset[target].to_numpy(float)
        folds = list(KFold(5, shuffle=True, random_state=2026101506).split(x))
        wake = None
        if t:
            wake = np.array([
                pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv")
                .set_index("alpha_deg").loc[[-1., 1.], "cl_wake"].to_numpy(float)
                for _, row in subset.iterrows()])
        configs = [("gp", "physics", k) for k in ("rbf", "matern32", "matern52")]
        configs += [("gp", "dimensionless", "rbf"),
                    ("boost", "physics", 7), ("boost", "physics", 15)]
        if t:
            configs += [("trend", "physics", "matern32")]

        def fit(config, train):
            method, mode, setting = config
            if method == "boost":
                estimator = HistGradientBoostingRegressor(
                    max_iter=180, max_leaf_nodes=setting, l2_regularization=1.,
                    learning_rate=.05, min_samples_leaf=10, random_state=20261015)
                estimator.fit(feature_map(x[train], mode), y[train])
                portable = boosting_payload(estimator, mode)
                np.testing.assert_allclose(predict_boosting(portable, x[train]),
                                           estimator.predict(feature_map(x[train], mode)),
                                           rtol=1e-12, atol=1e-12)
                return portable
            if method == "trend":
                gates = [fit_target(x[train], wake[train, i], "physics", "rbf")[1] for i in range(2)]
                return fit_trend(x[train], y[train], wake[train], .25, setting, gates)
            return fit_target(x[train], y[train], mode, setting)[1]

        def predict(model, values):
            if model.get("kind") == "hist-boosting":
                return predict_boosting(model, values)
            function = gated_prediction if model.get("kind") == "working-angle-gated" else scalar_prediction
            return np.array([function(model, row[None, :])[0] for row in values])

        def cv(config):
            predictions = np.zeros(len(y))
            for train, test in folds:
                predictions[test] = predict(fit(config, train), x[test])
            name = "_".join(map(str, config))
            record = dict(target=t, name=name, **metric(y, predictions))
            print(json.dumps(record), flush=True)
            return name, predictions, config, record

        with ThreadPoolExecutor(max_workers=6) as pool:
            fitted = list(pool.map(cv, configs))
        candidates = []
        for name, predictions, config, record in fitted:
            records.append(record)
            candidates.append((record["rmse"], predictions, [(1., config)], record))
        for i, left in enumerate(fitted):
            for right in fitted[i+1:]:
                for weight in (.25, .5, .75):
                    prediction = weight*left[1]+(1-weight)*right[1]
                    record = dict(target=t, name=f"{weight}*{left[0]}+{1-weight}*{right[0]}",
                                  **metric(y, prediction))
                    records.append(record)
                    candidates.append((record["rmse"], prediction,
                                       [(weight, left[2]), (1-weight, right[2])], record))
        _, predictions, components, selected = min(candidates, key=lambda v: v[0])
        models = [fit(config, np.arange(len(y))) for _, config in components]
        error = predictions-y
        empirical_std = max(1e-9, float(np.quantile(abs(error), .9, method="higher")/1.645))
        # The mixture band is an OOF empirical error band, not a GP posterior.
        target_payload = dict(kind="weighted-ensemble", models=models,
                              weights=[w for w, _ in components], interval_std=empirical_std)
        payload["targets"][t] = target_payload
        chosen_records.append(selected)
        pd.DataFrame(dict(actual=y, prediction=predictions)).to_csv(
            out / f"选定模型折外预测_{t}.csv", index=False)
        pd.DataFrame(records).to_csv(out / "训练集五折模型比较.csv", index=False)
    payload["training_samples"] = len(frame)
    payload["selection"] = "Per-target GP/boosting mixture selected by training-only five-fold RMSE"
    payload["uncertainty"] = "Fixed empirical OOF absolute-error 90% bands; no latent posterior claim"
    payload["development_sources"] = frame.development_source.value_counts().to_dict()
    (out / "增量构型气动代理.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf8")
    (out / "训练集模型选择.json").write_text(json.dumps(dict(
        selected=chosen_records, training_samples=len(frame), fresh_labels_used=False,
        target_definition="Unchanged lift slope and median VSPAERO Ew"), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
