"""Regularized surrogate selection inside random and spatial outer validation."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map, target_mean, TargetwiseAeroGP
from VSPAERO气动代理模型 import FEATURE_LABELS
from 训练_主动增量混合代理20261016 import boosting_payload, metric

TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
DATASETS = {
    "legacy": ROOT / "运行结果" / "百构型加密代理_20261016" / "一致性校核训练" / "low_labels" / "逐构型质量记录.csv",
    "medium": ROOT / "运行结果" / "百构型加密代理_20261016" / "中网格训练实算标签.csv",
}
REFERENCE = {
    "legacy": ROOT / "运行结果" / "主动增量代理_20261015" / "增量构型气动代理.json",
    "medium": ROOT / "运行结果" / "百构型加密代理_20261016" / "多精度候选代理.json",
}


def configurations():
    records = []
    for mode, ard, levels in (
        ("dimensionless", True, (.001, .01, .05)),
        ("dimensionless", False, (.01, .05, .1)),
        ("physics", False, (.01, .05)),
        ("raw", False, (.01, .05)),
    ):
        for noise in levels:
            records.append(dict(method="gp", mode=mode, ard=ard, penalty=noise))
    for degree in (1, 2):
        for alpha in (1., 10., 100.):
            records.append(dict(method="ridge", mode="dimensionless", degree=degree, penalty=alpha))
    for mode in ("raw", "dimensionless", "physics"):
        records.append(dict(method="boost", mode=mode, penalty=5.))
    return records


CONFIGS = configurations()


def optimize(objective, initial, bounds):
    fitted = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": 80, "ftol": 1e-8})
    return fitted.x, fitted.fun


def fit(config, x, y):
    features = feature_map(x, config["mode"])
    center, scale = features.mean(0), np.maximum(features.std(0), 1e-9)
    z = (features-center)/scale
    polynomial = None
    if config["method"] == "gp":
        length = np.ones(z.shape[1]) if config["ard"] else 1.
        estimator = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (.01, 100.))*Matern(length, (.1, 50.), nu=1.5)
            +WhiteKernel(config["penalty"], "fixed"), normalize_y=True,
            alpha=1e-9, optimizer=optimize, n_restarts_optimizer=0)
    elif config["method"] == "ridge":
        polynomial = PolynomialFeatures(config["degree"], include_bias=False)
        z = polynomial.fit_transform(z)
        estimator = Ridge(alpha=config["penalty"])
    else:
        estimator = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=.05, max_leaf_nodes=4, min_samples_leaf=15,
            l2_regularization=config["penalty"], early_stopping=False, random_state=2026101695)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(z, y)
    return dict(estimator=estimator, center=center, scale=scale, polynomial=polynomial,
                config=config, train_features=z)


def predict(fitted, x):
    z = (feature_map(x, fitted["config"]["mode"])-fitted["center"])/fitted["scale"]
    if fitted["polynomial"] is not None:
        z = fitted["polynomial"].transform(z)
    return fitted["estimator"].predict(z)


def geometry_groups(x):
    span = np.maximum(x.max(0)-x.min(0), 1e-9)
    return KMeans(n_clusters=5, n_init=10, random_state=2026101694).fit_predict((x-x.min(0))/span)


def selection_splits(x, groups):
    random = list(KFold(3, shuffle=True, random_state=2026101693).split(x))
    spatial = list(GroupKFold(n_splits=3).split(x, groups=groups))
    return random+spatial


def complexity(config):
    if config["method"] == "ridge":
        return (config["degree"]-1, -config["penalty"], 0)
    if config["method"] == "gp":
        return (3 if config["ard"] else 2, -config["penalty"],
                {"dimensionless": 0, "raw": 1, "physics": 2}[config["mode"]])
    return (4, -config["penalty"], 0)


def one_se_choice(scores):
    best = min(scores, key=lambda r: r["mean_rmse"])
    eligible = [r for r in scores if r["mean_rmse"] <= best["mean_rmse"]+best["rmse_se"]]
    choice = min(eligible, key=lambda r: (complexity(CONFIGS[r["config"]]), r["mean_rmse"]))
    return choice, best["mean_rmse"]+best["rmse_se"]


def select(x, y, groups):
    scores = []
    splits = selection_splits(x, groups)
    for index, config in enumerate(CONFIGS):
        errors = []
        for a, b in splits:
            errors.append(metric(y[b], predict(fit(config, x[a], y[a]), x[b]))["rmse"])
        scores.append(dict(config=index, mean_rmse=float(np.mean(errors)),
                           rmse_se=float(np.std(errors, ddof=1)/np.sqrt(len(errors)))))
    chosen, threshold = one_se_choice(scores)
    return chosen["config"], scores, threshold


def outer_task(task):
    x, y, groups, a, b, metadata = task
    selected, scores, threshold = select(x[a], y[a], groups[a])
    rows = []
    chosen_prediction = None
    chosen_train = None
    for index, config in enumerate(CONFIGS):
        fitted = fit(config, x[a], y[a])
        prediction = predict(fitted, x[b])
        train_prediction = predict(fitted, x[a])
        rows.append(dict(config=index, **metric(y[b], prediction),
                         train_r2=metric(y[a], train_prediction)["r2"],
                         train_rmse=metric(y[a], train_prediction)["rmse"],
                         validation=b.tolist(), actual=y[b].tolist(), prediction=prediction.tolist()))
        if index == selected:
            chosen_prediction, chosen_train = prediction, train_prediction
    return dict(**metadata, selected_config=selected, one_se_threshold=threshold,
                inner_scores=scores, candidates=rows, validation=b.tolist(),
                actual=y[b].tolist(), prediction=chosen_prediction.tolist(),
                selected_train_r2=metric(y[a], chosen_train)["r2"],
                selected_train_rmse=metric(y[a], chosen_train)["rmse"])


def scalar_zero():
    # A deterministic zero residual uses the existing polynomial-trend interface.
    return dict(features="dimensionless", kernel="matern32", feature_mean=[0.]*6,
                feature_scale=[1.]*6, train=[[0.]*6], mean=0., scale=1.,
                amplitude=1e-12, length=[1.]*6, noise=1e-12, dual=[0.],
                inverse=[[1.]], interval_scale=1., residual_mode="deterministic-zero")


def serialize(fitted, x, y, interval_std):
    config, estimator = fitted["config"], fitted["estimator"]
    if config["method"] == "ridge":
        value = dict(kind="trend-residual", trend=f"polynomial{config['degree']}",
                     transform="identity", trend_features="dimensionless",
                     trend_mean=fitted["center"].tolist(), trend_scale=fitted["scale"].tolist(),
                     powers=fitted["polynomial"].powers_.tolist(),
                     coefficients=estimator.coef_.tolist(), intercept=float(estimator.intercept_),
                     ridge_alpha=config["penalty"], residual=scalar_zero())
    elif config["method"] == "boost":
        value = boosting_payload(estimator, config["mode"])
        # The native boosting payload expects unscaled features.
        for tree in value["trees"]:
            for i, leaf in enumerate(tree["is_leaf"]):
                if not leaf:
                    column = tree["feature_idx"][i]
                    tree["num_threshold"][i] = (
                        tree["num_threshold"][i]*fitted["scale"][column]+fitted["center"][column])
    else:
        z = fitted["train_features"]
        length = np.atleast_1d(estimator.kernel_.k1.k2.length_scale)
        if len(length) == 1:
            length = np.repeat(length, z.shape[1])
        value = dict(features=config["mode"], kernel="matern32",
                     feature_mean=fitted["center"].tolist(), feature_scale=fitted["scale"].tolist(),
                     train=z.tolist(), mean=float(estimator._y_train_mean), scale=float(estimator._y_train_std),
                     amplitude=float(estimator.kernel_.k1.k1.constant_value),
                     length=length.tolist(), noise=float(estimator.kernel_.k2.noise_level),
                     dual=estimator.alpha_.tolist(),
                     inverse=cho_solve((estimator.L_, True), np.eye(len(z))).tolist(), interval_scale=1.)
    result = dict(kind="weighted-ensemble", models=[value], weights=[1.], interval_std=interval_std)
    portable = np.array([target_mean(result, row[None, :]) for row in x])
    np.testing.assert_allclose(portable, predict(fitted, x), rtol=1e-10, atol=1e-10)
    return result


def summaries(records, out):
    rows, candidates = [], []
    grouped = {}
    for record in records:
        grouped.setdefault((record["dataset"], record["target"], record["split"]), []).append(record)
    for key, values in grouped.items():
        if len(values) != 5:
            continue
        actual = np.concatenate([r["actual"] for r in values])
        predicted = np.concatenate([r["prediction"] for r in values])
        rows.append(dict(zip(("dataset", "target", "split"), key),
                         model="nested_selected", mean_train_r2=np.mean([r["selected_train_r2"] for r in values]),
                         **metric(actual, predicted)))
        for index in range(len(CONFIGS)):
            prediction = np.concatenate([r["candidates"][index]["prediction"] for r in values])
            candidates.append(dict(zip(("dataset", "target", "split"), key),
                                   config=index, **CONFIGS[index],
                                   mean_train_r2=np.mean([r["candidates"][index]["train_r2"] for r in values]),
                                   **metric(actual, prediction)))
    pd.DataFrame(rows).to_csv(out / "嵌套验证汇总.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "固定配置外层验证.csv", index=False)


def train(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    tasks, arrays = [], {}
    for name, path in DATASETS.items():
        frame = pd.read_csv(path)
        if frame.duplicated(list(FEATURE_LABELS)).any():
            raise ValueError("Duplicate geometry in training data")
        for target in TARGETS:
            mask = np.isfinite(frame[target].to_numpy(float))
            if target == TARGETS[1]:
                mask &= frame.wake_quality_pass.to_numpy(bool)
            subset = frame.loc[mask]
            x, y = subset[list(FEATURE_LABELS)].to_numpy(float), subset[target].to_numpy(float)
            groups = geometry_groups(x)
            arrays[name, target] = (x, y, groups)
            for split, folds in (
                ("random", KFold(5, shuffle=True, random_state=2026101692).split(x)),
                ("spatial", LeaveOneGroupOut().split(x, groups=groups)),
            ):
                for fold, (a, b) in enumerate(folds):
                    tasks.append((x, y, groups, a, b, dict(
                        dataset=name, target=target, split=split, fold=fold, count=len(y))))
    lock = dict(configurations=CONFIGS, sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), *DATASETS.values()]}, external_test_labels_used=False,
                selection="Random and geometry-group inner CV; one-standard-error simplicity rule",
                outer_random_folds=5, outer_spatial_groups=5, inner_folds_per_method=3, tasks=len(tasks))
    (out / "验证配置锁定.json").write_text(json.dumps(lock, indent=2), encoding="utf8")
    records = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(outer_task, task) for task in tasks]):
            records.append(future.result())
            (out / "逐折嵌套记录.json").write_text(json.dumps(records), encoding="utf8")
            (out / "进度.json").write_text(json.dumps(dict(
                stage="nested_validation", completed=len(records), requested=len(tasks),
                elapsed_s=time.monotonic()-started), indent=2), encoding="utf8")
            summaries(records, out)
            print("Nested validation", len(records), "/", len(tasks), flush=True)
    for name, path in DATASETS.items():
        payload = copy.deepcopy(json.loads(REFERENCE[name].read_text(encoding="utf8")))
        selections, targets = [], []
        for target in TARGETS:
            x, y, groups = arrays[name, target]
            index, scores, threshold = select(x, y, groups)
            fitted = fit(CONFIGS[index], x, y)
            selected_rows = [r for r in records if r["dataset"] == name and r["target"] == target]
            errors = np.concatenate([np.array(r["prediction"])-r["actual"] for r in selected_rows])
            interval_std = max(1e-9, float(np.quantile(abs(errors), .9)/1.645))
            targets.append(serialize(fitted, x, y, interval_std))
            selections.append(dict(target=target, config=index, settings=CONFIGS[index],
                                   one_se_threshold=threshold, cv_scores=scores,
                                   training_metrics=metric(y, predict(fitted, x))))
        payload.update(targets=targets, training_samples=len(pd.read_csv(path)),
                       selection="Nested random/spatial CV audit; inner one-SE regularized model selection",
                       uncertainty="Empirical nested OOF error band; not latent GP uncertainty")
        model = out / f"{name}_正则化候选.json"
        TargetwiseAeroGP(payload).to_json(model)
        TargetwiseAeroGP.from_json(model)
        (out / f"{name}_模型选择.json").write_text(json.dumps(selections, indent=2), encoding="utf8")
        (out / f"{name}_模型冻结.json").write_text(json.dumps(dict(
            sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            external_test_used_for_selection=False, automatic_production_promotion=False), indent=2), encoding="utf8")
    (out / "进度.json").write_text(json.dumps(dict(
        stage="completed", completed=len(records), requested=len(tasks),
        elapsed_s=time.monotonic()-started), indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "正则化与嵌套验证_20261016")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    train(args.output_dir, args.workers)
