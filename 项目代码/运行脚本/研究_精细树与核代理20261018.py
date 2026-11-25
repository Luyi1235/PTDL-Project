"""Compare higher-capacity trees and smooth kernels without using test labels."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.svm import SVR

import 研究_尺度约束与总体误差20261017 as previous
from 精细气动代理 import FineAeroSurrogate, component_prediction, kernel_matrix, shape_features
from 尺度约束气动代理 import geometry_transform

ROOT = previous.ROOT
DATA = ROOT / "运行结果/均匀覆盖3072代理_20261017/训练标签.csv"
OUT = ROOT / "运行结果/精细树与核代理_20261018"
FEATURES, TARGETS = previous.FEATURES, previous.prior.prior.TARGETS
CONFIGS = [
    dict(method="tree", transform=transform, depth=depth, penalty=penalty,
         iterations=iterations, ordered=ordered, feature_fraction=fraction)
    for transform in ("identity", "canonical-area")
    for depth, penalty, iterations, ordered, fraction in (
        (6, 20., 1500, False, 1.), (6, 20., 1500, True, 1.),
        (4, 5., 4000, False, 1.), (6, 20., 3500, False, .7),
        (8, 40., 2500, False, .7), (8, 80., 2000, True, .7),
        (10, 80., 1800, False, .7))
] + [
    dict(method="krr", kernel="rbf", gamma=gamma, alpha=alpha)
    for gamma in (.03, .1, .3, 1.) for alpha in (.01, .1)
] + [
    dict(method="krr", kernel="matern32", gamma=gamma, alpha=.03) for gamma in (.1, .3)
] + [
    dict(method="svr", kernel="rbf", gamma=gamma, C=c, epsilon=.05)
    for gamma in (.03, .1, .3) for c in (1., 5., 25.)
]


def fit(config, x, y):
    if config["method"] == "tree":
        f = previous.prior.prior.previous.features(
            geometry_transform(x, config["transform"]), "interference")
        estimator = CatBoostRegressor(
            loss_function="RMSE", iterations=config["iterations"], depth=config["depth"],
            learning_rate=.035, l2_leaf_reg=config["penalty"],
            boosting_type="Ordered" if config["ordered"] else "Plain",
            rsm=config["feature_fraction"], bootstrap_type="Bayesian",
            bagging_temperature=.5, random_strength=.5, random_seed=2026101802,
            thread_count=1, verbose=False, allow_writing_files=False)
        estimator.fit(f, y)
        return dict(config=config, estimator=estimator)
    f = shape_features(x)
    center, scale = f.mean(0), np.maximum(f.std(0), 1e-9)
    z = (f-center)/scale
    y_center, y_scale = float(y.mean()), max(float(y.std()), 1e-9)
    values = (y-y_center)/y_scale
    k = kernel_matrix(z, z, config["kernel"], config["gamma"])
    if config["method"] == "svr":
        estimator = SVR(kernel="precomputed", C=config["C"], epsilon=config["epsilon"],
                        cache_size=128)
    else:
        estimator = KernelRidge(kernel="precomputed", alpha=config["alpha"])
    estimator.fit(k, values)
    return dict(config=config, estimator=estimator, z=z, center=center, scale=scale,
                y_center=y_center, y_scale=y_scale)


def predict(fitted, x):
    config = fitted["config"]
    if config["method"] == "tree":
        f = previous.prior.prior.previous.features(
            geometry_transform(x, config["transform"]), "interference")
        return fitted["estimator"].predict(f)
    z = (shape_features(x)-fitted["center"])/fitted["scale"]
    return fitted["estimator"].predict(kernel_matrix(
        z, fitted["z"], config["kernel"], config["gamma"]))*fitted["y_scale"]+fitted["y_center"]


def portable(fitted):
    config, estimator = fitted["config"], fitted["estimator"]
    if config["method"] == "tree":
        return dict(kind="tree", transform=config["transform"],
                    model=previous.prior.export(estimator, "interference"))
    indices = (estimator.support_ if config["method"] == "svr"
               else np.arange(len(fitted["z"])))
    dual = estimator.dual_coef_.ravel()
    return dict(kind="kernel", kernel=config["kernel"], gamma=config["gamma"],
        center=fitted["center"].tolist(), scale=fitted["scale"].tolist(),
        support=fitted["z"][indices].tolist(), dual=dual.tolist(),
        intercept=float(estimator.intercept_[0]) if config["method"] == "svr" else 0.,
        y_center=fitted["y_center"], y_scale=fitted["y_scale"])


def task(args):
    target, i, split, fold, a, b, x, y = args
    start = time.monotonic()
    fitted = fit(CONFIGS[i], x[a], y[a])
    p = predict(fitted, x[b])
    return dict(target=target, config=i, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=p.tolist(), seconds=time.monotonic()-start,
                train_r2=previous.prior.prior.previous.prior.direct.base.metric(
                    y[a], predict(fitted, x[a]))["r2"],
                **previous.prior.prior.previous.prior.direct.base.metric(y[b], p))


def run(data, out, workers):
    out.mkdir(parents=True, exist_ok=True)
    hashes = {str(p): previous.prior.digest(p) for p in
              (Path(__file__), ROOT / "核心模型/精细气动代理.py", data)}
    freeze = out / "配置冻结.json"
    if freeze.exists():
        existing = json.loads(freeze.read_text(encoding="utf8"))
        if existing["hashes"] != hashes or existing["configs"] != CONFIGS:
            raise RuntimeError("Fine training sources changed")
    else:
        freeze.write_text(json.dumps(dict(hashes=hashes, configs=CONFIGS,
            external_labels_used=False, selection="Mean random/spatial OOF RMSE, all valid rows",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    frame = pd.read_csv(data)
    arrays, jobs = {}, []
    for target in TARGETS:
        valid = np.isfinite(frame[target].to_numpy(float))
        if target == TARGETS[1]:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        x, y = frame.loc[valid, FEATURES].to_numpy(float), frame.loc[valid, target].to_numpy(float)
        arrays[target] = x, y
        groups = previous.prior.prior.previous.prior.direct.base.geometry_groups(x)
        for split, folds in (
            ("random", KFold(3, shuffle=True, random_state=2026101803).split(x)),
            ("spatial", GroupKFold(3).split(x, groups=groups))):
            for fold, (a, b) in enumerate(folds):
                jobs.extend((target, i, split, fold, a, b, x, y) for i in range(len(CONFIGS)))
    records_path = out / "交叉验证逐折.json"
    records = json.loads(records_path.read_text(encoding="utf8")) if records_path.exists() else []
    done = {(r["target"], r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:4] not in done]
    total, start = len(records)+len(jobs), time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            records_path.write_text(json.dumps(records), encoding="utf8")
            (out / "进度.json").write_text(json.dumps(dict(stage="training_cv",
                completed=len(records), requested=total, elapsed_s=time.monotonic()-start)),
                encoding="utf8")
            print("Fine model CV", len(records), "/", total, flush=True)
    payload = dict(schema="ptdl-fine-aero-v1", training_samples=len(frame), targets=[],
        baseline_targets=json.loads(previous.prior.prior.REFERENCE.read_text(
            encoding="utf8"))["baseline_targets"], automatic_production_promotion=False)
    selection, summaries = [], []
    for target in TARGETS:
        x, y = arrays[target]
        subset = [r for r in records if r["target"] == target]
        predictions = [previous.prior.prior.previous.prior.records_matrix(subset, len(y), i)
                       for i in range(len(CONFIGS))]
        score, chosen, oof = previous.prior.prior.previous.prior.select_mixture(predictions, y)
        components, checks = [], []
        probe = x[np.linspace(0, len(x)-1, 40).astype(int)]
        probe = np.vstack((probe, (probe+np.roll(probe, 1, axis=0))/2))
        for i, weight in chosen:
            fitted = fit(CONFIGS[i], x, y)
            component = dict(portable(fitted), weight=weight)
            actual, expected = component_prediction(component, probe), predict(fitted, probe)
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
            components.append(component)
            checks.append(dict(config=i, weight=weight, max_abs_export_error=float(
                np.max(abs(actual-expected)))))
        interval = max(1e-9, float(np.quantile(abs(oof-y), .9)/1.645))
        payload["targets"].append(dict(components=components, interval_std=interval))
        selection.append(dict(target=target, count=len(y), score=score, components=checks))
        for i, values in enumerate(predictions):
            summaries.extend(dict(target=target, config=i, split=s, **CONFIGS[i],
                **previous.prior.prior.previous.prior.direct.base.metric(y, p))
                for s, p in zip(("random", "spatial"), values))
        selection[-1]["oof"] = [dict(split=s,
            **previous.prior.prior.previous.prior.direct.base.metric(y, p))
            for s, p in zip(("random", "spatial"), oof)]
    model_path = out / "精细气动候选代理.json"
    FineAeroSurrogate(payload).to_json(model_path)
    FineAeroSurrogate.from_json(model_path)
    pd.DataFrame(summaries).to_csv(out / "训练交叉验证汇总.csv", index=False)
    (out / "模型选择.json").write_text(json.dumps(dict(selected=selection,
        model_sha256=previous.prior.digest(model_path), external_labels_used=False,
        selected_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    (out / "进度.json").write_text(json.dumps(dict(stage="completed", completed=total,
        requested=total, elapsed_s=time.monotonic()-start)), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run(args.data.resolve(), args.output_dir.resolve(), args.workers)
