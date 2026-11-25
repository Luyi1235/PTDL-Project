"""Compare geometry-aware learners using training-only random and spatial folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import time

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold

import 推进_千构型主动补样20261017 as prior
from 气动几何特征 import interference_features

ROOT = prior.ROOT
DATA = prior.OUT / "中网格训练实算标签.csv"
OUT = ROOT / "运行结果/几何交互代理_20261017"
FEATURES = prior.direct.FEATURES
TARGETS = prior.direct.base.TARGETS


def features(x, mode):
    return interference_features(x) if mode == "interference" else prior.direct.base.feature_map(x, mode)


def configurations():
    configs = []
    for mode in ("raw", "physics", "interference"):
        for depth in (4, 6, 8):
            for penalty in (3., 20.):
                configs.append(dict(method="cat", mode=mode, depth=depth, penalty=penalty,
                                    transform="identity", iterations=1500))
    for transform in ("log", "inverse"):
        for depth in (4, 6):
            configs.append(dict(method="cat", mode="interference", depth=depth, penalty=20.,
                                transform=transform, iterations=1500))
    for mode in ("physics", "interference"):
        for leaves, penalty in ((4, 5.), (7, 1.), (15, 10.)):
            configs.append(dict(method="boost", mode=mode, leaves=leaves, penalty=penalty,
                                transform="identity", iterations=600))
    for mode in ("physics", "dimensionless"):
        for kernel in ("rbf", "matern32", "matern52"):
            configs.append(dict(method="gp_restart", mode=mode, kernel=kernel,
                                noise="learn", transform="identity"))
    return configs


CONFIGS = configurations()


def transform_y(y, mode):
    if mode == "log":
        return np.log(y)
    if mode == "inverse":
        return 1/y
    return y


def inverse_y(y, mode):
    if mode == "log":
        return np.exp(y)
    if mode == "inverse":
        # Invalid reciprocal predictions are surfaced, not silently clipped.
        if np.any(y <= 0):
            raise ValueError("Nonpositive inverse target prediction")
        return 1/y
    return y


def fit(config, x, y):
    values = transform_y(y, config["transform"])
    if config["method"] == "gp_restart":
        fitted = prior.physical.fit(config, x, values, np.empty((len(x), 5, 4)))
        return dict(config=config, gp=fitted)
    f = features(x, config["mode"])
    if config["method"] == "cat":
        estimator = CatBoostRegressor(
            loss_function="RMSE", iterations=config["iterations"], depth=config["depth"],
            learning_rate=.035, l2_leaf_reg=config["penalty"], bootstrap_type="Bayesian",
            bagging_temperature=.5, random_strength=.5, random_seed=2026101721,
            thread_count=1, verbose=False, allow_writing_files=False)
    else:
        estimator = HistGradientBoostingRegressor(
            max_iter=config["iterations"], max_leaf_nodes=config["leaves"],
            min_samples_leaf=15, l2_regularization=config["penalty"], learning_rate=.04,
            early_stopping=False, random_state=2026101721)
    estimator.fit(f, values)
    return dict(config=config, estimator=estimator)


def predict(model, x):
    config = model["config"]
    values = (prior.physical.predict(model["gp"], x) if "gp" in model else
              model["estimator"].predict(features(x, config["mode"])))
    return inverse_y(values, config["transform"])


def task(args):
    target, index, split, fold, a, b, x, y = args
    start = time.monotonic()
    fitted = fit(CONFIGS[index], x[a], y[a])
    pred = predict(fitted, x[b])
    return dict(target=target, config=index, split=split, fold=fold, validation=b.tolist(),
                prediction=pred.tolist(), actual=y[b].tolist(),
                train_r2=prior.direct.base.metric(y[a], predict(fitted, x[a]))["r2"],
                seconds=time.monotonic()-start, **prior.direct.base.metric(y[b], pred))


def summary(records, out):
    rows = []
    for target in TARGETS:
        for index in range(len(CONFIGS)):
            for split in ("random", "spatial"):
                parts = [r for r in records if (r["target"], r["config"], r["split"]) == (target, index, split)]
                if len(parts) != 3:
                    continue
                rows.append(dict(target=target, config=index, split=split, **CONFIGS[index],
                    train_r2=float(np.mean([r["train_r2"] for r in parts])),
                    **prior.direct.base.metric(np.concatenate([r["actual"] for r in parts]),
                                              np.concatenate([r["prediction"] for r in parts]))))
    pd.DataFrame(rows).to_csv(out / "训练交叉验证汇总.csv", index=False)


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    arrays, jobs = {}, []
    for target in TARGETS:
        mask = np.isfinite(frame[target].to_numpy(float))
        if target == TARGETS[1]:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        sub = frame.loc[mask]
        x, y = sub[FEATURES].to_numpy(float), sub[target].to_numpy(float)
        arrays[target] = (x, y)
        groups = prior.direct.base.geometry_groups(x)
        for split, folds in (
            ("random", KFold(3, shuffle=True, random_state=2026101722).split(x)),
            ("spatial", GroupKFold(3).split(x, groups=groups)),
        ):
            for fold, (a, b) in enumerate(folds):
                for index, config in enumerate(CONFIGS):
                    if target == TARGETS[0] and config["transform"] != "identity":
                        continue
                    jobs.append((target, index, split, fold, a, b, x, y))
    sources = [Path(__file__), Path(prior.__file__), Path(prior.physical.__file__),
               ROOT / "核心模型/气动几何特征.py", DATA]
    (out / "配置冻结.json").write_text(json.dumps(dict(
        configs=CONFIGS, selection="Minimum mean random/spatial OOF RMSE; single or two-model mixture",
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        counts={k: len(v[1]) for k, v in arrays.items()}, external_labels_used=False,
        fold_group="All angles of a geometry stay together", native_protocol="free-wake-legacy-v1"),
        indent=2), encoding="utf8")
    records = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, args) for args in jobs]):
            records.append(future.result())
            (out / "交叉验证逐折.json").write_text(json.dumps(records), encoding="utf8")
            summary(records, out)
            (out / "进度.json").write_text(json.dumps(dict(
                stage="training_cv", completed=len(records), requested=len(jobs),
                elapsed_s=time.monotonic()-start)), encoding="utf8")
            print("Geometry interactions CV", len(records), "/", len(jobs), flush=True)
    selected = []
    for target in TARGETS:
        x, y = arrays[target]
        results = [r for r in records if r["target"] == target]
        indices = [i for i in range(len(CONFIGS)) if
                   any(r["config"] == i for r in results)]
        predictions = [prior.records_matrix(results, len(y), i) for i in indices]
        score, chosen, oof = prior.select_mixture(predictions, y)
        components = []
        for i, weight in chosen:
            index = indices[i]
            fitted = fit(CONFIGS[index], x, y)
            path = out / f"target_{TARGETS.index(target)}_config_{index}.joblib"
            joblib.dump(fitted, path)
            components.append(dict(config=index, settings=CONFIGS[index], weight=weight,
                                   path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        selected.append(dict(target=target, score=score, components=components,
                             interval_std=max(1e-9, float(np.quantile(abs(oof-y), .9)/1.645)),
                             oof=[dict(split=s, **prior.direct.base.metric(y, p))
                                  for s, p in zip(("random", "spatial"), oof)]))
    (out / "模型选择.json").write_text(json.dumps(dict(
        selected=selected, external_labels_used=False,
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    (out / "进度.json").write_text(json.dumps(dict(
        stage="completed", completed=len(records), requested=len(jobs),
        elapsed_s=time.monotonic()-start)), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run(args.output_dir, args.workers)
