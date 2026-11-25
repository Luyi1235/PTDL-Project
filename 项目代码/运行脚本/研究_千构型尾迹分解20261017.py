"""Training-only model comparison for original VSPAERO wake efficiency labels."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import KFold, GroupKFold

import 研究_代理正则化与嵌套验证20261016 as base

ROOT = base.ROOT
SOURCE = ROOT / "运行结果/千构型原定义代理_20261017"
OUT = ROOT / "运行结果/千构型尾迹分解_20261017"
ANGLES = np.array([-3., -1., 1., 3., 5.])
FEATURES = list(base.FEATURE_LABELS)
TARGET = base.TARGETS[1]


def configs():
    values = []
    for mode in ("dimensionless", "raw"):
        for kernel, noise in (("matern32", .01), ("matern32", .2),
                              ("matern52", .01), ("matern52", "learn"),
                              ("rbf", .01), ("rbf", "learn")):
            values.append(dict(method="gp", mode=mode, kernel=kernel, noise=noise))
    for mode in ("dimensionless", "physics"):
        for leaves, penalty in ((7, 1.), (15, 1.), (31, 1.), (15, 10.)):
            values.append(dict(method="boost", mode=mode, leaves=leaves, penalty=penalty))
        for leaf in (2, 5):
            values.append(dict(method="extra", mode=mode, leaf=leaf))
        for leaves in (7, 15):
            for transform in ("identity", "log"):
                values.append(dict(method="angles", mode=mode, leaves=leaves,
                                   penalty=1., transform=transform))
    return values


CONFIGS = configs()


def read_data():
    frame = pd.read_csv(SOURCE / "原网格训练数据.csv")
    valid = frame.wake_quality_pass.to_numpy(bool).copy()
    valid &= np.isfinite(frame[TARGET].to_numpy(float))
    frame = frame.loc[valid].reset_index(drop=True)
    polars = []
    for _, row in frame.iterrows():
        path = Path(row.geometry_file).parent / "vspaero_polar.csv"
        polar = pd.read_csv(path).sort_values("alpha_deg")
        np.testing.assert_allclose(polar.alpha_deg, ANGLES)
        p = polar[["cl", "cl_wake", "cd_induced_wake", "e_wake"]].to_numpy(float)
        working = (abs(p[:, 1]) >= .15) & (p[:, 3] > 0)
        np.testing.assert_allclose(np.median(p[working, 3]), row[TARGET], atol=1e-10)
        polars.append(p)
    return frame, np.asarray(polars)


def boost(config):
    return HistGradientBoostingRegressor(
        max_iter=450, learning_rate=.045, max_leaf_nodes=config["leaves"],
        min_samples_leaf=10, l2_regularization=config["penalty"],
        early_stopping=False, random_state=2026101711)


def fit(config, x, y, polar):
    f = base.feature_map(x, config["mode"])
    center, scale = f.mean(0), np.maximum(f.std(0), 1e-9)
    z = (f-center)/scale
    model = dict(config=config, center=center, scale=scale)
    if config["method"] == "gp":
        length = np.ones(z.shape[1])
        kernel = (RBF(length, (.05, 100.)) if config["kernel"] == "rbf" else
                  Matern(length, (.05, 100.), nu=1.5 if config["kernel"] == "matern32" else 2.5))
        noise = (WhiteKernel(.03, (1e-5, .5)) if config["noise"] == "learn" else
                 WhiteKernel(config["noise"], "fixed"))
        estimator = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (.01, 100.))*kernel+noise,
            normalize_y=True, optimizer=base.optimize, alpha=1e-9,
            n_restarts_optimizer=0)
    elif config["method"] == "extra":
        estimator = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=config["leaf"],
                                        max_features=1., n_jobs=1, random_state=2026101711)
    else:
        estimator = boost(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(z, y)
    model["estimator"] = estimator
    if config["method"] == "angles":
        gates, components = [], []
        for angle in range(5):
            gate = boost(config).fit(z, polar[:, angle, 1])
            mask = (abs(polar[:, angle, 1]) >= .15) & (polar[:, angle, 3] > 0)
            values = polar[mask, angle, 3]
            if config["transform"] == "log":
                values = np.log(values)
            component = boost(config).fit(z[mask], values) if mask.sum() >= 20 else None
            gates.append(gate)
            components.append(component)
        model.update(gates=gates, components=components)
    return model


def predict(model, x):
    config = model["config"]
    z = (base.feature_map(x, config["mode"])-model["center"])/model["scale"]
    result = model["estimator"].predict(z)
    if config["method"] != "angles":
        return result
    gates = np.column_stack([m.predict(z) for m in model["gates"]])
    components = np.column_stack([m.predict(z) if m is not None else np.full(len(z), np.nan)
                                  for m in model["components"]])
    if config["transform"] == "log":
        components = np.exp(components)
    eligible = (abs(gates) >= .15) & np.isfinite(components) & (components > 0)
    for i in range(len(result)):
        if eligible[i].sum() >= 3:
            result[i] = np.median(components[i, eligible[i]])
    return result


def task(payload):
    index, split, fold, a, b, x, y, p = payload
    started = time.monotonic()
    model = fit(CONFIGS[index], x[a], y[a], p[a])
    prediction = predict(model, x[b])
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
                prediction=prediction.tolist(), actual=y[b].tolist(),
                train_r2=base.metric(y[a], predict(model, x[a]))["r2"],
                elapsed_s=time.monotonic()-started, **base.metric(y[b], prediction))


def summarize(records, out):
    rows = []
    for index in range(len(CONFIGS)):
        for split in ("random", "spatial"):
            parts = [r for r in records if r["config"] == index and r["split"] == split]
            if len(parts) != 3:
                continue
            actual = np.concatenate([r["actual"] for r in parts])
            prediction = np.concatenate([r["prediction"] for r in parts])
            rows.append(dict(config=index, split=split, **CONFIGS[index],
                             train_r2=np.mean([r["train_r2"] for r in parts]),
                             **base.metric(actual, prediction)))
    result = pd.DataFrame(rows)
    result.to_csv(out / "训练集交叉验证.csv", index=False)
    return result


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    frame, polar = read_data()
    x, y = frame[FEATURES].to_numpy(float), frame[TARGET].to_numpy(float)
    groups = base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101712).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((index, split, fold, a, b, x, y, polar) for index in range(len(CONFIGS)))
    freeze = dict(configurations=CONFIGS, train_count=len(y), external_labels_used=False,
                  target_definition="median VSPAERO Ew at abs(CLiw)>=0.15; original free wake",
                  selection="minimum average random/spatial OOF RMSE, training only",
                  hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), SOURCE / "原网格训练数据.csv")},
                  baseline_protocol="free-wake-legacy-v1", original96_reuse="retrospective only")
    (out / "配置冻结.json").write_text(json.dumps(freeze, indent=2), encoding="utf8")
    working = abs(polar[:, :, 1]) >= .15
    diagnostic = pd.DataFrame(dict(sample=frame["sample"], actual=y,
                                   working_count=working.sum(1),
                                   angle_mask=working.astype(int) @ (2**np.arange(5))))
    diagnostic.to_csv(out / "迎角组成.csv", index=False)
    records = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, j) for j in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            (out / "交叉验证逐折.json").write_text(json.dumps(records), encoding="utf8")
            summarize(records, out)
            (out / "进度.json").write_text(json.dumps(dict(
                stage="training_cv", completed=len(records), requested=len(jobs),
                elapsed_s=time.monotonic()-started)), encoding="utf8")
            print("Training CV", len(records), "/", len(jobs), flush=True)
    result = summarize(records, out)
    ranking = result.groupby("config")["rmse"].mean().sort_values()
    chosen = int(ranking.index[0])
    fitted = fit(CONFIGS[chosen], x, y, polar)
    joblib.dump(fitted, out / "尾迹候选.joblib")
    (out / "模型选择.json").write_text(json.dumps(dict(
        chosen=chosen, settings=CONFIGS[chosen], scores=ranking.to_dict(),
        model_sha256=hashlib.sha256((out / "尾迹候选.joblib").read_bytes()).hexdigest(),
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), external_labels_used=False),
        indent=2), encoding="utf8")
    (out / "进度.json").write_text(json.dumps(dict(
        stage="completed", completed=len(records), requested=len(jobs),
        elapsed_s=time.monotonic()-started)), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run(args.output_dir, args.workers)
