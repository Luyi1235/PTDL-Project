"""Training-only comparison of grouped and multiscale aerodynamic kernels."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel, Hyperparameter, ConstantKernel, Matern, RBF, WhiteKernel, Sum, Product,
)
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map
from VSPAERO气动代理模型 import FEATURE_LABELS
from 训练_主动增量混合代理20261016 import metric

TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


class FeatureSubset(Kernel):
    def __init__(self, base, columns):
        self.base = base
        self.columns = columns

    @property
    def hyperparameters(self):
        return [Hyperparameter("base__"+h.name, h.value_type, h.bounds, h.n_elements, h.fixed)
                for h in self.base.hyperparameters]

    @property
    def theta(self):
        return self.base.theta

    @theta.setter
    def theta(self, value):
        self.base.theta = value

    @property
    def bounds(self):
        return self.base.bounds

    def __call__(self, x, y=None, eval_gradient=False):
        return self.base(x[:, self.columns], None if y is None else y[:, self.columns],
                         eval_gradient=eval_gradient)

    def diag(self, x):
        return self.base.diag(x[:, self.columns])

    def is_stationary(self):
        return self.base.is_stationary()


def make_kernel(kind):
    def group(columns):
        return ConstantKernel(1., (.001, 100.))*FeatureSubset(
            Matern(np.ones(len(columns)), (.05, 50.), nu=1.5), columns)
    full = ConstantKernel(1., (.001, 100.))*Matern(np.ones(9), (.05, 50.), nu=1.5)
    grouped = group((0, 1, 2, 3))+group((4, 5, 6, 7, 8))
    if kind == "full":
        kernel = full
    elif kind == "grouped":
        kernel = grouped
    elif kind == "grouped-interaction":
        kernel = grouped+full
    elif kind == "multiscale":
        kernel = full+ConstantKernel(1., (.001, 100.))*RBF(np.ones(9), (.05, 50.))
    else:
        raise ValueError(kind)
    return kernel+WhiteKernel(.001, (1e-8, 1.))


def encode_kernel(kernel):
    if isinstance(kernel, Sum):
        return dict(kind="sum", left=encode_kernel(kernel.k1), right=encode_kernel(kernel.k2))
    if isinstance(kernel, Product):
        return dict(kind="product", left=encode_kernel(kernel.k1), right=encode_kernel(kernel.k2))
    if isinstance(kernel, FeatureSubset):
        return dict(kind="subset", columns=list(kernel.columns), base=encode_kernel(kernel.base))
    if isinstance(kernel, ConstantKernel):
        return dict(kind="constant", value=float(kernel.constant_value))
    if isinstance(kernel, WhiteKernel):
        return dict(kind="noise", value=float(kernel.noise_level))
    if isinstance(kernel, Matern):
        return dict(kind="matern", length=np.atleast_1d(kernel.length_scale).tolist(), nu=kernel.nu)
    if isinstance(kernel, RBF):
        return dict(kind="rbf", length=np.atleast_1d(kernel.length_scale).tolist())
    raise ValueError(type(kernel))


def cross_covariance(kernel, x, y):
    kind = kernel["kind"]
    if kind in ("sum", "product"):
        a = cross_covariance(kernel["left"], x, y)
        b = cross_covariance(kernel["right"], x, y)
        return a+b if kind == "sum" else a*b
    if kind == "subset":
        return cross_covariance(kernel["base"], x[:, kernel["columns"]], y[:, kernel["columns"]])
    if kind == "constant":
        return np.full((len(x), len(y)), kernel["value"])
    if kind == "noise":
        return np.zeros((len(x), len(y)))
    r2 = np.sum(((x[:, None, :]-y[None, :, :])/kernel["length"])**2, axis=2)
    if kind == "rbf":
        return np.exp(-.5*r2)
    if kind == "matern" and kernel["nu"] == 1.5:
        radius = np.sqrt(3*np.maximum(r2, 0))
        return (1+radius)*np.exp(-radius)
    raise ValueError(kind)


def predict(model, x):
    z = (feature_map(x, "physics")-model["feature_mean"])/model["feature_scale"]
    value = model["mean"]+model["scale"]*(
        cross_covariance(model["covariance"], z, np.asarray(model["train"])) @ model["dual"])
    return np.exp(value) if model["transform"] == "log" else value


def fit(x, y, kind, transform, restarts=2):
    features = feature_map(x, "physics")
    mean, scale = features.mean(0), np.maximum(features.std(0), 1e-9)
    z = (features-mean)/scale
    converted = np.log(y) if transform == "log" else y
    gp = GaussianProcessRegressor(kernel=make_kernel(kind), alpha=1e-8, normalize_y=True,
                                  n_restarts_optimizer=restarts, random_state=2026101641)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(z, converted)
    payload = dict(covariance=encode_kernel(gp.kernel_), feature_mean=mean.tolist(),
                   feature_scale=scale.tolist(), train=z.tolist(), dual=gp.alpha_.tolist(),
                   inverse=cho_solve((gp.L_, True), np.eye(len(z))).tolist(),
                   mean=float(gp._y_train_mean), scale=float(gp._y_train_std),
                   transform=transform, hyperparameter_count=len(gp.kernel_.theta),
                   log_marginal_likelihood=float(gp.log_marginal_likelihood_value_))
    actual = predict(payload, x[:5])
    reference = gp.predict(z[:5])
    if transform == "log":
        reference = np.exp(reference)
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
    return payload


def train(dataset, out, workers=8, fold_seed=2026101506):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(dataset)
    if frame.duplicated(list(FEATURE_LABELS)).any():
        raise ValueError("Duplicate training geometry")
    frame.to_csv(out / "冻结训练数据.csv", index=False)
    lock = dict(dataset=str(dataset), dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                samples=len(frame), folds=5, fold_seed=fold_seed,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                heldout_labels_used=False, target_definition="Unchanged lift slope and median VSPAERO Ew")
    (out / "训练前配置.json").write_text(json.dumps(lock, indent=2), encoding="utf8")
    arrays = {}
    tasks = []
    for t, target in enumerate(TARGETS):
        mask = np.isfinite(frame[target].to_numpy(float))
        if t:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        subset = frame.loc[mask]
        if len(subset) < 15:
            raise ValueError("At least fifteen usable labels required per target")
        x, y = subset[list(FEATURE_LABELS)].to_numpy(float), subset[target].to_numpy(float)
        arrays[t] = (x, y, list(KFold(5, shuffle=True, random_state=fold_seed).split(x)))
        for kind in ("full", "grouped", "grouped-interaction", "multiscale"):
            for transform in (("identity", "log") if t else ("identity",)):
                tasks.append((t, kind, transform))

    def run(task):
        t, kind, transform = task
        x, y, folds = arrays[t]
        started = time.monotonic()
        prediction = np.zeros(len(y))
        for a, b in folds:
            prediction[b] = predict(fit(x[a], y[a], kind, transform), x[b])
        record = dict(target=t, kernel=kind, transform=transform, count=len(y),
                      wall_time_s=time.monotonic()-started, **metric(y, prediction))
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            out / f"折外_{t}_{kind}_{transform}.csv", index=False)
        print(json.dumps(record), flush=True)
        return record, prediction

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, tasks))
    records = [record for record, _ in results]
    pd.DataFrame(records).to_csv(out / "分组多尺度五折比较.csv", index=False)
    winners, models = [], []
    for t in range(2):
        record, predictions = min((row for row in results if row[0]["target"] == t),
                                  key=lambda row: row[0]["rmse"])
        x, y, _ = arrays[t]
        model = fit(x, y, record["kernel"], record["transform"])
        model["empirical_interval_std"] = max(1e-9, float(np.quantile(abs(predictions-y), .9,
                                                                   method="higher")/1.645))
        models.append(model)
        winners.append(record)
    payload = dict(schema="ptdl_research_kernel_gp_v1", feature_labels=list(FEATURE_LABELS),
                   training_samples=len(frame), targets=models, selection=winners)
    path = out / "分组多尺度候选.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf8")
    (out / "模型冻结.json").write_text(json.dumps(dict(
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), selected=winners,
        automatic_production_promotion=False), indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "运行结果" / "主动增量代理_20261015" / "增量训练样本.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "分组多尺度Kriging_20261016")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fold-seed", type=int, default=2026101506)
    args = parser.parse_args()
    train(args.dataset, args.output_dir, args.workers, args.fold_seed)
