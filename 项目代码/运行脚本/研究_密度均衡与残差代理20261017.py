"""Training-only density balancing, robust trees and cross-fitted GP corrections."""

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

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import KFold, GroupKFold

import 研究_几何交互代理20261017 as previous

ROOT = previous.ROOT
OUT = ROOT / "运行结果/密度均衡与残差代理_20261017"
FEATURES, TARGETS = previous.FEATURES, previous.TARGETS
REFERENCE = previous.OUT / "几何交互候选代理.json"


def configurations():
    rows = []
    for mode, depth, penalty in (("physics", 4, 3.), ("interference", 6, 20.)):
        for weighting in ("none", "density", "acquisition"):
            for loss in ("RMSE", "Huber"):
                rows.append(dict(method="tree", mode=mode, depth=depth, penalty=penalty,
                                 weighting=weighting, loss=loss, ordered=False, iterations=1200))
        for weighting in ("none", "density"):
            rows.append(dict(method="tree", mode=mode, depth=depth, penalty=penalty,
                             weighting=weighting, loss="RMSE", ordered=True, iterations=1200))
    for mode, kernel in (("physics", "rbf"), ("dimensionless", "matern32")):
        for shrinkage in (.35, .7):
            rows.append(dict(method="residual", mode=mode, kernel=kernel, shrinkage=shrinkage,
                             inner_folds=3, iterations=1000))
    return rows


CONFIGS = configurations()


def sample_weights(x, acquisition, mode):
    if mode == "none":
        return np.ones(len(x))
    if mode == "acquisition":
        targeted = np.isin(acquisition, ("gp_uncertainty", "training_oof_error_neighborhood"))
        weights = np.where(targeted, .35, 1.)
    elif mode == "density":
        scale = np.maximum(np.ptp(x, axis=0), 1e-9)
        z = x/scale
        k = min(21, len(x))
        radius = cKDTree(z).query(z, k=k)[0][:, -1]
        weights = np.clip((radius/max(float(np.median(radius)), 1e-9))**2, .3, 3.)
    else:
        raise ValueError("Unknown acquisition weighting")
    return weights/weights.mean()


def fit_tree(config, x, y, acquisition):
    delta = max(.005, .5*float(np.std(y)))
    estimator = CatBoostRegressor(
        iterations=config["iterations"], depth=config["depth"], learning_rate=.035,
        l2_leaf_reg=config["penalty"], loss_function=(
            "RMSE" if config["loss"] == "RMSE" else f"Huber:delta={delta:.12g}"),
        boosting_type="Ordered" if config["ordered"] else "Plain",
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        random_seed=2026101728, thread_count=1, verbose=False, allow_writing_files=False,
    )
    weights = sample_weights(x, acquisition, config["weighting"])
    estimator.fit(previous.features(x, config["mode"]), y, sample_weight=weights)
    return dict(config=config, estimator=estimator)


def tree_predict(fitted, x):
    return fitted["estimator"].predict(previous.features(x, fitted["config"]["mode"]))


def fit(config, x, y, acquisition, target):
    if config["method"] == "tree":
        return fit_tree(config, x, y, acquisition)
    lift = target == TARGETS[0]
    starter = dict(method="tree", mode="physics" if lift else "interference",
                   depth=4 if lift else 6, penalty=3. if lift else 20.,
                   weighting="none", loss="RMSE", ordered=False, iterations=config["iterations"])
    oof = np.full(len(y), np.nan)
    for a, b in KFold(config["inner_folds"], shuffle=True, random_state=2026101729).split(x):
        fitted = fit_tree(starter, x[a], y[a], acquisition[a])
        oof[b] = tree_predict(fitted, x[b])
    if not np.isfinite(oof).all():
        raise RuntimeError("Residual training requires complete inner OOF predictions")
    f = previous.features(x, config["mode"])
    center, scale = f.mean(0), np.maximum(f.std(0), 1e-9)
    z = (f-center)/scale
    radial = (RBF(1., (.1, 25.)) if config["kernel"] == "rbf" else
              Matern(1., (.1, 25.), nu=1.5))
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (.01, 20.))*radial+WhiteKernel(.3, (.03, 2.)),
        normalize_y=True, alpha=1e-8, optimizer=previous.prior.direct.base.optimize,
        n_restarts_optimizer=0, random_state=2026101730)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(z, y-oof)
    return dict(config=config, base=fit_tree(starter, x, y, acquisition), estimator=gp,
                center=center, scale=scale, inner_oof=oof)


def predict(fitted, x):
    config = fitted["config"]
    if config["method"] == "tree":
        return tree_predict(fitted, x)
    z = (previous.features(x, config["mode"])-fitted["center"])/fitted["scale"]
    return tree_predict(fitted["base"], x)+config["shrinkage"]*fitted["estimator"].predict(z)


def task(args):
    target, index, split, fold, a, b, x, y, acquisition = args
    started = time.monotonic()
    fitted = fit(CONFIGS[index], x[a], y[a], acquisition[a], target)
    p = predict(fitted, x[b])
    return dict(target=target, config=index, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=p.tolist(),
                train_r2=previous.prior.direct.base.metric(y[a], predict(fitted, x[a]))["r2"],
                seconds=time.monotonic()-started, **previous.prior.direct.base.metric(y[b], p))


def reference_oof(target, n):
    selected = json.loads((previous.OUT / "模型选择.json").read_text(encoding="utf8"))["selected"]
    entry = next(s for s in selected if s["target"] == target)
    records = json.loads((previous.OUT / "交叉验证逐折.json").read_text(encoding="utf8"))
    records = [r for r in records if r["target"] == target]
    return sum(c["weight"]*previous.prior.records_matrix(records, n, c["config"])
               for c in entry["components"])


def main(out=OUT, workers=12):
    out.mkdir(parents=True, exist_ok=True)
    paths = (Path(__file__), Path(previous.__file__), previous.DATA, REFERENCE,
             previous.OUT / "模型选择.json", previous.OUT / "交叉验证逐折.json")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frozen_path = out / "配置冻结.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf8"))
        if frozen["hashes"] != manifest or frozen["configs"] != CONFIGS:
            raise RuntimeError("Frozen training configuration changed; use a new output directory")
    else:
        frozen_path.write_text(json.dumps(dict(
            configs=CONFIGS, hashes=manifest, external_labels_used=False,
            selection="Training-only mean random/spatial OOF RMSE, including frozen reference",
            residual_labels="Inner-fold prediction errors; each outer fit uses its training rows only",
            acquisition_weights="Targeted active samples 0.35; other rows 1.0; all rows retained",
            density_weights="Training-only geometry kNN radius squared, clipped to [0.3,3.0]",
            started_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    frame = pd.read_csv(previous.DATA)
    arrays, jobs = {}, []
    for target in TARGETS:
        mask = np.isfinite(frame[target].to_numpy(float))
        if target == TARGETS[1]:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        sub = frame.loc[mask]
        x, y = sub[FEATURES].to_numpy(float), sub[target].to_numpy(float)
        acquisition = sub.selection_basis.to_numpy(str)
        arrays[target] = (x, y, acquisition)
        groups = previous.prior.direct.base.geometry_groups(x)
        for split, folds in (
            ("random", KFold(3, shuffle=True, random_state=2026101722).split(x)),
            ("spatial", GroupKFold(3).split(x, groups=groups)),
        ):
            for fold, (a, b) in enumerate(folds):
                jobs.extend((target, i, split, fold, a, b, x, y, acquisition)
                            for i in range(len(CONFIGS)))
    started = time.monotonic()
    records_path = out / "交叉验证逐折.json"
    records = json.loads(records_path.read_text(encoding="utf8")) if records_path.exists() else []
    completed = {(r["target"], r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:4] not in completed]
    total = len(records)+len(jobs)
    def progress(stage):
        (out / "进度.json").write_text(json.dumps(dict(
            stage=stage, completed=len(records), requested=total, pid=os.getpid(),
            elapsed_s=time.monotonic()-started, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")),
            indent=2), encoding="utf8")
    progress("training")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            records_path.write_text(json.dumps(records), encoding="utf8")
            progress("training")
            print("Balanced residual CV", len(records), "/", total, flush=True)
    progress("selection")
    summary, selected = [], []
    for target in TARGETS:
        x, y, acquisition = arrays[target]
        subset = [r for r in records if r["target"] == target]
        predictions = [previous.prior.records_matrix(subset, len(y), i) for i in range(len(CONFIGS))]
        predictions.append(reference_oof(target, len(y)))
        for i, values in enumerate(predictions):
            for split, p in zip(("random", "spatial"), values):
                summary.append(dict(target=target, config=i, split=split,
                                    family="reference" if i == len(CONFIGS) else CONFIGS[i]["method"],
                                    **previous.prior.direct.base.metric(y, p)))
        score, chosen, oof = previous.prior.select_mixture(predictions, y)
        components = []
        for i, weight in chosen:
            if i == len(CONFIGS):
                components.append(dict(family="reference", weight=weight, path=str(REFERENCE),
                                       target_index=TARGETS.index(target),
                                       sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()))
            else:
                fitted = fit(CONFIGS[i], x, y, acquisition, target)
                path = out / f"target_{TARGETS.index(target)}_config_{i}.joblib"
                joblib.dump(fitted, path)
                components.append(dict(family="candidate", config=i, settings=CONFIGS[i], weight=weight,
                                       path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        selected.append(dict(target=target, score=score, components=components,
                             interval_std=max(1e-9, float(np.quantile(abs(oof-y), .9)/1.645)),
                             oof=[dict(split=s, **previous.prior.direct.base.metric(y, p))
                                  for s, p in zip(("random", "spatial"), oof)]))
    pd.DataFrame(summary).to_csv(out / "训练交叉验证汇总.csv", index=False)
    (out / "模型选择.json").write_text(json.dumps(dict(
        selected=selected, external_labels_used=False, training_samples=len(frame),
        selected_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    progress("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    main(args.output_dir, args.workers)
