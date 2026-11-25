"""Retain all labels and compare geometry-conditioned wake models and sampling risk."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
import 研究_局部线性与非对称树20261018 as utility
import 推进_分歧补样4096代理20261018 as batch
from 均匀风险尾迹代理 import UniformRiskWakeAero, features
from 尾迹射线网格特征 import FEATURE_NAMES, extract

OUT = ROOT / "运行结果/尾迹网格与均匀风险_20261018"
DATA = batch.OUT / "完整训练标签.csv"
BASELINE = batch.OUT / "精细训练/精细气动候选代理.json"
TARGET, FEATURES = batch.TARGETS[1], batch.FEATURES
helpers = utility.helpers
CONFIGS = [
    dict(mesh=mesh, weighting=weighting, transform=transform, policy=policy,
         depth=depth, penalty=penalty, iterations=iterations)
    for mesh in (False, True) for weighting in ("equal", "uniform-priority")
    for transform, policy, depth, penalty, iterations in (
        ("identity", "Ordered", 6, 40., 1300),
        ("canonical-area", "Plain", 5, 20., 1800),
        ("canonical-area", "Depthwise", 6, 60., 1300),
    )
]


def extract_row(item):
    index, row = item
    path = Path(row["geometry_file"]).with_suffix(".vspgeom") if isinstance(
        row.get("geometry_file"), str) else None
    try:
        value = extract(path, row["wing_area_m2"], row["aspect_ratio"])
        audit = dict(row_index=index, sample=row["sample"], mesh_path=str(path),
                     mesh_sha256=utility.digest(path), error=None)
    except Exception as error:
        value = np.zeros(len(FEATURE_NAMES))
        audit = dict(row_index=index, sample=row["sample"], mesh_path=str(path),
                     mesh_sha256=None, error=repr(error))
    return index, value, audit


def prepare_features(frame, out, workers):
    values_path, audit_path = out / "训练网格特征.npz", out / "训练网格特征记录.csv"
    if values_path.exists() and audit_path.exists():
        values = np.load(values_path)["features"]
        if values.shape != (len(frame), len(FEATURE_NAMES)):
            raise RuntimeError("Incorrect mesh feature cache dimensions")
        return values
    values, audit = np.zeros((len(frame), len(FEATURE_NAMES))), [None]*len(frame)
    jobs = list(enumerate(frame.to_dict("records")))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for completed, future in enumerate(as_completed([pool.submit(extract_row, j) for j in jobs]), 1):
            index, value, record = future.result()
            values[index], audit[index] = value, record
            if completed % 64 == 0 or completed == len(jobs):
                utility.save(out / "进度.json", dict(stage="geometry_features",
                    completed=completed, requested=len(jobs),
                    updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    np.savez_compressed(values_path, features=values)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    return values


def fit(config, x, y, mesh, uniform):
    settings = dict(
        loss_function="RMSE", iterations=config["iterations"], depth=config["depth"],
        learning_rate=.035, l2_leaf_reg=config["penalty"],
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        rsm=.5, random_seed=2026101826, thread_count=1, verbose=False,
        allow_writing_files=False,
    )
    if config["policy"] == "Depthwise":
        settings.update(grow_policy="Depthwise", min_data_in_leaf=30)
    else:
        settings["boosting_type"] = config["policy"]
    weight = np.ones(len(y))
    if config["weighting"] == "uniform-priority":
        weight[~uniform] = .35
    estimator = CatBoostRegressor(**settings)
    estimator.fit(features(x, mesh, config), y, sample_weight=weight/weight.mean())
    return estimator


def task(args):
    index, split, fold, a, b, x, y, mesh, uniform = args
    start = time.monotonic()
    config = CONFIGS[index]
    estimator = fit(config, x[a], y[a], mesh[a], uniform[a])
    p = estimator.predict(features(x[b], mesh[b], config))
    trained = estimator.predict(features(x[a], mesh[a], config))
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=p.tolist(),
                seconds=time.monotonic()-start,
                train_r2=helpers.direct.base.metric(y[a], trained)["r2"],
                **helpers.direct.base.metric(y[b], p))


def reference_oof(count):
    records = batch.previous.read(batch.OUT / "精细训练/交叉验证逐折.json")
    selected = batch.previous.read(batch.OUT / "精细训练/模型选择.json")["selected"][1]
    if selected["count"] != count:
        raise RuntimeError("Reference OOF labels differ")
    records = [r for r in records if r["target"] == TARGET]
    return sum(c["weight"]*helpers.records_matrix(records, count, c["config"])
               for c in selected["components"])


def train(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), DATA, BASELINE, ROOT / "核心模型/尾迹射线网格特征.py",
             ROOT / "核心模型/均匀风险尾迹代理.py",
             batch.OUT / "精细训练/交叉验证逐折.json",
             batch.OUT / "精细训练/模型选择.json"]
    hashes = {str(p): utility.digest(p) for p in paths}
    freeze = out / "训练配置冻结.json"
    if freeze.exists():
        old = batch.previous.read(freeze)
        if hashes != old["hashes"] or CONFIGS != old["configs"]:
            raise RuntimeError("Training sources changed")
    else:
        utility.save(freeze, dict(hashes=hashes, configs=CONFIGS,
            source_raw_rows=4096, test_labels_used_for_training=False,
            previous_192_used_for_diagnosis=True, uniform_subset_defined_by_acquisition=True,
            feature_inputs="Geometry coordinates, connectivity, normals, trailing-edge nodes",
            target_definition_unchanged=True, created_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    frame = pd.read_csv(DATA)
    if len(frame) != 4096 or frame.duplicated(FEATURES).any():
        raise RuntimeError("Training cohort changed")
    mesh = prepare_features(frame, out, workers)
    valid = np.isfinite(frame[TARGET].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    x, y = frame.loc[valid, FEATURES].to_numpy(float), frame.loc[valid, TARGET].to_numpy(float)
    mesh = mesh[valid]
    uniform = frame.loc[valid, "selection_basis"].eq("uniform_feasible_coverage").to_numpy(bool)
    if uniform.sum() < 1000:
        raise RuntimeError("Insufficient uniform validation coverage")
    groups = helpers.direct.base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101803).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, x, y, mesh, uniform) for i in range(len(CONFIGS)))
    record_path = out / "交叉验证逐折.json"
    records = batch.previous.read(record_path) if record_path.exists() else []
    done = {(r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:3] not in done]
    total, started = len(records)+len(jobs), time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            utility.save(record_path, records)
            utility.save(out / "进度.json", dict(stage="training_cv", completed=len(records),
                requested=total, elapsed_s=time.monotonic()-started,
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            print("Wake ray CV", len(records), "/", total, flush=True)
    utility.check_hashes(hashes)
    predictions = [helpers.records_matrix(records, len(y), i) for i in range(len(CONFIGS))]
    predictions.append(reference_oof(len(y)))
    summaries = []
    for index, p in enumerate(predictions):
        for split, v in zip(("random", "spatial"), p):
            for subset, mask in (("all", np.ones(len(y), bool)), ("uniform", uniform)):
                summaries.append(dict(config=index, split=split, subset=subset,
                    count=int(mask.sum()), **helpers.direct.base.metric(y[mask], v[mask])))
    pd.DataFrame(summaries).to_csv(out / "训练交叉验证汇总.csv", index=False)
    baseline = batch.previous.read(BASELINE)
    choices, fitted = [], {}
    for name, mask in (("uniform_risk", uniform), ("all_risk", np.ones(len(y), bool))):
        score, chosen, _ = helpers.select_mixture([p[:, mask] for p in predictions], y[mask])
        p = sum(weight*predictions[index] for index, weight in chosen)
        components = []
        utility.save(out / "进度.json", dict(stage="final_fit", completed=total, requested=total,
            model=name, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        for index, weight in chosen:
            if index == len(CONFIGS):
                components.append(dict(kind="reference", payload=baseline, weight=weight))
            else:
                if index not in fitted:
                    fitted[index] = fit(CONFIGS[index], x, y, mesh, uniform)
                components.append(dict(kind="tree", estimator=fitted[index],
                                       config=CONFIGS[index], weight=weight))
        path = out / (name+".joblib")
        payload = dict(schema="ptdl-uniform-risk-wake-v1", baseline=baseline,
                       wake_components=components, training_samples=len(frame))
        model = UniformRiskWakeAero(payload)
        expected = model.predict_many(x[:20], mesh[:20])
        joblib.dump(payload, path, compress=3)
        np.testing.assert_allclose(UniformRiskWakeAero.load(path).predict_many(x[:20], mesh[:20]),
                                   expected, atol=1e-10, rtol=1e-10)
        choices.append(dict(name=name, score=score, path=str(path), requires_mesh=model.requires_mesh,
            chosen=[dict(config=i, weight=w) for i, w in chosen],
            oof=[dict(split=split, subset=subset, count=int(m.sum()),
                 **helpers.direct.base.metric(y[m], v[m]))
                 for split, v in zip(("random", "spatial"), p)
                 for subset, m in (("all", np.ones(len(y), bool)), ("uniform", uniform))]))
    cache_paths = [out / "训练网格特征.npz", out / "训练网格特征记录.csv"]
    cache_paths += [Path(c["path"]) for c in choices]
    utility.save(out / "模型选择.json", dict(selected=choices,
        hashes={str(p): utility.digest(p) for p in cache_paths},
        target_definition_unchanged=True, test_labels_used_for_training=False,
        uniform_risk_selection_explicit=True, baseline_lift_preserved=True))
    utility.save(out / "进度.json", dict(stage="completed", completed=total, requested=total,
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        train(args.output_dir.resolve(), args.workers)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        utility.save(args.output_dir / "失败.json",
                     dict(error=repr(error), traceback=traceback.format_exc()))
        raise
