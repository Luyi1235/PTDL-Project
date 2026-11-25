"""Compare local tree policies using training-only random and spatial folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import tempfile
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

import 研究_密度均衡与残差代理20261017 as prior
from 局部几何气动代理 import LocalGeometryAero, model_features, numeric_tree_predict

ROOT = prior.ROOT
OUT = ROOT / "运行结果/局部细分尾迹代理_20261017"
TARGET = prior.TARGETS[1]
CONFIGS = [
    dict(mode=mode, policy=policy, weighting=weighting, iterations=1200,
         depth=6, penalty=20., min_data=15, max_leaves=24)
    for mode in ("interference", "wake_path")
    for policy in ("SymmetricTree", "Depthwise", "Lossguide")
    for weighting in ("none", "density")
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(config, x, y, acquisition):
    settings = dict(
        loss_function="RMSE", grow_policy=config["policy"], iterations=config["iterations"],
        depth=config["depth"], learning_rate=.035, l2_leaf_reg=config["penalty"],
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        random_seed=2026101740, thread_count=1, verbose=False, allow_writing_files=False)
    if config["policy"] != "SymmetricTree":
        settings["min_data_in_leaf"] = config["min_data"]
    if config["policy"] == "Lossguide":
        settings["max_leaves"] = config["max_leaves"]
    estimator = CatBoostRegressor(**settings)
    estimator.fit(model_features(x, config["mode"]), y,
                  sample_weight=prior.sample_weights(x, acquisition, config["weighting"]))
    return estimator


def export(estimator, mode):
    with tempfile.TemporaryDirectory(prefix="ptdl_local_") as folder:
        path = Path(folder) / "native.json"
        estimator.save_model(str(path), format="json")
        source = json.loads(path.read_text(encoding="utf8"))
    scale, bias = source["scale_and_bias"]
    key = "oblivious_trees" if "oblivious_trees" in source else "trees"
    return dict(features=mode, scale=scale, bias=bias[0], **{key: source[key]})


def task(args):
    index, split, fold, a, b, x, y, acquisition = args
    started = time.monotonic()
    config = CONFIGS[index]
    estimator = fit(config, x[a], y[a], acquisition[a])
    prediction = estimator.predict(model_features(x[b], config["mode"]))
    trained = estimator.predict(model_features(x[a], config["mode"]))
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=prediction.tolist(),
                train_r2=prior.previous.prior.direct.base.metric(y[a], trained)["r2"],
                seconds=time.monotonic()-started,
                **prior.previous.prior.direct.base.metric(y[b], prediction))


def reference_oof(n):
    selected = json.loads((prior.OUT / "模型选择.json").read_text(encoding="utf8"))["selected"][1]
    records = json.loads((prior.OUT / "交叉验证逐折.json").read_text(encoding="utf8"))
    records = [r for r in records if r["target"] == TARGET]
    values = np.zeros((2, n))
    for component in selected["components"]:
        prediction = (prior.reference_oof(TARGET, n) if component["family"] == "reference"
                      else prior.previous.prior.records_matrix(records, n, component["config"]))
        values += component["weight"] * prediction
    return values


def main(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    data = prior.previous.DATA
    frame = pd.read_csv(data)
    valid = frame.wake_quality_pass.fillna(False).to_numpy(bool).copy()
    valid &= np.isfinite(frame[TARGET].to_numpy(float))
    sub = frame.loc[valid]
    x, y = sub[prior.FEATURES].to_numpy(float), sub[TARGET].to_numpy(float)
    acquisition = sub.selection_basis.to_numpy(str)
    paths = (Path(__file__), ROOT / "核心模型/局部几何气动代理.py", data,
             prior.REFERENCE, prior.OUT / "密度均衡候选代理.json",
             prior.OUT / "模型选择.json", prior.OUT / "交叉验证逐折.json")
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    freeze = dict(configs=CONFIGS, hashes=hashes, training_count=len(y),
                  training_data_only=True, external_labels_used=False,
                  selection="Mean random/spatial OOF RMSE; convex pair including frozen reference",
                  source_training_count=len(frame), started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = out / "配置冻结.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        if old["hashes"] != hashes or old["configs"] != CONFIGS:
            raise RuntimeError("Local tree experiment changed; use a new output folder")
    else:
        path.write_text(json.dumps(freeze, indent=2), encoding="utf8")
    groups = prior.previous.prior.direct.base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101722).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, x, y, acquisition) for i in range(len(CONFIGS)))
    records_path = out / "交叉验证逐折.json"
    records = json.loads(records_path.read_text(encoding="utf8")) if records_path.exists() else []
    done = {(r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:3] not in done]
    requested = len(jobs) + len(records)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            records_path.write_text(json.dumps(records), encoding="utf8")
            (out / "进度.json").write_text(json.dumps(dict(
                stage="training_cv", completed=len(records), requested=requested,
                elapsed_s=time.monotonic()-started)), encoding="utf8")
            print("Local wake CV", len(records), "/", requested, flush=True)
    predictions = [prior.previous.prior.records_matrix(records, len(y), i)
                   for i in range(len(CONFIGS))]
    predictions.append(reference_oof(len(y)))
    score, chosen, oof = prior.previous.prior.select_mixture(predictions, y)
    baseline_score = np.mean([prior.previous.prior.direct.base.metric(y, p)["rmse"]
                              for p in predictions[-1]])
    interval = max(1e-9, float(np.quantile(abs(oof-y), .9)/1.645))
    baseline = json.loads(prior.REFERENCE.read_text(encoding="utf8"))
    wake_reference = json.loads(
        (prior.OUT / "密度均衡候选代理.json").read_text(encoding="utf8"))["targets"][1]
    components, expected = [], np.zeros(80)
    probe = x[np.linspace(0, len(x)-1, 40).astype(int)]
    probe = np.vstack((probe, (probe + np.roll(probe, 1, axis=0))/2))
    exported_checks = []
    for index, weight in chosen:
        if index == len(CONFIGS):
            model = wake_reference
            component = dict(kind="reference", model=model, weight=weight)
            native_prediction = np.array([prior.previous.prior.direct.base.target_mean(
                model, row[None]) for row in probe])
        else:
            config = CONFIGS[index]
            estimator = fit(config, x, y, acquisition)
            estimator.save_model(str(out / f"wake_config_{index}.cbm"))
            model = export(estimator, config["mode"])
            native_prediction = estimator.predict(model_features(probe, config["mode"]))
            portable_prediction = numeric_tree_predict(model, probe)
            np.testing.assert_allclose(portable_prediction, native_prediction, rtol=1e-9, atol=1e-9)
            exported_checks.append(dict(config=index, max_abs_difference=float(
                np.max(abs(portable_prediction-native_prediction)))))
            component = dict(kind="local-tree", model=model, weight=weight, config=index)
        components.append(component)
        expected += weight * native_prediction
    payload = dict(schema="ptdl-local-geometry-aero-v1", baseline=baseline,
                   training_samples=len(frame), wake_valid_samples=len(y),
                   wake_components=components, wake_interval_std=interval,
                   automatic_production_promotion=False, training_only_selection=True,
                   uncertainty="Fixed training OOF absolute error band; not a calibrated probability")
    candidate_path = out / "局部几何候选代理.json"
    LocalGeometryAero(payload).to_json(candidate_path)
    restored = LocalGeometryAero.from_json(candidate_path)
    np.testing.assert_allclose(restored.predict_many(probe)[:, 1], expected, rtol=1e-9, atol=1e-9)
    summaries = []
    for index, values in enumerate(predictions):
        for split, p in zip(("random", "spatial"), values):
            summaries.append(dict(config=index, split=split,
                **prior.previous.prior.direct.base.metric(y, p),
                **(CONFIGS[index] if index < len(CONFIGS) else {"policy": "reference"})))
    pd.DataFrame(summaries).to_csv(out / "训练交叉验证汇总.csv", index=False)
    selection = dict(
        components=[dict(config=i, weight=w) for i, w in chosen],
        score=score, reference_score=float(baseline_score), external_labels_used=False,
        oof=[dict(split=s, **prior.previous.prior.direct.base.metric(y, p))
             for s, p in zip(("random", "spatial"), oof)],
        model_sha256=digest(candidate_path), model_path=str(candidate_path),
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), export_checks=exported_checks,
        automatic_production_promotion=False)
    (out / "模型选择.json").write_text(json.dumps(selection, indent=2), encoding="utf8")
    (out / "进度.json").write_text(json.dumps(dict(stage="completed", completed=len(records),
        requested=requested, elapsed_s=time.monotonic()-started)), encoding="utf8")
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.workers)
