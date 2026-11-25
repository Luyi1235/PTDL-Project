"""Compare new tree families on frozen training data and lock fresh test predictions."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
import 研究_精细树与核代理20261018 as fine
import 推进_分歧补样4096代理20261018 as batch
from 局部线性树气动代理 import LocalLinearTreeAero, fitted_prediction, tree_features

OUT = ROOT / "运行结果/局部线性与非对称树_20261018"
FEATURES, TARGETS = batch.FEATURES, batch.TARGETS
helpers = fine.previous.prior.prior.previous.prior
CONFIGS = [
    dict(family="baseline", fine_index=i, **fine.CONFIGS[i]) for i in (1, 8, 9)
] + [
    dict(family="lightgbm", features=features, leaves=leaves, linear=linear,
         linear_penalty=penalty, min_leaf=min_leaf, iterations=1400)
    for features, leaves, linear, penalty, min_leaf in (
        ("shape", 15, False, 0., 25),
        ("interference", 15, False, 0., 25),
        ("interference", 31, False, 0., 35),
        ("wake_path", 15, False, 0., 35),
        ("shape", 7, True, 10., 40),
        ("shape", 15, True, 30., 40),
        ("interference", 7, True, 30., 50),
        ("interference", 15, True, 100., 60),
    )
] + [
    dict(family="asymmetric", features=features, grow_policy=policy,
         depth=depth, leaves=leaves, penalty=penalty, iterations=1700)
    for features, policy, depth, leaves, penalty in (
        ("interference", "Depthwise", 6, 31, 20.),
        ("interference", "Depthwise", 8, 63, 80.),
        ("interference", "Lossguide", 8, 31, 40.),
        ("wake_path", "Lossguide", 8, 31, 40.),
        ("wake_path", "Depthwise", 6, 31, 80.),
    )
] + [
    dict(family="extra_trees", features=features, min_leaf=min_leaf, iterations=600)
    for features, min_leaf in (("interference", 2), ("interference", 5), ("wake_path", 2))
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf8")
    temporary.replace(path)


def check_hashes(hashes):
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise RuntimeError("Frozen research artifact changed: " + name)


def snapshot(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "训练数据冻结.json"
    if manifest_path.exists():
        manifest = batch.previous.read(manifest_path)
        check_hashes(manifest["hashes"])
        return manifest
    original = batch.previous.read(batch.OUT / "配置冻结.json")
    jobs = [j for j in original["jobs"] if j["role"] == "train"][:768]
    rows, paths = [], []
    for job in jobs:
        path = batch.OUT / "solver" / job["sample"] / "record.json"
        row = batch.previous.read(path)
        batch.previous.validate_record(job, row)
        if row.get("wake_protocol") != original["protocol"] and not row.get("solver_failure"):
            raise RuntimeError("Native protocol changed")
        rows.append(row)
        paths.append(path)
    source = pd.read_csv(batch.SOURCE)
    frame = pd.concat((source, pd.DataFrame(rows)), ignore_index=True)
    if len(frame) != 3840 or frame.duplicated(FEATURES).any():
        raise RuntimeError("Unexpected snapshot size or duplicate geometry")
    data = out / "训练标签3840.csv"
    frame.to_csv(data, index=False)
    paths += [data, batch.SOURCE, batch.OUT / "配置冻结.json", Path(__file__),
              ROOT / "核心模型/局部线性树气动代理.py", Path(fine.__file__)]
    manifest = dict(
        hashes={str(p): digest(p) for p in paths},
        training_samples=len(frame), appended_training_samples=len(rows),
        appended_selection_basis=pd.DataFrame(rows).selection_basis.value_counts().to_dict(),
        configs=CONFIGS, test_labels_used=False,
        selection="Mean random and geometry-grouped three-fold OOF RMSE",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    save(manifest_path, manifest)
    return manifest


def fit(config, x, y):
    if config["family"] == "baseline":
        model = fine.fit(config, x, y)
        return dict(config=config, estimator=model["estimator"])
    values = tree_features(x, config["features"])
    columns = np.flatnonzero(values.std(0) > 1e-10)
    center = values[:, columns].mean(0)
    scale = np.maximum(values[:, columns].std(0), 1e-9)
    z = (values[:, columns] - center) / scale
    family = config["family"]
    if family == "lightgbm":
        estimator = LGBMRegressor(
            n_estimators=config["iterations"], learning_rate=.03,
            num_leaves=config["leaves"], min_child_samples=config["min_leaf"],
            reg_lambda=15., linear_tree=config["linear"],
            linear_lambda=config["linear_penalty"], max_bin=127,
            random_state=2026101804, n_jobs=1, verbosity=-1,
        )
    elif family == "asymmetric":
        settings = dict(
            iterations=config["iterations"], learning_rate=.035, depth=config["depth"],
            grow_policy=config["grow_policy"], min_data_in_leaf=25,
            l2_leaf_reg=config["penalty"], loss_function="RMSE",
            bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
            rsm=.8, thread_count=1, random_seed=2026101804, verbose=False,
            allow_writing_files=False,
        )
        if config["grow_policy"] == "Lossguide":
            settings["max_leaves"] = config["leaves"]
        estimator = CatBoostRegressor(**settings)
    else:
        estimator = ExtraTreesRegressor(
            n_estimators=config["iterations"], min_samples_leaf=config["min_leaf"],
            max_features=.8, random_state=2026101804, n_jobs=1,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        estimator.fit(z, y)
    return dict(config=config, estimator=estimator, columns=columns, center=center, scale=scale)


def task(args):
    target, index, split, fold, train, valid, x, y = args
    started = time.monotonic()
    model = fit(CONFIGS[index], x[train], y[train])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        p = fitted_prediction(model, x[valid])
        train_p = fitted_prediction(model, x[train])
    return dict(target=target, config=index, split=split, fold=fold,
                validation=valid.tolist(), actual=y[valid].tolist(), prediction=p.tolist(),
                seconds=time.monotonic()-started,
                train_r2=helpers.direct.base.metric(y[train], train_p)["r2"],
                **helpers.direct.base.metric(y[valid], p))


def train(out, workers):
    manifest = snapshot(out)
    if manifest["configs"] != CONFIGS:
        raise RuntimeError("Research configuration changed")
    frame = pd.read_csv(out / "训练标签3840.csv")
    arrays, jobs = {}, []
    for target in TARGETS:
        valid = np.isfinite(frame[target].to_numpy(float))
        if target == TARGETS[1]:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        x = frame.loc[valid, FEATURES].to_numpy(float)
        y = frame.loc[valid, target].to_numpy(float)
        arrays[target] = x, y
        groups = helpers.direct.base.geometry_groups(x)
        for split, folds in (
            ("random", KFold(3, shuffle=True, random_state=2026101803).split(x)),
            ("spatial", GroupKFold(3).split(x, groups=groups)),
        ):
            for fold, (a, b) in enumerate(folds):
                jobs.extend((target, i, split, fold, a, b, x, y) for i in range(len(CONFIGS)))
    record_path = out / "交叉验证逐折.json"
    records = batch.previous.read(record_path) if record_path.exists() else []
    done = {(r["target"], r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:4] not in done]
    total, started = len(records)+len(jobs), time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            save(record_path, records)
            save(out / "进度.json", dict(stage="training_cv", completed=len(records),
                requested=total, elapsed_s=time.monotonic()-started,
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            print("Local tree CV", len(records), "/", total, flush=True)
    check_hashes(manifest["hashes"])
    selection, summaries, targets, baseline_targets, oof_rows = [], [], [], [], []
    for target in TARGETS:
        x, y = arrays[target]
        subset = [r for r in records if r["target"] == target]
        predictions = [helpers.records_matrix(subset, len(y), i) for i in range(len(CONFIGS))]
        score, chosen, oof = helpers.select_mixture(predictions, y)
        baseline_score, baseline_chosen, baseline_oof = helpers.select_mixture(predictions[:3], y)
        targets.append(dict(components=[
            dict(fit(CONFIGS[i], x, y), weight=weight) for i, weight in chosen
        ], interval_std=max(1e-9, float(np.quantile(abs(oof-y), .9)/1.645))))
        baseline_targets.append(dict(components=[
            dict(fit(CONFIGS[i], x, y), weight=weight) for i, weight in baseline_chosen
        ], interval_std=max(1e-9, float(np.quantile(abs(baseline_oof-y), .9)/1.645))))
        entry = dict(target=target, count=len(y), mean_oof_rmse=score,
                     baseline_mean_oof_rmse=baseline_score,
                     relative_rmse_reduction=1-score/baseline_score,
                     components=[dict(config=i, weight=w) for i, w in chosen],
                     baseline_components=[dict(config=i, weight=w) for i, w in baseline_chosen],
                     oof=[], baseline_oof=[])
        for split_index, split in enumerate(("random", "spatial")):
            entry["oof"].append(dict(split=split, **helpers.direct.base.metric(y, oof[split_index])))
            entry["baseline_oof"].append(dict(
                split=split, **helpers.direct.base.metric(y, baseline_oof[split_index])))
            oof_rows.extend(dict(target=target, row_index=int(row), split=split,
                                 actual=float(y[row]), prediction=float(oof[split_index, row]),
                                 baseline_prediction=float(baseline_oof[split_index, row]))
                            for row in range(len(y)))
        selection.append(entry)
        for i, values in enumerate(predictions):
            for split, p in zip(("random", "spatial"), values):
                train_r2 = np.mean([r["train_r2"] for r in subset
                                    if r["config"] == i and r["split"] == split])
                summaries.append(dict(target=target, config=i, split=split, **CONFIGS[i],
                    mean_train_r2=float(train_r2), **helpers.direct.base.metric(y, p)))
    reference = batch.previous.read(fine.OUT / "精细气动候选代理.json")["baseline_targets"]
    model_paths = {}
    probe = frame.iloc[:40][FEATURES].to_numpy(float)
    for name, values in (("local3840", targets), ("same_data_baseline3840", baseline_targets)):
        path = out / (name + ".joblib")
        payload = dict(schema="ptdl-local-linear-tree-aero-v1", training_samples=len(frame),
                       targets=values, baseline_targets=reference)
        expected = LocalLinearTreeAero(payload).predict_many(probe)
        joblib.dump(payload, path, compress=3)
        np.testing.assert_allclose(LocalLinearTreeAero.load(path).predict_many(probe), expected,
                                   atol=1e-10, rtol=1e-10)
        model_paths[name] = str(path)
    pd.DataFrame(summaries).to_csv(out / "训练交叉验证汇总.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "训练交叉验证逐点.csv", index=False)
    save(out / "模型选择.json", dict(selected=selection, model_paths=model_paths,
         model_hashes={p: digest(p) for p in model_paths.values()}, test_labels_used=False))
    lock_predictions(out)
    save(out / "进度.json", dict(stage="models_frozen_waiting_fresh_test", completed=total,
         requested=total, elapsed_s=time.monotonic()-started,
         updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))


def lock_predictions(out):
    path = out / "独立预测冻结.json"
    if path.exists():
        check_hashes(batch.previous.read(path)["hashes"])
        return
    selection = batch.previous.read(out / "模型选择.json")
    check_hashes(selection["model_hashes"])
    manifest = batch.previous.read(batch.OUT / "配置冻结.json")
    jobs = [j for j in manifest["jobs"] if j["role"] == "test"]
    if len(jobs) != 192 or any((batch.OUT / "solver" / j["sample"]).exists() for j in jobs):
        raise RuntimeError("Fresh evaluation must be registered before test computation")
    frame = pd.DataFrame([dict(sample=j["sample"], **j["design"]) for j in jobs])
    x = frame[FEATURES].to_numpy(float)
    for name, model in selection["model_paths"].items():
        p = LocalLinearTreeAero.load(model).predict_many(x)
        for target_index in range(2):
            frame[f"{name}_prediction_{target_index}"] = p[:, target_index]
    csv = out / "独立锁定预测.csv"
    frame.to_csv(csv, index=False)
    paths = [csv, out / "模型选择.json", Path(__file__),
             ROOT / "核心模型/局部线性树气动代理.py", *map(Path, selection["model_paths"].values())]
    save(path, dict(jobs=jobs, models=list(selection["model_paths"]),
        hashes={str(p): digest(p) for p in paths}, locked_before_test_solver=True,
        test_labels_used=False, created_at=time.strftime("%Y-%m-%d %H:%M:%S")))


def evaluate_fresh(out, wait):
    manifest = batch.previous.read(out / "独立预测冻结.json")
    started = time.monotonic()
    while True:
        check_hashes(manifest["hashes"])
        paths = [batch.OUT / "solver" / j["sample"] / "record.json" for j in manifest["jobs"]]
        completed = sum(p.exists() for p in paths)
        save(out / "独立评价进度.json", dict(stage="waiting_fresh_test", completed=completed,
             requested=len(paths), updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        if completed == len(paths):
            rows = []
            for job, path in zip(manifest["jobs"], paths):
                row = batch.previous.read(path)
                batch.previous.validate_record(job, row)
                rows.append(row)
            labels = pd.DataFrame(rows).set_index("sample")
            predictions = pd.read_csv(out / "独立锁定预测.csv").set_index("sample").loc[labels.index]
            metrics = []
            for name in manifest["models"]:
                for i, target in enumerate(TARGETS):
                    y = labels[target].to_numpy(float)
                    valid = np.isfinite(y)
                    if i:
                        valid &= labels.wake_quality_pass.fillna(False).to_numpy(bool)
                    p = predictions[f"{name}_prediction_{i}"].to_numpy(float)
                    metrics.append(dict(model=name, target=target, count=int(valid.sum()),
                        **helpers.direct.base.metric(y[valid], p[valid])))
            pd.DataFrame(metrics).to_csv(out / "独立192构型指标.csv", index=False)
            predictions.join(labels[[*TARGETS, "wake_quality_pass"]]).to_csv(
                out / "独立192构型逐点.csv")
            save(out / "独立评价.json", dict(metrics=metrics, attempted=len(paths),
                 locked_before_test_solver=True, test_labels_used_for_training=False,
                 quality_rules_unchanged=True, automatic_production_promotion=False))
            save(out / "独立评价进度.json", dict(stage="completed", completed=completed,
                 requested=len(paths), updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            return
        progress = batch.previous.read(batch.OUT / "进度.json")
        if progress["stage"] == "failed":
            raise RuntimeError("Native batch failed: " + str(progress["error"]))
        if not wait:
            return
        if time.monotonic()-started > 24*3600:
            raise TimeoutError("Fresh evaluation timed out")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    try:
        if not args.evaluate_only:
            train(output, args.workers)
        evaluate_fresh(output, args.wait)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        save(output / "失败.json", dict(error=str(error), traceback=traceback.format_exc(),
             updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        raise
