"""Resume OOF fits with bounded concurrency, then finalize with CPU parallelism."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

import 研究_尾迹网格与均匀风险20261018 as study
from 均匀风险尾迹代理 import UniformRiskWakeAero


def fit(config, x, y, mesh, uniform, threads):
    settings = dict(
        loss_function="RMSE", iterations=config["iterations"], depth=config["depth"],
        learning_rate=.035, l2_leaf_reg=config["penalty"],
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        rsm=.5, random_seed=2026101826, thread_count=threads, verbose=False,
        allow_writing_files=False,
    )
    if config["policy"] == "Depthwise":
        settings.update(grow_policy="Depthwise", min_data_in_leaf=30)
    else:
        settings["boosting_type"] = config["policy"]
    weight = np.ones(len(y))
    if config["weighting"] == "uniform-priority":
        weight[~uniform] = .35
    model = CatBoostRegressor(**settings)
    model.fit(study.features(x, mesh, config), y, sample_weight=weight/weight.mean())
    return model


def cv_task(args):
    index, split, fold, a, b, x, y, mesh, uniform, threads = args
    started = time.monotonic()
    config = study.CONFIGS[index]
    model = fit(config, x[a], y[a], mesh[a], uniform[a], threads)
    prediction = model.predict(study.features(x[b], mesh[b], config), thread_count=threads)
    trained = model.predict(study.features(x[a], mesh[a], config), thread_count=threads)
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
        actual=y[b].tolist(), prediction=prediction.tolist(), cpu_threads=threads,
        seconds=time.monotonic()-started,
        train_r2=study.helpers.direct.base.metric(y[a], trained)["r2"],
        **study.helpers.direct.base.metric(y[b], prediction))


def resume_cv(workers, threads):
    out = study.OUT
    manifest = study.batch.previous.read(out / "训练配置冻结.json")
    study.utility.check_hashes(manifest["hashes"])
    if manifest["configs"] != study.CONFIGS:
        raise RuntimeError("OOF model configuration changed")
    frame = pd.read_csv(study.DATA)
    valid = np.isfinite(frame[study.TARGET].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    x = frame.loc[valid, study.FEATURES].to_numpy(float)
    y = frame.loc[valid, study.TARGET].to_numpy(float)
    uniform = frame.loc[valid, "selection_basis"].eq("uniform_feasible_coverage").to_numpy(bool)
    mesh_path, audit_path = out / "训练网格特征.npz", out / "训练网格特征记录.csv"
    mesh = np.load(mesh_path)["features"][valid]
    record_path = out / "交叉验证逐折.json"
    records = study.batch.previous.read(record_path)
    done = {(r["config"], r["split"], r["fold"]) for r in records}
    if len(done) != len(records):
        raise RuntimeError("Duplicate OOF record")
    for record in records:
        np.testing.assert_allclose(y[record["validation"]], record["actual"],
                                   atol=1e-12, rtol=1e-12)
    checkpoint = out / "断点初始逐折.json"
    if not checkpoint.exists():
        study.utility.save(checkpoint, records)
    execution_path = out / "续跑执行.json"
    hashes = dict(manifest["hashes"], **{str(p): study.utility.digest(p) for p in
        (Path(__file__), mesh_path, audit_path, checkpoint)})
    if execution_path.exists():
        previous = study.batch.previous.read(execution_path)
        if previous["hashes"] != hashes:
            raise RuntimeError("Resume sources changed")
    else:
        study.utility.save(execution_path, dict(hashes=hashes,
            cpu_threads_per_fit=threads, max_concurrent_fits=workers,
            retained_completed_fits=len(records), model_hyperparameters_unchanged=True,
            previous_failure="12 concurrent fits: CatBoost bad allocation",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    groups = study.helpers.direct.base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101803).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, x, y, mesh, uniform, threads)
                        for i in range(len(study.CONFIGS)) if (i, split, fold) not in done)
    errors = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cv_task, job): job[:3] for job in jobs}
        for future in as_completed(futures):
            try:
                records.append(future.result())
                study.utility.save(record_path, records)
            except Exception as error:
                errors.append(dict(job=futures[future], error=repr(error),
                                   traceback=traceback.format_exc()))
                study.utility.save(out / "续跑失败折.json", errors)
            study.utility.save(out / "进度.json", dict(
                stage="training_cv", completed=len(records), requested=72,
                failed_fits=len(errors), max_concurrent_fits=workers,
                cpu_threads_per_fit=threads, elapsed_s=time.monotonic()-started,
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            print("Resumed wake CV", len(records), "/ 72; failed", len(errors), flush=True)
    study.utility.check_hashes(hashes)
    if errors:
        raise RuntimeError("OOF failures retained; successful folds have been checkpointed")


def main(threads):
    out = study.OUT
    selection_path = out / "模型选择.json"
    if selection_path.exists() or (out / "新256构型独立验证/预测冻结.json").exists():
        raise RuntimeError("Final models or test predictions are already frozen")
    manifest = study.batch.previous.read(out / "训练配置冻结.json")
    study.utility.check_hashes(manifest["hashes"])
    if manifest["configs"] != study.CONFIGS:
        raise RuntimeError("OOF model configuration changed")
    record_path = out / "交叉验证逐折.json"
    records = study.batch.previous.read(record_path)
    expected = {(i, split, fold) for i in range(len(study.CONFIGS))
                for split in ("random", "spatial") for fold in range(3)}
    if len(records) != len(expected) or {
            (r["config"], r["split"], r["fold"]) for r in records} != expected:
        raise RuntimeError("All 72 OOF fits must be completed before finalization")
    frame = pd.read_csv(study.DATA)
    valid = np.isfinite(frame[study.TARGET].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    x = frame.loc[valid, study.FEATURES].to_numpy(float)
    y = frame.loc[valid, study.TARGET].to_numpy(float)
    uniform = frame.loc[valid, "selection_basis"].eq("uniform_feasible_coverage").to_numpy(bool)
    mesh_path, audit_path = out / "训练网格特征.npz", out / "训练网格特征记录.csv"
    mesh = np.load(mesh_path)["features"][valid]
    for record in records:
        np.testing.assert_allclose(y[record["validation"]], record["actual"],
                                   atol=1e-12, rtol=1e-12)
    paths = [Path(__file__), record_path, mesh_path, audit_path]
    if (out / "续跑执行.json").exists():
        paths.append(out / "续跑执行.json")
    hashes = dict(manifest["hashes"], **{str(p): study.utility.digest(p) for p in paths})
    execution_path = out / "全量训练执行.json"
    study.utility.save(execution_path, dict(
        hashes=hashes, oof_cpu_threads_per_fit=sorted({
            r.get("cpu_threads", 1) for r in records}), final_cpu_threads_per_fit=threads,
        model_hyperparameters_unchanged=True, target_definition_unchanged=True,
        test_labels_used_for_training=False,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    predictions = [study.helpers.records_matrix(records, len(y), i)
                   for i in range(len(study.CONFIGS))]
    predictions.append(study.reference_oof(len(y)))
    summaries = []
    for index, p in enumerate(predictions):
        for split, values in zip(("random", "spatial"), p):
            for subset, mask in (("all", np.ones(len(y), bool)), ("uniform", uniform)):
                summaries.append(dict(config=index, split=split, subset=subset,
                    count=int(mask.sum()),
                    **study.helpers.direct.base.metric(y[mask], values[mask])))
    pd.DataFrame(summaries).to_csv(out / "训练交叉验证汇总.csv", index=False)
    baseline = study.batch.previous.read(study.BASELINE)
    selected, fitted = [], {}
    for name, mask in (("uniform_risk", uniform), ("all_risk", np.ones(len(y), bool))):
        score, chosen, _ = study.helpers.select_mixture(
            [p[:, mask] for p in predictions], y[mask])
        p = sum(weight*predictions[index] for index, weight in chosen)
        components = []
        study.utility.save(out / "进度.json", dict(
            stage="final_fit", completed=len(records), requested=len(records),
            model=name, cpu_threads=threads, finalizer=str(Path(__file__)),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        print("Final fit", name, chosen, flush=True)
        for index, weight in chosen:
            if index == len(study.CONFIGS):
                components.append(dict(kind="reference", payload=baseline, weight=weight))
            else:
                if index not in fitted:
                    fitted[index] = fit(study.CONFIGS[index], x, y, mesh, uniform, threads)
                components.append(dict(kind="tree", estimator=fitted[index],
                    config=study.CONFIGS[index], weight=weight))
        payload = dict(schema="ptdl-uniform-risk-wake-v1", baseline=baseline,
            wake_components=components, training_samples=len(frame),
            execution_manifest=str(execution_path))
        path = out / (name+".joblib")
        model = UniformRiskWakeAero(payload)
        probes = model.predict_many(x[:20], mesh[:20])
        joblib.dump(payload, path, compress=3)
        np.testing.assert_allclose(
            UniformRiskWakeAero.load(path).predict_many(x[:20], mesh[:20]),
            probes, atol=1e-10, rtol=1e-10)
        selected.append(dict(name=name, score=score, path=str(path),
            requires_mesh=model.requires_mesh,
            chosen=[dict(config=i, weight=w) for i, w in chosen],
            oof=[dict(split=split, subset=subset, count=int(m.sum()),
                      **study.helpers.direct.base.metric(y[m], values[m]))
                 for split, values in zip(("random", "spatial"), p)
                 for subset, m in (("all", np.ones(len(y), bool)), ("uniform", uniform))]))
    study.utility.check_hashes(hashes)
    final_paths = [mesh_path, audit_path, execution_path, Path(__file__)]
    final_paths += [Path(entry["path"]) for entry in selected]
    study.utility.save(selection_path, dict(
        selected=selected, hashes={str(p): study.utility.digest(p) for p in final_paths},
        target_definition_unchanged=True, test_labels_used_for_training=False,
        uniform_risk_selection_explicit=True, baseline_lift_preserved=True,
        execution_manifest=str(execution_path)))
    study.utility.save(out / "进度.json", dict(
        stage="completed", completed=len(records), requested=len(records),
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--resume-cv", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cv-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.workers, args.cv_threads) < 1:
        raise ValueError("Positive CPU thread count required")
    try:
        if args.resume_cv:
            resume_cv(args.workers, args.cv_threads)
        main(args.threads)
    except Exception as error:
        study.utility.save(study.OUT / "失败.json",
            dict(error=repr(error), traceback=traceback.format_exc()))
        raise
