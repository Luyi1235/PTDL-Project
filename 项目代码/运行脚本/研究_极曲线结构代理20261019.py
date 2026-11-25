"""Learn force-profile structure and test selection using training OOF predictions."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import traceback

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold

import 研究_尾迹网格与均匀风险20261018 as previous
from 极曲线结构代理 import ANGLES, PolarStructureAero, fitted_prediction, force_coefficients, reconstruct

ROOT, DATA, FEATURES, TARGET = previous.ROOT, previous.DATA, previous.FEATURES, previous.TARGET
OUT = ROOT / "运行结果/极曲线结构代理_20261019"
MODES = ("direct", "profile", "half")
CONFIGS = [
    dict(mesh=mesh, selected=selected, transform="canonical-area", depth=depth,
         wake_weight=weight, penalty=40., iterations=1800)
    for mesh, selected in ((False, 0), (True, 64))
    for depth, weight in ((5, 1.), (5, 3.), (6, 1.), (6, 3.))
]


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "极曲线训练数据.npz"
    audit_path = out / "极曲线标签审计.json"
    if archive.exists():
        audit = previous.batch.previous.read(audit_path)
        previous.utility.check_hashes(audit["hashes"])
        with np.load(archive) as data:
            return {key: data[key] for key in data.files}
    frame = pd.read_csv(DATA)
    valid = np.isfinite(frame[TARGET].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    selected = frame.loc[valid]
    polars, source_hashes = [], {}
    for _, row in selected.iterrows():
        path = Path(row.geometry_file).parent / "vspaero_polar.csv"
        polar = pd.read_csv(path).set_index("alpha_deg").loc[ANGLES]
        working = (abs(polar.cl_wake.to_numpy(float)) >= .15) & (
            polar.e_wake.to_numpy(float) > 0)
        np.testing.assert_allclose(np.median(polar.e_wake.to_numpy(float)[working]),
                                   row[TARGET], atol=1e-8, rtol=0)
        values = polar[["cl", "cl_wake", "cd_induced_wake"]].to_numpy(float)
        polars.append(values)
        source_hashes[str(path)] = previous.utility.digest(path)
    x = selected[FEATURES].to_numpy(float)
    y = selected[TARGET].to_numpy(float)
    coefficients = force_coefficients(np.asarray(polars), x[:, 1])
    recovered = reconstruct(coefficients, y)
    source_mesh = previous.OUT / "训练网格特征.npz"
    mesh = np.load(source_mesh)["features"][valid]
    arrays = dict(x=x, y=y, coefficients=coefficients, mesh=mesh,
                  polar=np.asarray(polars), raw_indices=np.flatnonzero(valid))
    np.savez_compressed(archive, **arrays)
    paths = [DATA, source_mesh, archive, Path(__file__), ROOT / "核心模型/极曲线结构代理.py"]
    source_hashes.update({str(p): previous.utility.digest(p) for p in paths})
    previous.utility.save(audit_path, dict(hashes=source_hashes,
        raw_training_count=len(frame), valid_wake_count=len(y),
        original_target_definition_unchanged=True, external_test_labels_used=False,
        reconstruction=previous.helpers.direct.base.metric(y, recovered),
        reconstruction_max_error=float(abs(y-recovered).max())))
    pd.DataFrame(dict(sample=selected["sample"].to_numpy(), actual=y,
        reconstructed=recovered, error=recovered-y)).to_csv(
            out / "二次极曲线还原误差.csv", index=False)
    return arrays


def fit(config, x, y, mesh, coefficients, threads):
    values = previous.features(x, mesh, config)
    targets = np.column_stack((coefficients, y))
    center = targets.mean(0)
    scale = np.maximum(targets.std(0), 1e-8)
    scale[9] /= config["wake_weight"]
    z = (targets-center)/scale
    if config["selected"]:
        selector = ExtraTreesRegressor(
            n_estimators=128, max_features=.5, min_samples_leaf=8,
            max_depth=12, n_jobs=threads, random_state=2026101901)
        selector.fit(values, z)
        base_count = previous.features(x[:1], None, dict(config, mesh=False)).shape[1]
        ranked_mesh = np.argsort(-selector.feature_importances_[base_count:])[
            :config["selected"]]+base_count
        columns = np.unique(np.r_[np.arange(base_count), ranked_mesh])
    else:
        columns = np.arange(values.shape[1])
    estimator = CatBoostRegressor(
        loss_function="MultiRMSE", iterations=config["iterations"], depth=config["depth"],
        learning_rate=.035, l2_leaf_reg=config["penalty"], rsm=.7,
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        random_seed=2026101902, thread_count=threads, verbose=False,
        allow_writing_files=False)
    estimator.fit(values[:, columns], z)
    return dict(config=config, columns=columns, estimator=estimator, center=center, scale=scale)


def task(args):
    index, split, fold, a, b, data, threads = args
    started = time.monotonic()
    x, y, mesh, coefficients = (data[k] for k in ("x", "y", "mesh", "coefficients"))
    model = fit(CONFIGS[index], x[a], y[a], mesh[a], coefficients[a], threads)
    records = []
    for mode in MODES:
        p = fitted_prediction(model, x[b], mesh[b], mode)
        records.append(dict(config=index, mode=mode, split=split, fold=fold,
            validation=b.tolist(), actual=y[b].tolist(), prediction=p.tolist(),
            columns=model["columns"].tolist(), seconds=time.monotonic()-started,
            **previous.helpers.direct.base.metric(y[b], p)))
    return records


def reference_oof(count):
    records = previous.batch.previous.read(previous.OUT / "交叉验证逐折.json")
    selection = previous.batch.previous.read(previous.OUT / "模型选择.json")
    result = [previous.reference_oof(count)]
    for entry in selection["selected"]:
        result.append(sum(c["weight"]*(
            result[0] if c["config"] == len(previous.CONFIGS) else
            previous.helpers.records_matrix(records, count, c["config"]))
            for c in entry["chosen"]))
    return result


def main(out, workers, threads, prepare_only):
    data = prepare(out)
    if prepare_only:
        print(previous.batch.previous.read(out / "极曲线标签审计.json")["reconstruction"], flush=True)
        return
    audit = previous.batch.previous.read(out / "极曲线标签审计.json")
    previous.utility.check_hashes(audit["hashes"])
    execution = out / "配置冻结.json"
    freeze = dict(configs=CONFIGS, modes=MODES, workers=workers, threads=threads,
        hashes={str(p): previous.utility.digest(p) for p in (
            Path(__file__), ROOT / "核心模型/极曲线结构代理.py",
            out / "极曲线标签审计.json", previous.OUT / "模型选择.json")},
        target_definition_unchanged=True, feature_selection_within_training_fold=True,
        old_test_sets_diagnostic_only=True)
    if execution.exists():
        if previous.batch.previous.read(execution) != freeze:
            raise RuntimeError("Force-profile research configuration changed")
    else:
        previous.utility.save(execution, freeze)
    groups = previous.helpers.direct.base.geometry_groups(data["x"])
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101803).split(data["x"])),
        ("spatial", GroupKFold(3).split(data["x"], groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, data, threads) for i in range(len(CONFIGS)))
    path = out / "交叉验证逐折.json"
    records = previous.batch.previous.read(path) if path.exists() else []
    done = {(r["config"], r["split"], r["fold"]) for r in records if r["mode"] == "half"}
    jobs = [job for job in jobs if job[:3] not in done]
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, job): job[:3] for job in jobs}
        for future in as_completed(futures):
            try:
                records.extend(future.result())
                previous.utility.save(path, records)
            except Exception as error:
                errors.append(dict(job=futures[future], error=repr(error),
                                   traceback=traceback.format_exc()))
                previous.utility.save(out / "失败折.json", errors)
            previous.utility.save(out / "进度.json", dict(
                stage="training_cv", completed=len(records)//3, requested=48,
                failed=len(errors), updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            print("Profile CV", len(records)//3, "/ 48; failed", len(errors), flush=True)
    if errors:
        raise RuntimeError("Force-profile OOF failures recorded")
    previous.utility.check_hashes(freeze["hashes"])
    predictions, keys = [], []
    summaries = []
    y = data["y"]
    for i in range(len(CONFIGS)):
        for mode in MODES:
            subset = [r for r in records if r["mode"] == mode]
            p = previous.helpers.records_matrix(subset, len(y), i)
            predictions.append(p)
            keys.append(dict(kind="profile", config=i, mode=mode))
    for name, p in zip(("fine4096", "uniform_risk", "all_risk"), reference_oof(len(y))):
        predictions.append(p)
        keys.append(dict(kind="reference", name=name))
    for key, p in zip(keys, predictions):
        for split, values in zip(("random", "spatial"), p):
            summaries.append(dict(**key, split=split,
                **previous.helpers.direct.base.metric(y, values)))
    pd.DataFrame(summaries).to_csv(out / "模型比较.csv", index=False)
    score, chosen, oof = previous.helpers.select_mixture(predictions, y)
    baseline = previous.batch.previous.read(previous.BASELINE)
    components, fitted = [], {}
    for index, weight in chosen:
        key = keys[index]
        if key["kind"] == "profile":
            i = key["config"]
            if i not in fitted:
                previous.utility.save(out / "进度.json", dict(
                    stage="final_fit", completed=48, requested=48, config=i,
                    updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
                fitted[i] = fit(CONFIGS[i], data["x"], y, data["mesh"],
                                data["coefficients"], max(threads, 8))
            components.append(dict(kind="profile", fitted=fitted[i],
                                   mode=key["mode"], weight=weight))
        elif key["name"] == "fine4096":
            components.append(dict(kind="reference", payload=baseline, weight=weight))
        else:
            components.append(dict(kind="wake_reference", weight=weight,
                payload=joblib.load(previous.OUT / (key["name"]+".joblib"))))
    payload = dict(schema="ptdl-polar-structure-v1", baseline=baseline,
                   wake_components=components, training_samples=4096)
    model_path = out / "极曲线结构候选.joblib"
    model = PolarStructureAero(payload)
    joblib.dump(payload, model_path, compress=3)
    np.testing.assert_allclose(model.predict_many(data["x"][:20], data["mesh"][:20]),
        PolarStructureAero.load(model_path).predict_many(data["x"][:20], data["mesh"][:20]),
        atol=1e-10, rtol=1e-10)
    previous.utility.save(out / "模型选择.json", dict(
        score=score, selected=[dict(**keys[i], weight=w) for i, w in chosen],
        path=str(model_path), requires_mesh=model.requires_mesh,
        hashes={str(p): previous.utility.digest(p) for p in
                (model_path, ROOT / "核心模型/极曲线结构代理.py", execution)},
        oof=[dict(split=split, **previous.helpers.direct.base.metric(y, p))
             for split, p in zip(("random", "spatial"), oof)],
        test_labels_used_for_selection=False, automatic_production_promotion=False))
    previous.utility.save(out / "进度.json", dict(stage="completed", completed=48, requested=48,
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        main(args.output_dir.resolve(), args.workers, args.threads, args.prepare_only)
    except Exception:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        previous.utility.save(args.output_dir / "失败.json", dict(error=traceback.format_exc()))
        raise
