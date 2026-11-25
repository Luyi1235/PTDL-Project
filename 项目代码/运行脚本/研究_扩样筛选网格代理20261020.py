"""Expand mesh training with fold-local feature selection and a fresh stratified holdout."""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import traceback

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

import 推进_覆盖扩样6144代理20261019 as expansion
from 筛选网格气动代理 import SelectedMeshAero, predict_component, select_columns
from 稳定网格尾迹代理 import features
from 均匀风险尾迹代理 import features as raw_features
from 尾迹射线网格特征 import FEATURE_NAMES, extract

ROOT, utility, batch = expansion.ROOT, expansion.utility, expansion.batch
SOURCE, STUDY = expansion.OUT, expansion.study.previous
FEATURES, TARGETS = expansion.FEATURES, expansion.TARGETS
OUT = ROOT / "运行结果/扩样筛选网格代理_20261020"
DATA = SOURCE / "完整训练标签.csv"
BASELINE = SOURCE / "精细训练/精细气动候选代理.json"
CONFIGS = [
    dict(mesh=True, transform="canonical-area", zero_tolerance=1e-8,
         keep_mesh=keep, depth=depth, policy=policy, penalty=penalty,
         iterations=iterations)
    for keep, depth, policy, penalty, iterations in (
        (None, 5, "Plain", 20., 1800),
        (64, 6, "Plain", 40., 1800),
        (128, 6, "Depthwise", 60., 1600),
    )
]


def progress(stage, **details):
    utility.save(OUT / "进度.json", dict(stage=stage, pid=os.getpid(),
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **details))


def holdout_jobs(frame, forbidden, centers, count_per_stratum=128):
    x = frame[FEATURES].to_numpy(float)
    lo, hi, span = x.min(0), x.max(0), np.ptp(x, axis=0)
    avoid = np.asarray(forbidden, float).copy()
    rng = np.random.default_rng(2026102002)
    fixed = {key: float(frame.iloc[0][key]) for key in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    jobs = []
    for basis in ("fresh_uniform_feasible", "fresh_training_error_neighborhood"):
        completed = 0
        for _ in range(100000):
            row = (rng.uniform(lo, hi) if basis == "fresh_uniform_feasible"
                   else centers[rng.integers(len(centers))]
                   + rng.normal(0, rng.uniform(.04, .08), len(FEATURES))*span)
            if np.any(row < lo) or np.any(row > hi):
                continue
            design = dict(fixed, **dict(zip(FEATURES, row)))
            if (max(expansion.original.prior.geometry_constraints(
                    expansion.original.prior.AirframeDesign(**design)).values()) > 1e-9
                    or np.linalg.norm((avoid-row)/span, axis=1).min() <= .02):
                continue
            jobs.append(dict(sample=f"selected6144_test_{len(jobs):04d}",
                             design=design, role="test", basis=basis))
            avoid = np.vstack((avoid, row))
            completed += 1
            if completed == count_per_stratum:
                break
        if completed != count_per_stratum:
            raise RuntimeError("Independent stratum sampling exhausted")
    return jobs


def plan():
    path = OUT / "构型与方法冻结.json"
    if path.exists():
        frozen = batch.previous.read(path)
        utility.check_hashes(frozen["hashes"])
        return frozen
    frame = pd.read_csv(STUDY.DATA)
    avoid = batch.forbidden_geometries(frame)
    manifests = [
        batch.OUT / "配置冻结.json",
        expansion.native.OUT / "构型与计算冻结.json",
        expansion.evaluation.OUT / "构型与计算冻结.json",
        SOURCE / "配置冻结.json",
    ]
    for manifest in manifests:
        jobs = batch.previous.read(manifest)["jobs"]
        avoid = np.vstack((avoid, [[j["design"][k] for k in FEATURES] for j in jobs]))
    centers = expansion.training_error_centers(frame)
    jobs = holdout_jobs(frame, avoid, centers)
    paths = [Path(__file__), ROOT / "核心模型/筛选网格气动代理.py",
             ROOT / "核心模型/稳定网格尾迹代理.py", STUDY.DATA,
             STUDY.OUT / "训练网格特征.npz", STUDY.OUT / "训练网格特征记录.csv",
             *manifests]
    frozen = dict(jobs=jobs, configs=CONFIGS,
        hashes={str(p): utility.digest(p) for p in paths},
        training_rows_planned=6144, test_count=256,
        test_strata=dict(uniform=128, training_error_neighborhood=128),
        acquisition_from_training_oof_only=True, target_definition_unchanged=True,
        quality_rules_unchanged=True, feature_selection_inside_training_folds=True,
        native_protocol="free-wake-legacy-v1", wake_nodes=8, wake_iterations=30,
        mesh="legacy", native_cores=8, automatic_production_promotion=False,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(path, frozen)
    return frozen


def wait_for_inputs():
    deadline = time.monotonic()+48*3600
    while True:
        stage = batch.previous.read(SOURCE / "进度.json")
        if stage["stage"] == "failed":
            raise RuntimeError("Source 6144 training pipeline failed")
        if DATA.exists() and stage["stage"] not in ("solver_train", "planned", "collection_limit"):
            frame = pd.read_csv(DATA)
            if len(frame) == 6144:
                return frame
        progress("waiting_expanded_training_data",
                 source_completed=stage.get("new_training_completed"),
                 source_requested=2048)
        if time.monotonic() >= deadline:
            raise TimeoutError("Expanded training labels exceeded forty-eight hours")
        time.sleep(30)


def extract_item(item):
    index, row = item
    try:
        path = Path(row["geometry_file"]).with_suffix(".vspgeom")
        value = extract(path, row["wing_area_m2"], row["aspect_ratio"])
        return index, value, dict(error=None, mesh_sha256=utility.digest(path))
    except Exception as error:
        if bool(row.get("wake_quality_pass", False)) and np.isfinite(row.get(TARGETS[1], np.nan)):
            raise RuntimeError("Valid training label lacks geometry descriptors") from error
        return index, np.zeros(len(FEATURE_NAMES)), dict(error=repr(error))


def prepare_mesh(frame):
    prior = pd.read_csv(STUDY.DATA)
    if len(frame) != 6144 or not frame.iloc[:4096]["sample"].equals(prior["sample"]):
        raise RuntimeError("Expanded training table has unexpected source order")
    columns = FEATURES+list(TARGETS)
    np.testing.assert_allclose(frame.iloc[:4096][columns],
                               prior[columns], rtol=1e-12, atol=1e-12,
                               equal_nan=True)
    cache = OUT / "训练网格特征.npz"
    audit_path = OUT / "新增网格审计.json"
    mesh, done = np.zeros((6144, len(FEATURE_NAMES))), np.zeros(6144, bool)
    audits = {}
    if cache.exists():
        with np.load(cache) as saved:
            if saved["features"].shape != mesh.shape or saved["done"].shape != done.shape:
                raise RuntimeError("Expanded mesh cache shape changed")
            mesh, done = saved["features"], saved["done"]
        audits = batch.previous.read(audit_path)
    else:
        mesh[:4096] = np.load(STUDY.OUT / "训练网格特征.npz")["features"]
        done[:4096] = True
    jobs = [(i, frame.iloc[i].to_dict()) for i in range(4096, len(frame)) if not done[i]]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract_item, job) for job in jobs]
        for future in as_completed(futures):
            index, values, audit = future.result()
            mesh[index], done[index], audits[str(index)] = values, True, audit
            if int(done.sum()) % 32 == 0 or done.all():
                utility.save(audit_path, audits)
                temporary = OUT / "训练网格特征.partial.npz"
                np.savez_compressed(temporary, features=mesh, done=done)
                temporary.replace(cache)
                progress("mesh_features", completed=int(done.sum()), requested=len(frame))
    if not done.all():
        raise RuntimeError("Expanded geometry feature coverage incomplete")
    return mesh


def fit(config, x, y, mesh, threads=2):
    matrix = features(x, mesh, config)
    geometry_columns = raw_features(x, None, dict(config, mesh=False)).shape[1]
    columns = select_columns(matrix, y, geometry_columns, config["keep_mesh"], threads)
    options = dict(
        loss_function="RMSE", iterations=config["iterations"], depth=config["depth"],
        learning_rate=.035, l2_leaf_reg=config["penalty"], rsm=.7,
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        random_seed=2026102003, thread_count=threads, verbose=False,
        allow_writing_files=False)
    if config["policy"] == "Depthwise":
        options.update(grow_policy="Depthwise", min_data_in_leaf=30)
    else:
        options["boosting_type"] = "Plain"
    estimator = CatBoostRegressor(**options).fit(matrix[:, columns], y)
    return dict(kind="tree", config=config, columns=columns, estimator=estimator)


def task(job):
    index, split, fold, a, b, x, y, mesh = job
    started = time.monotonic()
    component = fit(CONFIGS[index], x[a], y[a], mesh[a])
    prediction = predict_component(component, x[b], mesh[b])
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
        actual=y[b].tolist(), prediction=prediction.tolist(),
        selected_columns=component["columns"].tolist(), seconds=time.monotonic()-started,
        train_r2=STUDY.helpers.direct.base.metric(
            y[a], predict_component(component, x[a], mesh[a]))["r2"],
        **STUDY.helpers.direct.base.metric(y[b], prediction))


def baseline_ready():
    selection = SOURCE / "精细训练/模型选择.json"
    deadline = time.monotonic()+24*3600
    while not selection.exists():
        progress("waiting_expanded_geometry_model")
        stage = batch.previous.read(SOURCE / "进度.json")
        if stage["stage"] == "failed":
            raise RuntimeError("Geometry-only 6144 training failed")
        if time.monotonic() >= deadline:
            raise TimeoutError("Geometry baseline training exceeded twenty-four hours")
        time.sleep(30)
    selected = batch.previous.read(selection)
    if selected["model_sha256"] != utility.digest(BASELINE):
        raise RuntimeError("Expanded geometry baseline hash differs")
    return selected


def train(frame, mesh):
    valid = np.isfinite(frame[TARGETS[1]].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    x, y = frame.loc[valid, FEATURES].to_numpy(float), frame.loc[valid, TARGETS[1]].to_numpy(float)
    mesh = mesh[valid]
    groups = STUDY.helpers.direct.base.geometry_groups(x)
    records_path = OUT / "交叉验证逐折.json"
    records = batch.previous.read(records_path) if records_path.exists() else []
    completed = {(r["config"], r["split"], r["fold"]) for r in records}
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101803).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, x, y, mesh)
                        for i in range(len(CONFIGS)) if (i, split, fold) not in completed)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(task, job) for job in jobs]):
            records.append(future.result())
            utility.save(records_path, records)
            progress("training_cv", completed=len(records), requested=18)
    selected = baseline_ready()
    base_selection = selected["selected"][1]
    if base_selection["count"] != len(y):
        raise RuntimeError("Geometry reference and mesh training labels differ")
    base_records = [r for r in batch.previous.read(SOURCE / "精细训练/交叉验证逐折.json")
                    if r["target"] == TARGETS[1]]
    predictions = [STUDY.helpers.records_matrix(records, len(y), i) for i in range(len(CONFIGS))]
    reference = sum(c["weight"]*STUDY.helpers.records_matrix(
        base_records, len(y), c["config"]) for c in base_selection["components"])
    predictions.append(reference)
    score, chosen, oof = STUDY.helpers.select_mixture(predictions, y)
    components = []
    for index, weight in chosen:
        component = (dict(kind="reference") if index == len(CONFIGS)
                     else fit(CONFIGS[index], x, y, mesh))
        components.append(dict(component, weight=weight))
    payload = dict(schema="ptdl-selected-mesh-aero-v1",
        baseline=batch.previous.read(BASELINE), wake_components=components,
        training_samples=6144, wake_training_valid=int(valid.sum()))
    model_path = OUT / "筛选网格候选.joblib"
    model = SelectedMeshAero(payload)
    probe = model.predict_many(x[:20], mesh[:20])
    joblib.dump(payload, model_path, compress=3)
    np.testing.assert_allclose(SelectedMeshAero.load(model_path).predict_many(x[:20], mesh[:20]),
                               probe, atol=1e-12, rtol=1e-12)
    rows = [dict(config=i, split=split, **STUDY.helpers.direct.base.metric(y, values))
            for i, prediction in enumerate(predictions)
            for split, values in zip(("random", "spatial"), prediction)]
    pd.DataFrame(rows).to_csv(OUT / "模型比较.csv", index=False)
    paths = [model_path, DATA, BASELINE, records_path, OUT / "训练网格特征.npz",
             OUT / "训练数据冻结.json", OUT / "构型与方法冻结.json",
             ROOT / "核心模型/筛选网格气动代理.py"]
    result = dict(path=str(model_path), score=score,
        selected=[dict(config=i, weight=w) for i, w in chosen],
        training_samples=6144, wake_training_valid=int(valid.sum()),
        oof=[dict(split=s, **STUDY.helpers.direct.base.metric(y, p))
             for s, p in zip(("random", "spatial"), oof)],
        test_labels_used_for_selection=False, automatic_production_promotion=False,
        hashes={str(p): utility.digest(p) for p in paths})
    utility.save(OUT / "模型选择.json", result)
    return result


def diagnostics(selected):
    models, _ = expansion.evaluation.predictors()
    from 精细气动代理 import FineAeroSurrogate
    models.update(selected_mesh=SelectedMeshAero.load(selected["path"]),
                  fine6144=FineAeroSurrogate.from_json(BASELINE))
    sources = [
        ("known192", batch.OUT / "独立评价逐点.csv", None),
        ("known256", expansion.native.OUT / "独立逐点.csv", expansion.native.OUT),
        ("known384", expansion.evaluation.OUT / "独立逐点.csv", expansion.evaluation.OUT),
    ]
    metrics = []
    for name, path, cache in sources:
        frame = pd.read_csv(path)
        if cache is not None:
            mesh = np.load(cache / "测试几何特征.npz")["features"]
            locked = pd.read_csv(cache / "锁定预测.csv")
            np.testing.assert_array_equal(frame["sample"], locked["sample"])
        else:
            values = []
            for _, row in frame.iterrows():
                try:
                    values.append(extract(Path(row.geometry_file).with_suffix(".vspgeom"),
                                          row.wing_area_m2, row.aspect_ratio))
                except Exception:
                    if bool(row.wake_quality_pass) and np.isfinite(row[TARGETS[1]]):
                        raise
                    values.append(np.zeros(len(FEATURE_NAMES)))
            mesh = np.asarray(values)
        x = frame[FEATURES].to_numpy(float)
        for key, model in models.items():
            p = (model.predict_many(x, mesh) if getattr(model, "requires_mesh", False)
                 else model.predict_many(x))
            for index in range(2):
                frame[f"{key}_prediction_{index}"] = p[:, index]
        metrics.extend(expansion.evaluation.metric_rows(frame, list(models), name))
        frame.to_csv(OUT / f"{name}_诊断逐点.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "既有构型诊断指标.csv", index=False)
    utility.save(OUT / "既有构型诊断.json", dict(
        metrics=metrics, previously_inspected_cohorts=True,
        test_labels_used_for_selection=False))


def lock_predictions(plan, selected):
    test_out = OUT / "独立256构型"
    test_out.mkdir(exist_ok=True)
    lock = test_out / "预测冻结.json"
    if lock.exists():
        frozen = batch.previous.read(lock)
        utility.check_hashes(frozen["hashes"])
        return frozen
    if any((OUT / "solver" / job["sample"]).exists() for job in plan["jobs"]):
        raise RuntimeError("New test labels exist before prediction registration")
    for start in range(0, len(plan["jobs"]), 8):
        request = dict(jobs=plan["jobs"][start:start+8],
                       output_root=str(test_out / "geometry"))
        folder = test_out / "geometry_worker" / f"batch_{start:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        expansion.native.original.controlled.run_process(
            [sys.executable, "-X", "utf8", str(expansion.native.WORKER)],
            request, folder, 1800)
        progress("prediction_geometry", completed=min(start+8, len(plan["jobs"])),
                 requested=len(plan["jobs"]))
    models, paths = expansion.evaluation.predictors()
    from 精细气动代理 import FineAeroSurrogate
    models.update(selected_mesh=SelectedMeshAero.load(selected["path"]),
                  fine6144=FineAeroSurrogate.from_json(BASELINE))
    paths.update(selected_mesh=Path(selected["path"]), fine6144=BASELINE)
    frame = pd.DataFrame([dict(sample=j["sample"], selection_basis=j["basis"], **j["design"])
                          for j in plan["jobs"]])
    mesh, audit = [], []
    for job in plan["jobs"]:
        path = test_out / "geometry" / job["sample"] / "airframe.vspgeom"
        mesh.append(extract(path, job["design"]["wing_area_m2"], job["design"]["aspect_ratio"]))
        audit.append(dict(sample=job["sample"], mesh_sha256=utility.digest(path)))
    mesh, x = np.asarray(mesh), frame[FEATURES].to_numpy(float)
    for name, model in models.items():
        p = model.predict_many(x, mesh) if getattr(model, "requires_mesh", False) else model.predict_many(x)
        for index in range(2):
            frame[f"{name}_prediction_{index}"] = p[:, index]
    predictions, audit_path = test_out / "锁定预测.csv", test_out / "测试几何记录.csv"
    frame.to_csv(predictions, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    frozen = dict(models=list(models), locked_before_test_solver=True,
        test_labels_seen=False, hashes={str(p): utility.digest(p) for p in
            [predictions, audit_path, OUT / "模型选择.json", OUT / "构型与方法冻结.json",
             *paths.values()]}, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(lock, frozen)
    return frozen


def evaluate(plan, frozen):
    test_out = OUT / "独立256构型"
    predictions = pd.read_csv(test_out / "锁定预测.csv").set_index("sample")
    audit = pd.read_csv(test_out / "测试几何记录.csv").set_index("sample")
    labels = []
    expansion.original.controlled.original.WORKER = expansion.original.WORKER
    for job in plan["jobs"]:
        utility.check_hashes(frozen["hashes"])
        row = expansion.run_case(OUT, job)
        if bool(row["wake_quality_pass"]) and np.isfinite(row[TARGETS[1]]):
            geometry = OUT / "solver" / job["sample"] / "case/airframe.vspgeom"
            if utility.digest(geometry) != audit.loc[job["sample"], "mesh_sha256"]:
                raise RuntimeError("Valid native test mesh differs from frozen predictor input")
        labels.append(row)
        pd.DataFrame(labels).to_csv(test_out / "独立标签.csv", index=False)
        progress("native_test", completed=len(labels), requested=len(plan["jobs"]))
    frame = pd.DataFrame(labels).set_index("sample")
    for column in predictions:
        if "_prediction_" in column:
            frame[column] = predictions.loc[frame.index, column]
    frame["stratum"] = [j["basis"] for j in plan["jobs"]]
    rows, comparisons = [], []
    for cohort, mask in (
        ("all", np.ones(len(frame), bool)),
        ("uniform", frame.stratum.eq("fresh_uniform_feasible").to_numpy()),
        ("training_error_neighborhood",
         frame.stratum.eq("fresh_training_error_neighborhood").to_numpy()),
    ):
        subset = frame.loc[mask]
        rows.extend(expansion.evaluation.metric_rows(subset, frozen["models"], cohort))
        valid = np.isfinite(subset[TARGETS[1]]) & subset.wake_quality_pass.fillna(False)
        for name in frozen["models"]:
            if name != "selected_mesh":
                comparisons.append(dict(cohort=cohort, reference=name,
                    **batch.previous.evaluator.paired_bootstrap(
                        subset.loc[valid, TARGETS[1]].to_numpy(float),
                        subset.loc[valid, "selected_mesh_prediction_1"].to_numpy(float),
                        subset.loc[valid, f"{name}_prediction_1"].to_numpy(float))))
    frame.to_csv(test_out / "独立逐点.csv")
    pd.DataFrame(rows).to_csv(test_out / "独立指标.csv", index=False)
    utility.save(test_out / "独立评价.json", dict(metrics=rows, comparisons=comparisons,
        prediction_locked_before_labels=True, all_valid_labels_retained=True,
        test_labels_used_for_selection=False, automatic_production_promotion=False))


def main(prepare_only=False):
    OUT.mkdir(exist_ok=True)
    manifest = plan()
    if prepare_only:
        progress("planned", training_rows=6144, test_requested=256, cv_fits=18)
        return
    frame = wait_for_inputs()
    freeze = OUT / "训练数据冻结.json"
    data_hash = utility.digest(DATA)
    if freeze.exists() and batch.previous.read(freeze)["data_sha256"] != data_hash:
        raise RuntimeError("Expanded training data changed during research")
    utility.save(freeze, dict(data_path=str(DATA), data_sha256=data_hash,
                              rows=len(frame), test_labels_used=False))
    mesh = prepare_mesh(frame)
    selected = (batch.previous.read(OUT / "模型选择.json")
                if (OUT / "模型选择.json").exists() else train(frame, mesh))
    utility.check_hashes(selected["hashes"])
    if not (OUT / "既有构型诊断.json").exists():
        progress("known_cohort_diagnostics")
        diagnostics(selected)
    frozen = lock_predictions(manifest, selected)
    evaluate(manifest, frozen)
    progress("completed", training_rows=6144, test_completed=256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args()
    try:
        main(options.prepare_only)
    except Exception:
        OUT.mkdir(exist_ok=True)
        progress("failed", error=traceback.format_exc())
        raise
