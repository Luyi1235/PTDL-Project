"""Expand training coverage and retain a separate prediction-locked native holdout."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import qmc

import 验证_极曲线结构代理20261019 as evaluation
from 极曲线结构代理 import PolarStructureAero
from 精细气动代理 import FineAeroSurrogate
from 尾迹射线网格特征 import extract

study, native = evaluation.study, evaluation.native
batch, utility, original = evaluation.batch, evaluation.utility, evaluation.original
ROOT, FEATURES, TARGETS = study.ROOT, study.FEATURES, batch.TARGETS
OUT = ROOT / "运行结果/覆盖扩样6144代理_20261019"
TRAINER = Path(__file__).with_name("训练_覆盖扩样6144代理20261019.py")
TRAIN_COUNT, TEST_COUNT = 2048, 256


def progress(out, stage, **details):
    utility.save(out / "进度.json", dict(stage=stage, pid=os.getpid(),
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **details))


def training_error_centers(frame):
    valid = np.isfinite(frame[study.TARGET].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    y = frame.loc[valid, study.TARGET].to_numpy(float)
    selection = batch.previous.read(study.OUT / "模型选择.json")
    references = study.reference_oof(len(y))
    records = batch.previous.read(study.OUT / "交叉验证逐折.json")
    prediction = np.zeros((2, len(y)))
    for component in selection["selected"]:
        if component["kind"] == "reference":
            index = ("fine4096", "uniform_risk", "all_risk").index(component["name"])
            values = references[index]
        else:
            values = study.previous.helpers.records_matrix(
                [r for r in records if r["mode"] == component["mode"]],
                len(y), component["config"])
        prediction += component["weight"]*values
    error = np.mean(abs(prediction-y), axis=0)
    return frame.loc[valid, FEATURES].to_numpy(float)[np.argsort(-error)[:256]]


def plan(out):
    frame = pd.read_csv(study.DATA)
    if len(frame) != 4096:
        raise RuntimeError("Expected the frozen 4096-row source cohort")
    x = frame[FEATURES].to_numpy(float)
    lo, hi, span = x.min(0), x.max(0), np.ptp(x, axis=0)
    avoid = batch.forbidden_geometries(frame)
    prior_paths = [
        batch.OUT / "配置冻结.json", native.OUT / "构型与计算冻结.json",
        evaluation.OUT / "构型与计算冻结.json",
    ]
    for path in prior_paths:
        avoid = np.vstack((avoid, [[j["design"][k] for k in FEATURES]
            for j in batch.previous.read(path)["jobs"]]))
    fixed = {k: float(frame.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    jobs = []

    def add(row, basis, role):
        nonlocal avoid
        if np.any(row < lo) or np.any(row > hi):
            return False
        design = dict(fixed, **dict(zip(FEATURES, row)))
        minimum_distance = .02 if role == "train" else 1e-5
        if (max(original.prior.geometry_constraints(
                original.prior.AirframeDesign(**design)).values()) > 1e-9
                or np.linalg.norm((avoid-row)/span, axis=1).min() <= minimum_distance):
            return False
        number = sum(j["role"] == role for j in jobs)
        jobs.append(dict(sample=f"coverage6144_{role}_{number:04d}",
                         design=design, basis=basis, role=role))
        avoid = np.vstack((avoid, row))
        return True

    pool = qmc.scale(qmc.Sobol(7, scramble=True, seed=2026101910).random_base2(16), lo, hi)
    for row in pool:
        add(row, "uniform_feasible_coverage", "train")
        if len(jobs) == 1536:
            break
    if len(jobs) != 1536:
        raise RuntimeError("Uniform coverage pool exhausted")
    centers = training_error_centers(frame)
    rng = np.random.default_rng(2026101911)
    for _ in range(100000):
        row = centers[rng.integers(len(centers))]+rng.normal(0, rng.uniform(.04, .08), 7)*span
        add(row, "training_oof_error_neighborhood", "train")
        if len(jobs) == TRAIN_COUNT:
            break
    if len(jobs) != TRAIN_COUNT:
        raise RuntimeError("Training-neighborhood pool exhausted")
    rng = np.random.default_rng(2026101912)
    for _ in range(100000):
        add(rng.uniform(lo, hi), "fresh_uniform_feasible", "test")
        if len(jobs) == TRAIN_COUNT+TEST_COUNT:
            break
    if len(jobs) != TRAIN_COUNT+TEST_COUNT:
        raise RuntimeError("Independent test pool exhausted")
    paths = [Path(__file__), TRAINER, Path(study.__file__), Path(evaluation.__file__),
        Path(native.__file__), native.WORKER, study.DATA,
        study.OUT / "模型选择.json", study.OUT / "交叉验证逐折.json",
        ROOT / "运行脚本/研究_精细树与核代理20261018.py",
        ROOT / "核心模型/精细气动代理.py", *prior_paths]
    manifest = dict(jobs=jobs, hashes={str(p): utility.digest(p) for p in paths},
        source_train_count=4096, new_train_count=TRAIN_COUNT, target_train_count=6144,
        test_count=TEST_COUNT, acquisition_counts=dict(uniform=1536, error_neighborhood=512),
        test_labels_used_for_acquisition=False, inspected_test_geometries_excluded=True,
        target_definition_unchanged=True, quality_rules_unchanged=True,
        native_protocol="free-wake-legacy-v1", mesh="legacy", wake_nodes=8,
        wake_iterations=30, native_cores=8,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(out / "配置冻结.json", manifest)
    return manifest


def run_case(out, job):
    folder = out / "solver" / job["sample"]
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "record.json"
    if cache.exists():
        row = batch.previous.read(cache)
        batch.previous.validate_record(job, row)
        return row
    try:
        row = original.controlled.run_case(original.prior.solver.DEFAULT_PYTHON,
            job["design"], job["sample"], folder / "case", 600,
            job["role"], job["basis"], (0., float("inf")), 30, "legacy")
        row.update(original.verify(folder / "case"))
        row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
        row["wake_quality_pass"] = bool(row["quality_pass"])
    except Exception as error:
        row = dict(sample=job["sample"], **job["design"], solver_failure=repr(error),
            wake_quality_pass=False, **{key: None for key in TARGETS})
    row.update(role=job["role"], selection_basis=job["basis"])
    utility.save(cache, row)
    return row


def lock(out, jobs, model_path):
    path = out / "预测冻结.json"
    if path.exists():
        frozen = batch.previous.read(path)
        utility.check_hashes(frozen["hashes"])
        return frozen
    if any((out.parent / "solver" / job["sample"]).exists() for job in jobs):
        raise RuntimeError("Independent force labels exist before prediction registration")
    native.prepare(out, dict(jobs=jobs))
    models, paths = evaluation.predictors()
    models["fine6144"] = FineAeroSurrogate.from_json(model_path)
    paths["fine6144"] = model_path
    model_selection = model_path.parent / "模型选择.json"
    if batch.previous.read(model_selection)["model_sha256"] != utility.digest(model_path):
        raise RuntimeError("Expanded model artifact changed")
    frame = pd.DataFrame([dict(sample=job["sample"], **job["design"]) for job in jobs])
    mesh, audit = [], []
    for job in jobs:
        geometry = out / "geometry" / job["sample"] / "airframe.vspgeom"
        mesh.append(extract(geometry, job["design"]["wing_area_m2"], job["design"]["aspect_ratio"]))
        audit.append(dict(sample=job["sample"], mesh_sha256=utility.digest(geometry)))
    mesh = np.asarray(mesh)
    x = frame[FEATURES].to_numpy(float)
    metadata = {}
    for name, model in models.items():
        p = (model.predict_many(x, mesh) if name in ("polar_structure", "all_risk")
             else model.predict_many(x))
        metadata[name] = dict(requires_mesh=getattr(model, "requires_mesh", False))
        for index in range(2):
            frame[f"{name}_prediction_{index}"] = p[:, index]
    prediction, audit_path, feature_path = (
        out / "锁定预测.csv", out / "测试几何记录.csv", out / "测试几何特征.npz")
    frame.to_csv(prediction, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    np.savez_compressed(feature_path, features=mesh)
    hashes = {str(p): utility.digest(p) for p in (
        Path(__file__), model_selection, out.parent / "完整训练标签.csv",
        out.parent / "配置冻结.json", prediction, audit_path, feature_path, *paths.values())}
    frozen = dict(models=metadata, hashes=hashes, locked_before_test_solver=True,
        test_aerodynamic_labels_seen=False, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(path, frozen)
    return frozen


def main(out, prepare_only, collect_limit):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "配置冻结.json"
    manifest = batch.previous.read(manifest_path) if manifest_path.exists() else plan(out)
    utility.check_hashes(manifest["hashes"])
    if prepare_only:
        progress(out, "planned", new_training_requested=TRAIN_COUNT, test_requested=TEST_COUNT)
        return
    original.controlled.original.WORKER = original.WORKER
    training, tests = ([j for j in manifest["jobs"] if j["role"] == role]
                       for role in ("train", "test"))
    records = []
    for job in training:
        utility.check_hashes(manifest["hashes"])
        records.append(run_case(out, job))
        pd.DataFrame(records).to_csv(out / "新增训练标签.csv", index=False)
        progress(out, "solver_train", new_training_completed=len(records),
            new_training_requested=TRAIN_COUNT,
            valid_new_wake=sum(bool(r["wake_quality_pass"]) for r in records))
        print("Coverage native training", len(records), "/", TRAIN_COUNT, flush=True)
        if collect_limit and len(records) >= collect_limit:
            progress(out, "collection_limit", new_training_completed=len(records))
            return
    dataset = out / "完整训练标签.csv"
    if not dataset.exists():
        pd.concat((pd.read_csv(study.DATA), pd.DataFrame(records)), ignore_index=True).to_csv(
            dataset, index=False)
    else:
        existing = pd.read_csv(dataset)
        if len(existing) != 6144 or set(existing["sample"]) != set(
                pd.read_csv(study.DATA)["sample"]) | {r["sample"] for r in records}:
            raise RuntimeError("Expanded training table differs")
    model_folder = out / "精细训练"
    model_folder.mkdir(exist_ok=True)
    model_path = model_folder / "精细气动候选代理.json"
    if not (model_folder / "模型选择.json").exists():
        progress(out, "training", training_rows=6144)
        with (model_folder / "训练输出.log").open("w", encoding="utf8") as stream:
            subprocess.run([sys.executable, "-X", "utf8", "-u", str(TRAINER),
                "--data", str(dataset), "--output-dir", str(model_folder)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=21600)
    test_out = out / "独立256构型"
    test_out.mkdir(exist_ok=True)
    progress(out, "prediction_registration", test_requested=TEST_COUNT)
    frozen = lock(test_out, tests, model_path)
    audit = pd.read_csv(test_out / "测试几何记录.csv").set_index("sample")
    labels = []
    for job in tests:
        utility.check_hashes(manifest["hashes"])
        utility.check_hashes(frozen["hashes"])
        row = run_case(out, job)
        geometry = out / "solver" / job["sample"] / "case/airframe.vspgeom"
        row["mesh_matches_prediction"] = (
            geometry.exists() and utility.digest(geometry) == audit.loc[job["sample"], "mesh_sha256"])
        labels.append(row)
        pd.DataFrame(labels).to_csv(test_out / "独立标签.csv", index=False)
        progress(out, "solver_test", test_completed=len(labels), test_requested=TEST_COUNT)
        print("Coverage native test", len(labels), "/", TEST_COUNT, flush=True)
    native.evaluate(test_out, labels, frozen)
    progress(out, "completed", training_rows=6144, test_completed=len(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collect-limit", type=int, default=0)
    args = parser.parse_args()
    try:
        main(args.output_dir.resolve(), args.prepare_only, args.collect_limit)
    except Exception:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        progress(args.output_dir, "failed", error=traceback.format_exc())
        raise
