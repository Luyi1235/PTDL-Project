"""Add training-only disagreement and error-neighborhood samples with a fresh holdout."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc

import 研究_精细树与核代理20261018 as fine
import 推进_尺度约束完整训练20261017 as previous
from 精细气动代理 import FineAeroSurrogate
from 尺度约束气动代理 import ScalePopulationAero, geometry_transform
from 分目标气动高斯过程 import TargetwiseAeroGP

ROOT = fine.ROOT
OUT = ROOT / "运行结果/分歧补样4096代理_20261018"
SOURCE = fine.DATA
FEATURES, TARGETS = fine.FEATURES, fine.TARGETS
BASE = previous.BATCH
OLD_TRAINER = ROOT / "运行脚本/训练_均匀覆盖代理20261017.py"


def old_batch_prediction(x):
    selection = previous.read(BASE / "模型选择.json")
    values = np.empty((len(x), 2))
    for i, entry in enumerate(selection["selected"]):
        values[:, i] = sum(c["weight"]*previous.expansion.balanced.tree_predict(
            joblib.load(c["path"]), x) for c in entry["components"])
    return values


def error_centers(frame):
    records = previous.read(BASE / "训练交叉验证逐折.json")
    selection = previous.read(BASE / "模型选择.json")
    error = np.zeros(len(frame))
    for entry in selection["selected"]:
        target = entry["target"]
        valid = np.isfinite(frame[target].to_numpy(float))
        if target == TARGETS[1]:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        y = frame.loc[valid, target].to_numpy(float)
        subset = [r for r in records if r["target"] == target]
        p = sum(c["weight"]*fine.previous.prior.prior.previous.prior.records_matrix(
            subset, len(y), c["config"]) for c in entry["components"])
        error[valid] += np.mean(abs(p-y), axis=0)/max(float(np.std(y)), 1e-9)
    return np.argsort(-error)[:200]


def forbidden_geometries(frame):
    arrays = [frame[FEATURES].to_numpy(float)]
    folders = [BASE, previous.OUT]
    for folder in folders:
        manifest = previous.read(folder / ("冻结配置.json" if folder == BASE else "流程冻结.json"))
        jobs = manifest["jobs"] if folder == BASE else manifest["fresh_test_jobs"]
        arrays.append(np.array([[j["design"][k] for k in FEATURES] for j in jobs]))
    old = previous.expansion.balanced.previous.prior
    for folder in (old.SOURCE, old.OUT):
        arrays.append(pd.read_csv(folder / "中网格独立测试标签.csv")[FEATURES].to_numpy(float))
    arrays.append(pd.read_csv(old.original.prior.driver.LOW_DIR / "合并留出验证.csv")[
        FEATURES].to_numpy(float))
    for folder in (old.original.prior.FIRST, ROOT / "运行结果/多精度气动代理_20261016"):
        manifest = previous.read(folder / "冻结采样与计算配置.json")
        arrays.append(np.array([[j["design"][k] for k in FEATURES]
                                for j in manifest["jobs"] if j["role"] == "test"]))
    manifest = previous.read(ROOT / "运行结果/几何交互独立验证_20261017/冻结配置.json")
    arrays.append(np.array([[j["design"][k] for k in FEATURES] for j in manifest["jobs"]]))
    return np.vstack(arrays)


def plan():
    frame = pd.read_csv(SOURCE)
    if len(frame) != 3072:
        raise RuntimeError("Expected the completed 3072-row training cohort")
    x = frame[FEATURES].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    span = hi-lo
    avoid = forbidden_geometries(frame)
    fixed = {k: float(frame.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    original = previous.expansion.balanced.previous.prior.original
    jobs = []
    def add(row, basis, role="train"):
        nonlocal avoid
        if np.any(row < lo) or np.any(row > hi):
            return False
        design = {**fixed, **dict(zip(FEATURES, row))}
        if (max(original.prior.geometry_constraints(original.prior.AirframeDesign(
                **design)).values()) > 1e-9
                or np.linalg.norm((avoid-row)/span, axis=1).min() <= 1e-5):
            return False
        jobs.append(dict(sample=f"adaptive4096_{role}_{len(jobs):04d}",
                         role=role, basis=basis, design=design))
        avoid = np.vstack((avoid, row))
        return True
    pool = qmc.scale(qmc.Sobol(7, scramble=True, seed=2026101811).random_base2(14), lo, hi)
    feasible = []
    for row in pool:
        design = {**fixed, **dict(zip(FEATURES, row))}
        if max(original.prior.geometry_constraints(original.prior.AirframeDesign(
                **design)).values()) <= 1e-9:
            feasible.append(row)
    feasible = np.array(feasible)
    for row in feasible:
        add(row, "uniform_feasible_coverage")
        if len(jobs) == 512:
            break
    reference = old_batch_prediction(feasible)
    model = CatBoostRegressor()
    model.load_model(str(previous.OUT / "完整训练/wake_config_6.cbm"))
    canonical = model.predict(fine.previous.prior.prior.previous.features(
        geometry_transform(feasible, "canonical-area"), "interference"))
    disagreement = abs(reference[:, 1]-canonical)
    for row in feasible[np.argsort(-disagreement)]:
        add(row, "training_models_disagreement")
        if len(jobs) == 768:
            break
    centers = x[error_centers(frame)]
    rng = np.random.default_rng(2026101812)
    for _ in range(100000):
        row = centers[rng.integers(len(centers))] + rng.normal(0, .025, 7)*span
        add(row, "training_oof_error_neighborhood")
        if len(jobs) == 1024:
            break
    if len(jobs) != 1024:
        raise RuntimeError("Adaptive training pool exhausted")
    for _ in range(100000):
        add(rng.uniform(lo, hi), "fresh_sealed_uniform", "test")
        if len(jobs) == 1216:
            break
    if len(jobs) != 1216:
        raise RuntimeError("Fresh test pool exhausted")
    sources = [Path(__file__), Path(fine.__file__), ROOT / "核心模型/精细气动代理.py",
               SOURCE, BASE / "模型选择.json", BASE / "训练交叉验证逐折.json",
               previous.OUT / "完整训练/wake_config_6.cbm"]
    hashes = {str(p): previous.expansion.digest(p) for p in sources}
    prior_manifest = previous.read(BASE / "冻结配置.json")
    hashes.update({str(ROOT / name): value for name, value in prior_manifest["source_hashes"].items()})
    return dict(jobs=jobs, hashes=hashes, new_train_count=1024, target_train_count=4096,
                test_count=192, acquisition_counts=dict(uniform=512, disagreement=256,
                                                       training_oof_neighborhood=256),
                external_error_labels_used=False, original_labels_replaced=False,
                automatic_production_promotion=False,
                protocol="free-wake-legacy-v1", cores=8, wake_nodes=8, wake_iterations=30,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"))


def predictions_and_lock(out, jobs, paths):
    frozen = out / "独立预测冻结.json"
    if frozen.exists():
        result = previous.read(frozen)
        previous.evaluator.check_hashes(result["hashes"])
        return result
    frame = pd.DataFrame([dict(sample=j["sample"], **j["design"]) for j in jobs])
    x = frame[FEATURES].to_numpy(float)
    design_class = previous.expansion.balanced.previous.prior.original.prior.AirframeDesign
    for name, path in paths.items():
        schema = previous.read(path).get("schema")
        if schema == "ptdl-fine-aero-v1":
            p = FineAeroSurrogate.from_json(path).predict_many(x)
        elif schema == "ptdl-scale-population-aero-v1":
            p = ScalePopulationAero.from_json(path).predict_many(x)
        else:
            model = TargetwiseAeroGP.from_json(path)
            p = np.array([model.predict(design_class(**j["design"])) for j in jobs])
        for i in range(2):
            frame[f"{name}_prediction_{i}"] = p[:, i]
    csv = out / "独立锁定预测.csv"
    frame.to_csv(csv, index=False)
    result = dict(models=list(paths), hashes={str(p): previous.expansion.digest(p)
        for p in [csv, *paths.values()]}, locked_before_test_solver=True,
        time=time.strftime("%Y-%m-%d %H:%M:%S"))
    previous.save(frozen, result)
    return result


def evaluate(out, records, lock):
    previous.evaluator.check_hashes(lock["hashes"])
    data = pd.DataFrame(records).set_index("sample")
    p = pd.read_csv(out / "独立锁定预测.csv").set_index("sample").loc[data.index]
    metrics = []
    for i, target in enumerate(TARGETS):
        y = data[target].to_numpy(float)
        valid = np.isfinite(y)
        if i:
            valid &= data.wake_quality_pass.fillna(False).to_numpy(bool)
        if valid.sum() < 3:
            raise RuntimeError("Not enough independent labels")
        for model in lock["models"]:
            metrics.append(dict(model=model, target=target, count=int(valid.sum()),
                **previous.expansion.balanced.previous.prior.direct.base.metric(
                    y[valid], p[f"{model}_prediction_{i}"].to_numpy(float)[valid])))
    previous.save(out / "独立评价.json", dict(metrics=metrics, attempted=len(data),
        goal_r2=.9, same_quality_rules=True, selection_uses_training_only=True,
        automatic_production_promotion=False))
    data.join(p[[c for c in p if "_prediction_" in c]]).to_csv(out / "独立评价逐点.csv")


def main(out, prepare_only=False):
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "配置冻结.json"
    if not frozen.exists():
        previous.save(frozen, plan())
    manifest = previous.read(frozen)
    previous.evaluator.check_hashes(manifest["hashes"])
    if prepare_only:
        return
    original = previous.expansion.balanced.previous.prior.original
    original.controlled.original.WORKER = original.WORKER
    start, records, lock = time.monotonic(), [], None
    source = pd.read_csv(SOURCE)
    def progress(stage, error=None):
        previous.save(out / "进度.json", dict(stage=stage, pid=os.getpid(),
            native_completed=len(records), native_requested=1216,
            train_completed=3072+min(len(records), 1024), train_requested=4096,
            test_completed=max(0, len(records)-1024), test_requested=192,
            error=error, elapsed_s=time.monotonic()-start,
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        for job in manifest["jobs"]:
            previous.evaluator.check_hashes(manifest["hashes"])
            if job["role"] == "test" and lock is None:
                progress("training")
                data = out / "完整训练标签.csv"
                pd.concat((source, pd.DataFrame(records)), ignore_index=True).to_csv(data, index=False)
                for script, folder, result in (
                    (Path(fine.__file__), out / "精细训练", "精细气动候选代理.json"),
                    (OLD_TRAINER, out / "旧结构对照", "均匀覆盖候选代理.json")):
                    folder.mkdir(exist_ok=True)
                    if not (folder / "模型选择.json").exists():
                        with (folder / "训练输出.log").open("w", encoding="utf8") as stream:
                            subprocess.run([sys.executable, str(script), "--data", str(data),
                                "--output-dir", str(folder), "--workers", "12"],
                                stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT,
                                check=True, timeout=21600)
                paths = dict(fine4096=out / "精细训练/精细气动候选代理.json",
                    same_data_baseline=out / "旧结构对照/均匀覆盖候选代理.json",
                    original3072=BASE / "均匀覆盖候选代理.json",
                    canonical3072=previous.OUT / "完整训练/尺度与总体加权候选代理.json")
                if (fine.OUT / "模型选择.json").exists():
                    paths["fine3072"] = fine.OUT / "精细气动候选代理.json"
                lock = predictions_and_lock(out, [j for j in manifest["jobs"] if j["role"] == "test"], paths)
            if lock:
                previous.evaluator.check_hashes(lock["hashes"])
            folder = out / "solver" / job["sample"]
            folder.mkdir(parents=True, exist_ok=True)
            cache = folder / "record.json"
            if cache.exists():
                row = previous.read(cache)
                previous.validate_record(job, row)
            else:
                try:
                    row = original.controlled.run_case(original.prior.solver.DEFAULT_PYTHON,
                        job["design"], job["sample"], folder / "case", 600,
                        job["role"], job["basis"], (0., float("inf")), 30, "legacy")
                    row.update(original.verify(folder / "case"))
                    row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
                    row["wake_quality_pass"] = bool(row["quality_pass"])
                except Exception as exc:
                    row = dict(sample=job["sample"], **job["design"],
                        solver_failure=repr(exc), wake_quality_pass=False,
                        **{k: None for k in TARGETS})
                row.update(role=job["role"], selection_basis=job["basis"])
                previous.save(cache, row)
            records.append(row)
            if job["role"] == "test":
                pd.DataFrame([r for r in records if r["role"] == "test"]).to_csv(
                    out / "独立测试标签.csv", index=False)
            progress("solver_train" if job["role"] == "train" else "solver_test")
            print("Adaptive native", len(records), "/ 1216", flush=True)
        evaluate(out, [r for r in records if r["role"] == "test"], lock)
        progress("completed")
    except Exception:
        progress("failed", traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.prepare_only)
