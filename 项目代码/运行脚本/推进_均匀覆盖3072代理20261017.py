"""Expand uniform coverage, refit on training data, then solve a new sealed holdout."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import qmc

import 研究_密度均衡与残差代理20261017 as balanced
from 分目标气动高斯过程 import TargetwiseAeroGP

ROOT = balanced.ROOT
OUT = ROOT / "运行结果/均匀覆盖3072代理_20261017"
FEATURES, TARGETS = balanced.FEATURES, balanced.TARGETS
TRAINER = Path(__file__).with_name("训练_均匀覆盖代理20261017.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(training_total=3072, tests=128):
    frame = pd.read_csv(balanced.previous.DATA)
    x = frame[FEATURES].to_numpy(float)
    if training_total <= len(frame) or tests < 1:
        raise ValueError("Expansion requires additional training geometries and a holdout")
    lo, hi = x.min(0), x.max(0)
    fixed = {k: float(frame.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    forbidden = [x]
    for folder in (balanced.previous.prior.SOURCE, balanced.previous.prior.OUT):
        forbidden.append(pd.read_csv(folder / "中网格独立测试标签.csv")[FEATURES].to_numpy(float))
    fresh = ROOT / "运行结果/几何交互独立验证_20261017/冻结配置.json"
    forbidden.append(np.array([[j["design"][k] for k in FEATURES]
                              for j in json.loads(fresh.read_text(encoding="utf8"))["jobs"]]))
    driver = balanced.previous.prior.original.prior.driver
    forbidden.append(pd.read_csv(driver.LOW_DIR / "合并留出验证.csv")[FEATURES].to_numpy(float))
    for source in (balanced.previous.prior.original.prior.FIRST, ROOT / "运行结果/多精度气动代理_20261016"):
        old = json.loads((source / "冻结采样与计算配置.json").read_text(encoding="utf8"))
        forbidden.append(np.array([[j["design"][k] for k in FEATURES] for j in old["jobs"]
                                   if j["role"] == "test"]))
    avoid = np.vstack(forbidden)
    jobs = []
    pool = qmc.scale(qmc.Sobol(7, scramble=True, seed=2026101733).random_base2(15), lo, hi)
    ntrain = training_total-len(frame)
    for row in pool:
        design = {**fixed, **dict(zip(FEATURES, row))}
        violations = balanced.previous.prior.original.prior.geometry_constraints(
            balanced.previous.prior.original.prior.AirframeDesign(**design))
        if max(violations.values()) <= 1e-9 and np.linalg.norm((avoid-row)/(hi-lo), axis=1).min() > 1e-5:
            jobs.append(dict(sample=f"uniform3072_train_{len(jobs):04d}", role="train", design=design))
            avoid = np.vstack((avoid, row))
        if len(jobs) == ntrain:
            break
    if len(jobs) != ntrain:
        raise RuntimeError("Uniform candidate pool exhausted")
    rng = np.random.default_rng(2026101734)
    while len(jobs) < ntrain+tests:
        row = rng.uniform(lo, hi)
        design = {**fixed, **dict(zip(FEATURES, row))}
        violations = balanced.previous.prior.original.prior.geometry_constraints(
            balanced.previous.prior.original.prior.AirframeDesign(**design))
        if max(violations.values()) <= 1e-9 and np.linalg.norm((avoid-row)/(hi-lo), axis=1).min() > 1e-5:
            jobs.append(dict(sample=f"uniform3072_test_{len(jobs)-ntrain:04d}", role="test", design=design))
            avoid = np.vstack((avoid, row))
    sources = driver.sources()
    for path in (Path(__file__), TRAINER, Path(balanced.__file__), Path(balanced.previous.__file__),
                 Path(__file__).with_name("导出_密度均衡与残差代理20261017.py"),
                 Path(__file__).with_name("导出_几何交互代理20261017.py"),
                 balanced.previous.prior.original.WORKER,
                 Path(balanced.previous.prior.original.__file__),
                 Path(balanced.previous.prior.original.controlled.__file__)):
        sources[str(path.relative_to(ROOT))] = digest(path)
    models = dict(geometry_interaction=balanced.REFERENCE,
                  balanced1512=balanced.OUT / "密度均衡候选代理.json",
                  workbench289=driver.LOW_MODEL)
    return dict(
        jobs=jobs, training_total=training_total, existing_count=len(frame), test_count=tests,
        source_data=str(balanced.previous.DATA), source_data_hash=digest(balanced.previous.DATA),
        source_hashes=sources, model_paths={k: str(v) for k, v in models.items()},
        model_hashes={k: digest(v) for k, v in models.items()},
        protocol="free-wake-legacy-v1", mesh="legacy", wake_nodes=8, wake_iterations=30, cores=8,
        acquisition="Uniform feasible Sobol coverage; no external error labels used",
        automatic_production_promotion=False)


def main(out=OUT, training_total=3072, tests=128, prepare_only=False):
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / "冻结配置.json"
    if not frozen_path.exists():
        frozen_path.write_text(json.dumps(plan(training_total, tests), indent=2), encoding="utf8")
    frozen = json.loads(frozen_path.read_text(encoding="utf8"))
    if (frozen["training_total"], frozen["test_count"]) != (training_total, tests):
        raise RuntimeError("Requested counts differ from frozen expansion")
    test_lock = None
    def check():
        for name, expected in frozen["source_hashes"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"Frozen expansion source changed: {name}")
        if digest(frozen["source_data"]) != frozen["source_data_hash"]:
            raise RuntimeError("Frozen source dataset changed")
        for name, expected in frozen["model_hashes"].items():
            if digest(frozen["model_paths"][name]) != expected:
                raise RuntimeError(f"Frozen reference changed: {name}")
        if test_lock is not None:
            for name, expected in test_lock["hashes"].items():
                if digest(test_lock["paths"][name]) != expected:
                    raise RuntimeError(f"Frozen test model changed: {name}")
    check()
    if prepare_only:
        print(frozen_path)
        return
    started = time.monotonic()
    records = []
    source = pd.read_csv(frozen["source_data"])
    nnew = frozen["training_total"]-len(source)
    def progress(stage, error=None):
        (out / "进度.json").write_text(json.dumps(dict(
            stage=stage, pid=os.getpid(), native_completed=len(records),
            native_requested=len(frozen["jobs"]),
            train_completed=len(source)+min(len(records), nnew),
            train_requested=frozen["training_total"],
            test_completed=max(0, len(records)-nnew), test_requested=frozen["test_count"],
            elapsed_s=time.monotonic()-started, error=error,
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    try:
        snapshot = out / "源文件快照"
        for relative, expected in frozen["source_hashes"].items():
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                shutil.copy2(ROOT / relative, path)
            if digest(path) != expected:
                raise RuntimeError("Frozen source snapshot mismatch")
        original = balanced.previous.prior.original
        original.prior.solver.WORKER = original.WORKER
        progress("solver_train")
        models = None
        for job in frozen["jobs"]:
            check()
            if job["role"] == "test" and models is None:
                progress("training")
                data = out / "训练标签.csv"
                candidate = out / "均匀覆盖候选代理.json"
                if not candidate.exists():
                    with (out / "训练输出.log").open("w", encoding="utf8") as stream:
                        subprocess.run([sys.executable, str(TRAINER), "--data", str(data),
                                        "--output-dir", str(out), "--workers", "16"], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=10800, check=True)
                selected = json.loads((out / "模型选择.json").read_text(encoding="utf8"))
                if digest(candidate) != selected["model_sha256"]:
                    raise RuntimeError("Expanded model changed after selection")
                paths = {**frozen["model_paths"], "expanded3072": str(candidate)}
                lock = dict(paths=paths, hashes={k: digest(v) for k, v in paths.items()},
                            locked_before_new_test=True,
                            time=time.strftime("%Y-%m-%d %H:%M:%S"))
                lock_path = out / "独立评价模型锁定.json"
                if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf8"))["hashes"] != lock["hashes"]:
                    raise RuntimeError("Test model hashes changed across resume")
                lock_path.write_text(json.dumps(lock, indent=2), encoding="utf8")
                test_lock = lock
                models = {k: TargetwiseAeroGP.from_json(path) for k, path in paths.items()}
            folder = out / "solver" / job["sample"]
            cache = folder / "record.json"
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf8"))
                if (row["sample"] != job["sample"] or row["role"] != job["role"]
                        or any(abs(float(row[k])-value) > 1e-10 for k, value in job["design"].items())):
                    raise RuntimeError("Cached native result belongs to a different geometry")
            else:
                folder.mkdir(parents=True)
                try:
                    row = original.native(
                        original.prior.solver.DEFAULT_PYTHON, job["design"], job["sample"],
                        folder / "case", 3600, job["role"], "uniform_feasible_coverage",
                        (0., float("inf")), 30, "legacy")
                    row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
                    row["wake_quality_pass"] = bool(row["quality_pass"])
                except Exception as exc:
                    row = dict(sample=job["sample"], **job["design"], solver_failure=repr(exc),
                               wake_quality_pass=False, **{k: None for k in TARGETS})
                row.update(role=job["role"])
                cache.write_text(json.dumps(row, indent=2), encoding="utf8")
            if job["role"] == "test":
                design = original.prior.AirframeDesign(**job["design"])
                for name, model in models.items():
                    for index, value in enumerate(model.predict(design)):
                        row[f"{name}_prediction_{index}"] = value
            records.append(row)
            training = pd.DataFrame([r for r in records if r["role"] == "train"])
            pd.concat((source, training), ignore_index=True).to_csv(out / "训练标签.csv", index=False)
            if job["role"] == "test":
                pd.DataFrame([r for r in records if r["role"] == "test"]).to_csv(
                    out / "独立测试逐点.csv", index=False)
            progress("solver_train" if job["role"] == "train" else "solver_test")
            print("Uniform expansion native", len(records), "/", len(frozen["jobs"]), flush=True)
        metrics = []
        test = pd.DataFrame([r for r in records if r["role"] == "test"])
        for name in models:
            for index, target in enumerate(TARGETS):
                y = test[target].to_numpy(float)
                valid = np.isfinite(y)
                if index:
                    valid &= test.wake_quality_pass.fillna(False).to_numpy(bool)
                if valid.sum() < 3:
                    raise RuntimeError("Insufficient valid independent labels")
                p = test[f"{name}_prediction_{index}"].to_numpy(float)
                metrics.append(dict(model=name, target=target, count=int(valid.sum()),
                                    **balanced.previous.prior.direct.base.metric(y[valid], p[valid])))
        (out / "独立评价.json").write_text(json.dumps(dict(
            metrics=metrics, training_only_selection=True, goal_r2=.9,
            automatic_production_promotion=False), indent=2), encoding="utf8")
        progress("completed")
    except Exception:
        progress("failed", traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--training-total", type=int, default=3072)
    parser.add_argument("--tests", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.training_total, args.tests, args.prepare_only)
