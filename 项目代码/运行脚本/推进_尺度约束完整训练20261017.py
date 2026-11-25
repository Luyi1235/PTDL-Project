"""Validate the 2152 fit, then refit all 3072 rows and solve a new sealed cohort."""

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

import numpy as np
import pandas as pd

import 研究_尺度约束与总体误差20261017 as study
import 推进_均匀覆盖3072代理20261017 as expansion
import 评价_局部细分代理新构型20261017 as evaluator
from 尺度约束气动代理 import ScalePopulationAero
from 分目标气动高斯过程 import TargetwiseAeroGP

ROOT, BATCH = study.ROOT, study.BATCH
OUT = ROOT / "运行结果/尺度约束完整训练_20261017"
FEATURES, TARGETS = expansion.FEATURES, expansion.TARGETS
TRAINER = Path(study.__file__)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf8")


def validate_record(job, row):
    if row["sample"] != job["sample"] or row["role"] != job["role"]:
        raise RuntimeError("Incorrect native record identity")
    np.testing.assert_allclose([row[k] for k in FEATURES],
                               [job["design"][k] for k in FEATURES], atol=1e-10, rtol=1e-12)


def fit_control(data, out):
    path = out / "同数据旧结构对照.json"
    if path.exists():
        return path
    frame = pd.read_csv(data)
    valid = np.isfinite(frame[TARGETS[1]].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    sub = frame.loc[valid]
    x, y = sub[FEATURES].to_numpy(float), sub[TARGETS[1]].to_numpy(float)
    fitted = study.fit(study.CONFIGS[12], x, y, sub.selection_basis.to_numpy(str))
    payload = read(study.prior.prior.REFERENCE)
    payload["targets"][1] = dict(
        kind="weighted-ensemble", weights=[.75, .25], interval_std=.06,
        models=[study.historical_portable(m, .06) for m in fitted["models"]])
    payload["training_samples"] = len(frame)
    save(path, payload)
    model = TargetwiseAeroGP.from_json(path)
    probes = x[np.linspace(0, len(x)-1, 40).astype(int)]
    actual = np.array([model.predict(
        expansion.balanced.previous.prior.original.prior.AirframeDesign(
            **dict(zip(FEATURES, row))))[1] for row in probes])
    np.testing.assert_allclose(actual, study.predict(fitted, probes), atol=1e-9, rtol=1e-9)
    return path


def register_predictions(folder, jobs, paths, data):
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / "预测冻结.json"
    if frozen.exists():
        manifest = read(frozen)
        evaluator.check_hashes(manifest["hashes"])
        if manifest["jobs"] != jobs:
            raise RuntimeError("Evaluation cohort changed")
        return manifest
    training = pd.read_csv(data)[FEATURES].to_numpy(float)
    x = np.array([[j["design"][k] for k in FEATURES] for j in jobs])
    if any(np.any(np.all(np.isclose(training, row, rtol=0, atol=1e-10), axis=1)) for row in x):
        raise RuntimeError("Evaluation geometry appears in training")
    frame = pd.DataFrame([dict(sample=j["sample"], **j["design"]) for j in jobs])
    design_class = expansion.balanced.previous.prior.original.prior.AirframeDesign
    for name, path in paths.items():
        if read(path).get("schema") == "ptdl-scale-population-aero-v1":
            prediction = ScalePopulationAero.from_json(path).predict_many(x)
        else:
            model = TargetwiseAeroGP.from_json(path)
            prediction = np.array([model.predict(design_class(**j["design"])) for j in jobs])
        for i in range(2):
            frame[f"{name}_prediction_{i}"] = prediction[:, i]
    csv = folder / "锁定预测.csv"
    frame.to_csv(csv, index=False)
    sources = [Path(__file__), TRAINER, Path(evaluator.__file__), data, csv,
               ROOT / "核心模型/尺度约束气动代理.py", *paths.values()]
    manifest = dict(jobs=jobs, models=list(paths),
                    hashes={str(p): expansion.digest(p) for p in sources},
                    selection_uses_evaluation_labels=False,
                    automatic_production_promotion=False,
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(frozen, manifest)
    return manifest


def evaluate(folder, native_root, filename):
    manifest = read(folder / "预测冻结.json")
    evaluator.check_hashes(manifest["hashes"])
    records = []
    for job in manifest["jobs"]:
        cache = native_root / job["sample"] / "record.json"
        if not cache.exists():
            return False
        row = read(cache)
        validate_record(job, row)
        records.append(row)
    predictions = pd.read_csv(folder / "锁定预测.csv").set_index("sample")
    labels = pd.DataFrame(records).set_index("sample")
    predictions = predictions.loc[labels.index]
    metrics = []
    for index, target in enumerate(TARGETS):
        y = labels[target].to_numpy(float)
        valid = np.isfinite(y)
        if index == 1:
            valid &= labels.wake_quality_pass.fillna(False).to_numpy(bool)
        if valid.sum() < 3:
            raise RuntimeError("Insufficient valid labels in the frozen evaluation cohort")
        for name in manifest["models"]:
            p = predictions[f"{name}_prediction_{index}"].to_numpy(float)
            metrics.append(dict(model=name, target=target, count=int(valid.sum()),
                **expansion.balanced.previous.prior.direct.base.metric(y[valid], p[valid])))
    y = labels[TARGETS[1]].to_numpy(float)
    valid = np.isfinite(y) & labels.wake_quality_pass.fillna(False).to_numpy(bool)
    candidate = predictions.scale_candidate_prediction_1.to_numpy(float)[valid]
    differences = [dict(reference=name, **evaluator.paired_bootstrap(
        y[valid], candidate, predictions[f"{name}_prediction_1"].to_numpy(float)[valid]))
        for name in manifest["models"] if name != "scale_candidate"]
    save(folder / f"{filename}.json", dict(
        metrics=metrics, attempted=len(labels), wake_r2_differences=differences,
        training_selection_only=True, quality_rules_unchanged=True,
        independent_final_test=filename == "新增独立测试",
        target_r2=.9, automatic_production_promotion=False,
        completed_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    pd.DataFrame(metrics).to_csv(folder / f"{filename}指标.csv", index=False)
    labels.join(predictions[[c for c in predictions if "_prediction_" in c]]).to_csv(
        folder / f"{filename}逐点.csv")
    return True


def fresh_plan(primary):
    source = pd.read_csv(primary["source_data"])
    x = source[FEATURES].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    forbidden = [x, np.array([[j["design"][k] for k in FEATURES] for j in primary["jobs"]])]
    previous = expansion.balanced.previous.prior
    for folder in (previous.SOURCE, previous.OUT):
        forbidden.append(pd.read_csv(folder / "中网格独立测试标签.csv")[FEATURES].to_numpy(float))
    driver = previous.original.prior.driver
    forbidden.append(pd.read_csv(driver.LOW_DIR / "合并留出验证.csv")[FEATURES].to_numpy(float))
    for folder in (previous.original.prior.FIRST, ROOT / "运行结果/多精度气动代理_20261016"):
        old = read(folder / "冻结采样与计算配置.json")
        forbidden.append(np.array([[j["design"][k] for k in FEATURES]
                                   for j in old["jobs"] if j["role"] == "test"]))
    old = read(ROOT / "运行结果/几何交互独立验证_20261017/冻结配置.json")
    forbidden.append(np.array([[j["design"][k] for k in FEATURES] for j in old["jobs"]]))
    avoid = np.vstack(forbidden)
    fixed = {k: float(source.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    rng = np.random.default_rng(2026101763)
    jobs = []
    for _ in range(100000):
        row = rng.uniform(lo, hi)
        design = {**fixed, **dict(zip(FEATURES, row))}
        violations = previous.original.prior.geometry_constraints(
            previous.original.prior.AirframeDesign(**design))
        if max(violations.values()) <= 1e-9 and np.linalg.norm((avoid-row)/(hi-lo), axis=1).min() > 1e-5:
            jobs.append(dict(sample=f"scale3072_test_{len(jobs):04d}", role="test", design=design))
            avoid = np.vstack((avoid, row))
        if len(jobs) == 128:
            return jobs
    raise RuntimeError("Unable to construct a fresh feasible cohort")


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    primary = read(BATCH / "冻结配置.json")
    frozen = out / "流程冻结.json"
    if frozen.exists():
        evaluator.check_hashes(read(frozen)["hashes"])
        return read(frozen)
    training = study.OUT / "训练标签冻结.csv"
    control = fit_control(training, out)
    paths = dict(scale_candidate=study.OUT / "尺度与总体加权候选代理.json",
                 same_data_control=control, geometry1512=study.prior.prior.REFERENCE,
                 interim1640=ROOT / "运行结果/首批均匀增量代理_20261017/训练/均匀覆盖候选代理.json",
                 workbench289=Path(primary["model_paths"]["workbench289"]))
    jobs = [j for j in primary["jobs"] if j["role"] == "train"][640:768]
    manifest = register_predictions(out / "2152留出诊断", jobs, paths, training)
    plan = dict(fresh_test_jobs=fresh_plan(primary),
                hashes={**manifest["hashes"], str(BATCH / "冻结配置.json"):
                        expansion.digest(BATCH / "冻结配置.json")},
                target_training_count=3072, original_protocol_unchanged=True,
                extra_test_selection_uses_training_only=True,
                extra_test_solver_starts_after_full_model_lock=True,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(frozen, plan)
    return plan


def full_snapshot(out):
    folder = out / "完整训练"
    folder.mkdir(exist_ok=True)
    frozen = folder / "数据冻结.json"
    if frozen.exists():
        evaluator.check_hashes(read(frozen)["hashes"])
        return folder / "训练标签冻结.csv"
    primary = read(BATCH / "冻结配置.json")
    source = Path(primary["source_data"])
    rows, paths = [], []
    for job in primary["jobs"]:
        if job["role"] != "train":
            continue
        path = BATCH / "solver" / job["sample"] / "record.json"
        if not path.exists():
            return None
        row = read(path)
        validate_record(job, row)
        rows.append(row)
        paths.append(path)
    frame = pd.concat((pd.read_csv(source), pd.DataFrame(rows)), ignore_index=True)
    if len(frame) != 3072 or frame.duplicated(FEATURES).any():
        raise RuntimeError("Full training cohort has an incorrect size or repeated geometry")
    data = folder / "训练标签冻结.csv"
    frame.to_csv(data, index=False)
    save(frozen, dict(hashes={str(p): expansion.digest(p) for p in [source, data, *paths]},
        original_samples=len(frame)-len(rows), new_samples=len(rows),
        all_failed_records_retained=True, sampled_by_external_errors=False,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    return data


def run(out, prepare_only=False):
    start = time.monotonic()
    plan = prepare(out)
    def progress(stage, **values):
        save(out / "进度.json", dict(stage=stage, pid=os.getpid(), **values,
            elapsed_s=time.monotonic()-start, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    diagnostic = out / "2152留出诊断"
    if prepare_only:
        progress("prepared")
        return
    try:
        while True:
            evaluator.check_hashes(plan["hashes"])
            if not (diagnostic / "后续留出诊断.json").exists():
                evaluate(diagnostic, BATCH / "solver", "后续留出诊断")
            data = full_snapshot(out)
            if data is not None:
                break
            state = read(BATCH / "进度.json")
            if state["stage"] == "failed":
                raise RuntimeError("Primary native expansion failed: " + str(state["error"]))
            progress("waiting_full_training", train_completed=state["train_completed"],
                     train_requested=3072,
                     interim_diagnostic_completed=(diagnostic / "后续留出诊断.json").exists())
            time.sleep(30)
        training = out / "完整训练"
        progress("full_training_cv")
        if not (training / "模型选择.json").exists():
            with (out / "完整训练输出.log").open("w", encoding="utf8") as stream:
                subprocess.run([sys.executable, str(TRAINER), "--output-dir", str(training),
                                "--workers", "12"], stdout=stream, stderr=subprocess.STDOUT,
                               cwd=ROOT, timeout=14400, check=True)
        evaluator.check_hashes(plan["hashes"])
        control = fit_control(data, training)
        paths = dict(scale_candidate=training / "尺度与总体加权候选代理.json",
                     same_data_control=control, geometry1512=study.prior.prior.REFERENCE,
                     workbench289=Path(read(BATCH / "冻结配置.json")["model_paths"]["workbench289"]))
        if (BATCH / "均匀覆盖候选代理.json").exists():
            paths["expanded3072"] = BATCH / "均匀覆盖候选代理.json"
        evaluation = out / "3072新增独立测试"
        lock = register_predictions(evaluation, plan["fresh_test_jobs"], paths, data)
        original = expansion.balanced.previous.prior.original
        original.controlled.original.WORKER = original.WORKER
        completed = 0
        for job in plan["fresh_test_jobs"]:
            evaluator.check_hashes(plan["hashes"])
            evaluator.check_hashes(lock["hashes"])
            folder = out / "solver" / job["sample"]
            folder.mkdir(parents=True, exist_ok=True)
            cache = folder / "record.json"
            if cache.exists():
                validate_record(job, read(cache))
            else:
                try:
                    row = original.controlled.run_case(
                        original.prior.solver.DEFAULT_PYTHON, job["design"], job["sample"],
                        folder / "case", 600, "test", "sealed_uniform_scale_test",
                        (0., float("inf")), 30, "legacy")
                    row.update(original.verify(folder / "case"))
                    row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
                    row["wake_quality_pass"] = bool(row["quality_pass"])
                except Exception as exc:
                    row = dict(sample=job["sample"], **job["design"], solver_failure=repr(exc),
                               wake_quality_pass=False, **{key: None for key in TARGETS})
                row.update(role="test")
                save(cache, row)
            completed += 1
            progress("sealed_native_test", completed=completed, requested=128)
        if not evaluate(evaluation, out / "solver", "新增独立测试"):
            raise RuntimeError("Sealed evaluation labels missing")
        progress("completed", completed=128, requested=128)
    except Exception:
        progress("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.prepare_only)
