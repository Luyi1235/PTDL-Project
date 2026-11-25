"""Refit with the first uniform block and pre-register a future incremental comparison."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import 评价_代理候选队列20261017 as queue
from 分目标气动高斯过程 import TargetwiseAeroGP
from 局部几何气动代理 import LocalGeometryAero
from 线性叶节点气动代理 import LinearLeafAero

ROOT, BATCH = queue.ROOT, queue.BATCH
OUT = ROOT / "运行结果/首批均匀增量代理_20261017"
TRAINER = ROOT / "运行脚本/训练_均匀覆盖代理20261017.py"
FEATURES, TARGETS = queue.FEATURES, queue.TARGETS


def prepare_training(out):
    out.mkdir(parents=True, exist_ok=True)
    freeze = out / "增量训练冻结.json"
    if freeze.exists():
        queue.evaluator.check_hashes(json.loads(freeze.read_text(encoding="utf8"))["hashes"])
        return
    plan_path = BATCH / "冻结配置.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    source = Path(plan["source_data"])
    rows, records = [], []
    for job in [j for j in plan["jobs"] if j["role"] == "train"][:128]:
        record_path = BATCH / "solver" / job["sample"] / "record.json"
        row = json.loads(record_path.read_text(encoding="utf8"))
        if row["sample"] != job["sample"] or row["role"] != "train":
            raise RuntimeError("Incorrect incremental training sample")
        np.testing.assert_allclose([row[k] for k in FEATURES],
                                   [job["design"][k] for k in FEATURES], rtol=1e-12, atol=1e-10)
        rows.append(row)
        records.append(record_path)
    frame = pd.concat((pd.read_csv(source), pd.DataFrame(rows)), ignore_index=True)
    if frame.duplicated(FEATURES).any():
        raise RuntimeError("Duplicate incremental training geometries")
    data = out / "训练标签冻结.csv"
    frame.to_csv(data, index=False)
    freeze.write_text(json.dumps(dict(
        training_samples=len(frame), new_samples=len(rows), failures_retained=True,
        selection_data_only=True, sources_count=len(records),
        hashes={str(p): queue.study.prior.digest(p) for p in
                [Path(__file__), TRAINER, plan_path, source, data, *records]},
        started_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")


def fit_and_register(out):
    prepare_training(out)
    training_out = out / "训练"
    model_path = training_out / "均匀覆盖候选代理.json"
    if not model_path.exists():
        with (out / "训练输出.log").open("w", encoding="utf8") as stream:
            subprocess.run([sys.executable, str(TRAINER), "--data", str(out / "训练标签冻结.csv"),
                            "--output-dir", str(training_out), "--workers", "12"],
                           stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT,
                           timeout=10800, check=True)
    selection_path = training_out / "模型选择.json"
    selection = json.loads(selection_path.read_text(encoding="utf8"))
    if queue.study.prior.digest(model_path) != selection["model_sha256"]:
        raise RuntimeError("Interim surrogate changed after model selection")
    evaluation = out / "预先登记评价"
    evaluation.mkdir(exist_ok=True)
    freeze = evaluation / "预测冻结.json"
    if freeze.exists():
        queue.evaluator.check_hashes(json.loads(freeze.read_text(encoding="utf8"))["hashes"])
        return evaluation
    plan_path = BATCH / "冻结配置.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    train_jobs = [j for j in plan["jobs"] if j["role"] == "train"]
    jobs = train_jobs[256:384] + [j for j in plan["jobs"] if j["role"] == "test"]
    # These labels do not exist at registration, including the next training block.
    if any((BATCH / "solver" / job["sample"]).exists() for job in jobs):
        raise RuntimeError("Prospective cohort computation started before registration")
    paths = dict(
        interim1640=model_path,
        linear_leaf=queue.study.OUT / "线性叶节点候选代理.json",
        local_geometry=queue.study.prior.OUT / "局部几何候选代理.json",
        balanced1512=queue.study.prior.prior.OUT / "密度均衡候选代理.json",
        geometry_interaction=queue.study.prior.prior.REFERENCE,
        workbench289=Path(plan["model_paths"]["workbench289"]))
    frame = pd.DataFrame([dict(sample=j["sample"], role=j["role"], **j["design"]) for j in jobs])
    x = frame[FEATURES].to_numpy(float)
    for name, path in paths.items():
        if name == "linear_leaf":
            prediction = LinearLeafAero.from_json(path).predict_many(x)
        elif name == "local_geometry":
            prediction = LocalGeometryAero.from_json(path).predict_many(x)
        else:
            model = TargetwiseAeroGP.from_json(path)
            prediction = np.array([model.predict(
                queue.study.prior.prior.previous.prior.original.prior.AirframeDesign(**j["design"]))
                for j in jobs])
        for i in range(2):
            frame[f"{name}_prediction_{i}"] = prediction[:, i]
    predictions = evaluation / "预先锁定预测.csv"
    frame.to_csv(predictions, index=False)
    sources = [Path(__file__), TRAINER, Path(queue.__file__), Path(queue.evaluator.__file__),
               out / "训练标签冻结.csv", out / "增量训练冻结.json", selection_path,
               predictions, plan_path,
               ROOT / "核心模型/局部几何气动代理.py", ROOT / "核心模型/线性叶节点气动代理.py",
               *paths.values()]
    freeze.write_text(json.dumps(dict(
        hashes={str(p): queue.study.prior.digest(p) for p in sources},
        models=list(paths), model_paths={k: str(v) for k, v in paths.items()},
        frozen_before_all_evaluation_computation=True, test_labels_used_for_selection=False,
        training_samples=1640, prospective_incremental_holdout_samples=128,
        train_block="Uniform new training rows 256..383; excluded from every interim model",
        independent_test_samples=128, automatic_production_promotion=False,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    return evaluation


def additional_comparison(out, role):
    name = "新增训练区块诊断" if role == "train" else "独立测试"
    result_path = out / f"{name}.json"
    result = json.loads(result_path.read_text(encoding="utf8"))
    frame = pd.read_csv(out / f"{name}逐点.csv")
    y = frame[TARGETS[1]].to_numpy(float)
    valid = np.isfinite(y) & frame.wake_quality_pass.fillna(False).to_numpy(bool)
    result["interim_wake_differences"] = [
        dict(reference=name, **queue.evaluator.paired_bootstrap(
            y[valid], frame.interim1640_prediction_1.to_numpy(float)[valid],
            frame[f"{name}_prediction_1"].to_numpy(float)[valid]))
        for name in ("geometry_interaction", "balanced1512")]
    result["prospectively_excluded_from_interim_training"] = True
    result["later_used_in_main3072_training"] = role == "train"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf8")


def run(out, wait):
    evaluation = fit_and_register(out)
    manifest = json.loads((evaluation / "预测冻结.json").read_text(encoding="utf8"))
    completed, start = set(), time.monotonic()
    while True:
        queue.evaluator.check_hashes(manifest["hashes"])
        for role in ("train", "test"):
            if role not in completed and queue.evaluate(role, evaluation, manifest):
                additional_comparison(evaluation, role)
                completed.add(role)
                print("Incremental prospective cohort completed:", role, flush=True)
        (out / "进度.json").write_text(json.dumps(dict(
            stage="completed" if len(completed) == 2 else "waiting_prospective_native",
            completed_blocks=sorted(completed), pid=os.getpid(),
            training_samples=1640, elapsed_s=time.monotonic()-start,
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))), encoding="utf8")
        if len(completed) == 2 or not wait:
            return
        progress = json.loads((BATCH / "进度.json").read_text(encoding="utf8"))
        if progress["stage"] == "failed":
            raise RuntimeError("Native batch failed: " + str(progress["error"]))
        if time.monotonic()-start > 20*3600:
            raise TimeoutError("Prospective incremental evaluation timed out")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.wait)
