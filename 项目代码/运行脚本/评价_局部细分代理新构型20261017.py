"""Freeze predictions before the queued holdout is solved and evaluate them later."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import 研究_局部细分尾迹代理20261017 as study
from 局部几何气动代理 import LocalGeometryAero
from 分目标气动高斯过程 import TargetwiseAeroGP

ROOT = study.ROOT
BATCH = ROOT / "运行结果/均匀覆盖3072代理_20261017"
OUT = study.OUT / "新构型评价"
FEATURES, TARGETS = study.prior.FEATURES, study.prior.TARGETS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen evaluation artifact changed: {path}")


def prepare(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "预测冻结.json"
    if frozen.exists():
        check_hashes(json.loads(frozen.read_text(encoding="utf8"))["hashes"])
        return
    native_plan = BATCH / "冻结配置.json"
    plan = json.loads(native_plan.read_text(encoding="utf8"))
    jobs = [j for j in plan["jobs"] if j["role"] == "train"][:128]
    jobs += [j for j in plan["jobs"] if j["role"] == "test"]
    test_directories = [BATCH / "solver" / j["sample"] for j in jobs if j["role"] == "test"]
    if any(path.exists() for path in test_directories):
        raise RuntimeError("Cannot pre-register after the sealed test computation started")
    paths = {
        "local_geometry": study.OUT / "局部几何候选代理.json",
        "geometry_interaction": study.prior.REFERENCE,
        "balanced1512": study.prior.OUT / "密度均衡候选代理.json",
        "workbench289": Path(plan["model_paths"]["workbench289"]),
    }
    models = {name: (LocalGeometryAero.from_json(path) if name == "local_geometry"
                     else TargetwiseAeroGP.from_json(path)) for name, path in paths.items()}
    predictions = []
    for job in jobs:
        row = dict(sample=job["sample"], role=job["role"], **job["design"])
        design = study.prior.previous.prior.original.prior.AirframeDesign(**job["design"])
        for name, model in models.items():
            for i, value in enumerate(model.predict(design)):
                row[f"{name}_prediction_{i}"] = value
        predictions.append(row)
    prediction_path = out / "预先锁定预测.csv"
    pd.DataFrame(predictions).to_csv(prediction_path, index=False)
    sources = [Path(__file__), Path(study.__file__),
               ROOT / "核心模型/局部几何气动代理.py", native_plan,
               study.OUT / "配置冻结.json", study.OUT / "模型选择.json",
               prediction_path, *paths.values()]
    manifest = dict(
        hashes={str(p): digest(p) for p in sources},
        model_paths={k: str(v) for k, v in paths.items()},
        models=list(paths), frozen_before_new_test=True,
        training_block="First 128 new uniform geometries; diagnostic only, later training rows",
        independent_block="All 128 new sealed geometries from the expansion plan",
        test_labels_used_for_selection=False, automatic_production_promotion=False,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    frozen.write_text(json.dumps(manifest, indent=2), encoding="utf8")


def paired_bootstrap(y, candidate, reference, repetitions=2000):
    rng = np.random.default_rng(2026101741)
    delta = []
    for _ in range(repetitions):
        index = rng.integers(0, len(y), len(y))
        variance = np.sum((y[index] - y[index].mean())**2)
        if variance <= 1e-15:
            continue
        delta.append((np.sum((y[index]-reference[index])**2)
                      - np.sum((y[index]-candidate[index])**2))/variance)
    return dict(mean=float(np.mean(delta)),
                lower95=float(np.quantile(delta, .025)),
                upper95=float(np.quantile(delta, .975)), repetitions=len(delta))


def evaluate_block(frame, role, manifest, out):
    records = []
    for sample in frame["sample"]:
        path = BATCH / "solver" / sample / "record.json"
        if not path.exists():
            return False
        records.append(json.loads(path.read_text(encoding="utf8")))
    labels = pd.DataFrame(records)
    if labels["sample"].duplicated().any() or (labels.role != role).any():
        raise RuntimeError("Evaluation labels belong to a different cohort")
    labels = labels.set_index("sample").loc[frame["sample"]]
    np.testing.assert_allclose(labels[FEATURES].to_numpy(float),
                               frame[FEATURES].to_numpy(float), atol=1e-10, rtol=1e-12)
    metrics = []
    for index, target in enumerate(TARGETS):
        y = labels[target].to_numpy(float)
        valid = np.isfinite(y)
        if index == 1:
            valid &= labels.wake_quality_pass.fillna(False).to_numpy(bool)
        if valid.sum() < 3:
            raise RuntimeError("Too few valid evaluation labels")
        for name in manifest["models"]:
            p = frame[f"{name}_prediction_{index}"].to_numpy(float)
            metrics.append(dict(model=name, target=target, count=int(valid.sum()),
                                **study.prior.previous.prior.direct.base.metric(y[valid], p[valid])))
    valid = np.isfinite(labels[TARGETS[1]].to_numpy(float))
    valid &= labels.wake_quality_pass.fillna(False).to_numpy(bool)
    bootstrap = paired_bootstrap(
        labels[TARGETS[1]].to_numpy(float)[valid],
        frame["local_geometry_prediction_1"].to_numpy(float)[valid],
        frame["balanced1512_prediction_1"].to_numpy(float)[valid])
    result = dict(
        role=role, independent_test=(role == "test"), attempted=len(frame), metrics=metrics,
        wake_r2_difference_vs_balanced=bootstrap, test_labels_used_for_selection=False,
        quality_rules_unchanged=True, automatic_production_promotion=False,
        completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    name = "新增训练区块诊断" if role == "train" else "独立测试"
    (out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    pd.DataFrame(metrics).to_csv(out / f"{name}指标.csv", index=False)
    labeled = frame.copy()
    for target in TARGETS:
        labeled[target] = labels[target].to_numpy()
    labeled["wake_quality_pass"] = labels.wake_quality_pass.to_numpy()
    labeled.to_csv(out / f"{name}逐点.csv", index=False)
    return True


def run(out=OUT, wait=False, max_hours=20):
    prepare(out)
    manifest = json.loads((out / "预测冻结.json").read_text(encoding="utf8"))
    started = time.monotonic()
    completed = set()
    while True:
        check_hashes(manifest["hashes"])
        frame = pd.read_csv(out / "预先锁定预测.csv")
        for role in ("train", "test"):
            if role not in completed and evaluate_block(
                    frame.loc[frame.role == role].reset_index(drop=True), role, manifest, out):
                completed.add(role)
                print("Local wake evaluation completed:", role, flush=True)
        (out / "进度.json").write_text(json.dumps(dict(
            stage="completed" if len(completed) == 2 else "waiting_native_labels",
            completed_blocks=sorted(completed), pid=os.getpid(),
            elapsed_s=time.monotonic()-started,
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))), encoding="utf8")
        if len(completed) == 2 or not wait:
            return
        progress = json.loads((BATCH / "进度.json").read_text(encoding="utf8"))
        if progress["stage"] == "failed":
            raise RuntimeError("Uniform expansion batch failed: " + str(progress["error"]))
        if time.monotonic()-started > max_hours*3600:
            raise TimeoutError("Timed out waiting for native holdout labels")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--max-hours", type=float, default=20.)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.wait, args.max_hours)
