"""Compare frozen geometry candidates against a shared upcoming native holdout."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import 研究_线性叶节点代理20261017 as study
import 评价_局部细分代理新构型20261017 as evaluator
from 局部几何气动代理 import LocalGeometryAero
from 线性叶节点气动代理 import LinearLeafAero
from 分目标气动高斯过程 import TargetwiseAeroGP

ROOT, BATCH = study.ROOT, evaluator.BATCH
OUT = ROOT / "运行结果/代理候选统一评价_20261017"
FEATURES, TARGETS = evaluator.FEATURES, evaluator.TARGETS


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    freeze = out / "预测冻结.json"
    if freeze.exists():
        evaluator.check_hashes(json.loads(freeze.read_text(encoding="utf8"))["hashes"])
        return
    plan_path = BATCH / "冻结配置.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    jobs = [j for j in plan["jobs"] if j["role"] == "train"][:128]
    jobs += [j for j in plan["jobs"] if j["role"] == "test"]
    if any((BATCH / "solver" / j["sample"]).exists() for j in jobs if j["role"] == "test"):
        raise RuntimeError("Independent computation started before prediction registration")
    paths = dict(
        linear_leaf=study.OUT / "线性叶节点候选代理.json",
        local_geometry=study.prior.OUT / "局部几何候选代理.json",
        balanced1512=study.prior.prior.OUT / "密度均衡候选代理.json",
        geometry_interaction=study.prior.prior.REFERENCE,
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
                study.prior.prior.previous.prior.original.prior.AirframeDesign(**j["design"]))
                for j in jobs])
        for index in range(2):
            frame[f"{name}_prediction_{index}"] = prediction[:, index]
    predictions = out / "预先锁定预测.csv"
    frame.to_csv(predictions, index=False)
    sources = [
        Path(__file__), Path(study.__file__), Path(evaluator.__file__),
        ROOT / "核心模型/局部几何气动代理.py", ROOT / "核心模型/线性叶节点气动代理.py",
        plan_path, predictions, study.OUT / "配置冻结.json", study.OUT / "模型选择.json",
        *paths.values()]
    freeze.write_text(json.dumps(dict(
        hashes={str(p): study.prior.digest(p) for p in sources},
        models=list(paths), model_paths={k: str(v) for k, v in paths.items()},
        frozen_before_test_computation=True, test_labels_used_for_selection=False,
        train_block="First 128 additional training geometries, diagnostic only",
        test_block="128 entirely new sealed geometries",
        automatic_production_promotion=False, created_at=time.strftime("%Y-%m-%d %H:%M:%S")),
        indent=2), encoding="utf8")


def evaluate(role, out, manifest):
    frame = pd.read_csv(out / "预先锁定预测.csv")
    frame = frame.loc[frame.role == role].reset_index(drop=True)
    records = []
    for sample in frame["sample"]:
        path = BATCH / "solver" / sample / "record.json"
        if not path.exists():
            return False
        records.append(json.loads(path.read_text(encoding="utf8")))
    labels = pd.DataFrame(records)
    if not np.array_equal(labels["sample"].to_numpy(), frame["sample"].to_numpy()):
        raise RuntimeError("Native cohort order changed")
    if (labels.role != role).any():
        raise RuntimeError("Native cohort role changed")
    np.testing.assert_allclose(labels[FEATURES].to_numpy(float), frame[FEATURES].to_numpy(float),
                               rtol=1e-12, atol=1e-10)
    metrics, deltas = [], []
    for index, target in enumerate(TARGETS):
        y = labels[target].to_numpy(float)
        valid = np.isfinite(y)
        if index:
            valid &= labels.wake_quality_pass.fillna(False).to_numpy(bool)
        if valid.sum() < 3:
            raise RuntimeError("Insufficient valid labels")
        for name in manifest["models"]:
            p = frame[f"{name}_prediction_{index}"].to_numpy(float)
            metrics.append(dict(model=name, target=target, count=int(valid.sum()),
                **study.prior.prior.previous.prior.direct.base.metric(y[valid], p[valid])))
            if index and name in ("local_geometry", "linear_leaf"):
                reference = frame["balanced1512_prediction_1"].to_numpy(float)
                deltas.append(dict(model=name, reference="balanced1512",
                    **evaluator.paired_bootstrap(y[valid], p[valid], reference[valid])))
        frame[target] = y
    frame["wake_quality_pass"] = labels.wake_quality_pass.to_numpy()
    basename = "新增训练区块诊断" if role == "train" else "独立测试"
    frame.to_csv(out / f"{basename}逐点.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / f"{basename}指标.csv", index=False)
    (out / f"{basename}.json").write_text(json.dumps(dict(
        role=role, independent_test=role == "test", attempted=len(frame),
        metrics=metrics, paired_wake_r2_differences=deltas,
        predictions_frozen_before_new_test=True, test_labels_used_for_selection=False,
        automatic_production_promotion=False, completed_at=time.strftime("%Y-%m-%d %H:%M:%S")),
        indent=2), encoding="utf8")
    return True


def run(out, wait=False, max_hours=20.):
    prepare(out)
    manifest = json.loads((out / "预测冻结.json").read_text(encoding="utf8"))
    started, completed = time.monotonic(), set()
    while True:
        evaluator.check_hashes(manifest["hashes"])
        for role in ("train", "test"):
            if role not in completed and evaluate(role, out, manifest):
                completed.add(role)
                print("Candidate cohort completed:", role, flush=True)
        (out / "进度.json").write_text(json.dumps(dict(
            stage="completed" if len(completed) == 2 else "waiting_native_labels",
            completed_blocks=sorted(completed), pid=os.getpid(),
            elapsed_s=time.monotonic()-started, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))),
            encoding="utf8")
        if len(completed) == 2 or not wait:
            return
        progress = json.loads((BATCH / "进度.json").read_text(encoding="utf8"))
        if progress["stage"] == "failed":
            raise RuntimeError("Native expansion failed: " + str(progress["error"]))
        if time.monotonic()-started > max_hours*3600:
            raise TimeoutError("Waiting for the sealed native cohort timed out")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--max-hours", type=float, default=20.)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.wait, args.max_hours)
