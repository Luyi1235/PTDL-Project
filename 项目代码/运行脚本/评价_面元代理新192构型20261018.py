"""Lock mesh-conditioned models before new test solves, then evaluate unchanged models."""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

import 研究_面元邻近代理20261018 as panel
import 推进_分歧补样4096代理20261018 as batch

ROOT = batch.ROOT
OUT = panel.OUT / "新192构型评价"


def prepare():
    OUT.mkdir(exist_ok=True)
    frozen = OUT / "模型冻结.json"
    if frozen.exists():
        manifest = batch.previous.read(frozen)
        batch.previous.evaluator.check_hashes(manifest["hashes"])
        return manifest
    selection_path = panel.OUT / "模型选择.json"
    if not selection_path.exists():
        return None
    manifest_path = batch.OUT / "配置冻结.json"
    jobs = [j for j in batch.previous.read(manifest_path)["jobs"] if j["role"] == "test"]
    if any((batch.OUT / "solver" / j["sample"]).exists() for j in jobs):
        raise RuntimeError("Cannot register the panel model after new test computation starts")
    selection = batch.previous.read(selection_path)
    paths = [Path(__file__), Path(panel.__file__), selection_path, manifest_path,
             ROOT / "核心模型/气动面元邻近特征.py"]
    paths += [Path(c["path"]) for entry in selection["selected"] for c in entry["components"]]
    manifest = dict(jobs=jobs, selection=selection,
        hashes={str(p): batch.previous.expansion.digest(p) for p in paths},
        models_frozen_before_test_computation=True,
        predictors_use_geometry_only=True, label_access_after_prediction_files_saved=True,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    batch.previous.save(frozen, manifest)
    return manifest


def predict_case(job, selection, output):
    frozen = output / f"{job['sample']}.json"
    if frozen.exists():
        return batch.previous.read(frozen)
    folder = batch.OUT / "solver" / job["sample"]
    if not (folder / "record.json").exists():
        return None
    geometry = dict(job["design"], sample=job["sample"],
                    geometry_file=str(folder / "case/airframe.vsp3"))
    descriptors = panel.extract(geometry)
    x = np.array([[job["design"][k] for k in batch.FEATURES]])
    mesh = np.array([[descriptors[k] for k in panel.FEATURE_NAMES]])
    p = [sum(c["weight"]*panel.predict(joblib.load(c["path"]), x, mesh)[0]
             for c in entry["components"]) for entry in selection["selected"]]
    record = dict(sample=job["sample"], **job["design"],
        panel_prediction_0=float(p[0]), panel_prediction_1=float(p[1]),
        geometry_available=bool(descriptors["mesh_available"]),
        geometric_preprocessing_error=descriptors["extraction_error"])
    batch.previous.save(frozen, record)
    return record


def evaluate(manifest, predictions):
    frame = pd.DataFrame(predictions).set_index("sample")
    rows = []
    for job in manifest["jobs"]:
        row = batch.previous.read(batch.OUT / "solver" / job["sample"] / "record.json")
        batch.previous.validate_record(job, row)
        rows.append(row)
    labels = pd.DataFrame(rows).set_index("sample").loc[frame.index]
    reference = pd.read_csv(batch.OUT / "独立锁定预测.csv").set_index("sample").loc[frame.index]
    metrics, comparisons = [], []
    for index, target in enumerate(batch.TARGETS):
        y = labels[target].to_numpy(float)
        valid = np.isfinite(y)
        if index:
            valid &= labels.wake_quality_pass.fillna(False).to_numpy(bool)
        p = frame[f"panel_prediction_{index}"].to_numpy(float)
        metrics.append(dict(target=target, count=int(valid.sum()),
            **batch.previous.expansion.balanced.previous.prior.direct.base.metric(y[valid], p[valid])))
        for model in ("fine4096", "same_data_baseline", "original3072"):
            comparisons.append(dict(target=target, reference=model,
                **batch.previous.evaluator.paired_bootstrap(
                    y[valid], p[valid], reference[f"{model}_prediction_{index}"].to_numpy(float)[valid])))
    batch.previous.save(OUT / "独立评价.json", dict(metrics=metrics, attempted=len(rows),
        trained_on_3072=True, compared_with_4096_models=True, comparisons=comparisons,
        quality_rules_unchanged=True, models_frozen_before_test_computation=True))
    frame.join(labels[[*batch.TARGETS, "wake_quality_pass"]]).to_csv(OUT / "独立评价逐点.csv")


def main(wait):
    OUT.mkdir(exist_ok=True)
    predictions_dir = OUT / "逐点预测"
    predictions_dir.mkdir(exist_ok=True)
    start = time.monotonic()
    while True:
        manifest = prepare()
        predictions = []
        if manifest is not None:
            batch.previous.evaluator.check_hashes(manifest["hashes"])
            for job in manifest["jobs"]:
                row = predict_case(job, manifest["selection"], predictions_dir)
                if row is not None:
                    predictions.append(row)
        complete = len(predictions) == 192
        if complete:
            evaluate(manifest, predictions)
        batch.previous.save(OUT / "进度.json", dict(
            stage="completed" if complete else
                  "waiting_test_geometry" if manifest is not None else "waiting_training",
            completed_predictions=len(predictions), requested=192,
            elapsed_s=time.monotonic()-start, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        if complete or not wait:
            return
        if time.monotonic()-start > 24*3600:
            raise TimeoutError("Panel independent evaluation timed out")
        progress_path = batch.OUT / "进度.json"
        if progress_path.exists():
            state = batch.previous.read(progress_path)
            if state["stage"] == "failed":
                raise RuntimeError("Adaptive native batch failed: " + str(state["error"]))
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    main(args.wait)
