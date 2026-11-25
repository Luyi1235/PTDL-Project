"""Register stable-model predictions before the shared new holdout is solved."""

from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

import 研究_稳定网格尾迹代理20261019 as study
import 推进_覆盖扩样6144代理20261019 as expansion
from 稳定网格尾迹代理 import StableMeshWakeAero
from 尾迹射线网格特征 import extract

utility, batch = study.utility, study.study.batch
OUT = study.OUT / "独立256构型"


def progress(stage, **details):
    utility.save(OUT / "进度.json", dict(stage=stage,
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **details))


def diagnostic(model):
    metrics = []
    for name, path in (
        ("diagnostic192", study.profile.OUT / "既有构型诊断/diagnostic192逐点.csv"),
        ("diagnostic256", study.profile.OUT / "既有构型诊断/diagnostic256逐点.csv"),
    ):
        frame = pd.read_csv(path)
        x = frame[study.profile.FEATURES].to_numpy(float)
        if name == "diagnostic256":
            mesh = np.load(expansion.native.OUT / "测试几何特征.npz")["features"]
            locked = pd.read_csv(expansion.native.OUT / "锁定预测.csv")
            np.testing.assert_array_equal(frame["sample"], locked["sample"])
        else:
            source = pd.read_csv(batch.OUT / "独立评价逐点.csv").set_index("sample")
            mesh = []
            for _, row in frame.iterrows():
                raw = source.loc[row["sample"]]
                try:
                    value = extract(Path(raw.geometry_file).with_suffix(".vspgeom"),
                                    raw.wing_area_m2, raw.aspect_ratio)
                except Exception:
                    if bool(row.wake_quality_pass):
                        raise
                    value = np.zeros(len(expansion.evaluation.FEATURE_NAMES))
                mesh.append(value)
            mesh = np.asarray(mesh)
        p = model.predict_many(x, mesh)
        for index in range(2):
            frame[f"stable_mesh_prediction_{index}"] = p[:, index]
        names = ("stable_mesh", "fine3072", "fine4096", "all_risk", "polar_structure")
        metrics.extend(expansion.evaluation.metric_rows(frame, names, name))
        frame.to_csv(study.OUT / (name+"逐点.csv"), index=False)
    pd.DataFrame(metrics).to_csv(study.OUT / "既有构型诊断指标.csv", index=False)
    utility.save(study.OUT / "既有构型诊断.json",
        dict(metrics=metrics, reused_sets=True, test_labels_used_for_selection=False))


def register(model, selection_path):
    path = OUT / "预测冻结.json"
    if path.exists():
        frozen = batch.previous.read(path)
        utility.check_hashes(frozen["hashes"])
        return frozen
    manifest_path = expansion.OUT / "配置冻结.json"
    manifest = batch.previous.read(manifest_path)
    utility.check_hashes(manifest["hashes"])
    jobs = [job for job in manifest["jobs"] if job["role"] == "test"]
    if any((expansion.OUT / "solver" / job["sample"]).exists() for job in jobs):
        raise RuntimeError("Holdout solving started before stable-model prediction registration")
    expansion.native.prepare(OUT, dict(jobs=jobs))
    frame = pd.DataFrame([dict(sample=job["sample"], **job["design"]) for job in jobs])
    mesh, audit = [], []
    for job in jobs:
        path = OUT / "geometry" / job["sample"] / "airframe.vspgeom"
        mesh.append(extract(path, job["design"]["wing_area_m2"], job["design"]["aspect_ratio"]))
        audit.append(dict(sample=job["sample"], mesh_sha256=utility.digest(path)))
    mesh = np.asarray(mesh)
    p = model.predict_many(frame[study.profile.FEATURES].to_numpy(float), mesh)
    for index in range(2):
        frame[f"stable_mesh_prediction_{index}"] = p[:, index]
    predictions, audits, features = (
        OUT / "锁定预测.csv", OUT / "测试几何记录.csv", OUT / "测试几何特征.npz")
    frame.to_csv(predictions, index=False)
    pd.DataFrame(audit).to_csv(audits, index=False)
    np.savez_compressed(features, features=mesh)
    selection = batch.previous.read(selection_path)
    paths = [Path(__file__), selection_path, Path(selection["path"]), manifest_path,
        predictions, audits, features, study.study.ROOT / "核心模型/稳定网格尾迹代理.py"]
    frozen = dict(hashes={str(p): utility.digest(p) for p in paths},
        locked_before_test_solver=True, test_aerodynamic_labels_seen=False,
        shared_with_6144_holdout=True, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(OUT / "预测冻结.json", frozen)
    return frozen


def evaluate(frozen):
    utility.check_hashes(frozen["hashes"])
    source = expansion.OUT / "独立256构型/独立逐点.csv"
    data = pd.read_csv(source).set_index("sample")
    prediction = pd.read_csv(OUT / "锁定预测.csv").set_index("sample").loc[data.index]
    np.testing.assert_allclose(data[study.profile.FEATURES],
                               prediction[study.profile.FEATURES], atol=1e-10, rtol=0)
    for index in range(2):
        data[f"stable_mesh_prediction_{index}"] = prediction[f"stable_mesh_prediction_{index}"]
    audit = pd.read_csv(OUT / "测试几何记录.csv").set_index("sample")
    valid = np.isfinite(data[study.profile.TARGET]) & data.wake_quality_pass.fillna(False)
    for name in data.index[valid]:
        mesh = expansion.OUT / "solver" / name / "case/airframe.vspgeom"
        if utility.digest(mesh) != audit.loc[name, "mesh_sha256"]:
            raise RuntimeError("Valid holdout mesh differs from registered model input")
    names = ("stable_mesh", "fine6144", "fine3072", "fine4096", "all_risk", "polar_structure")
    metrics = expansion.evaluation.metric_rows(data, names, "fresh256")
    pd.DataFrame(metrics).to_csv(OUT / "独立指标.csv", index=False)
    data.to_csv(OUT / "独立逐点.csv")
    comparisons = []
    y = data.loc[valid, study.profile.TARGET].to_numpy(float)
    for name in names[1:]:
        comparisons.append(dict(reference=name,
            **batch.previous.evaluator.paired_bootstrap(y,
                data.loc[valid, "stable_mesh_prediction_1"].to_numpy(float),
                data.loc[valid, f"{name}_prediction_1"].to_numpy(float))))
    utility.save(OUT / "独立评价.json", dict(metrics=metrics, comparisons=comparisons,
        locked_before_test_solver=True, test_labels_used_for_selection=False,
        valid_wake_geometry_mismatches=0, automatic_production_promotion=False))


def main():
    OUT.mkdir(exist_ok=True)
    selection_path = study.OUT / "模型选择.json"
    started = time.monotonic()
    while not selection_path.exists():
        progress("waiting_training")
        if (study.OUT / "失败.json").exists():
            raise RuntimeError("Stable-descriptor training failed")
        if time.monotonic()-started > 12*3600:
            raise TimeoutError("Stable-descriptor training exceeded twelve hours")
        time.sleep(30)
    selection = batch.previous.read(selection_path)
    utility.check_hashes(selection["hashes"])
    model = StableMeshWakeAero.load(selection["path"])
    progress("diagnostic")
    diagnostic(model)
    progress("prediction_registration")
    frozen = register(model, selection_path)
    while not (expansion.OUT / "独立256构型/独立评价.json").exists():
        utility.check_hashes(frozen["hashes"])
        progress("waiting_shared_native_holdout")
        stage = batch.previous.read(expansion.OUT / "进度.json")
        if stage["stage"] == "failed":
            raise RuntimeError("6144 expansion pipeline failed")
        if time.monotonic()-started > 48*3600:
            raise TimeoutError("Shared native holdout exceeded forty-eight hours")
        time.sleep(30)
    evaluate(frozen)
    progress("completed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        progress("failed", error=traceback.format_exc())
        raise
