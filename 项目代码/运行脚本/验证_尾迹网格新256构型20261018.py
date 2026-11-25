"""Precompute input geometry, lock predictions, then solve a new independent cohort."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

import 研究_尾迹网格与均匀风险20261018 as study
from 均匀风险尾迹代理 import UniformRiskWakeAero
from 尾迹射线网格特征 import FEATURE_NAMES, extract
from 精细气动代理 import FineAeroSurrogate
from 局部线性树气动代理 import LocalLinearTreeAero

batch, utility = study.batch, study.utility
ROOT = study.ROOT
OUT = study.OUT / "新256构型独立验证"
WORKER = Path(__file__).with_name("运行_尾迹射线几何预处理20261018.py")
original = batch.previous.expansion.balanced.previous.prior.original
TARGETS, FEATURES = batch.TARGETS, batch.FEATURES


def plan(out):
    frame = pd.read_csv(study.DATA)
    x = frame[FEATURES].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    span = hi-lo
    avoid = batch.forbidden_geometries(frame)
    previous_jobs = batch.previous.read(batch.OUT / "配置冻结.json")["jobs"]
    avoid = np.vstack((avoid, np.array([[j["design"][k] for k in FEATURES] for j in previous_jobs])))
    fixed = {k: float(frame.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    jobs, rng = [], np.random.default_rng(2026101828)
    for _ in range(100000):
        row = rng.uniform(lo, hi)
        design = dict(fixed, **dict(zip(FEATURES, row)))
        if (max(original.prior.geometry_constraints(
                original.prior.AirframeDesign(**design)).values()) > 1e-9
                or np.linalg.norm((avoid-row)/span, axis=1).min() <= 1e-5):
            continue
        jobs.append(dict(sample=f"wake_ray_test_{len(jobs):04d}", design=design,
                         role="test", basis="fresh_uniform_feasible"))
        avoid = np.vstack((avoid, row))
        if len(jobs) == 256:
            break
    if len(jobs) != 256:
        raise RuntimeError("New independent geometry pool exhausted")
    paths = [Path(__file__), WORKER, study.DATA, Path(study.__file__),
             ROOT / "核心模型/尾迹射线网格特征.py",
             ROOT / "核心模型/均匀风险尾迹代理.py"]
    manifest = dict(jobs=jobs, hashes={str(p): utility.digest(p) for p in paths},
        target_definition_unchanged=True, quality_rules_unchanged=True,
        native_protocol="free-wake-legacy-v1", mesh="legacy", wake_nodes=8,
        wake_iterations=30, native_cores=8, fresh_uniform_test=True,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(out / "构型与计算冻结.json", manifest)
    return manifest


def prepare(out, manifest):
    request = dict(jobs=manifest["jobs"], output_root=str(out / "geometry"))
    log_folder = out / "geometry_worker"
    log_folder.mkdir(exist_ok=True)
    original.controlled.run_process([sys.executable, str(WORKER)], request, log_folder, 1800)


def lock(out, manifest):
    path = out / "预测冻结.json"
    if path.exists():
        result = batch.previous.read(path)
        utility.check_hashes(result["hashes"])
        return result
    if any((out / "solver" / j["sample"]).exists() for j in manifest["jobs"]):
        raise RuntimeError("Test solving started before prediction lock")
    selection_path = study.OUT / "模型选择.json"
    selection = batch.previous.read(selection_path)
    utility.check_hashes(selection["hashes"])
    frame = pd.DataFrame([dict(sample=j["sample"], **j["design"]) for j in manifest["jobs"]])
    x = frame[FEATURES].to_numpy(float)
    mesh, audit = [], []
    for job in manifest["jobs"]:
        mesh_path = out / "geometry" / job["sample"] / "airframe.vspgeom"
        try:
            values = extract(mesh_path, job["design"]["wing_area_m2"], job["design"]["aspect_ratio"])
            error = None
        except Exception as exc:
            values, error = np.zeros(len(FEATURE_NAMES)), repr(exc)
        mesh.append(values)
        audit.append(dict(sample=job["sample"], mesh_sha256=utility.digest(mesh_path),
                          feature_error=error))
    mesh = np.asarray(mesh)
    model_paths = {entry["name"]: entry["path"] for entry in selection["selected"]}
    model_paths.update(fine4096=str(study.BASELINE),
                       fine3072=str(batch.fine.OUT / "精细气动候选代理.json"),
                       local3840=str(utility.OUT / "local3840.joblib"))
    metadata = {}
    for name, model_path in model_paths.items():
        if name in ("uniform_risk", "all_risk"):
            model = UniformRiskWakeAero.load(model_path)
            p = model.predict_many(x, mesh)
            metadata[name] = dict(requires_mesh=model.requires_mesh)
        elif name == "local3840":
            p = LocalLinearTreeAero.load(model_path).predict_many(x)
            metadata[name] = dict(requires_mesh=False)
        else:
            p = FineAeroSurrogate.from_json(model_path).predict_many(x)
            metadata[name] = dict(requires_mesh=False)
        for i in range(2):
            frame[f"{name}_prediction_{i}"] = p[:, i]
    prediction_csv = out / "锁定预测.csv"
    frame.to_csv(prediction_csv, index=False)
    feature_path, audit_path = out / "测试几何特征.npz", out / "测试几何记录.csv"
    np.savez_compressed(feature_path, features=mesh)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    paths = [Path(__file__), WORKER, selection_path, prediction_csv, feature_path, audit_path]
    paths += list(map(Path, model_paths.values()))
    result = dict(models=metadata, hashes={str(p): utility.digest(p) for p in paths},
        locked_before_test_solver=True, test_aerodynamic_labels_seen=False,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(path, result)
    return result


def evaluate(out, records, frozen):
    labels = pd.DataFrame(records).set_index("sample")
    predictions = pd.read_csv(out / "锁定预测.csv").set_index("sample").loc[labels.index]
    metrics = []
    for i, target in enumerate(TARGETS):
        y = labels[target].to_numpy(float)
        valid = np.isfinite(y)
        if i:
            valid &= labels.wake_quality_pass.fillna(False).to_numpy(bool)
        for name, metadata in frozen["models"].items():
            p = predictions[f"{name}_prediction_{i}"].to_numpy(float)
            metrics.append(dict(model=name, target=target, count=int(valid.sum()), **metadata,
                **study.helpers.direct.base.metric(y[valid], p[valid])))
    pd.DataFrame(metrics).to_csv(out / "独立指标.csv", index=False)
    predictions.join(labels[[*TARGETS, "wake_quality_pass", "mesh_matches_prediction"]]).to_csv(
        out / "独立逐点.csv")
    utility.save(out / "独立评价.json", dict(metrics=metrics, attempted=len(records),
        locked_before_test_solver=True, target_definition_unchanged=True,
        quality_rules_unchanged=True, test_labels_used_for_training=False,
        geometry_mismatch_count=int((~labels.mesh_matches_prediction.fillna(False)).sum()),
        automatic_production_promotion=False))


def main(out, wait, prepare_only):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "构型与计算冻结.json"
    manifest = batch.previous.read(manifest_path) if manifest_path.exists() else plan(out)
    utility.check_hashes(manifest["hashes"])
    started, records = time.monotonic(), []
    def progress(stage, error=None):
        utility.save(out / "进度.json", dict(stage=stage, native_completed=len(records),
            native_requested=256, pid=os.getpid(), elapsed_s=time.monotonic()-started,
            error=error, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        progress("geometry_preprocessing")
        prepare(out, manifest)
        if prepare_only:
            progress("geometry_prepared")
            return
        while not (study.OUT / "模型选择.json").exists():
            progress("waiting_training")
            if (study.OUT / "失败.json").exists():
                raise RuntimeError("Training failed")
            if not wait:
                return
            if time.monotonic()-started > 12*3600:
                raise TimeoutError("Training wait timed out")
            time.sleep(30)
        utility.check_hashes(manifest["hashes"])
        frozen = lock(out, manifest)
        geometry_audit = pd.read_csv(out / "测试几何记录.csv").set_index("sample")
        original.controlled.original.WORKER = original.WORKER
        for job in manifest["jobs"]:
            utility.check_hashes(manifest["hashes"])
            utility.check_hashes(frozen["hashes"])
            folder = out / "solver" / job["sample"]
            folder.mkdir(parents=True, exist_ok=True)
            cache = folder / "record.json"
            if cache.exists():
                row = batch.previous.read(cache)
                batch.previous.validate_record(job, row)
            else:
                try:
                    row = original.controlled.run_case(original.prior.solver.DEFAULT_PYTHON,
                        job["design"], job["sample"], folder / "case", 600,
                        "test", job["basis"], (0., float("inf")), 30, "legacy")
                    row.update(original.verify(folder / "case"))
                    row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
                    row["wake_quality_pass"] = bool(row["quality_pass"])
                    row["mesh_matches_prediction"] = (
                        utility.digest(folder / "case/airframe.vspgeom")
                        == geometry_audit.loc[job["sample"], "mesh_sha256"])
                except Exception as error:
                    row = dict(sample=job["sample"], **job["design"],
                        solver_failure=repr(error), wake_quality_pass=False,
                        mesh_matches_prediction=False, **{k: None for k in TARGETS})
                row.update(role="test", selection_basis=job["basis"])
                utility.save(cache, row)
            records.append(row)
            pd.DataFrame(records).to_csv(out / "独立标签.csv", index=False)
            progress("native_test")
            print("New wake test", len(records), "/ 256", flush=True)
        evaluate(out, records, frozen)
        progress("completed")
    except Exception:
        progress("failed", traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.wait, args.prepare_only)
