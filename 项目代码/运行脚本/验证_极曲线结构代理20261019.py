"""Reused-cohort diagnostics followed by prediction-locked fresh native validation."""

import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

import 研究_极曲线结构代理20261019 as study
import 验证_尾迹网格新256构型20261018 as native
from 极曲线结构代理 import PolarStructureAero
from 均匀风险尾迹代理 import UniformRiskWakeAero
from 尾迹射线网格特征 import FEATURE_NAMES, extract
from 精细气动代理 import FineAeroSurrogate

OUT = study.OUT / "新384构型独立验证"
COUNT = 384
utility, batch, original = native.utility, native.batch, native.original


def progress(out, stage, **details):
    utility.save(out / "进度.json", dict(stage=stage, pid=os.getpid(),
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **details))


def plan(out):
    frame = pd.read_csv(study.DATA)
    values = frame[study.FEATURES].to_numpy(float)
    lo, hi = values.min(0), values.max(0)
    span = hi-lo
    avoid = batch.forbidden_geometries(frame)
    old_manifests = [
        batch.OUT / "配置冻结.json",
        native.OUT / "构型与计算冻结.json",
    ]
    for path in old_manifests:
        jobs = batch.previous.read(path)["jobs"]
        avoid = np.vstack((avoid, [[j["design"][k] for k in study.FEATURES] for j in jobs]))
    fixed = {k: float(frame.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    jobs, rng = [], np.random.default_rng(2026101903)
    for _ in range(100000):
        row = rng.uniform(lo, hi)
        design = dict(fixed, **dict(zip(study.FEATURES, row)))
        if (max(original.prior.geometry_constraints(
                original.prior.AirframeDesign(**design)).values()) > 1e-9
                or np.linalg.norm((avoid-row)/span, axis=1).min() <= 1e-5):
            continue
        jobs.append(dict(sample=f"polar_structure_test_{len(jobs):04d}",
            design=design, role="test", basis="fresh_uniform_feasible"))
        avoid = np.vstack((avoid, row))
        if len(jobs) == COUNT:
            break
    if len(jobs) != COUNT:
        raise RuntimeError("Independent geometry pool exhausted")
    paths = [
        Path(__file__), Path(native.__file__), native.WORKER, study.DATA,
        Path(study.__file__), study.ROOT / "核心模型/极曲线结构代理.py",
        study.ROOT / "核心模型/尾迹射线网格特征.py", *old_manifests,
    ]
    manifest = dict(jobs=jobs, hashes={str(p): utility.digest(p) for p in paths},
        target_definition_unchanged=True, quality_rules_unchanged=True,
        previously_inspected_192_and_256_excluded=True,
        native_protocol="free-wake-legacy-v1", mesh="legacy", wake_nodes=8,
        wake_iterations=30, native_cores=8, fresh_uniform_test=True,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(out / "构型与计算冻结.json", manifest)
    return manifest


def predictors():
    selection_path = study.OUT / "模型选择.json"
    selection = batch.previous.read(selection_path)
    utility.check_hashes(selection["hashes"])
    old_selection = batch.previous.read(study.previous.OUT / "模型选择.json")
    utility.check_hashes(old_selection["hashes"])
    paths = dict(
        polar_structure=Path(selection["path"]),
        fine3072=batch.fine.OUT / "精细气动候选代理.json",
        fine4096=study.previous.BASELINE,
        all_risk=study.previous.OUT / "all_risk.joblib",
    )
    models = dict(
        polar_structure=PolarStructureAero.load(paths["polar_structure"]),
        fine3072=FineAeroSurrogate.from_json(paths["fine3072"]),
        fine4096=FineAeroSurrogate.from_json(paths["fine4096"]),
        all_risk=UniformRiskWakeAero.load(paths["all_risk"]),
    )
    return models, paths


def metric_rows(frame, names, cohort):
    rows = []
    for index, target in enumerate(batch.TARGETS):
        y = frame[target].to_numpy(float)
        valid = np.isfinite(y)
        if index:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        for name in names:
            p = frame[f"{name}_prediction_{index}"].to_numpy(float)
            rows.append(dict(cohort=cohort, model=name, target=target,
                count=int(valid.sum()),
                **study.previous.helpers.direct.base.metric(y[valid], p[valid])))
    return rows


def diagnostic():
    out = study.OUT / "既有构型诊断"
    out.mkdir(exist_ok=True)
    models, model_paths = predictors()
    sources = dict(
        diagnostic192=study.previous.OUT / "192构型诊断复评/诊断逐点.csv",
        diagnostic256=native.OUT / "独立逐点.csv",
    )
    metrics, comparisons = [], []
    for name, source in sources.items():
        frame = pd.read_csv(source)
        x = frame[study.FEATURES].to_numpy(float)
        if name == "diagnostic256":
            locked = pd.read_csv(native.OUT / "锁定预测.csv")
            np.testing.assert_array_equal(frame["sample"], locked["sample"])
            np.testing.assert_allclose(x, locked[study.FEATURES].to_numpy(float),
                                      atol=1e-10, rtol=0)
            mesh = np.load(native.OUT / "测试几何特征.npz")["features"]
        else:
            mesh = []
            for _, row in frame.iterrows():
                try:
                    value = extract(Path(row.geometry_file).with_suffix(".vspgeom"),
                                    row.wing_area_m2, row.aspect_ratio)
                except Exception:
                    if bool(row.wake_quality_pass):
                        raise
                    value = np.zeros(len(FEATURE_NAMES))
                mesh.append(value)
            mesh = np.asarray(mesh)
        for key, model in models.items():
            p = (model.predict_many(x, mesh) if key in ("polar_structure", "all_risk")
                 else model.predict_many(x))
            for index in range(2):
                frame[f"{key}_prediction_{index}"] = p[:, index]
        metrics.extend(metric_rows(frame, models, name))
        y = frame[study.TARGET].to_numpy(float)
        valid = np.isfinite(y) & frame.wake_quality_pass.fillna(False).to_numpy(bool)
        for reference in ("fine3072", "fine4096", "all_risk"):
            comparisons.append(dict(cohort=name, reference=reference,
                **batch.previous.evaluator.paired_bootstrap(y[valid],
                    frame.polar_structure_prediction_1.to_numpy(float)[valid],
                    frame[f"{reference}_prediction_1"].to_numpy(float)[valid])))
        frame.to_csv(out / (name+"逐点.csv"), index=False)
    pd.DataFrame(metrics).to_csv(out / "诊断指标.csv", index=False)
    paths = [Path(__file__), study.OUT / "模型选择.json",
             *sources.values(), *model_paths.values()]
    utility.save(out / "诊断评价.json", dict(metrics=metrics, comparisons=comparisons,
        reused_test_sets=True, test_labels_used_for_selection=False,
        hashes={str(p): utility.digest(p) for p in paths}))
    return metrics


def lock(out, manifest):
    path = out / "预测冻结.json"
    if path.exists():
        frozen = batch.previous.read(path)
        utility.check_hashes(frozen["hashes"])
        return frozen
    if any((out / "solver" / job["sample"]).exists() for job in manifest["jobs"]):
        raise RuntimeError("Native solving started before prediction lock")
    models, model_paths = predictors()
    frame = pd.DataFrame([dict(sample=job["sample"], **job["design"])
                          for job in manifest["jobs"]])
    x = frame[study.FEATURES].to_numpy(float)
    mesh, audit = [], []
    for job in manifest["jobs"]:
        mesh_path = out / "geometry" / job["sample"] / "airframe.vspgeom"
        value = extract(mesh_path, job["design"]["wing_area_m2"], job["design"]["aspect_ratio"])
        mesh.append(value)
        audit.append(dict(sample=job["sample"], mesh_sha256=utility.digest(mesh_path)))
    mesh = np.asarray(mesh)
    metadata = {}
    for name, model in models.items():
        p = (model.predict_many(x, mesh) if name in ("polar_structure", "all_risk")
             else model.predict_many(x))
        metadata[name] = dict(requires_mesh=getattr(model, "requires_mesh", False))
        for index in range(2):
            frame[f"{name}_prediction_{index}"] = p[:, index]
    prediction_path = out / "锁定预测.csv"
    feature_path, audit_path = out / "测试几何特征.npz", out / "测试几何记录.csv"
    frame.to_csv(prediction_path, index=False)
    np.savez_compressed(feature_path, features=mesh)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    paths = [
        Path(__file__), Path(native.__file__), native.WORKER,
        out / "构型与计算冻结.json", study.OUT / "模型选择.json",
        prediction_path, feature_path, audit_path, *model_paths.values(),
        study.ROOT / "核心模型/极曲线结构代理.py",
        study.ROOT / "核心模型/均匀风险尾迹代理.py",
        study.ROOT / "核心模型/精细气动代理.py",
    ]
    frozen = dict(models=metadata, hashes={str(p): utility.digest(p) for p in paths},
        locked_before_test_solver=True, test_aerodynamic_labels_seen=False,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    utility.save(path, frozen)
    return frozen


def summarize(out):
    frame = pd.read_csv(out / "独立逐点.csv")
    report = batch.previous.read(out / "独立评价.json")
    y = frame[study.TARGET].to_numpy(float)
    valid = np.isfinite(y) & frame.wake_quality_pass.fillna(False).to_numpy(bool)
    comparisons = []
    for reference in ("fine3072", "fine4096", "all_risk"):
        comparisons.append(dict(reference=reference,
            **batch.previous.evaluator.paired_bootstrap(y[valid],
                frame.polar_structure_prediction_1.to_numpy(float)[valid],
                frame[f"{reference}_prediction_1"].to_numpy(float)[valid])))
    report["comparisons"] = comparisons
    report["valid_wake_geometry_mismatch_count"] = int(
        (~frame.mesh_matches_prediction.fillna(False).to_numpy(bool)[valid]).sum())
    utility.save(out / "独立评价.json", report)
    lines = [
        "# 极曲线结构代理独立验证", "",
        "全部模型预测在本组气动力求解前固定。保留原始尾迹效率定义和质量规则。",
        "此前192和256构型仅用于另行诊断，未混入本组结果。", "",
        "| 模型 | 预测量 | 构型数 | R² | MAE | RMSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["metrics"]:
        target = "升力斜率" if row["target"] == batch.TARGETS[0] else "尾迹效率"
        lines.append(f"| {row['model']} | {target} | {row['count']} | "
                     f"{row['r2']:.4f} | {row['mae']:.5f} | {row['rmse']:.5f} |")
    lines.extend(["", f"有效尾迹构型的几何不一致数为{report['valid_wake_geometry_mismatch_count']}。"])
    (out / "独立验证结论.md").write_text("\n".join(lines)+"\n", encoding="utf8")


def main(out, prepare_only):
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "构型与计算冻结.json"
    manifest = batch.previous.read(manifest_path) if manifest_path.exists() else plan(out)
    utility.check_hashes(manifest["hashes"])
    progress(out, "geometry_preprocessing")
    native.prepare(out, manifest)
    if prepare_only:
        progress(out, "geometry_prepared")
        return
    started = time.monotonic()
    while not (study.OUT / "模型选择.json").exists():
        progress(out, "waiting_training")
        if (study.OUT / "失败.json").exists():
            raise RuntimeError("Force-profile training failed")
        if time.monotonic()-started > 12*3600:
            raise TimeoutError("Training wait exceeded twelve hours")
        time.sleep(30)
    utility.check_hashes(manifest["hashes"])
    progress(out, "diagnostic")
    diagnostic()
    frozen = lock(out, manifest)
    audit = pd.read_csv(out / "测试几何记录.csv").set_index("sample")
    original.controlled.original.WORKER = original.WORKER
    records = []
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
                    == audit.loc[job["sample"], "mesh_sha256"])
            except Exception as error:
                row = dict(sample=job["sample"], **job["design"], solver_failure=repr(error),
                    wake_quality_pass=False, mesh_matches_prediction=False,
                    **{key: None for key in batch.TARGETS})
            row.update(role="test", selection_basis=job["basis"])
            utility.save(cache, row)
        records.append(row)
        pd.DataFrame(records).to_csv(out / "独立标签.csv", index=False)
        progress(out, "native_test", native_completed=len(records), native_requested=COUNT)
        print("Profile native test", len(records), "/", COUNT, flush=True)
    native.evaluate(out, records, frozen)
    summarize(out)
    progress(out, "completed", native_completed=len(records), native_requested=COUNT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        main(args.output_dir.resolve(), args.prepare_only)
    except Exception:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        progress(args.output_dir, "failed", error=traceback.format_exc())
        raise
