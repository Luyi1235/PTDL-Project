"""Paired wake/grid diagnostics, then conditional higher-quality DOE expansion."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import pandas as pd

import 运行_多精度气动代理20261016 as driver
import 重算VSPAERO_Ew_DOE并训练代理 as solver
from 检查_尾迹标签质量 import polar_quality as basic_quality
from 检查_极曲线力一致性20261016 import polar_quality as force_diagnostic
from VSPAERO气动代理模型 import FEATURE_LABELS
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "运行结果" / "百构型加密代理_20261016"
OUT = ROOT / "运行结果" / "尾迹分辨率与扩样_20261016"
WORKER = ROOT / "运行脚本" / "运行_VSPAERO尾迹节点探针20261016.py"
LOW_DATA = FIRST / "一致性校核训练" / "low_labels" / "逐构型质量记录.csv"


def diagnostic_quality(path):
    result = force_diagnostic(path)
    # Near-field and wake lift need not agree closely. Keep their difference
    # as a diagnostic; acceptance uses finite/positive working-drag checks.
    result["quality_pass"] = basic_quality(path)["quality_pass"]
    result["wake_quality_pass"] = result["quality_pass"]
    result["force_discrepancy_is_acceptance_rule"] = False
    return result


def status(out, stage, **values):
    temp = out / "进度.tmp"
    temp.write_text(json.dumps(dict(pid=os.getpid(), stage=stage,
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **values),
        ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(out / "进度.json")


def prepare_jobs():
    previous = json.loads((FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    designs = {j["sample"]: j["design"] for j in previous["jobs"] if j["role"] == "train"}
    jobs = []
    # The baseline runs first so the effect of streamwise resolution becomes
    # available early. All probe geometries come from training, not tests.
    for sample, mesh, nodes, repeat in [
        ("medium_train_000", "medium", 16, 0),
        ("medium_train_000", "medium", 32, 0),
        ("medium_train_000", "medium", 64, 0),
        ("medium_train_000", "medium", 32, 1),
        *[(s, m, n, 0) for s in ("medium_train_000", "medium_train_033", "medium_train_066")
          for m, n in (("fine", 32), ("fine", 64), ("verification", 64))],
        *[(s, "medium", n, 0) for s in ("medium_train_033", "medium_train_066") for n in (32, 64)],
        ("medium_train_029", "medium", 32, 0),
        ("medium_train_029", "medium", 64, 0),
    ]:
        jobs.append(dict(sample=f"{sample}_{mesh}_wake{nodes}_repeat{repeat}",
                         source_sample=sample, design=designs[sample], mesh=mesh,
                         nodes=nodes, repeat=repeat))
    return jobs


def native_probe(job, out):
    folder = out / "solver" / job["sample"]
    saved = folder / "record.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf8"))
    if folder.exists():
        raise RuntimeError(f"Interrupted probe folder requires inspection: {folder}")
    os.environ["PTDL_DIAGNOSTIC_WAKE_NODES"] = str(job["nodes"])
    try:
        row = solver._run_case(solver.DEFAULT_PYTHON, job["design"], job["sample"], folder, 3600,
                               "training_geometry_diagnostic", "streamwise_wake_grid_pair",
                               (0., float("inf")), 30, job["mesh"])
        settings = (folder / "airframe.vspaero").read_text(encoding="utf8")
        applied = int(re.search(r"^NumWakeNodes\s*=\s*(\d+)", settings, re.M).group(1))
        log = (folder / "worker.log").read_text(encoding="utf8")
        observed = re.search(r"NumberOfWakeTrailingNodes_:\s*(\d+)", log)
        if applied != job["nodes"] or observed is None or int(observed.group(1)) != applied:
            raise RuntimeError("Native wake-node setting differs from requested value")
        row.update(diagnostic_quality(folder / "vspaero_polar.csv"),
                   applied_wake_nodes=applied, mesh_level=job["mesh"])
    except Exception:
        row = dict(sample=job["sample"], **job["design"], quality_pass=False,
                   solver_error=traceback.format_exc())
    row.update(source_sample=job["source_sample"], nodes=job["nodes"],
               mesh=job["mesh"], repeat=job["repeat"])
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
    return row


def convergence(rows):
    results = []
    for sample in ("medium_train_000", "medium_train_033", "medium_train_066"):
        def find(mesh, nodes):
            return next((r for r in rows if r["source_sample"] == sample
                         and r["mesh"] == mesh and r["nodes"] == nodes and r["repeat"] == 0), {})
        for label, a, b in (
            ("wake_32_to_64_fine", find("fine", 32), find("fine", 64)),
            ("fine_to_verification_wake64", find("fine", 64), find("verification", 64)),
        ):
            valid = bool(a.get("quality_pass") and b.get("quality_pass"))
            record = dict(sample=sample, comparison=label, valid=valid, passed=False)
            if valid:
                lift = abs(b[driver.TARGETS[0]]-a[driver.TARGETS[0]])/abs(b[driver.TARGETS[0]])
                wake = abs(b[driver.TARGETS[1]]-a[driver.TARGETS[1]])
                record.update(relative_lift_change=lift, absolute_wake_change=wake,
                              passed=bool(lift <= .01 and wake <= max(.01, .03*abs(b[driver.TARGETS[1]]))))
            results.append(record)
    return dict(passed=all(r["passed"] for r in results), pairs=results,
                criterion="Lift relative change <=1%; Ew absolute change <=max(0.01,3% of finer value)",
                scope="Three training geometries; provisional dataset protocol selection",
                fixed_efficency_definition=driver.TARGETS[1])


def expansion_plan(out, total, tests, probe_manifest):
    old = json.loads((FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    low = pd.read_csv(LOW_DATA)
    x = low[list(FEATURE_LABELS)].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    span = hi-lo
    jobs = [dict(sample=f"fine64_train_{i:03d}", role="train", mesh="fine", design=j["design"],
                 basis="same_training_geometry_new_solver_protocol")
            for i, j in enumerate(j for j in old["jobs"] if j["role"] == "train")]
    if total < len(jobs):
        raise ValueError("Expansion must retain the original training geometries")
    excluded = [x]
    for folder in (FIRST, ROOT / "运行结果" / "多精度气动代理_20261016"):
        plan = json.loads((folder / "冻结采样与计算配置.json").read_text(encoding="utf8"))
        excluded.append(np.array([[j["design"][k] for k in FEATURE_LABELS]
                                  for j in plan["jobs"] if j["role"] == "test"]))
    excluded.append(pd.read_csv(driver.LOW_DIR / "合并留出验证.csv")[list(FEATURE_LABELS)].to_numpy(float))
    excluded = np.vstack(excluded)
    fixed = {k: jobs[0]["design"][k] for k in ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    rng = np.random.default_rng(2026101691)
    pool = []
    while len(pool) < 10000:
        row = rng.uniform(lo, hi)
        if (np.linalg.norm((excluded-row)/span, axis=1).min() > 1e-5
                and all(v <= 1e-9 for v in geometry_constraints(AirframeDesign(
                    **fixed, **dict(zip(FEATURE_LABELS, row)))).values())):
            pool.append(row)
    pool = np.array(pool)
    selected = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in jobs])
    distances = np.linalg.norm((pool[:, None]-selected[None])/span, axis=2).min(1)
    while len(jobs) < total:
        index = int(distances.argmax())
        row = pool[index]
        jobs.append(dict(sample=f"fine64_train_{len(jobs):03d}", role="train", mesh="fine",
                         design={**fixed, **dict(zip(FEATURE_LABELS, row))}, basis="new_global_maximin"))
        distances = np.minimum(distances, np.linalg.norm((pool-row)/span, axis=1))
        distances[index] = -1.
    selected = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in jobs])
    excluded = np.vstack((excluded, selected))
    for i in range(tests):
        while True:
            row = rng.uniform(lo, hi)
            design = {**fixed, **dict(zip(FEATURE_LABELS, row))}
            if (np.linalg.norm((excluded-row)/span, axis=1).min() > 1e-5
                    and all(v <= 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values())):
                break
        jobs.append(dict(sample=f"fine64_test_{i:03d}", role="test", mesh="fine", design=design,
                         basis="fresh_sealed_uniform_test"))
        excluded = np.vstack((excluded, row))
    manifest = driver.sources()
    for path in (Path(__file__), WORKER, out / "冻结探针计划.json", out / "收敛检查.json",
                 Path(__file__).parent / "检查_极曲线力一致性20261016.py"):
        manifest[str(path.relative_to(ROOT))] = driver.digest(path)
    return dict(jobs=jobs, source_manifest=manifest, train_count=total, test_count=tests,
                fine_count=0, wake_iterations=30, wake_nodes=64, mesh_reference="fine",
                low_data_sha256=driver.digest(LOW_DATA), native_workers=1,
                old_labels_reused_as_high_fidelity=False, old_test_labels_used_for_training=False,
                protocol="All high-fidelity labels recomputed at fine mesh / 64 streamwise wake nodes",
                diagnostic_protocol_sha256=driver.digest(out / "冻结探针计划.json"))


def install_expansion_hooks():
    native_case = driver._run_case
    fit_models = driver.fit_models

    def verified_native(*args, **kwargs):
        row = native_case(*args, **kwargs)
        folder = Path(args[3] if len(args) > 3 else kwargs["case_dir"])
        settings = (folder / "airframe.vspaero").read_text(encoding="utf8")
        applied = int(re.search(r"^NumWakeNodes\s*=\s*(\d+)", settings, re.M).group(1))
        if applied != 64:
            raise RuntimeError("High-resolution training case did not use 64 wake nodes")
        row.update(applied_wake_nodes=64, mesh_level="fine")
        return row

    def fit_fine_models(frame, output):
        internal = frame.copy()
        baseline = internal["sample"] == "fine64_train_000"
        if baseline.sum() != 1:
            raise RuntimeError("Unique higher-quality baseline required")
        # The shared trainer uses this identifier to locate its baseline.
        # Raw records keep their actual fine-grid sample names.
        internal.loc[baseline, "sample"] = "medium_train_000"
        paths = fit_models(internal, output)
        for name in ("high_candidate", "direct_high"):
            path = paths[name]
            payload = json.loads(path.read_text(encoding="utf8"))
            payload.update(mesh_reference="fine", wake_nodes=64,
                           label_quality="Finite polar and positive working induced drag; lift discrepancy diagnostic only")
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
        return paths

    driver._run_case = verified_native
    driver.fit_models = fit_fine_models


def main(out, budget, tests):
    out.mkdir(parents=True, exist_ok=True)
    solver.WORKER = WORKER
    driver.LOW_DATA = LOW_DATA
    driver.polar_quality = diagnostic_quality
    path = out / "冻结探针计划.json"
    manifest = dict(jobs=prepare_jobs(), training_geometries_only=True, native_workers=1,
                    source_sha256=driver.digest(Path(__file__)), worker_sha256=driver.digest(WORKER),
                    target_definition="Original median VSPAERO Ew; unchanged",
                    conditional_training_budget=budget, conditional_test_budget=tests)
    if path.exists() and json.loads(path.read_text(encoding="utf8")) != manifest:
        raise RuntimeError("Probe protocol changed; use a new output directory")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    rows = []
    try:
        for job in manifest["jobs"]:
            status(out, "native_wake_grid_diagnostics", completed=len(rows), requested=len(manifest["jobs"]))
            rows.append(native_probe(job, out))
            pd.DataFrame(rows).to_csv(out / "尾迹网格配对结果.csv", index=False)
            print(job["sample"], rows[-1].get("quality_pass"), flush=True)
        assessment = convergence(rows)
        (out / "收敛检查.json").write_text(json.dumps(assessment, indent=2), encoding="utf8")
        if not assessment["passed"]:
            status(out, "diagnostics_complete_protocol_unresolved",
                   assessment=str(out / "收敛检查.json"), full_expansion_started=False)
            return
        if budget <= 0:
            status(out, "diagnostics_completed", assessment=str(out / "收敛检查.json"))
            return
        expanded = out / f"加密64节点{budget}构型"
        expanded.mkdir(parents=True, exist_ok=True)
        plan = expanded / "冻结采样与计算配置.json"
        if not plan.exists():
            plan.write_text(json.dumps(expansion_plan(out, budget, tests, manifest),
                                       ensure_ascii=False, indent=2), encoding="utf8")
        os.environ["PTDL_DIAGNOSTIC_WAKE_NODES"] = "64"
        install_expansion_hooks()
        status(out, "higher_quality_expansion_running", progress_file=str(expanded / "进度.json"))
        driver.run(expanded, (budget, tests, 0))
        evaluation = expanded / "中网格盲测评估.json"
        assessment = json.loads(evaluation.read_text(encoding="utf8"))
        assessment.update(mesh_reference="fine", wake_nodes=64,
                          force_discrepancy_is_acceptance_rule=False)
        evaluation.write_text(json.dumps(assessment, indent=2), encoding="utf8")
        (expanded / "高分辨率独立评价.json").write_bytes(evaluation.read_bytes())
        (expanded / "高分辨率训练标签.csv").write_bytes((expanded / "中网格训练实算标签.csv").read_bytes())
        status(out, "completed", assessment=str(expanded / "高分辨率独立评价.json"))
    except Exception:
        status(out, "failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--training-budget", type=int, default=300)
    parser.add_argument("--test-budget", type=int, default=48)
    args = parser.parse_args()
    main(args.output_dir, args.training_budget, args.test_budget)
