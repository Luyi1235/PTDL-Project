"""Resolve remaining discretization sensitivity before balanced DOE and nested selection."""

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

import 研究_代理强正则化20261016 as strong
import 运行_多精度气动代理20261016 as driver
import 重算VSPAERO_Ew_DOE并训练代理 as solver
from 推进_尾迹分辨率与扩样20261016 import diagnostic_quality
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints
from VSPAERO气动代理模型 import FEATURE_LABELS

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "运行结果" / "尾迹分辨率与扩样_20261016"
FIRST = ROOT / "运行结果" / "百构型加密代理_20261016"
OUT = ROOT / "运行结果" / "稳定标签与均衡采样_20261016"
WORKER = ROOT / "运行脚本" / "运行_VSPAERO自适应分辨率20261016.py"
LOW_DATA = FIRST / "一致性校核训练" / "low_labels" / "逐构型质量记录.csv"
ANCHORS = ("medium_train_000", "medium_train_033", "medium_train_066")


def status(out, stage, **values):
    temp = out / "进度.tmp"
    temp.write_text(json.dumps(dict(pid=os.getpid(), stage=stage,
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **values),
        ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(out / "进度.json")


def compare(a, b):
    if not a.get("quality_pass") or not b.get("quality_pass"):
        return dict(passed=False, valid=False)
    lift = abs(b[driver.TARGETS[0]]-a[driver.TARGETS[0]])/abs(b[driver.TARGETS[0]])
    wake = abs(b[driver.TARGETS[1]]-a[driver.TARGETS[1]])
    return dict(valid=True, relative_lift_change=lift, absolute_wake_change=wake,
                passed=bool(lift <= .01 and wake <= max(.01, .03*abs(b[driver.TARGETS[1]]))))


def verify_protocol(folder, mesh, nodes):
    settings = (folder / "airframe.vspaero").read_text(encoding="utf8")
    observed = re.search(r"^NumWakeNodes\s*=\s*(\d+)", settings, re.M)
    receipt = json.loads((folder / "自适应求解设置.json").read_text(encoding="utf8"))
    log = (folder / "worker.log").read_text(encoding="utf8")
    trailing = re.search(r"NumberOfWakeTrailingNodes_:\s*(\d+)", log)
    threads = re.search(r"\s-omp\s+(\d+)\s", log)
    if (observed is None or int(observed.group(1)) != nodes or trailing is None
            or int(trailing.group(1)) != nodes or threads is None or int(threads.group(1)) != 8
            or receipt["wake_nodes"] != nodes or receipt["mesh_level"] != mesh):
        raise RuntimeError("Adaptive native protocol changed or was not applied")
    return dict(applied_wake_nodes=nodes, mesh_level=mesh, native_cores_requested=8,
                native_command_cores=8)


def solve(design, sample, mesh, nodes, out):
    name = f"{sample}_{mesh}_wake{nodes}_cores8"
    folder = out / "solver" / name
    saved = folder / "record.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf8"))
    if folder.exists():
        raise RuntimeError(f"Inspect incomplete adaptive solver folder: {folder}")
    os.environ["PTDL_ADAPTIVE_WAKE_NODES"] = str(nodes)
    os.environ["PTDL_ADAPTIVE_NATIVE_CORES"] = "8"
    try:
        row = solver._run_case(solver.DEFAULT_PYTHON, design, name, folder, 7200,
                               "training_only_resolution_diagnostic", "adaptive_wake_and_grid",
                               (0., float("inf")), 30, mesh)
        row.update(diagnostic_quality(folder / "vspaero_polar.csv"),
                   **verify_protocol(folder, mesh, nodes))
    except Exception:
        row = dict(sample=name, **design, quality_pass=False, error=traceback.format_exc(),
                   mesh_level=mesh, applied_wake_nodes=nodes)
    row["source_sample"] = sample
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
    return row


def balanced_plan(out, total, tests, mesh, nodes):
    old = json.loads((FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    jobs = [dict(sample=f"stable_train_{i:03d}", role="train", mesh=mesh, design=j["design"],
                 basis="same_training_geometry_new_protocol")
            for i, j in enumerate(j for j in old["jobs"] if j["role"] == "train")]
    if total <= len(jobs):
        raise ValueError("Balanced expansion requires additional training geometries")
    low = pd.read_csv(LOW_DATA)
    x = low[list(FEATURE_LABELS)].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    span = hi-lo
    excluded = [x]
    for source in (FIRST, ROOT / "运行结果" / "多精度气动代理_20261016"):
        plan = json.loads((source / "冻结采样与计算配置.json").read_text(encoding="utf8"))
        excluded.append(np.array([[j["design"][k] for k in FEATURE_LABELS]
                                  for j in plan["jobs"] if j["role"] == "test"]))
    excluded.append(pd.read_csv(driver.LOW_DIR / "合并留出验证.csv")[list(FEATURE_LABELS)].to_numpy(float))
    excluded = np.vstack(excluded)
    selected = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in jobs])
    fixed = {k: jobs[0]["design"][k] for k in ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    rng = np.random.default_rng(2026101697)

    def draw(avoid):
        while True:
            row = rng.uniform(lo, hi)
            design = {**fixed, **dict(zip(FEATURE_LABELS, row))}
            if (np.linalg.norm((avoid-row)/span, axis=1).min() > 1e-5
                    and all(v <= 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values())):
                return row, design

    iid = min(total-len(jobs), int(round(total*.5)))
    for _ in range(iid):
        row, design = draw(np.vstack((excluded, selected)))
        jobs.append(dict(sample=f"stable_train_{len(jobs):03d}", role="train", mesh=mesh,
                         design=design, basis="uniform_feasible_training"))
        selected = np.vstack((selected, row))
    if len(jobs) < total:
        pool = np.array([draw(np.vstack((excluded, selected)))[0] for _ in range(6000)])
        distances = np.linalg.norm((pool[:, None]-selected[None])/span, axis=2).min(1)
        while len(jobs) < total:
            index = int(distances.argmax())
            row = pool[index]
            jobs.append(dict(sample=f"stable_train_{len(jobs):03d}", role="train", mesh=mesh,
                             design={**fixed, **dict(zip(FEATURE_LABELS, row))}, basis="global_maximin_training"))
            distances = np.minimum(distances, np.linalg.norm((pool-row)/span, axis=1))
            distances[index] = -1.
            selected = np.vstack((selected, row))
    excluded = np.vstack((excluded, selected))
    for i in range(tests):
        row, design = draw(excluded)
        jobs.append(dict(sample=f"stable_test_{i:03d}", role="test", mesh=mesh, design=design,
                         basis="fresh_sealed_uniform_test"))
        excluded = np.vstack((excluded, row))
    manifest = driver.sources()
    for path in (Path(__file__), WORKER, out / "离散精度选择.json",
                 Path(__file__).parent / "研究_代理正则化与嵌套验证20261016.py",
                 Path(__file__).parent / "研究_代理强正则化20261016.py",
                 Path(__file__).parent / "推进_尾迹分辨率与扩样20261016.py",
                 Path(__file__).parent / "检查_极曲线力一致性20261016.py"):
        manifest[str(path.relative_to(ROOT))] = driver.digest(path)
    return dict(jobs=jobs, source_manifest=manifest, train_count=total, test_count=tests, fine_count=0,
                wake_nodes=nodes, wake_iterations=30, mesh_reference=mesh, native_workers=1,
                native_cores_requested=8, low_data_sha256=driver.digest(LOW_DATA),
                new_uniform_training_count=iid,
                old_test_labels_used_for_training=False, old_high_labels_reused=False,
                protocol="Recompute anchors; balanced uniform/maximin DOE; nested random/spatial regularized selection")


def install_training(mesh, nodes):
    native = driver._run_case
    original_fit = driver.fit_models

    def verified_native(*args, **kwargs):
        positional = list(args)
        if len(positional) > 4:
            positional[4] = 7200
        else:
            kwargs["timeout_s"] = 7200
        row = native(*positional, **kwargs)
        folder = Path(args[3])
        row.update(verify_protocol(folder, mesh, nodes))
        return row

    def selected_models(frame, output):
        internal = frame.copy()
        baseline = internal["sample"] == "stable_train_000"
        if baseline.sum() != 1:
            raise RuntimeError("Stable dataset has no unique baseline")
        internal.loc[baseline, "sample"] = "medium_train_000"
        paths = original_fit(internal, output)
        for name in ("high_candidate", "direct_high"):
            path = paths[name]
            payload = json.loads(path.read_text(encoding="utf8"))
            payload.update(mesh_reference=mesh, wake_nodes=nodes)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
        (output / "条件高斯候选原始.json").write_bytes(paths["high_candidate"].read_bytes())
        dataset = output / "正则化训练数据.csv"
        frame.to_csv(dataset, index=False)
        regularized = strong.regularized
        regularized.DATASETS = {"stable": dataset}
        regularized.REFERENCE = {"stable": paths["high_candidate"]}
        selection = output / "嵌套正则化"
        regularized.train(selection, 12)
        candidate = json.loads((selection / "stable_正则化候选.json").read_text(encoding="utf8"))
        candidate.update(low_fidelity_used_by_target=[False, False],
                         low_data_role="Baseline comparison only; excluded from selected regularized GP",
                         mesh_reference=mesh, wake_nodes=nodes)
        paths["high_candidate"].write_text(json.dumps(candidate, indent=2), encoding="utf8")
        old_choice = json.loads((output / "训练模型选择.json").read_text(encoding="utf8"))
        (output / "训练模型选择.json").write_text(json.dumps(dict(
            candidate="Nested random/spatial regularized selection, one-SE rule",
            reference_candidates=old_choice, aerodynamic_test_labels_used=False,
            regularized_selection=str(selection / "stable_模型选择.json")), indent=2), encoding="utf8")
        return paths

    driver._run_case, driver.fit_models = verified_native, selected_models


def main(out, budget, tests):
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        previous = json.loads((PREVIOUS / "进度.json").read_text(encoding="utf8"))
        if previous["stage"] == "failed":
            raise RuntimeError("Previous controller failed; inspect its error before continuation")
        if previous["stage"] in ("diagnostics_complete_protocol_unresolved", "diagnostics_completed", "completed"):
            break
        status(out, "waiting_existing_diagnostics", previous=previous)
        if time.monotonic()-started > 48*3600:
            raise TimeoutError("Existing diagnostics wait exceeded 48 hours")
        time.sleep(20)
    if previous["stage"] == "completed":
        status(out, "previous_expansion_completed", previous=previous)
        return
    solver.WORKER = WORKER
    driver.LOW_DATA = LOW_DATA
    driver.polar_quality = diagnostic_quality
    source_manifest = {str(p): driver.digest(p) for p in (Path(__file__), WORKER)}
    lock = out / "自适应计算锁定.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf8")) != source_manifest:
        raise RuntimeError("Adaptive protocol changed; use a new output directory")
    lock.write_text(json.dumps(source_manifest, indent=2), encoding="utf8")
    old_plan = json.loads((FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    designs = {j["sample"]: j["design"] for j in old_plan["jobs"] if j["role"] == "train"}
    rows, checks, chosen = [], [], None

    def case(sample, mesh, nodes):
        status(out, "adaptive_resolution_diagnostics", completed=len(rows),
               running_sample=sample, running_mesh=mesh, running_nodes=nodes)
        record = solve(designs[sample], sample, mesh, nodes, out)
        if not any(r["sample"] == record["sample"] for r in rows):
            rows.append(record)
        pd.DataFrame(rows).to_csv(out / "自适应配对结果.csv", index=False)
        status(out, "adaptive_resolution_diagnostics", completed=len(rows), last=record["sample"])
        return record

    for nodes in (128, 256):
        for mesh, finer in (("fine", "verification"), ("verification", "ultra")):
            phase = []
            for sample in ANCHORS:
                lower = case(sample, mesh, nodes//2)
                current = case(sample, mesh, nodes)
                refined = case(sample, finer, nodes)
                phase.extend([dict(sample=sample, mesh=mesh, nodes=nodes, comparison="wake",
                                   **compare(lower, current)),
                              dict(sample=sample, mesh=mesh, nodes=nodes, comparison="surface_grid",
                                   **compare(current, refined))])
            checks.extend(phase)
            (out / "自适应精度检查.json").write_text(json.dumps(checks, indent=2), encoding="utf8")
            if all(c["passed"] for c in phase):
                chosen = dict(mesh=mesh, nodes=nodes, checks=phase,
                              scope="Three training anchors; provisional sampling protocol",
                              target_definition="Original median VSPAERO Ew, unchanged")
                break
        if chosen:
            break
    if chosen is None:
        status(out, "resolution_unsettled", checks=str(out / "自适应精度检查.json"))
        return
    (out / "离散精度选择.json").write_text(json.dumps(chosen, indent=2), encoding="utf8")
    expanded = out / f"均衡{budget}构型_{chosen['mesh']}_wake{chosen['nodes']}"
    expanded.mkdir(parents=True, exist_ok=True)
    plan = expanded / "冻结采样与计算配置.json"
    if not plan.exists():
        plan.write_text(json.dumps(balanced_plan(out, budget, tests, chosen["mesh"], chosen["nodes"]),
                                   ensure_ascii=False, indent=2), encoding="utf8")
    os.environ["PTDL_ADAPTIVE_WAKE_NODES"] = str(chosen["nodes"])
    os.environ["PTDL_ADAPTIVE_NATIVE_CORES"] = "8"
    install_training(chosen["mesh"], chosen["nodes"])
    status(out, "balanced_training_running", progress_file=str(expanded / "进度.json"))
    driver.run(expanded, (budget, tests, 0))
    path = expanded / "中网格盲测评估.json"
    report = json.loads(path.read_text(encoding="utf8"))
    report.update(mesh_reference=chosen["mesh"], wake_nodes=chosen["nodes"],
                  selection="Nested random/spatial regularized model; unchanged test targets")
    path.write_text(json.dumps(report, indent=2), encoding="utf8")
    (expanded / "稳定标签独立评价.json").write_bytes(path.read_bytes())
    status(out, "completed", assessment=str(expanded / "稳定标签独立评价.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--training-budget", type=int, default=300)
    parser.add_argument("--test-budget", type=int, default=48)
    args = parser.parse_args()
    try:
        main(args.output_dir, args.training_budget, args.test_budget)
    except Exception:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        status(args.output_dir, "failed", error=traceback.format_exc())
        raise
