"""Audit fixed-wake discretization, then train and independently test a separate surrogate."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import re
import traceback

import pandas as pd

import 续跑_稳定标签与均衡采样20261016 as prior
import VSPAERO受控子进程20261017 as controlled

ROOT = prior.ROOT
WORKER = Path(__file__).with_name("运行_VSPAERO固定尾迹20261017.py")
OUT = ROOT / "运行结果" / "固定尾迹均衡代理_20261017"
_original_verify = prior.verify_protocol


def fixed_receipt(folder, mesh, nodes):
    result = _original_verify(folder, mesh, nodes)
    receipt = json.loads((folder / "自适应求解设置.json").read_text(encoding="utf8"))
    settings = (folder / "airframe.vspaero").read_text(encoding="utf8")
    wake_iters = re.search(r"^WakeIters\s*=\s*(\d+)", settings, re.M)
    far = re.search(r"^FarDist\s*=\s*([0-9.eE+-]+)", settings, re.M)
    if (receipt["wake_protocol"] != "fixed-wake-v1" or receipt["applied"]["FixedWakeFlag"] != 1
            or receipt["applied"]["FarDistToggle"] != 1 or receipt["far_distance_spans"] != 20.
            or wake_iters is None or int(wake_iters.group(1)) != 0 or far is None
            or abs(float(far.group(1))-receipt["applied"]["FarDist"]) > 1e-6):
        raise RuntimeError("Fixed-wake protocol mismatch")
    return dict(result, wake_protocol="fixed-wake-v1", far_distance_spans=20.)


def probe(out, sample, design, mesh, nodes):
    folder = out / "solver" / f"{sample}_{mesh}_{nodes}"
    saved = folder / "record.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf8"))
    os.environ["PTDL_FIXED_WAKE_NODES"] = str(nodes)
    try:
        row = controlled.run_case(prior.solver.DEFAULT_PYTHON, design, sample, folder, 3600,
                                  "training_protocol_diagnostic", "fixed_wake_grid_pair",
                                  (0., float("inf")), 30, mesh)
        row.update(prior.diagnostic_quality(folder / "vspaero_polar.csv"),
                   **fixed_receipt(folder, mesh, nodes))
    except Exception:
        row = dict(sample=sample, quality_pass=False, error=traceback.format_exc(),
                   mesh_level=mesh, applied_wake_nodes=nodes)
    row["source_sample"] = sample
    folder.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
    return row


def main(out, budget, tests):
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "固定尾迹计算锁定.json"
    sources = {str(p.relative_to(ROOT)): prior.driver.digest(p)
               for p in (Path(__file__), WORKER, Path(controlled.__file__), Path(prior.__file__))}
    if lock.exists() and json.loads(lock.read_text(encoding="utf8")) != sources:
        raise RuntimeError("Fixed-wake controller source changed")
    lock.write_text(json.dumps(sources, indent=2), encoding="utf8")
    prior.solver.WORKER = WORKER
    prior.driver.LOW_DATA = prior.LOW_DATA
    prior.driver._run_case = controlled.run_case
    prior.driver.polar_quality = prior.diagnostic_quality
    old = json.loads((prior.FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    designs = {j["sample"]: j["design"] for j in old["jobs"] if j["role"] == "train"}
    records, checks, chosen = [], [], None

    def case(sample, mesh, nodes):
        prior.status(out, "fixed_wake_diagnostics", completed=len(records),
                     running_sample=sample, running_mesh=mesh, running_nodes=nodes)
        row = probe(out, sample, designs[sample], mesh, nodes)
        if not any(r["source_sample"] == sample and r["mesh_level"] == mesh
                   and r["applied_wake_nodes"] == nodes for r in records):
            records.append(row)
        pd.DataFrame(records).to_csv(out / "固定尾迹配对结果.csv", index=False)
        return row

    for mesh, finer in (("medium", "fine"), ("fine", "verification")):
        phase = []
        for sample in prior.ANCHORS:
            coarse = case(sample, mesh, 64)
            wake = case(sample, mesh, 128)
            surface = case(sample, finer, 64)
            phase.extend([dict(sample=sample, mesh=mesh, comparison="wake", **prior.compare(coarse, wake)),
                          dict(sample=sample, mesh=mesh, comparison="grid", **prior.compare(coarse, surface))])
            if not all(r["passed"] for r in phase):
                break
        checks.extend(phase)
        (out / "固定尾迹一致性检查.json").write_text(json.dumps(checks, indent=2), encoding="utf8")
        if len(phase) == 6 and all(r["passed"] for r in phase):
            chosen = dict(mesh=mesh, nodes=64, wake_protocol="fixed-wake-v1",
                          far_distance_spans=20., checks=phase,
                          scope="Separate fixed-wake approximation; no production promotion")
            break
    if chosen is None:
        prior.status(out, "fixed_protocol_unresolved", assessment=str(out / "固定尾迹一致性检查.json"))
        return
    (out / "离散精度选择.json").write_text(json.dumps(chosen, indent=2), encoding="utf8")
    expanded = out / f"均衡{budget}构型_{chosen['mesh']}_fixed64"
    expanded.mkdir(parents=True, exist_ok=True)
    plan_path = expanded / "冻结采样与计算配置.json"
    if not plan_path.exists():
        plan = prior.balanced_plan(out, budget, tests, chosen["mesh"], 64)
        plan.update(wake_protocol="fixed-wake-v1", far_distance_spans=20.,
                    protocol="Separate fixed-wake labels; random/spatial nested selection; sealed new test")
        plan["source_manifest"].update(sources)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
    os.environ["PTDL_FIXED_WAKE_NODES"] = "64"
    prior.verify_protocol = fixed_receipt
    prior.install_training(chosen["mesh"], 64)
    train = prior.driver.fit_models

    def fit(frame, output):
        paths = train(frame, output)
        for name in ("high_candidate", "direct_high"):
            payload = json.loads(paths[name].read_text(encoding="utf8"))
            payload.update(wake_protocol="fixed-wake-v1", far_distance_spans=20.)
            paths[name].write_text(json.dumps(payload, indent=2), encoding="utf8")
        return paths

    prior.driver.fit_models = fit
    prior.status(out, "fixed_balanced_training", progress_file=str(expanded / "进度.json"))
    prior.driver.run(expanded, (budget, tests, 0))
    report = json.loads((expanded / "中网格盲测评估.json").read_text(encoding="utf8"))
    report.update(wake_protocol="fixed-wake-v1", mesh_reference=chosen["mesh"],
                  far_distance_spans=20., automatic_production_promotion=False)
    (expanded / "固定尾迹独立评价.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    prior.status(out, "completed", assessment=str(expanded / "固定尾迹独立评价.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--training-budget", type=int, default=300)
    parser.add_argument("--test-budget", type=int, default=48)
    args = parser.parse_args()
    try:
        main(args.output_dir, args.training_budget, args.test_budget)
    except Exception:
        prior.status(args.output_dir, "failed", error=traceback.format_exc())
        raise
