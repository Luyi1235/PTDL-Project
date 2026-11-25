"""Frozen original-resolution free-wake DOE with fresh labels and sealed validation."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import copy
import json
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd
from scipy.stats import qmc

import 续跑_稳定标签与均衡采样20261016 as prior
import VSPAERO受控子进程20261017 as controlled
from 分目标气动高斯过程 import TargetwiseAeroGP

ROOT = prior.ROOT
OUT = ROOT / "运行结果" / "千构型原定义代理_20261017"
WORKER = Path(__file__).with_name("运行_VSPAERO原网格并行核20261017.py")
regularized = prior.strong.regularized
regularized.CONFIGS = [
    dict(method="gp", mode="dimensionless", ard=True, penalty=value) for value in (.05, .2, .8)
] + [
    dict(method="gp", mode="raw", ard=False, penalty=value) for value in (.05, .4)
] + [
    dict(method="gp", mode="physics", ard=False, penalty=.2),
    dict(method="ridge", mode="dimensionless", degree=2, penalty=100.),
    dict(method="boost", mode="physics", penalty=5.),
]


def plan(out, total, tests):
    source = pd.read_csv(prior.LOW_DATA)
    x = source[list(prior.FEATURE_LABELS)].to_numpy(float)
    if total <= len(x):
        raise ValueError("Expansion must exceed the existing geometry count")
    old = json.loads((prior.FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    fixed = {k: old["jobs"][0]["design"][k]
             for k in ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    lo, hi = x.min(0), x.max(0)
    span = hi-lo
    forbidden = []
    for folder in (prior.FIRST, ROOT / "运行结果" / "多精度气动代理_20261016"):
        archived = json.loads((folder / "冻结采样与计算配置.json").read_text(encoding="utf8"))
        forbidden.extend([[j["design"][k] for k in prior.FEATURE_LABELS]
                          for j in archived["jobs"] if j["role"] == "test"])
    forbidden.extend(pd.read_csv(prior.driver.LOW_DIR / "合并留出验证.csv")[
        list(prior.FEATURE_LABELS)].to_numpy(float).tolist())
    forbidden = np.asarray(forbidden)
    jobs = [dict(sample=f"stable_train_{i:04d}", role="train", mesh="legacy",
                 design={**fixed, **dict(zip(prior.FEATURE_LABELS, row))},
                 basis="existing_geometry_recomputed_original_free_wake")
            for i, row in enumerate(x)]
    selected = x.copy()
    pool = qmc.scale(qmc.Sobol(d=7, scramble=True, seed=2026101703).random_base2(14), lo, hi)
    for row in pool:
        if len(jobs) == total:
            break
        design = {**fixed, **dict(zip(prior.FEATURE_LABELS, row))}
        if (all(v <= 1e-9 for v in prior.geometry_constraints(prior.AirframeDesign(**design)).values())
                and np.linalg.norm((forbidden-row)/span, axis=1).min() > 1e-5
                and np.linalg.norm((selected-row)/span, axis=1).min() > 1e-5):
            jobs.append(dict(sample=f"stable_train_{len(jobs):04d}", role="train",
                             mesh="legacy", design=design, basis="scrambled_sobol_feasible_interior"))
            selected = np.vstack((selected, row))
    if len(jobs) != total:
        raise RuntimeError("Insufficient feasible Sobol pool")
    avoid = np.vstack((selected, forbidden))
    rng = np.random.default_rng(2026101704)
    while len(jobs) < total+tests:
        row = rng.uniform(lo, hi)
        design = {**fixed, **dict(zip(prior.FEATURE_LABELS, row))}
        if (all(v <= 1e-9 for v in prior.geometry_constraints(prior.AirframeDesign(**design)).values())
                and np.linalg.norm((avoid-row)/span, axis=1).min() > 1e-5):
            jobs.append(dict(sample=f"stable_test_{len(jobs)-total:04d}", role="test",
                             mesh="legacy", design=design, basis="fresh_sealed_uniform_feasible"))
            avoid = np.vstack((avoid, row))
    sources = prior.driver.sources()
    for p in (Path(__file__), WORKER, Path(controlled.__file__), Path(prior.__file__),
              Path(prior.strong.__file__), Path(regularized.__file__)):
        sources[str(p.relative_to(ROOT))] = prior.driver.digest(p)
    return dict(jobs=jobs, source_manifest=sources, train_count=total, test_count=tests,
                fine_count=0, wake_protocol="free-wake-legacy-v1",
                mesh_reference="legacy", wake_nodes=8, wake_iterations=30,
                native_workers=1, native_cores=8, old_labels_reused=False,
                sealed_test_labels_used_for_selection=False, old_geometry_count=len(x),
                new_interior_geometry_count=total-len(x), selected_configs=regularized.CONFIGS,
                purpose="Predict original-resolution solver labels; grid uncertainty evaluated separately",
                production_promotion=False)


def verify(folder):
    receipt = json.loads((folder / "冻结求解设置.json").read_text(encoding="utf8"))
    settings = (folder / "airframe.vspaero").read_text(encoding="utf8")
    log = (folder / "worker.log").read_text(encoding="utf8")
    expected = dict(FixedWakeFlag=0, NumWakeNodes=8, NCPU=8, FarDistToggle=0, WakeNumIter=30)
    if (receipt["wake_protocol"] != "free-wake-legacy-v1"
            or receipt["applied"] != expected
            or re.search(r"^NumWakeNodes\s*=\s*8\s*$", settings, re.M) is None
            or re.search(r"^WakeIters\s*=\s*30\s*$", settings, re.M) is None
            or re.search(r"^FarDist\s*=\s*-1\s*$", settings, re.M) is None
            or re.search(r"\s-omp\s+8\s", log) is None):
        raise RuntimeError("Original free-wake protocol mismatch")
    return dict(wake_protocol="free-wake-legacy-v1", applied_wake_nodes=8,
                mesh_level="legacy", native_command_cores=8)


def native(*args, **kwargs):
    args = list(args)
    args[4] = 3600
    result = controlled.run_case(*args, **kwargs)
    result.update(verify(Path(args[3])))
    return result


def fit_models(frame, out):
    dataset = out / "原网格训练数据.csv"
    frame.to_csv(dataset, index=False)
    baseline = frame.loc[frame["sample"] == "stable_train_0000"].iloc[0]
    if not baseline.wake_quality_pass:
        raise RuntimeError("Newly computed baseline is invalid")
    reference = copy.deepcopy(json.loads(prior.driver.LOW_MODEL.read_text(encoding="utf8")))
    reference.update(baseline_targets=baseline[list(prior.driver.TARGETS)].to_numpy(float).tolist(),
                     mesh_reference="legacy", wake_protocol="free-wake-legacy-v1",
                     wake_nodes=8, high_training_samples=int(np.isfinite(frame[
                         prior.driver.TARGETS[0]].to_numpy(float)).sum()),
                     low_fidelity_used_by_target=[False, False])
    template = out / "原定义模型模板.json"
    template.write_text(json.dumps(reference), encoding="utf8")
    regularized.DATASETS = {"expanded": dataset}
    regularized.REFERENCE = {"expanded": template}
    selection = out / "嵌套正则化"
    regularized.train(selection, 12)
    candidate = out / "多精度候选代理.json"
    candidate.write_bytes((selection / "expanded_正则化候选.json").read_bytes())
    direct = copy.deepcopy(reference)
    direct["targets"] = []
    for target in prior.driver.TARGETS:
        mask = np.isfinite(frame[target].to_numpy(float))
        if target == prior.driver.TARGETS[1]:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        x = frame.loc[mask, list(prior.FEATURE_LABELS)].to_numpy(float)
        y = frame.loc[mask, target].to_numpy(float)
        fitted = regularized.fit(dict(method="gp", mode="dimensionless", ard=True, penalty=.05), x, y)
        direct["targets"].append(regularized.serialize(fitted, x, y, .1))
    direct["selection"] = "Predefined direct ARD-GP reference; test labels excluded"
    direct_path = out / "仅高精度候选代理.json"
    TargetwiseAeroGP(direct).to_json(direct_path)
    (out / "训练模型选择.json").write_text(json.dumps(dict(
        selection="Nested random/spatial validation; training labels only",
        configurations=regularized.CONFIGS, old_labels_reused=False)), encoding="utf8")
    return dict(high_candidate=candidate, direct_high=direct_path, low_reference=prior.driver.LOW_MODEL)


def main(out, total, tests):
    out.mkdir(parents=True, exist_ok=True)
    prior.driver.LOW_DATA = prior.LOW_DATA
    path = out / "冻结采样与计算配置.json"
    if not path.exists():
        path.write_text(json.dumps(plan(out, total, tests), ensure_ascii=False, indent=2), encoding="utf8")
    prior.solver.WORKER = WORKER
    prior.driver._run_case = native
    prior.driver.polar_quality = prior.diagnostic_quality
    prior.driver.fit_models = fit_models
    prior.driver.run(out, (total, tests, 0))
    report = json.loads((out / "中网格盲测评估.json").read_text(encoding="utf8"))
    report.update(mesh_reference="legacy", wake_protocol="free-wake-legacy-v1",
                  all_labels_recomputed=True, grid_convergence_claimed=False)
    (out / "千构型独立评价.json").write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--training-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=96)
    args = parser.parse_args()
    try:
        main(args.output_dir, args.training_count, args.test_count)
    except Exception:
        prior.status(args.output_dir, "failed", error=traceback.format_exc())
        raise
