"""Component-backed equal-budget experiments and block/operator ablations."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from MOEAD多目标联合优化 import (
    MOEAD, NSGAII, MOPSO, HybridNSGAIIMOPSO, PTDLEnduranceJointProblem,
    OBJECTIVE_LABELS, _mopso_archive,
)
from PTDL分块自适应优化 import PTDLBlockAdaptive
from PTDL耦合代理辅助优化 import PTDLCoupledSurrogateSearch
from PTDL任务耦合自适应优化 import PTDLTaskAdaptiveSearch
from VSPAERO气动代理模型 import load_vsp_surrogate, FEATURE_LABELS
from ptdl_hybrid_system_adapter import HybridSystemPowertrain, HybridSystemAdapterConfig
from 规则混动协同控制 import RuleBasedHybridCoordinator
from 响应指标 import project_response_stress_scenario
from 运行_多随机种子算法对比 import _hypervolume_3d
from 优化算法配置库 import summarize_equal_budget_benchmark, ALGORITHM_PROFILES
from 耦合搜索去重试验20261016 import UniqueCoupledProblem

ALGORITHMS = ("NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO",
              "PTDL-自适应分块", "PTDL-无分块", "PTDL-固定算子",
              "PTDL-耦合代理辅助", "PTDL-取消在线代理",
              "PTDL-任务耦合自适应", "PTDL-任务耦合固定",
              "PTDL-耦合贡献学习", "PTDL-耦合局部灵敏度", "PTDL-耦合贡献学习保先验",
              "PTDL-去重贡献学习")
ALGORITHM_PROFILES["PTDL-去重贡献学习"] = {
    "method": "保留提案先验的贡献学习，对重复提案进行局部变量组调整",
    "recommended_use": "减少构型与功率边界重复评价，在相同任务评价预算内增加有效探索",
    "caveat": "保持相同初始种群；收益通过独立种子与去重消融比较检验",
}


def worker(job):
    algorithm, seed, resolution, generations, output, model_path, response_definition = job
    folder = Path(output) / f"a{ALGORITHMS.index(algorithm)}" / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "统计.json"
    if done.exists():
        record = json.loads(done.read_text(encoding="utf8"))
        expected_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if (record["model_sha256"] != expected_hash or record["generations"] != generations
                or record["resolution"] != resolution or record["algorithm"] != algorithm
                or record["seed"] != seed
                or record.get("response_definition", "legacy") != response_definition):
            raise RuntimeError("Resume configuration changed; choose a new output folder")
        return record
    model = load_vsp_surrogate(model_path)
    frame = pd.read_csv(ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv")
    bounds = {k: (float(frame[k].min()), float(frame[k].max())) for k in FEATURE_LABELS}
    seed_path = ROOT / "运行结果" / "H6_三项风险闭环复验" / "NSGA-II与MOEA-D择优可行候选.json"
    anchor = json.loads(seed_path.read_text(encoding="utf8"))["decision"]
    problem = PTDLEnduranceJointProblem(
        aero_provider=model.corrected_aero,
        powertrain_factory=partial(HybridSystemPowertrain, config=HybridSystemAdapterConfig()),
        dispatch_policy_factory=RuleBasedHybridCoordinator, seed_decisions=[anchor],
        design_feature_bounds=bounds, disturbances=project_response_stress_scenario(12.),
        response_definition=response_definition,
    )
    n = (resolution + 1) * (resolution + 2) // 2
    if algorithm == "PTDL-去重贡献学习":
        problem = UniqueCoupledProblem(problem, seed, n)
    if algorithm == "NSGA-II":
        optimizer = NSGAII(population_size=n, generations=generations, seed=seed)
    elif algorithm == "MOEA/D":
        optimizer = MOEAD(resolution=resolution, generations=generations, seed=seed)
    elif algorithm == "MOPSO":
        optimizer = MOPSO(resolution=resolution, generations=generations, seed=seed)
    elif algorithm.startswith("Hybrid"):
        optimizer = HybridNSGAIIMOPSO(resolution=resolution, generations=generations, seed=seed)
    elif algorithm in ("PTDL-耦合代理辅助", "PTDL-取消在线代理"):
        optimizer = PTDLCoupledSurrogateSearch(
            resolution=resolution, generations=generations, seed=seed,
            use_surrogate=algorithm != "PTDL-取消在线代理")
    elif algorithm.startswith("PTDL-任务耦合"):
        optimizer = PTDLTaskAdaptiveSearch(
            resolution=resolution, generations=generations, seed=seed,
            adaptive=algorithm.endswith("自适应"))
    elif algorithm in ("PTDL-耦合贡献学习", "PTDL-耦合局部灵敏度", "PTDL-耦合贡献学习保先验",
                        "PTDL-去重贡献学习"):
        optimizer = PTDLCoupledSurrogateSearch(
            resolution=resolution, generations=generations, seed=seed, use_surrogate=False,
            proposal_policy={"PTDL-耦合贡献学习": "credit",
                             "PTDL-耦合局部灵敏度": "local-linear",
                             "PTDL-耦合贡献学习保先验": "credit-prior",
                             "PTDL-去重贡献学习": "credit-prior"}[algorithm])
    else:
        optimizer = PTDLBlockAdaptive(
            resolution=resolution, generations=generations, seed=seed,
            block_search=algorithm != "PTDL-无分块", adaptive_credit=algorithm != "PTDL-固定算子")
    def progress(generation, total):
        (folder / "进度.json").write_text(json.dumps(dict(generation=generation, total=total)),
                                       encoding="utf8")
    optimizer.progress_callback = progress
    started = time.perf_counter()
    run = optimizer.run(problem)
    elapsed = time.perf_counter() - started
    calls = list(run.evaluated_archive)
    if len(calls) != n * (generations + 1):
        raise RuntimeError("Optimizer evaluation budget mismatch")
    all_rows = []
    for i, e in enumerate(calls):
        all_rows.append(dict(evaluation=i, feasible=e.feasible, violation=e.violation,
                             **e.result.objectives, **e.result.diagnostics,
                             **dict(zip(problem.decision_labels, e.decision))))
    pd.DataFrame(all_rows).to_csv(folder / "全部评价.csv", index=False)
    generations_rows = []
    for generation, archive in run.generation_pareto_archive.items():
        for item in archive:
            if item.feasible:
                generations_rows.append(dict(generation=generation, **item.result.objectives,
                                              **dict(zip(problem.decision_labels, item.decision))))
    pd.DataFrame(generations_rows).to_csv(folder / "逐代档案.csv", index=False)
    run.history.to_csv(folder / "收敛.csv", index=False)
    record = dict(algorithm=algorithm, seed=seed, population=n, resolution=resolution,
                  generations=generations, evaluations=len(calls), runtime_s=elapsed,
                  feasible_ratio=float(np.mean([e.feasible for e in run.evaluations])),
                  archive_count=len([e for e in _mopso_archive(calls, 10000) if e.feasible]),
                  model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                  model_path=model_path, folder=str(folder))
    record["code_sha256"] = hashlib.sha256(b"".join(
        path.name.encode("utf8") + path.read_bytes()
        for path in sorted((ROOT / "核心模型").glob("*.py"))
    )).hexdigest()
    record["response_definition"] = response_definition
    if isinstance(problem, UniqueCoupledProblem):
        record["duplicate_repairs"] = problem.repairs
        record["unique_decisions"] = len(problem.seen)
        record["repair_source_sha256"] = hashlib.sha256(
            (Path(__file__).parent / "耦合搜索去重试验20261016.py").read_bytes()).hexdigest()
    done.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
    return record


def summarize(out, records):
    frames = []
    for r in records:
        p = Path(r["folder"]) / "逐代档案.csv"
        try:
            f = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if len(f):
            f["algorithm"], f["seed"] = r["algorithm"], r["seed"]
            frames.append(f)
    if not frames:
        raise RuntimeError("No feasible archive found")
    fronts = pd.concat(frames, ignore_index=True)
    values = fronts[list(OBJECTIVE_LABELS)].to_numpy(float)
    ideal, nadir = values.min(0), values.max(0)
    span = np.maximum(nadir - ideal, 1e-12)
    hv = []
    for (a, seed, g), f in fronts.groupby(["algorithm", "seed", "generation"]):
        volume = _hypervolume_3d((f[list(OBJECTIVE_LABELS)].to_numpy(float) - ideal)/span,
                                np.full(3, 1.05))
        hv.append(dict(algorithm=a, seed=int(seed), generation=int(g), hypervolume=volume))
    fronts.to_csv(out / "逐代可行Pareto.csv", index=False)
    hv = pd.DataFrame(hv)
    hv.to_csv(out / "共享基准逐代超体积.csv", index=False)
    for r in records:
        last = hv[(hv.algorithm == r["algorithm"]) & (hv.seed == r["seed"]) &
                  (hv.generation == r["generations"])]
        r["hypervolume"] = float(last.hypervolume.iloc[0]) if len(last) else 0.
    runs = pd.DataFrame(records)
    runs.to_csv(out / "多种子算法结果.csv", index=False)
    runs.groupby("algorithm").agg(
        seeds=("seed", "count"), hv_mean=("hypervolume", "mean"), hv_std=("hypervolume", "std"),
        feasible_mean=("feasible_ratio", "mean"), runtime_mean_s=("runtime_s", "mean"),
    ).to_csv(out / "算法汇总.csv")
    tests = []
    focus_name = "PTDL-耦合代理辅助" if "PTDL-耦合代理辅助" in runs.algorithm.values else "PTDL-自适应分块"
    focus = runs[runs.algorithm == focus_name].set_index("seed")
    for a in runs.algorithm.unique():
        if a == focus_name:
            continue
        other = runs[runs.algorithm == a].set_index("seed")
        seeds = focus.index.intersection(other.index)
        delta = focus.loc[seeds].hypervolume.to_numpy() - other.loc[seeds].hypervolume.to_numpy()
        if not len(delta):
            continue
        p = 1. if np.all(abs(delta) < 1e-12) else float(wilcoxon(delta).pvalue)
        tests.append(dict(comparator=a, paired_seeds=len(delta), hv_delta_mean=float(delta.mean()),
                          wins=int((delta > 1e-12).sum()), losses=int((delta < -1e-12).sum()), p_raw=p))
    ordered = sorted(tests, key=lambda r: r["p_raw"])
    adjusted = 0.
    for i, r in enumerate(ordered):
        adjusted = max(adjusted, min(1., (len(ordered)-i) * r["p_raw"]))
        r["p_holm"] = adjusted
    pd.DataFrame(tests).to_csv(out / "配对比较与消融.csv", index=False)
    (out / "评价设置.json").write_text(json.dumps(dict(
        algorithms=sorted(runs.algorithm.unique().tolist()),
        seeds=sorted(runs.seed.unique().tolist()),
        population_size=int(runs.population.iloc[0]),
        resolution=int(runs.resolution.iloc[0]), generations=int(runs.generations.iloc[0]),
        aero_training_samples=load_vsp_surrogate(records[0]["model_path"]).training_samples,
        objectives=list(OBJECTIVE_LABELS), ideal=ideal.tolist(), nadir=nadir.tolist(),
        reference=[1.05]*3, normalization="pooled feasible historical archives across all runs",
        dynamics="component-backed low-order 80 h; no full-mission six-DoF claim",
        response=records[0].get("response_definition", "legacy"),
        budget="initial population plus generations; all algorithms exactly same evaluation calls",
        novelty="project-specific combination under test; theoretical novelty and general superiority not established",
    ), ensure_ascii=False, indent=2), encoding="utf8")
    export_catalog(out)


def export_catalog(out):
    stats = pd.read_csv(out / "算法汇总.csv").rename(columns={
        "seeds": "seed_count", "hv_mean": "final_shared_hypervolume_mean",
        "hv_std": "final_shared_hypervolume_std",
        "feasible_mean": "final_feasible_ratio_mean", "runtime_mean_s": "runtime_s_mean",
    })
    stats.to_csv(out / "算法库统计.csv", index=False)
    catalog = summarize_equal_budget_benchmark(out / "算法库统计.csv", out / "评价设置.json")
    (out / "算法配置库.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "气动代理模型数据" /
                   "气动代理模型_211构型_分目标GP_20261013.json")
    p.add_argument("--resolution", type=int, default=6)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20261015, 20261023)))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    p.add_argument("--response-definition", choices=("legacy", "recovery-integral"), default="legacy")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "核心模型").glob("*.py"))
    }
    frozen = dict(algorithms=list(args.algorithms), seeds=args.seeds,
                  resolution=args.resolution, generations=args.generations,
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  response_definition=args.response_definition, source_manifest=manifest)
    frozen_path = args.output_dir / "运行前冻结配置.json"
    if frozen_path.exists():
        prior = json.loads(frozen_path.read_text(encoding="utf8"))
        if prior != frozen:
            raise RuntimeError("Frozen experiment changed; choose a new output directory")
    else:
        frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    snapshot = args.output_dir / "冻结源代码"
    snapshot.mkdir(exist_ok=True)
    for name, digest in manifest.items():
        source = ROOT / "核心模型" / name
        target = snapshot / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Frozen source snapshot changed")
        else:
            shutil.copy2(source, target)
    jobs = [(a, s, args.resolution, args.generations, str(args.output_dir), str(args.model), args.response_definition)
            for a in args.algorithms for s in args.seeds]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for f in as_completed(futures):
            try:
                rows.append(f.result())
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            except Exception as e:
                errors.append(dict(job=futures[f], error=repr(e)))
                print(errors[-1], flush=True)
            (args.output_dir / "进度.json").write_text(json.dumps(dict(
                completed=len(rows), total=len(jobs), errors=errors), ensure_ascii=False, indent=2),
                encoding="utf8")
    if rows:
        summarize(args.output_dir, rows)
    if errors:
        raise RuntimeError(f"{len(errors)} experiment runs failed")
