"""Process-isolated same-budget optimizer comparison with full JSBSim constraints."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from MOEAD多目标联合优化 import run_ptdl_80h_joint_moead, OBJECTIVE_LABELS
from PTDL联合优化高保真接口 import PTDL6DoFFullMissionOptimizerValidator
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 规则混动协同控制 import RuleBasedHybridCoordinator
from VSPAERO气动代理模型 import load_vsp_surrogate, FEATURE_LABELS
from 响应指标 import project_response_stress_scenario
from 运行_多随机种子算法对比 import _hypervolume_3d

ALGORITHMS = {
    "NSGA-II": "ptdl_80h_joint_nsga_ii",
    "MOEA/D": "ptdl_80h_joint_moead",
    "MOPSO": "ptdl_80h_joint_mopso",
    "Hybrid NSGA-II/MOPSO": "ptdl_80h_joint_hybrid_nsga_ii_mopso",
}


def worker(job):
    algorithm, seed, output, resolution, generations, hours = job
    folder = Path(output) / ALGORITHMS[algorithm] / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    # Native solver output remains in its own process log.
    with (folder / "运行日志.txt").open("w", encoding="utf-8") as log:
        import contextlib
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return run_one(algorithm, seed, folder, resolution, generations, hours)


def run_one(algorithm, seed, folder, resolution, generations, hours, surrogate_path=None, domain_path=None):
    data = ROOT / "气动代理模型数据"
    surrogate_path=Path(surrogate_path) if surrogate_path else data / "气动代理模型_扩充DOE131点_ARD_RBF_GP_20261012.json"
    domain_path=Path(domain_path) if domain_path else data / "VSPAERO_DOE扩充样本_131点_Ew_物理选点_网格修复_自适应补样_20261011.csv"
    surrogate = load_vsp_surrogate(surrogate_path)
    frame = pd.read_csv(domain_path)
    bounds = {label: (float(frame[label].min()), float(frame[label].max())) for label in FEATURE_LABELS}
    seed_path = ROOT / "运行结果" / "H6_三项风险闭环复验" / "NSGA-II与MOEA-D择优可行候选.json"
    decision = json.loads(seed_path.read_text(encoding="utf-8"))["decision"]
    validator = PTDL6DoFFullMissionOptimizerValidator(
        folder / "逐候选六自由度", mission_hours=hours,
        mission_duration_basis="endurance-before-landing",
        integration_step_s=0.05, control_period_s=2, sample_period_s=30,
        dispatch_policy_factory=RuleBasedHybridCoordinator,
    )
    started = time.perf_counter()
    result = run_ptdl_80h_joint_moead(
        folder, resolution=resolution, generations=generations, seed=seed,
        aero_provider=lambda d, c: surrogate.corrected_aero(d, c),
        aero_model_label=f"{type(surrogate).__name__} {surrogate.training_samples}",
        powertrain_factory=partial(HybridSystemPowertrain, config=HybridSystemAdapterConfig()),
        dispatch_policy_factory=RuleBasedHybridCoordinator,
        seed_decisions=[decision], algorithm=algorithm, design_feature_bounds=bounds,
        disturbances=project_response_stress_scenario(12.0),
        sixdof_validator=validator,
    )
    run = result["run"]
    pd.DataFrame(validator.records).to_csv(folder / "逐候选六自由度汇总.csv", index=False)
    record = {
        "algorithm": algorithm, "seed": seed,
        "population": (resolution+1)*(resolution+2)//2,
        "generations": generations, "sixdof_endurance_hours": hours,
        "evaluation_calls": len(run.evaluated_archive or run.evaluations),
        "sixdof_candidate_calls": len(validator.records),
        "final_feasible_ratio": float(np.mean([e.feasible for e in run.evaluations])),
        "runtime_s": time.perf_counter()-started, "folder": str(folder),
        "surrogate_path":str(surrogate_path),
        "surrogate_sha256":hashlib.sha256(surrogate_path.read_bytes()).hexdigest(),
        "domain_path":str(domain_path),
    }
    (folder / "单次统计.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def summarize(output, records):
    frames = []
    for record in records:
        f = pd.read_csv(Path(record["folder"]) / (ALGORITHMS[record["algorithm"]] + "_generation_feasible_pareto.csv"))
        if len(f):
            f["algorithm"], f["seed"] = record["algorithm"], record["seed"]
            frames.append(f)
    runs = pd.DataFrame(records)
    if frames:
        fronts = pd.concat(frames, ignore_index=True)
        values = fronts[list(OBJECTIVE_LABELS)].to_numpy(float)
        ideal, nadir = values.min(axis=0), values.max(axis=0)
        span = np.maximum(nadir-ideal, 1e-9)
        hv = []
        for (algorithm, seed, generation), f in fronts.groupby(["algorithm", "seed", "generation"]):
            hv.append(dict(algorithm=algorithm, seed=int(seed), generation=int(generation),
                           hypervolume=_hypervolume_3d((f[list(OBJECTIVE_LABELS)].to_numpy(float)-ideal)/span, np.full(3,1.05))))
        pd.DataFrame(hv).to_csv(output / "共享基准逐代超体积.csv", index=False)
        fronts.to_csv(output / "逐代可行档案.csv", index=False)
        (output / "归一化基准.json").write_text(json.dumps({"ideal":ideal.tolist(),"nadir":nadir.tolist(),"reference":[1.05]*3}),encoding="utf-8")
        for row in records:
            values = [v for v in hv if v["algorithm"]==row["algorithm"] and v["seed"]==row["seed"] and v["generation"]==row["generations"]]
            row["final_hypervolume"] = values[0]["hypervolume"] if values else 0.0
    runs = pd.DataFrame(records)
    runs.to_csv(output / "多种子运行比较.csv", index=False)
    runs.groupby("algorithm").agg(
        seeds=("seed","count"), feasible_mean=("final_feasible_ratio","mean"),
        feasible_std=("final_feasible_ratio","std"), runtime_mean_s=("runtime_s","mean")
    ).to_csv(output / "算法统计.csv")
    (output / "评价口径.json").write_text(json.dumps({
        "objectives": "Preserved PPT objectives from component-backed mission and paired wind response",
        "constraints": "Every unique candidate additionally runs ground takeoff, JSBSim full mission, landing roll",
        "sixdof_endurance_hours": records[0]["sixdof_endurance_hours"] if records else None,
        "selection": "Historical feasible archive, shared normalization; failed candidates retained",
        "scope": "Six-DoF constrained joint optimization; objectives are not replaced by six-DoF-only metrics",
        "initialization": "Shared known candidate anchor plus same-seed random population",
    },ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=list(range(20261013,20261021)))
    p.add_argument("--resolution",type=int,default=2)
    p.add_argument("--generations",type=int,default=3)
    p.add_argument("--mission-hours",type=float,default=80)
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--algorithms",nargs="+",choices=tuple(ALGORITHMS),default=tuple(ALGORITHMS))
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    jobs=[(a,s,str(args.output_dir),args.resolution,args.generations,args.mission_hours) for a in args.algorithms for s in args.seeds]
    rows, errors=[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(worker,j):j for j in jobs}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
                print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
                pd.DataFrame(rows).to_csv(args.output_dir/"多种子运行比较.csv",index=False)
            except Exception as error:
                errors.append({"job":futures[future],"error":repr(error)})
                print(errors[-1],flush=True)
            (args.output_dir/"进度.json").write_text(json.dumps({"completed":len(rows),"total":len(jobs),"errors":errors},ensure_ascii=False,indent=2),encoding="utf-8")
    if rows:
        summarize(args.output_dir,rows)
    if errors:
        raise RuntimeError(f"{len(errors)} optimizer runs failed; see progress record")


if __name__=="__main__":
    main()
