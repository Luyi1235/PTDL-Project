"""Compare saved historical fronts with a single cross-budget reference box."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"核心模型"))
from MOEAD多目标联合优化 import OBJECTIVE_LABELS
from 运行_六自由度多种子闭环比较 import ALGORITHMS
from 运行_多随机种子算法对比 import _hypervolume_3d


def tail_assessment(values, tolerance=.01, window=2):
    """Numerical plateau is descriptive, not proof of a global optimum."""
    values=np.asarray(values,float)
    if len(values)<window+1 or not np.isfinite(values).all() or values[-1]<=0:
        return dict(tail_relative_change=None, tail_plateau=False)
    change=float(np.max(abs(values[-window:]-values[-window-1:-1])) / max(values[-1],1e-12))
    return dict(tail_relative_change=change,tail_plateau=bool(change<=tolerance))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--expanded",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    runs,fronts=[],[]
    for budget,path in (("H2G3",args.baseline),("H4G7",args.expanded)):
        progress=json.loads((path/"进度.json").read_text(encoding="utf-8"))
        if progress["completed"]!=progress["total"] or progress["errors"]:
            raise RuntimeError(f"Incomplete optimizer batch: {path}")
        records=pd.read_csv(path/"多种子运行比较.csv")
        records["budget"]=budget
        runs.append(records)
        for row in records.itertuples():
            f=pd.read_csv(Path(row.folder)/(ALGORITHMS[row.algorithm]+"_generation_feasible_pareto.csv"))
            f["budget"],f["seed"],f["algorithm"]=budget,row.seed,row.algorithm
            fronts.append(f)
    runs=pd.concat(runs,ignore_index=True)
    runs["tail_plateau"]=False
    data=pd.concat(fronts,ignore_index=True)
    values=data[list(OBJECTIVE_LABELS)].to_numpy(float)
    ideal,nadir=values.min(0),values.max(0)
    scale=np.maximum(nadir-ideal,1e-9)
    records=[]
    for row in runs.itertuples():
        group=data[(data.budget==row.budget)&(data.algorithm==row.algorithm)&(data.seed==row.seed)]
        hv=[]
        for generation in range(row.generations+1):
            f=group[group.generation==generation]
            value=_hypervolume_3d((f[list(OBJECTIVE_LABELS)].to_numpy(float)-ideal)/scale,np.full(3,1.05)) if len(f) else 0.
            hv.append(value)
            records.append(dict(budget=row.budget,algorithm=row.algorithm,seed=row.seed,generation=generation,
                                evaluations=(generation+1)*row.population,hypervolume=value,archive_size=len(f)))
        idx=(runs.budget==row.budget)&(runs.algorithm==row.algorithm)&(runs.seed==row.seed)
        for key,value in dict(shared_final_hv=hv[-1],**tail_assessment(hv)).items():
            runs.loc[idx,key]=value
    convergence=pd.DataFrame(records)
    convergence.to_csv(out/"跨预算统一超体积.csv",index=False)
    runs.to_csv(out/"跨预算逐次统计.csv",index=False)
    stats=runs.groupby(["budget","algorithm"]).agg(
        seeds=("seed","count"),evaluations=("evaluation_calls","sum"),
        feasibility_mean=("final_feasible_ratio","mean"),hv_mean=("shared_final_hv","mean"),
        hv_std=("shared_final_hv","std"),runtime_mean_s=("runtime_s","mean"),
        plateau_fraction=("tail_plateau","mean")).reset_index()
    stats.to_csv(out/"跨预算算法统计.csv",index=False)
    pairs=[]
    rng=np.random.default_rng(20261013)
    for algorithm in ALGORITHMS:
        base=runs[(runs.algorithm==algorithm)&(runs.budget=="H2G3")].set_index("seed")
        larger=runs[(runs.algorithm==algorithm)&(runs.budget=="H4G7")].set_index("seed")
        seeds=sorted(set(base.index)&set(larger.index))
        if not seeds:
            continue
        delta=larger.loc[seeds,"shared_final_hv"].to_numpy()-base.loc[seeds,"shared_final_hv"].to_numpy()
        boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
        pairs.append(dict(algorithm=algorithm,paired_seeds=len(seeds),mean_hv_delta=float(delta.mean()),
                          bootstrap_low=float(np.quantile(boot,.025)),bootstrap_high=float(np.quantile(boot,.975)),
                          improved_seeds=int(np.sum(delta>0)),plateau_seeds=int(larger.tail_plateau.sum())))
    pd.DataFrame(pairs).to_csv(out/"同种子配对预算收益.csv",index=False)
    (out/"分析口径.json").write_text(json.dumps(dict(
        ideal=ideal.tolist(),nadir=nadir.tolist(),reference=[1.05]*3,
        archive="Recorded historical feasible Pareto archives; final population used only for feasibility ratio",
        plateau="Maximum relative HV step change in final two generations <= 1%; positive HV required",
        scope="Descriptive finite-budget plateau, not global convergence or final engineering qualification",
        uncertainty="Paired bootstrap with only four seeds; exploratory interval",
        objectives="Existing PPT low-order component-backed objectives; full 6DoF enters candidate constraints",
    ),ensure_ascii=False,indent=2),encoding="utf-8")
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
    plt.rcParams["axes.unicode_minus"]=False
    plt.rcParams.update({"font.size":12,"axes.titlesize":13,"axes.labelsize":12,
                         "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11})
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout="constrained")
    colors=["#206b5b","#bd6740"]
    for ax,algorithm in zip(axes.flat,ALGORITHMS):
        for color,budget in zip(colors,("H2G3","H4G7")):
            f=convergence[(convergence.algorithm==algorithm)&(convergence.budget==budget)]
            if f.empty:
                continue
            grouped=f.groupby("evaluations").hypervolume
            mean=grouped.mean()
            low,high=grouped.min(),grouped.max()
            ax.plot(mean.index,mean.values,label="6个体3代" if budget=="H2G3" else "15个体7代",color=color)
            ax.fill_between(mean.index,low.values,high.values,color=color,alpha=.15)
        title=algorithm.replace("Hybrid NSGA-II/MOPSO","混合算法")
        if not ((convergence.algorithm==algorithm)&(convergence.budget=="H4G7")).any():
            title+="（扩大预算未完成）"
        ax.set_title(title)
        ax.set_xlabel("累计候选评价次数")
        ax.set_ylabel("共享基准超体积")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out/"跨预算收敛比较.png",dpi=160)
    plt.close(fig)
    print(stats.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__=="__main__":
    main()
