"""Fuse power, SOC and flight residuals for model-based lag/delay identification."""

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"核心模型"))
from 运行_PTDL估计六自由度全任务闭环 import run
from 规则混动协同控制 import RuleBasedHybridCoordinator


def simulate(job):
    folder,tau,delay,slow=job
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"日志.txt").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):
        result=run(
            folder,integration_step_s=.05,control_period_s=2,terminal_control_period_s=.5,
            sample_period_s=2,mission_hours=4,mission_mode="full",pitch_kp=4,pitch_kd=4,
            powertrain_config={"engine_time_constant_s":tau,"engine_command_delay_s":delay},
            dispatch_policy=RuleBasedHybridCoordinator(),
            phase_overrides={"approach_descent":{"speed_mps":30}} if slow else None,
        )
    return {"path":str(folder),"tau":tau,"delay":delay,"completed":result["mission_completed"]}


def residual(candidate,reference):
    columns=["engine_power_kw","soc","speed_mps","altitude_m"]
    scales=np.array([20.,.05,2.,100.])
    ref=reference[reference.time_s<=candidate.time_s.max()]
    predicted=np.column_stack([np.interp(ref.time_s,candidate.time_s,candidate[k]) for k in columns])
    error=(predicted-ref[columns].to_numpy())/scales
    split=int(len(error)*.7)
    return float(np.mean(error[:split]**2)),float(np.mean(error[split:]**2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    truth=simulate((out/"模拟观测",2.5,1.,False))
    filename="PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv"
    reference=pd.read_csv(Path(truth["path"])/filename)
    rng=np.random.default_rng(20261013)
    for name,sigma in (("engine_power_kw",.05),("soc",.0001),("speed_mps",.01),("altitude_m",.1)):
        reference[name]+=rng.normal(0,sigma,len(reference))
    reference.to_csv(out/"带噪声模拟观测.csv",index=False)
    jobs=[(out/f"candidate_{i}",tau,delay,False)
          for i,(tau,delay) in enumerate(( (t,d) for t in (1.5,2.5,3.5) for d in (0.,1.,2.)))]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(simulate,jobs):
            train,test=residual(pd.read_csv(Path(row["path"])/filename),reference)
            rows.append(row|{"training_residual":train,"validation_residual":test})
    winner=min(rows,key=lambda r:r["training_residual"])
    selected={"engine_time_constant_s":winner["tau"],"engine_command_delay_s":winner["delay"]}
    (out/"校正后动力参数.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    independent=simulate((out/"独立慢速进近",winner["tau"],winner["delay"],True))
    pd.DataFrame(rows).to_csv(out/"动态参数融合校正比较.csv",index=False)
    (out/"融合校正说明.json").write_text(json.dumps({
        "source":"Known estimated-parameter six-DoF simulation with added observation noise",
        "signals":["engine power","SOC","airspeed","altitude"],
        "split":"First 70 percent for grid selection, final 30 percent for temporal validation",
        "selected":selected,"validation_residual":winner["validation_residual"],
        "independent_mission_completed":independent["completed"],
        "scope":"Synthetic method verification, not identification from measured hardware",
    },indent=2),encoding="utf-8")
    print(selected,independent["completed"],flush=True)


if __name__=="__main__":
    main()
