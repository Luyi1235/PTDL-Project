"""Unified staged command perturbations, local lift margin and touchdown checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from 运行_蓝色方法综合攻关 import execute,NEURAL_PHASES


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--neural",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((args.source/"分阶段MPC配置.json").read_text(encoding="utf-8"))
    jobs=[]
    scenarios=["nominal","dynamic_delay","battery_health","recovery",
               "glide_28","glide_30","glide_32","glide_34"]
    scenarios += ["response_"+p for p in NEURAL_PHASES]
    for scenario in scenarios:
        for policy in ("rule","fuzzy","mpc","scheduled","neural"):
            config=selected if policy=="scheduled" else {"model":str(args.neural)} if policy=="neural" else {}
            jobs.append((str(out),scenario+"_"+policy,policy,config,scenario))
    rows=execute(jobs,args.workers,out)
    responses=[]
    for r in rows:
        if not r["scenario"].startswith("response_") or "error" in r:
            continue
        baseline=pd.read_csv(out/("nominal_"+r["policy"])/"PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
        f=pd.read_csv(Path(r["path"])/"PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
        f=f[f.time_s<=baseline.time_s.max()].copy()
        active=f[f.speed_command_event_active>0]
        rec={"scenario":r["scenario"],"policy":r["policy"],"event_observed":len(active)>0,
             "band_mps":.3,"hold_s":30.,"window_s":600.,"recovery_s":None,"censored":True}
        if len(active):
            end=float(active.time_s.min()+60)
            t=f.time_s.to_numpy()
            error=f.speed_mps.to_numpy()-np.interp(t,baseline.time_s,baseline.speed_mps)
            rec["peak_delta_speed_mps"]=float(np.max(abs(error[t>=active.time_s.min()])))
            for i in np.flatnonzero((t>=end)&(t<=end+600)):
                window=(t>=t[i])&(t<=t[i]+30)
                if t[window].max()<t[i]+30-1e-6:
                    continue
                if np.all(abs(error[window])<=.3):
                    rec["recovery_s"]=float(t[i]-end)
                    rec["censored"]=False
                    break
        responses.append(rec)
    pd.DataFrame(responses).to_csv(out/"六阶段配对响应.csv",index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ("config","phase_metrics","health_estimator")} for r in rows]).to_csv(out/"统一验收比较.csv",index=False)
    (out/"验收定义.json").write_text(json.dumps({
        "lift_screen":"1.10 * stall speed with instantaneous mass, altitude density and bank; CLmax=1.42 estimated",
        "alpha_max_deg":12,"touchdown_sink_max_mps":2,"SOCmin":.2,
        "perturbation":"Stage-relative speed command +3 m/s for 60 s, start 30 s after actual stage entry",
        "recovery":"Paired nominal actual-speed residual <=0.3 m/s for 30 s; 600 s observation; censored when insufficient",
        "glide_scope":"Zero shaft propulsion from long descent through touchdown; preceding climb and cruise powered",
    },ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
