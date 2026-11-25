"""Recovery trajectory comparison and evidence-scoped control configuration library."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from 运行_蓝色方法综合攻关 import execute,NEURAL_PHASES


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--tuning",type=Path,required=True)
    p.add_argument("--neural",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((args.tuning/"分阶段MPC配置.json").read_text(encoding="utf-8"))
    jobs=[]
    for scenario in ("fixed_recovery","adaptive_recovery"):
        for policy in ("rule","fuzzy","mpc","scheduled","neural"):
            config=selected if policy=="scheduled" else {"model":str(args.neural)} if policy=="neural" else {}
            jobs.append((str(out),scenario+"_"+policy,policy,config,scenario))
    rows=execute(jobs,5,out)
    for r in rows:
        if "error" in r:
            continue
        f=pd.read_csv(Path(r["path"])/"PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
        descent=f[f.phase=="long_descent"]
        powered=descent[descent.power_loss_active==0]
        r["descent_end_height_m"]=float(descent.altitude_m.iloc[-1])
        r["descent_end_error_to_common_800m"]=float(abs(descent.altitude_m.iloc[-1]-800))
        r["power_restored_at_s"]=float(powered.time_s.min())
        r["post_restore_tracking_rmse_m"]=float(np.sqrt(np.mean(powered.altitude_error_m**2)))
    (out/"恢复轨迹比较.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    all_rows=json.loads((args.study/"试验汇总.json").read_text(encoding="utf-8"))+rows
    catalog={}
    for scenario in sorted(set(r["scenario"] for r in all_rows)):
        group=[r for r in all_rows if r["scenario"]==scenario]
        accepted=[r for r in group if r["screen_pass"]]
        catalog[scenario]={
            "status":"available" if accepted else "no_passed_configuration",
            "accepted_policies":[r["policy"] for r in accepted],
            "ranked_by_fuel":[r["policy"] for r in sorted(accepted,key=lambda r:r["fuel_kg"])],
            "ranked_by_battery_loss":[r["policy"] for r in sorted(accepted,key=lambda r:r["battery_loss_mj"])],
            "records":[r["path"] for r in group],
        }
    responses=pd.read_csv(args.study/"六阶段配对响应.csv")
    catalog_payload={
        "schema":"ptdl_evidence_scoped_configuration_library_v1",
        "scope":"Estimated reference aircraft, 4-hour planned six-stage profile, fixed phase constraints",
        "screening":str(args.study/"验收定义.json"),"phase_mpc_parameters":selected,
        "neural_model":str(args.neural),"scenarios":catalog,
        "response_comparison":responses.where(pd.notnull(responses),None).to_dict("records"),
        "selection_rule":"Filter feasibility and reserve first, then apply declared fuel/loss preference; no universal winner",
        "optimizer_catalog_status":"Separate 80-hour multi-seed optimization statistics required",
    }
    (out/"控制算法配置库.json").write_text(json.dumps(catalog_payload,ensure_ascii=False,indent=2),encoding="utf-8")
    # Matched reserve screening precedes the diagnostic fuel/loss decision scan.
    baseline=[r for r in all_rows if r["scenario"]=="nominal" and r["screen_pass"] and r["minimum_soc"]>=.30]
    scan=[]
    if baseline:
        values=np.asarray([[r["fuel_kg"],r["battery_loss_mj"],r["speed_error_max"]] for r in baseline])
        lo,hi=values.min(0),values.max(0)
        normalized=(values-lo)/np.maximum(hi-lo,1e-9)
        for i in range(11):
            for j in range(11-i):
                w=np.asarray([i,j,10-i-j])/10
                chosen=int(np.argmin(normalized@w))
                scan.append({"fuel_weight":w[0],"loss_weight":w[1],"tracking_weight":w[2],
                             "selected_policy":baseline[chosen]["policy"]})
        pd.DataFrame(scan).to_csv(out/"同储备门槛控制方案权重扫描.csv",index=False)
    print("Recovery and control catalog complete",flush=True)


if __name__=="__main__":
    main()
