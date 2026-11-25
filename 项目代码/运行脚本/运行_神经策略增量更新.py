"""Train on selected boundary errors; choose on validation, test once by scenario."""

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"核心模型"))
from 智能能量管理与容错调度 import NEURAL_PHASES,FUZZY_SOC_TARGETS,neural_imitation_feature_vector
from 运行_蓝色方法综合攻关 import job_run


def load_case(folder):
    summary=json.loads((folder/"PTDL_JSBSim全任务闭环摘要.json").read_text(encoding="utf-8"))
    f=pd.read_csv(folder/"PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
    f=f[f.engine_power_cap_kw>1].copy()
    x=np.asarray([neural_imitation_feature_vector(
        r.phase,r.control_period_s,r.soc_before,r.altitude_error_m,
        r.speed_target_mps-r.speed_mps,r.required_propulsion_power_kw,
        r.engine_power_cap_kw,r.previous_engine_target_kw,r.phase_elapsed_s,
        summary["phase_duration_s"][r.phase],
    ) for r in f.itertuples()])
    return x,np.clip(f.engine_target_kw.to_numpy()/f.engine_power_cap_kw.to_numpy(),0,1),f,summary


def scores(y,pred):
    e=pred-y
    return dict(rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),
                r2=float(1-np.sum(e*e)/max(np.sum((y-y.mean())**2),1e-12)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    train_cases=[load_case(args.source/(n+"_scheduled")) for n in ("nominal","recovery")]
    x=np.vstack([v[0] for v in train_cases])
    y=np.concatenate([v[1] for v in train_cases])
    frame=pd.concat([v[2] for v in train_cases],ignore_index=True)
    vx,vy,_,_=load_case(args.source/"battery_health_scheduled")
    tx,ty,test_frame,summary=load_case(args.source/"slow_approach_scheduled")
    mean=x.mean(0)
    scale=np.maximum(x.std(0),1e-8)
    xx=(x-mean)/scale
    model=MLPRegressor(hidden_layer_sizes=(32,16),activation="tanh",max_iter=400,
                       early_stopping=True,random_state=20261013,n_iter_no_change=20)
    model.fit(xx,y)
    residual=abs(model.predict(xx)-y)
    boundary=(frame.soc_before.to_numpy()<.35)|(frame.phase_elapsed_s.to_numpy()<120)|(
        frame.required_propulsion_power_kw.to_numpy()<15)
    selected=np.flatnonzero(boundary & (residual>=np.quantile(residual[boundary],.65)))
    candidates=[model]
    for alpha in (1e-4,1e-3):
        enriched=MLPRegressor(hidden_layer_sizes=(48,24),activation="tanh",max_iter=500,
                              alpha=alpha,early_stopping=True,random_state=20261014,n_iter_no_change=25)
        indices=np.concatenate((np.arange(len(y)),selected,selected))
        enriched.fit(xx[indices],y[indices])
        candidates.append(enriched)
    validation=[scores(vy,np.clip(m.predict((vx-mean)/scale),0,1)) for m in candidates]
    chosen=int(np.argmin([v["rmse"] for v in validation]))
    model=candidates[chosen]
    pred=np.clip(model.predict((tx-mean)/scale),0,1)
    test=scores(ty,pred)
    phases={phase:scores(ty[test_frame.phase.to_numpy()==phase],pred[test_frame.phase.to_numpy()==phase])
            for phase in NEURAL_PHASES if (test_frame.phase==phase).any()}
    model_path=out/"增量神经策略.json"
    payload={
        "schema":"ptdl_mpc_imitation_mlp_v1","phase_labels":list(NEURAL_PHASES),
        "feature_mean":mean.tolist(),"feature_scale":scale.tolist(),
        "target_mean":0.0,"target_scale":1.0,
        "coefs":[v.tolist() for v in model.coefs_],
        "intercepts":[v.tolist() for v in model.intercepts_],
        "phase_durations_s":summary["phase_duration_s"],
        "phase_soc_targets":FUZZY_SOC_TARGETS,
    }
    model_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    result={
        "training_scenarios":["nominal","recovery"],"validation_scenario":"battery_health",
        "test_scenario":"slow_approach","training_samples":len(y),
        "boundary_error_samples_reweighted":len(selected),
        "validation_candidates":validation,"selected_candidate":chosen,
        "test_metrics":test,"test_by_phase":phases,
        "scope":"MPC behavior imitation, boundary-error reweighting, no independent reward-trained controller",
    }
    (out/"增量训练评价.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    runs=[job_run((str(out),scenario+"_neural","neural",{"model":str(model_path)},scenario))
          for scenario in ("nominal","slow_approach","recovery","glide_30","battery_health")]
    (out/"神经策略闭环评价.json").write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
