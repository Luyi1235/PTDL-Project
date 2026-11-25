"""Residual/uncertainty acquisition and a fresh sealed solver holdout."""

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"核心模型"))
from VSPAERO气动代理模型 import FEATURE_LABELS,load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign,geometry_constraints
from 训练部署VSPAERO_ARD_RBF代理 import fit_model,TARGETS,DEFAULT_DATA,DEFAULT_OUTPUT
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case,DEFAULT_PYTHON


def prediction(model,frame):
    return np.asarray([model.predict(AirframeDesign(**{k:float(r[k]) for k in FEATURE_LABELS}))
                       for _,r in frame.iterrows()])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DEFAULT_DATA)
    incumbent=load_vsp_surrogate(DEFAULT_OUTPUT)
    x=data[list(FEATURE_LABELS)].to_numpy(float)
    lo,hi=x.min(0),x.max(0)
    span=hi-lo
    rng=np.random.default_rng(20261013)
    pool=[]
    for values in rng.uniform(lo,hi,(6000,len(lo))):
        design=AirframeDesign(**dict(zip(FEATURE_LABELS,values)))
        if all(v<=1e-9 for v in geometry_constraints(design).values()):
            pool.append(values)
    pool=np.asarray(pool)
    distances=np.linalg.norm(((pool[:,None,:]-x[None,:,:])/span),axis=2)
    nearest=distances.min(1)
    # Holdout sealed using geometry alone, before any prediction or residual ranking.
    test_indices=[]
    coverage=nearest.copy()
    for _ in range(8):
        index=int(np.argmax(coverage))
        test_indices.append(index)
        coverage=np.minimum(coverage,np.linalg.norm((pool-pool[index])/span,axis=1))
        coverage[test_indices]=-1
    candidates=np.argsort(nearest)[-1200:]
    candidates=np.asarray([i for i in candidates if i not in test_indices])
    # Cross-fit residuals, so acquisition does not rely on in-sample GP errors.
    residual=np.zeros_like(data[list(TARGETS)].to_numpy(float))
    splits=np.arange(len(data))%5
    baseline_index=int(np.argmin(np.linalg.norm(x-np.asarray([getattr(AirframeDesign(),k) for k in FEATURE_LABELS]),axis=1)))
    for fold in range(5):
        test=(splits==fold)
        test[baseline_index]=False
        train=~test
        model=fit_model(data.loc[train])
        residual[test]=abs(prediction(model,data.loc[test])-data.loc[test,list(TARGETS)].to_numpy(float))
    target_scale=np.maximum(data[list(TARGETS)].std().to_numpy(float),1e-6)
    nearest_idx=np.argmin(distances[candidates],axis=1)
    uncertainty=[]
    for i in candidates:
        pred=incumbent.predict_with_uncertainty(AirframeDesign(**dict(zip(FEATURE_LABELS,pool[i]))))
        uncertainty.append(np.asarray(pred[2:])/target_scale)
    score=np.mean(residual[nearest_idx]/target_scale,axis=1)+np.mean(uncertainty,axis=1)
    selected=[]
    for _ in range(16):
        ranked=score*np.minimum(nearest[candidates],1)
        index=int(np.argmax(ranked))
        chosen=int(candidates[index])
        selected.append(chosen)
        nearest=np.minimum(nearest,np.linalg.norm((pool-pool[chosen])/span,axis=1))
        score[index]=-1
    fixed={k:float(data.iloc[0][k]) for k in ("propeller_diameter_m","payload_mass_kg","hybrid_system_mass_kg")}
    jobs=[]
    for role,indices in (("adaptive_train",selected),("sealed_test",test_indices)):
        for n,i in enumerate(indices):
            jobs.append((role,f"{role}_{n:02d}",{**fixed,**dict(zip(FEATURE_LABELS,pool[i]))}))
    (out/"锁定选点与门槛.json").write_text(json.dumps({
        "seed":20261013,"jobs":jobs,"selection":"5-fold residual plus GP uncertainty and coverage",
        "gate":{"per_target_rmse_ratio_max":1.05,"normalized_rmse_max":.65,"interval_90_coverage_min":.75},
        "test_use":"Fixed independent validation batch; excluded from training and acquisition; compare only after all 24 labels are valid.",
    },ensure_ascii=False,indent=2),encoding="utf-8")
    existing=out/"新增有效标签.csv"
    try:
        rows=pd.read_csv(existing).to_dict("records") if existing.exists() else []
    except pd.errors.EmptyDataError:
        rows=[]
    failures=[]
    completed={r["sample"] for r in rows}
    pending=[j for j in jobs if j[1] not in completed]
    def attempt_directory(name):
        path=out/"cases"/name
        attempt=0
        while path.exists():
            attempt+=1
            path=out/"cases"/f"{name}_retry_{attempt}"
        return path
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(_run_case,DEFAULT_PYTHON,d,n,attempt_directory(n),600,r,"residual_uncertainty_coverage"):n for r,n,d in pending}
        for future in as_completed(futures):
            try:
                row=future.result()
                rows.append(row)
                print(row["sample"],"valid",flush=True)
            except Exception as error:
                failures.append({"sample":futures[future],"error":repr(error)})
                print(failures[-1],flush=True)
            pd.DataFrame(rows).to_csv(out/"新增有效标签.csv",index=False)
            (out/"求解失败.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding="utf-8")
    labels=pd.DataFrame(rows)
    train=labels[labels.selection_role=="adaptive_train"]
    test=labels[labels.selection_role=="sealed_test"]
    if len(train)!=16 or len(test)!=8:
        raise RuntimeError("Insufficient valid labels for the locked deployment gate")
    combined=pd.concat([data,train],ignore_index=True)
    combined.to_csv(out/"扩充训练样本.csv",index=False)
    candidate=fit_model(combined)
    candidate_path=out/"候选气动代理.json"
    candidate.to_json(candidate_path)
    metrics={}
    truth=test[list(TARGETS)].to_numpy(float)
    for name,model in (("incumbent",incumbent),("candidate",candidate)):
        pred=prediction(model,test)
        std=np.asarray([model.predict_with_uncertainty(AirframeDesign(**{k:float(r[k]) for k in FEATURE_LABELS}))[2:]
                        for _,r in test.iterrows()])
        errors=pred-truth
        metrics[name]={
            "rmse":np.sqrt(np.mean(errors**2,axis=0)).tolist(),
            "r2":(1-np.sum(errors**2,axis=0)/np.maximum(np.sum((truth-truth.mean(0))**2,axis=0),1e-12)).tolist(),
            "coverage90":np.mean(abs(errors)<=1.645*std,axis=0).tolist(),
        }
    new=np.asarray(metrics["candidate"]["rmse"])
    old=np.asarray(metrics["incumbent"]["rmse"])
    accepted=bool(np.all(new<=1.05*old) and np.all(new/target_scale<=.65)
                  and np.all(np.asarray(metrics["candidate"]["coverage90"])>=.75))
    selected_path=candidate_path if accepted else DEFAULT_OUTPUT
    reload=load_vsp_surrogate(selected_path)
    smoke=reload.predict(AirframeDesign())
    (out/"代理部署决策.json").write_text(json.dumps({
        "candidate_accepted":accepted,"selected_model":str(selected_path),
        "metrics":metrics,"training_count":len(combined),"fresh_test_count":len(test),
        "reload_smoke":list(smoke),"default_replaced":False,
        "usage":"Use selected_model as explicit surrogate; incumbent retained on rejection",
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metrics),accepted,flush=True)


if __name__=="__main__":
    main()
