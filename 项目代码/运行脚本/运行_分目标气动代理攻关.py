"""Training-only model selection, locked new solver test, and explicit deployment gate."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import TargetwiseAeroGP, feature_map, reconstructed_ew
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints
from 训练部署VSPAERO_ARD_RBF代理 import TARGETS, DEFAULT_OUTPUT, DEFAULT_DATA
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, _load_saved_case, _extract_worker_result, DEFAULT_PYTHON
from 运行_代理更新部署门槛 import prediction
from 检查_尾迹标签质量 import polar_quality

PRIOR = ROOT / "运行结果" / "蓝色攻关_代理更新门槛_20261013"
TRAIN = PRIOR / "扩充训练样本.csv"
CONFIGS = [(mode, kernel) for mode in ("raw", "dimensionless")
           for kernel in ("rbf", "matern32", "matern52")]


def fit_target(x, y, mode, kind):
    features = feature_map(x, mode)
    mean, scale = features.mean(0), np.maximum(features.std(0), 1e-9)
    z = (features-mean)/scale
    covariance = RBF(np.ones(z.shape[1]), (.05, 50)) if kind == "rbf" else Matern(
        np.ones(z.shape[1]), (.05, 50), nu=1.5 if kind == "matern32" else 2.5)
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (.01, 100))*covariance + WhiteKernel(.001, (1e-8, 1)),
        alpha=1e-8, normalize_y=True, n_restarts_optimizer=3, random_state=20261013)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(z, y)
    matrix = gp.kernel_(z) + np.eye(len(z))*gp.alpha
    payload = dict(features=mode, kernel=kind, feature_mean=mean.tolist(), feature_scale=scale.tolist(),
                   train=z.tolist(), mean=float(gp._y_train_mean), scale=float(gp._y_train_std),
                   amplitude=float(gp.kernel_.k1.k1.constant_value),
                   length=np.asarray(gp.kernel_.k1.k2.length_scale).tolist(),
                   noise=float(gp.kernel_.k2.noise_level), dual=gp.alpha_.tolist(),
                   inverse=np.linalg.solve(matrix, np.eye(len(z))).tolist(), interval_scale=1.)
    return gp, payload


def polar_coefficients(frame):
    coefficients=[]
    for _,row in frame.iterrows():
        polar=pd.read_csv(Path(row.geometry_file).parent/"vspaero_polar.csv")
        coefficients.append(np.concatenate([np.polyfit(polar.alpha_deg,polar[k],2)
                                            for k in ("cl","cl_wake","cd_induced_wake")]))
    return np.asarray(coefficients)


def fit_polar(x, coefficients, mode):
    return [fit_target(x,coefficients[:,i],mode,"rbf") for i in range(9)]


def select_model(frame, out):
    all_x, all_y = frame[list(FEATURE_LABELS)].to_numpy(float), frame[list(TARGETS)].to_numpy(float)
    results, selected, residuals = [], [], []
    for target in range(2):
        valid=np.ones(len(frame),dtype=bool) if target==0 else frame.wake_quality_pass.to_numpy(bool)
        x,y=all_x[valid],all_y[valid]
        candidates = []
        for mode, kind in CONFIGS:
            pred, std = np.zeros(len(x)), np.zeros(len(x))
            for train, test in KFold(5, shuffle=True, random_state=20261013).split(x):
                gp, params = fit_target(x[train], y[train, target], mode, kind)
                z = (feature_map(x[test], mode)-params["feature_mean"])/params["feature_scale"]
                pred[test], std[test] = gp.predict(z, return_std=True)
            err = pred-y[:,target]
            rmse = float(np.sqrt(np.mean(err**2)))
            interval_scale = max(1., float(np.quantile(abs(err)/np.maximum(std, 1e-9), .90, method="higher")/1.645))
            record = dict(target=TARGETS[target], training_count=len(x), features=mode, kernel=kind, rmse=rmse,
                          r2=float(1-np.sum(err**2)/np.sum((y[:,target]-y[:,target].mean())**2)),
                          raw_coverage90=float(np.mean(abs(err)<=1.645*std)), interval_scale=interval_scale)
            results.append(record)
            candidates.append((rmse, record, err))
            pd.DataFrame(results).to_csv(out/"训练集五折模型比较.csv", index=False)
            print(json.dumps(record), flush=True)
        _, chosen, err = min(candidates, key=lambda a:a[0])
        _, params = fit_target(x, y[:,target], chosen["features"], chosen["kernel"])
        params["interval_scale"] = chosen["interval_scale"]
        if target==1:
            coefficients=polar_coefficients(frame.loc[valid])
            scalar_oof=y[:,target]+err
            for mode in ("raw","dimensionless"):
                pred=np.zeros(len(x))
                fallbacks=0
                for train,test in KFold(5,shuffle=True,random_state=20261013).split(x):
                    fits=fit_polar(x[train],coefficients[train],mode)
                    predicted=[]
                    for gp,component in fits:
                        z=(feature_map(x[test],mode)-component["feature_mean"])/component["feature_scale"]
                        predicted.append(gp.predict(z))
                    for index,values in zip(test,np.asarray(predicted).T):
                        estimate=reconstructed_ew(values,x[index,1])
                        if not np.isfinite(estimate):
                            estimate=scalar_oof[index]
                            fallbacks+=1
                        pred[index]=estimate
                error=pred-y[:,target]
                rmse=float(np.sqrt(np.mean(error**2)))
                interval_std=max(1e-9,float(np.quantile(abs(error),.9,method="higher")/1.645))
                record=dict(target=TARGETS[target],training_count=len(x),features=mode,kernel="polar_rbf",
                            rmse=rmse,r2=float(1-np.sum(error**2)/np.sum((y[:,target]-y[:,target].mean())**2)),
                            raw_coverage90=float(np.mean(abs(error)<=1.645*interval_std)),interval_scale=1.,
                            fallback_count=fallbacks)
                results.append(record)
                pd.DataFrame(results).to_csv(out/"训练集五折模型比较.csv",index=False)
                print(json.dumps(record),flush=True)
                candidates.append((rmse,record,error))
            best_rmse, best_record, best_error=min(candidates,key=lambda a:a[0])
            if best_record["kernel"]=="polar_rbf":
                components=[component for gp,component in fit_polar(x,coefficients,best_record["features"])]
                params=dict(kind="polar",features=best_record["features"],kernel="polar_rbf",
                            components=components,fallback=params,
                            interval_std=max(1e-9,float(np.quantile(abs(best_error),.9,method="higher")/1.645)),
                            interval_note="Training-only OOF absolute-error 90th percentile; fixed empirical band, not latent GP posterior",
                            train_count=len(x))
                err=best_error
        selected.append(params)
        complete=np.full(len(frame),np.nan)
        complete[valid]=err
        residuals.append(complete)
    baseline = np.array([getattr(AirframeDesign(), k) for k in FEATURE_LABELS])
    idx = np.flatnonzero(np.linalg.norm(all_x-baseline, axis=1)<1e-8)
    if len(idx) != 1:
        raise ValueError("Unique baseline required")
    payload = dict(schema="ptdl_targetwise_gp_v1", feature_labels=list(FEATURE_LABELS),
                   training_samples=len(frame), baseline_targets=all_y[idx[0]].tolist(), targets=selected,
                   uncertainty="Scalar GP includes nugget and OOF scale; polar reconstruction uses a fixed OOF absolute-error 90% band",
                   selection="Per-target minimum five-fold RMSE; test labels excluded")
    model = TargetwiseAeroGP(payload)
    model.to_json(out/"分目标候选代理.json")
    pd.DataFrame(np.asarray(residuals).T, columns=TARGETS).to_csv(out/"训练集折外残差.csv", index=False)
    return model


def annotate_quality(frame):
    records=[]
    for _,row in frame.iterrows():
        quality=polar_quality(Path(row.geometry_file).parent/"vspaero_polar.csv")
        records.append({**row.to_dict(),"wake_quality_pass":quality["quality_pass"],
                        "wake_valid_points":quality["valid_points"],
                        "wake_negative_working_drag_points":quality["negative_working_drag_points"]})
    return pd.DataFrame(records)


def repair_quality(frame, out, workers):
    frame=annotate_quality(frame)
    audit=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={}
        for index,row in frame[~frame.wake_quality_pass].iterrows():
            name=f"row_{index:03d}_{row['sample']}_wake120"
            folder=out/"quality_repair"/name
            design={k:float(row[k]) for k in (*FEATURE_LABELS,"propeller_diameter_m","payload_mass_kg","hybrid_system_mass_kg")}
            role=str(row.get("selection_role","training"))
            if folder.exists():
                try:
                    _extract_worker_result((folder/"worker.log").read_text(encoding="utf-8"))
                    saved=_load_saved_case(row["sample"],design,folder,role,"quality_repair_wake120",(0.,float("inf")))
                    passed=polar_quality(folder/"vspaero_polar.csv")["quality_pass"]
                    audit.append(dict(index=int(index),sample=row["sample"],accepted=passed,source="saved",old_geometry=row.geometry_file,new_geometry=saved["geometry_file"]))
                    if passed:
                        for key,value in saved.items():
                            frame.loc[index,key]=value
                    continue
                except (OSError,RuntimeError,ValueError):
                    suffix=1
                    while folder.with_name(name+f"_retry{suffix}").exists():
                        suffix+=1
                    folder=folder.with_name(name+f"_retry{suffix}")
            futures[executor.submit(_run_case,DEFAULT_PYTHON,design,row["sample"],folder,900,role,
                                    "quality_repair_wake120",(0.,float("inf")),120)]=(index,folder,row)
        for future in as_completed(futures):
            index,folder,old=futures[future]
            try:
                replacement=future.result()
                passed=polar_quality(folder/"vspaero_polar.csv")["quality_pass"]
                audit.append(dict(index=int(index),sample=old["sample"],accepted=passed,
                                  old_geometry=old.geometry_file,new_geometry=replacement["geometry_file"]))
                if passed:
                    for key,value in replacement.items():
                        frame.loc[index,key]=value
            except Exception as error:
                audit.append(dict(index=int(index),sample=old["sample"],accepted=False,error=repr(error)))
            (out/"极曲线修复记录.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
            print("quality_repair",old["sample"],audit[-1]["accepted"],flush=True)
    (out/"极曲线修复记录.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    return annotate_quality(frame)


def lock_jobs(out, frame):
    path = out/"锁定选点.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["jobs"]
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    old_test = pd.read_csv(PRIOR/"新增有效标签.csv")
    old_test = old_test[old_test.selection_role=="sealed_test"]
    excluded = np.vstack((x, old_test[list(FEATURE_LABELS)].to_numpy(float)))
    lo, hi = x.min(0), x.max(0)
    span = hi-lo
    rng = np.random.default_rng(2026101302)
    pool = np.array([v for v in rng.uniform(lo,hi,(12000,7))
                     if all(g<=1e-9 for g in geometry_constraints(AirframeDesign(**dict(zip(FEATURE_LABELS,v)))).values())])
    distance = np.linalg.norm((pool[:,None,:]-excluded[None,:,:])/span,axis=2).min(1)
    # Half random and half maximin tests: lock geometry before solver labels.
    test = list(rng.choice(np.flatnonzero(distance>1e-5), 12, replace=False))
    for i in test:
        distance = np.minimum(distance, np.linalg.norm((pool-pool[i])/span,axis=1))
    distance[test] = -1
    for _ in range(12):
        i = int(distance.argmax())
        test.append(i)
        distance = np.minimum(distance, np.linalg.norm((pool-pool[i])/span,axis=1))
        distance[test] = -1
    train = []
    for _ in range(64):
        i = int(distance.argmax())
        train.append(i)
        distance = np.minimum(distance, np.linalg.norm((pool-pool[i])/span,axis=1))
        distance[test+train] = -1
    fixed={k:float(frame.iloc[0][k]) for k in ("propeller_diameter_m","payload_mass_kg","hybrid_system_mass_kg")}
    jobs = [(role, f"{role}_{n:03d}", {**fixed,**dict(zip(FEATURE_LABELS,pool[i]))})
            for role,indices in (("train",train),("test",test)) for n,i in enumerate(indices)]
    path.write_text(json.dumps(dict(jobs=jobs,seed=2026101302,train_count=64,test_count=24,
                                   gate=dict(rmse_ratio_max=1.05,normalized_rmse_max=.65,coverage90_min=.75),
                                   test_use="Single final evaluation after selecting and freezing model on training-only CV",
                                   test_design="12 IID admissible box draws and 12 maximin points"),
                               ensure_ascii=False,indent=2),encoding="utf-8")
    return jobs


def collect(out, jobs, workers):
    path=out/"新增有效标签.csv"
    rows=pd.read_csv(path).to_dict("records") if path.exists() else []
    failed_path=out/"未求得标签的锁定构型.json"
    exhausted={r["sample"] for r in json.loads(failed_path.read_text(encoding="utf-8"))
               if r["role"]=="test"} if failed_path.exists() else set()
    # Keep valid low-efficiency solver labels, separately from deployment clamps.
    done={r["sample"] for r in rows}
    for role,name,design in jobs:
        if name in done:
            continue
        for folder in sorted((out/"cases").glob(name+"*")):
            try:
                _extract_worker_result((folder/"worker.log").read_text(encoding="utf-8"))
                row=_load_saved_case(name,design,folder,role,"locked_space_filling",(0.,float("inf")))
                row["outside_previous_label_envelope"]=not (.45<=row[TARGETS[1]]<=1.4)
                rows.append(row)
                break
            except (RuntimeError, OSError, ValueError):
                continue
    if rows:
        pd.DataFrame(rows).to_csv(path,index=False)
    for attempt in range(4):
        done={r["sample"] for r in rows}
        pending=[j for j in jobs if j[1] not in done and j[1] not in exhausted]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=workers if attempt<2 else 1) as executor:
            futures={}
            for role,name,design in pending:
                folder=out/"cases"/name
                suffix=0
                while folder.exists():
                    suffix+=1
                    folder=out/"cases"/f"{name}_retry{suffix}"
                futures[executor.submit(_run_case,DEFAULT_PYTHON,design,name,folder,600,role,
                                        "locked_space_filling",(0.,float("inf")),30 if attempt<2 else 120)]=name
            for future in as_completed(futures):
                try:
                    rows.append(future.result())
                    pd.DataFrame(rows).to_csv(path,index=False)
                    print("label",len(rows),len(jobs),rows[-1]["sample"],flush=True)
                except Exception as error:
                    with (out/"求解重试记录.jsonl").open("a",encoding="utf-8") as log:
                        log.write(json.dumps(dict(sample=futures[future],attempt=attempt,error=repr(error)))+"\n")
    done={r["sample"] for r in rows}
    missing=[dict(role=r,sample=n,design=d) for r,n,d in jobs if n not in done]
    (out/"未求得标签的锁定构型.json").write_text(json.dumps(missing,ensure_ascii=False,indent=2),encoding="utf-8")
    if any(r["role"]=="train" for r in missing) or sum(r["role"]=="test" for r in missing)>4:
        raise RuntimeError("Training collection incomplete or fewer than 20 solver-valid tests")
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=3)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    (out/"极曲线质量预检规则.json").write_text(json.dumps(dict(
        basis="Rule derived from training-polar audit before new-test scoring",
        rule="Five finite polar points; >=3 abs(CLiw)>=0.15 points with positive CDiw and Ew; no nonpositive CDiw in that working subset",
        training="Retain CL-slope rows; exclude invalid wake labels from Ew training only",
        testing="Identical quality mask for incumbent and candidate; retain failed records; require >=20 scorable cases per target",
        scalar_gate="Original RMSE ratio 1.05, normalized RMSE 0.65, empirical coverage 0.75 retained",
        normalization_data=str(DEFAULT_DATA),
    ),ensure_ascii=False,indent=2),encoding="utf-8")
    frame=pd.read_csv(TRAIN)
    jobs=lock_jobs(out,frame)
    labels=collect(out,jobs,args.workers)
    all_rows=pd.concat([frame,labels],ignore_index=True)
    all_rows=repair_quality(all_rows,out,args.workers)
    all_rows.to_csv(out/"全部构型与质量审计.csv",index=False)
    combined=all_rows[all_rows.selection_role!="test"].reset_index(drop=True)
    combined.to_csv(out/"训练样本211点.csv",index=False)
    candidate=select_model(combined,out)
    candidate_path=out/"分目标候选代理.json"
    frozen_hash=hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    test=all_rows[all_rows.selection_role=="test"].reset_index(drop=True)
    test.to_csv(out/"独立验证极曲线质量.csv",index=False)
    valid=np.column_stack((np.ones(len(test),dtype=bool),test.wake_quality_pass.to_numpy(bool)))
    truth=test[list(TARGETS)].to_numpy(float)
    metrics={}
    predictions=test[["sample",*FEATURE_LABELS,*TARGETS]].copy()
    for name,model in (("incumbent",load_vsp_surrogate(DEFAULT_OUTPUT)),("candidate",candidate)):
        pred=prediction(model,test)
        std=np.array([model.predict_with_uncertainty(AirframeDesign(**{k:float(r[k]) for k in FEATURE_LABELS}))[2:]
                      for _,r in test.iterrows()])
        error=pred-truth
        metrics[name]=dict(rmse=[],mae=[],r2=[],coverage90=[],mean_interval_width90=[])
        for i in range(2):
            mask=valid[:,i]
            e,t,s=error[mask,i],truth[mask,i],std[mask,i]
            metrics[name]["rmse"].append(float(np.sqrt(np.mean(e**2))))
            metrics[name]["mae"].append(float(np.mean(abs(e))))
            metrics[name]["r2"].append(float(1-np.sum(e**2)/max(np.sum((t-t.mean())**2),1e-12)))
            metrics[name]["coverage90"].append(float(np.mean(abs(e)<=1.645*s)))
            metrics[name]["mean_interval_width90"].append(float(np.mean(3.29*s)))
        for i,target in enumerate(TARGETS):
            predictions[f"{name}_{target}"]=pred[:,i]
            predictions[f"{name}_{target}_std"]=std[:,i]
            predictions[f"scorable_{target}"]=valid[:,i]
    new=np.array(metrics["candidate"]["rmse"])
    old=np.array(metrics["incumbent"]["rmse"])
    scale=pd.read_csv(DEFAULT_DATA)[list(TARGETS)].std().to_numpy(float)
    checks=dict(rmse_noninferiority=(new<=1.05*old).tolist(),
                normalized_accuracy=(new/scale<=.65).tolist(),
                interval_coverage=(np.array(metrics["candidate"]["coverage90"])>=.75).tolist(),
                valid_test_count=(valid.sum(0)>=20).tolist())
    accepted=all(all(v) for v in checks.values())
    result=dict(candidate_accepted=accepted,checks=checks,metrics=metrics,training_count=len(combined),
                per_target_training_count=[m["train_count"] if m.get("kind")=="polar" else len(m["train"]) for m in candidate.payload["targets"]],
                scored_test_counts=valid.sum(0).tolist(),
                fresh_test_count=sum(role=="test" for role,name,design in jobs),
                collected_test_count=len(test),frozen_model_sha256=frozen_hash,
                selected_model=str(candidate_path if accepted else DEFAULT_OUTPUT),default_replaced=False,
                uncertainty_comparison="Incumbent latent intervals; candidate scalar OOF-scaled or polar OOF fixed-width bands; widths reported",
                test_use="No test-based model selection, refit or threshold change after this evaluation")
    predictions.to_csv(out/"独立验证逐点预测.csv",index=False)
    (out/"部署门槛决策.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
