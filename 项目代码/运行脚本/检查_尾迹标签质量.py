"""Audit full polar quality and probe wake-iteration sensitivity at fixed geometry."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"核心模型"))
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case,DEFAULT_PYTHON,FEATURE_LABELS


def polar_quality(path):
    polar=pd.read_csv(path)
    finite=np.isfinite(polar.to_numpy(float)).all()
    working=polar.cl_wake.abs()>=.15
    valid=working & (polar.cd_induced_wake>0) & (polar.e_wake>0)
    return dict(quality_pass=bool(finite and len(polar)>=5 and valid.sum()>=3 and valid.sum()==working.sum()),
                working_points=int(working.sum()),valid_points=int(valid.sum()),
                negative_working_drag_points=int((working & (polar.cd_induced_wake<=0)).sum()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--probe",action="store_true")
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.data)
    records=[]
    for _,row in frame.iterrows():
        record=dict(sample=row["sample"],**polar_quality(Path(row.geometry_file).parent/"vspaero_polar.csv"))
        records.append(record)
    pd.DataFrame(records).to_csv(out/"全极曲线质量.csv",index=False)
    failed={r["sample"] for r in records if not r["quality_pass"]}
    print("quality",len(frame)-len(failed),"/",len(frame),flush=True)
    if not args.probe:
        return
    jobs=[]
    for _,row in frame[frame["sample"].isin(failed)].head(3).iterrows():
        design={k:float(row[k]) for k in (*FEATURE_LABELS,"propeller_diameter_m","payload_mass_kg","hybrid_system_mass_kg")}
        for iterations in (60,120):
            name=f"{row['sample']}_wake{iterations}"
            jobs.append((design,name,iterations))
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(_run_case,DEFAULT_PYTHON,d,n,out/n,900,"quality_probe","wake_convergence",(0,float("inf")),w): (n,w)
                 for d,n,w in jobs}
        for future in as_completed(futures):
            name,wake=futures[future]
            try:
                row=future.result()
                rows.append(dict(**row,wake_iterations=wake,**polar_quality(out/name/"vspaero_polar.csv")))
                print(name,rows[-1]["quality_pass"],flush=True)
            except Exception as error:
                rows.append(dict(sample=name,wake_iterations=wake,error=repr(error),quality_pass=False))
            pd.DataFrame(rows).to_csv(out/"尾迹迭代探测.csv",index=False)


if __name__=="__main__":
    main()
