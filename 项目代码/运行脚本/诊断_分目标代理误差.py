"""Training-label diagnostics and final frozen-test visualizations."""

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
from 训练部署VSPAERO_ARD_RBF代理 import TARGETS


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--study",type=Path,required=True)
    args=p.parse_args()
    out=args.study
    frame=pd.read_csv(out/"训练样本211点.csv")
    residual=pd.read_csv(out/"训练集折外残差.csv")
    rows=[]
    for i,row in frame.iterrows():
        polar_path=Path(row.geometry_file).parent/"vspaero_polar.csv"
        if not polar_path.exists():
            continue
        f=pd.read_csv(polar_path)
        eligible=(f.cl_wake.abs()>=.15)&(f.e_wake>0)
        alpha=np.radians(f.alpha_deg.to_numpy())
        fit=np.polyfit(alpha,f.cl,1)
        cl_error=f.cl-np.polyval(fit,alpha)
        rows.append(dict(sample=row["sample"],eligible_polar_points=int(eligible.sum()),
                         wake_quality_pass=bool(row.wake_quality_pass),
                         threshold_distance=float(abs(f.cl_wake.abs()-.15).min()),
                         ew_range=float(f.loc[eligible,"e_wake"].max()-f.loc[eligible,"e_wake"].min()),
                         cl_linearity_rmse=float(np.sqrt(np.mean(cl_error**2))),
                         ew=float(row[TARGETS[1]]),
                         cl_oof_error=float(residual.iloc[i,0]),
                         ew_oof_error=float(residual.iloc[i,1])))
    diagnostic=pd.DataFrame(rows)
    diagnostic.to_csv(out/"训练标签与折外误差诊断.csv",index=False)
    summary=dict(
        labels_with_available_polars=len(diagnostic),
        eligibility_counts={str(k):int(v) for k,v in diagnostic.eligible_polar_points.value_counts().items()},
        near_threshold_count=int((diagnostic.threshold_distance<.03).sum()),
        median_ew_polar_range=float(diagnostic.ew_range.median()),
        maximum_ew_polar_range=float(diagnostic.ew_range.max()),
        maximum_cl_linearity_rmse=float(diagnostic.cl_linearity_rmse.max()),
        usable_wake_training_count=int(diagnostic.wake_quality_pass.sum()),
        low_ew_training_count=int((frame[TARGETS[1]]<.45).sum()),
        interpretation="Ew remains the legacy thresholded polar median. Threshold crossings can change membership discontinuously; this is a candidate error mechanism, not a causal attribution of all residuals.",
        preserved="Wake Ew definition, raw solver labels, frozen test and deployment thresholds retained",
    )
    (out/"误差机制诊断.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    test=pd.read_csv(out/"独立验证逐点预测.csv")
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for i,(ax,target,label) in enumerate(zip(axes,TARGETS,("升力斜率","尾迹效率因子"))):
        scored=test[test[f"scorable_{target}"].astype(bool)]
        truth=scored[target].to_numpy()
        for model,color,title in (("incumbent","#a96638","原131点代理"),("candidate","#1b776b","211点分目标代理")):
            ax.scatter(truth,scored[f"{model}_{target}"],color=color,label=title,s=25,alpha=.85)
        lo=min(truth.min(),scored[f"candidate_{target}"].min(),scored[f"incumbent_{target}"].min())
        hi=max(truth.max(),scored[f"candidate_{target}"].max(),scored[f"incumbent_{target}"].max())
        ax.plot([lo,hi],[lo,hi],"--",color="#666666",linewidth=1)
        ax.set_title(f"{label}  可评分 {len(scored)}/24")
        ax.set_xlabel("独立构型 VSPAERO 值")
        ax.set_ylabel("代理预测值")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out/"独立验证预测对比.png",dpi=170)
    plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
