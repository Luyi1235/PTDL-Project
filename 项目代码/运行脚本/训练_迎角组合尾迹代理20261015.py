"""Share all training geometries across possible Ew working-angle combinations."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import TargetwiseAeroGP, gated_prediction
from VSPAERO气动代理模型 import FEATURE_LABELS
from 运行_分目标气动代理攻关 import fit_target

OUT = ROOT / "运行结果" / "迎角组合尾迹代理_20261015"
SOURCE = ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv"
BASE = ROOT / "运行结果" / "物理特征代理_20261015" / "物理特征气动代理.json"


def fit_shared(x, alternatives, gates, mode, kernel, fallback):
    return dict(kind="working-angle-gated", gates=gates, fallback=fallback, experts={
        str(i): fit_target(x, alternatives[:, i], mode, kernel)[1] for i in range(3)
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    frame = frame[frame.wake_quality_pass]
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    y = frame.vspaero_oswald_efficiency.to_numpy(float)
    wake, alternatives = [], []
    for _, row in frame.iterrows():
        polar = pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv").set_index("alpha_deg")
        values = polar.loc[[-3., 3., 5.], "e_wake"].to_numpy(float)
        wake.append(polar.loc[[-1., 1.], "cl_wake"].to_numpy(float))
        alternatives.append([np.median(values), np.median(np.r_[values, polar.loc[-1., "e_wake"]]),
                             np.median(np.r_[values, polar.loc[1., "e_wake"]])])
    wake, alternatives = np.asarray(wake), np.asarray(alternatives)
    splits = list(KFold(5, shuffle=True, random_state=20261015).split(x))
    folds = []
    for a, b in splits:
        gates = [fit_target(x[a], wake[a, i], "physics", "rbf")[1] for i in range(2)]
        fallback = fit_target(x[a], y[a], "physics", "matern32")[1]
        folds.append((a, b, gates, fallback))

    def cv(job):
        mode, kernel = job
        prediction = np.zeros(len(y))
        for a, b, gates, fallback in folds:
            model = fit_shared(x[a], alternatives[a], gates, mode, kernel, fallback)
            for i in b:
                prediction[i] = gated_prediction(model, x[i:i+1])[0]
        error = prediction - y
        record = dict(features=mode, kernel=kernel, r2=float(r2_score(y, prediction)),
                      rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))))
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            OUT / f"折外预测_{mode}_{kernel}.csv", index=False)
        print(json.dumps(record), flush=True)
        return record

    jobs = [(mode, kernel) for mode in ("physics", "dimensionless")
            for kernel in ("rbf", "matern32", "matern52")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(cv, jobs))
    pd.DataFrame(records).to_csv(OUT / "训练集五折模型比较.csv", index=False)
    chosen = min(records, key=lambda item: item["rmse"])
    baseline_rmse = float(pd.read_csv(BASE.parent / "训练集五折模型比较.csv")
                          .query("target == 1").rmse.min())
    decision = dict(chosen=chosen, baseline_rmse=baseline_rmse,
                    training_cv_improved=chosen["rmse"] < baseline_rmse,
                    label_definition="Original Ew median; all possible angle subsets trained separately",
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    (OUT / "训练集模型选择.json").write_text(json.dumps(decision, indent=2), encoding="utf8")
    gates = [fit_target(x, wake[:, i], "physics", "rbf")[1] for i in range(2)]
    fallback = fit_target(x, y, "physics", "matern32")[1]
    payload = json.loads(BASE.read_text(encoding="utf8"))
    payload["targets"][1] = fit_shared(x, alternatives, gates, chosen["features"], chosen["kernel"], fallback)
    payload["selection"] = "Minimum training-only five-fold RMSE; shared angle-subset targets"
    TargetwiseAeroGP(payload).to_json(OUT / "迎角组合候选代理.json")
    print("selection", json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
