"""Fit a shared smooth residual and a geometry-predicted discrete angle trend."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import copy
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

OUT = ROOT / "运行结果" / "尾迹分区趋势代理_20261015"
BASE = ROOT / "运行结果" / "物理特征代理_20261015" / "物理特征气动代理.json"


def fit_trend(x, y, wake, multiplier, kernel, gates):
    regimes = (abs(wake[:, 0]) >= .15).astype(int) + 2 * (abs(wake[:, 1]) >= .15)
    reference = float(np.mean(y[regimes == 0]))
    offsets = {str(i): multiplier * (float(np.mean(y[regimes == i])) - reference)
               for i in np.unique(regimes)}
    adjusted = y - np.array([offsets[str(i)] for i in regimes])
    base = fit_target(x, adjusted, "physics", kernel)[1]
    experts = {}
    for key, offset in offsets.items():
        experts[key] = dict(base, mean=base["mean"] + offset)
    return dict(kind="working-angle-gated", gates=gates, experts=experts,
                fallback=dict(base, mean=base["mean"] + float(np.mean(list(offsets.values())))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv")
    frame = frame[frame.wake_quality_pass]
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    y = frame.vspaero_oswald_efficiency.to_numpy(float)
    wake = np.array([
        pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv")
        .set_index("alpha_deg").loc[[-1., 1.], "cl_wake"].to_numpy(float)
        for _, row in frame.iterrows()
    ])
    folds = []
    for a, b in KFold(5, shuffle=True, random_state=20261015).split(x):
        gates = [fit_target(x[a], wake[a, i], "physics", "rbf")[1] for i in range(2)]
        folds.append((a, b, gates))

    def cv(job):
        multiplier, kernel = job
        prediction = np.zeros(len(y))
        for a, b, gates in folds:
            model = fit_trend(x[a], y[a], wake[a], multiplier, kernel, gates)
            for i in b:
                prediction[i] = gated_prediction(model, x[i:i+1])[0]
        error = prediction - y
        record = dict(multiplier=multiplier, kernel=kernel, r2=float(r2_score(y, prediction)),
                      rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))))
        print(json.dumps(record), flush=True)
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            OUT / f"折外预测_{multiplier}_{kernel}.csv", index=False)
        return record
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(cv, [(a, k) for a in (.25, .5, .75, 1.)
                                    for k in ("rbf", "matern32", "matern52")]))
    pd.DataFrame(records).to_csv(OUT / "训练集五折模型比较.csv", index=False)
    chosen = min(records, key=lambda item: item["rmse"])
    payload = json.loads(BASE.read_text(encoding="utf8"))
    gates = [fit_target(x, wake[:, i], "physics", "rbf")[1] for i in range(2)]
    payload["targets"][1] = fit_trend(x, y, wake, chosen["multiplier"], chosen["kernel"], gates)
    payload["selection"] = "Training-only CV; discrete working-angle mean trend and shared physical GP residual"
    TargetwiseAeroGP(payload).to_json(OUT / "尾迹分区趋势候选代理.json")
    (OUT / "训练集模型选择.json").write_text(json.dumps(chosen, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
