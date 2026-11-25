"""Model the actual working-angle selection without changing the Ew label."""

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
from 分目标气动高斯过程 import TargetwiseAeroGP, scalar_prediction, gated_prediction
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign
from 运行_分目标气动代理攻关 import fit_target

OUT = ROOT / "运行结果" / "分区尾迹代理_20261015"
SOURCE = ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv"
BASE = ROOT / "运行结果" / "物理特征代理_20261015" / "物理特征气动代理.json"
TARGET = "vspaero_oswald_efficiency"


def fit_gated(x, y, wake, kernel):
    regimes = (abs(wake[:, 0]) >= .15).astype(int) + 2 * (abs(wake[:, 1]) >= .15)
    gates = [fit_target(x, wake[:, i], "physics", "rbf")[1] for i in range(2)]
    fallback = fit_target(x, y, "physics", kernel)[1]
    experts = {
        str(regime): fit_target(x[regimes == regime], y[regimes == regime], "physics", kernel)[1]
        for regime in np.unique(regimes) if (regimes == regime).sum() >= 12
    }
    return dict(kind="working-angle-gated", gates=gates, experts=experts, fallback=fallback)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    valid = frame.wake_quality_pass.to_numpy(bool)
    train = frame.loc[valid]
    x, y = train[list(FEATURE_LABELS)].to_numpy(float), train[TARGET].to_numpy(float)
    wake, audit = [], []
    for _, row in train.iterrows():
        polar = pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv")
        gate = [float(polar.loc[np.isclose(polar.alpha_deg, a), "cl_wake"].iloc[0]) for a in (-1, 1)]
        wake.append(gate)
        working = polar.cl_wake.abs() >= .15
        audit.append(dict(sample=row["sample"], working_count=int(working.sum()),
                          label=float(row[TARGET]),
                          angle_efficiency_range=float(polar.loc[working, "e_wake"].max()
                                                       - polar.loc[working, "e_wake"].min())))
    wake = np.asarray(wake)
    pd.DataFrame(audit).to_csv(OUT / "工作迎角与标签诊断.csv", index=False)

    def cv(kernel):
        prediction = np.zeros(len(y))
        predicted_regime = np.zeros(len(y), dtype=int)
        for a, b in KFold(5, shuffle=True, random_state=20261015).split(x):
            model = fit_gated(x[a], y[a], wake[a], kernel)
            for index in b:
                prediction[index] = gated_prediction(model, x[index:index+1])[0]
                gate = [scalar_prediction(m, x[index:index+1])[0] for m in model["gates"]]
                predicted_regime[index] = sum(1 << i for i, v in enumerate(gate) if abs(v) >= .15)
        error = prediction - y
        actual_regime = (abs(wake[:, 0]) >= .15).astype(int) + 2 * (abs(wake[:, 1]) >= .15)
        record = dict(kernel=kernel, r2=float(r2_score(y, prediction)),
                      mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                      regime_accuracy=float(np.mean(predicted_regime == actual_regime)))
        pd.DataFrame(dict(actual=y, prediction=prediction, actual_regime=actual_regime,
                          predicted_regime=predicted_regime)).to_csv(OUT / f"折外预测_{kernel}.csv", index=False)
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(cv, ("rbf", "matern32", "matern52")))
    pd.DataFrame(records).to_csv(OUT / "训练集五折模型比较.csv", index=False)
    chosen = min(records, key=lambda item: item["rmse"])
    payload = json.loads(BASE.read_text(encoding="utf8"))
    payload["targets"][1] = fit_gated(x, y, wake, chosen["kernel"])
    payload["selection"] = "Training-only five-fold RMSE; geometry-predicted working-angle gates"
    payload["label_definition"] = "Original median Ew with abs(CLiw)>=0.15, unchanged"
    path = OUT / "分区尾迹气动代理.json"
    TargetwiseAeroGP(payload).to_json(path)
    # Freeze the chosen model before opening held-out labels.
    (OUT / "验证前模型锁定.json").write_text(json.dumps(dict(
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), chosen=chosen,
        training_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        evaluation_policy="One selected model; same held-out batch for paired comparison. No refit after evaluation.",
    ), indent=2), encoding="utf8")
    test = pd.read_csv(ROOT / "运行结果" / "物理特征代理_20261015" / "新独立验证_逐点.csv")
    model = load_vsp_surrogate(path)
    estimates = np.asarray([
        model.predict(AirframeDesign(**{k: float(row[k]) for k in FEATURE_LABELS}))
        for _, row in test.iterrows()
    ])
    metrics = []
    for i, target in enumerate(("vspaero_cl_alpha_per_rad", TARGET)):
        mask = np.isfinite(test[target].to_numpy(float))
        if i:
            mask &= test.wake_quality_pass.to_numpy(bool)
        actual = test.loc[mask, target].to_numpy(float)
        error = estimates[mask, i] - actual
        metrics.append(dict(target=target, count=int(mask.sum()), r2=float(r2_score(actual, estimates[mask, i])),
                            mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2)))))
        test[f"gated_prediction_{i}"] = estimates[:, i]
    test.to_csv(OUT / "同批独立验证_逐点.csv", index=False)
    (OUT / "同批独立验证.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print("heldout", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
