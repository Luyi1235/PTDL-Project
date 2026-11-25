"""Retire old tests into development data, then fit for a newly sealed solver test."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import TargetwiseAeroGP, scalar_prediction, gated_prediction
from VSPAERO气动代理模型 import FEATURE_LABELS
from 运行_分目标气动代理攻关 import fit_target
from 训练_尾迹分区趋势代理20261015 import fit_trend
from 检查_尾迹标签质量 import polar_quality

OUT = ROOT / "运行结果" / "增量构型气动代理_20261015"
PRIOR = ROOT / "运行结果" / "分目标代理211点_20261013"
BASE = ROOT / "运行结果" / "物理特征代理_20261015"
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--extra-csv", type=Path)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PRIOR / "训练样本211点.csv")
    prior_test = pd.read_csv(PRIOR / "新增有效标签.csv").query("selection_role == 'test'").copy()
    prior_test["wake_quality_pass"] = [
        polar_quality(Path(row.geometry_file).parent / "vspaero_polar.csv")["quality_pass"]
        for _, row in prior_test.iterrows()
    ]
    recent_test = pd.read_csv(BASE / "新独立验证_逐点.csv")
    columns = list(FEATURE_LABELS) + list(TARGETS) + ["geometry_file", "wake_quality_pass", "sample"]
    frame = pd.concat([f[columns].assign(development_source=name) for name, f in
                       (("original_training", train), ("retired_test_20261013", prior_test),
                        ("retired_test_20261015", recent_test))], ignore_index=True)
    if args.extra_csv:
        extra = pd.read_csv(args.extra_csv)
        frame = pd.concat([frame, extra[columns].assign(
            development_source="active_solver_20261015")], ignore_index=True)
    if frame.duplicated(list(FEATURE_LABELS)).any():
        raise RuntimeError("Repeated geometry requires a label consistency audit")
    frame.to_csv(OUT / "增量训练样本.csv", index=False)
    folds = {}
    arrays = {}
    for t, target in enumerate(TARGETS):
        mask = np.isfinite(frame[target].to_numpy(float))
        if t:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        subset = frame.loc[mask]
        x = subset[list(FEATURE_LABELS)].to_numpy(float)
        y = subset[target].to_numpy(float)
        wake = None
        if t:
            wake = np.array([
                pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv").set_index("alpha_deg")
                .loc[[-1., 1.], "cl_wake"].to_numpy(float) for _, row in subset.iterrows()
            ])
        arrays[t] = (x, y, wake)
        folds[t] = []
        for a, b in KFold(5, shuffle=True, random_state=2026101506).split(x):
            gates = [fit_target(x[a], wake[a, i], "physics", "rbf")[1] for i in range(2)] if t else None
            folds[t].append((a, b, gates))
    jobs = [(0, "physics", "rbf", 0.), (0, "physics", "matern52", 0.),
            (0, "dimensionless", "rbf", 0.), (1, "physics", "matern32", 0.),
            (1, "physics", "matern52", 0.), (1, "physics", "matern32", .25),
            (1, "physics", "matern32", .5)]

    def cv(job):
        t, mode, kernel, multiplier = job
        x, y, wake = arrays[t]
        predictions = np.zeros(len(y))
        for a, b, gates in folds[t]:
            if multiplier:
                model = fit_trend(x[a], y[a], wake[a], multiplier, kernel, gates)
            else:
                model = fit_target(x[a], y[a], mode, kernel)[1]
            for index in b:
                predictions[index] = (gated_prediction if multiplier else scalar_prediction)(
                    model, x[index:index+1])[0]
        error = predictions - y
        record = dict(target=t, features=mode, kernel=kernel, multiplier=multiplier,
                      count=len(y), mae=float(np.mean(abs(error))),
                      rmse=float(np.sqrt(np.mean(error**2))),
                      r2=float(1 - np.sum(error**2) / np.sum((y-y.mean())**2)))
        print(json.dumps(record), flush=True)
        return record
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(cv, jobs))
    pd.DataFrame(records).to_csv(OUT / "训练集五折模型比较.csv", index=False)
    payload = json.loads((BASE / "物理特征气动代理.json").read_text(encoding="utf8"))
    chosen_records = []
    for t in range(2):
        chosen = min((r for r in records if r["target"] == t), key=lambda r: r["rmse"])
        x, y, wake = arrays[t]
        if chosen["multiplier"]:
            gates = [fit_target(x, wake[:, i], "physics", "rbf")[1] for i in range(2)]
            target = fit_trend(x, y, wake, chosen["multiplier"], chosen["kernel"], gates)
        else:
            target = fit_target(x, y, chosen["features"], chosen["kernel"])[1]
        payload["targets"][t] = target
        chosen_records.append(chosen)
    payload["training_samples"] = len(frame)
    payload["selection"] = "Training-only five-fold RMSE; old test batches retired into development"
    payload["development_sources"] = frame.development_source.value_counts().to_dict()
    TargetwiseAeroGP(payload).to_json(OUT / "增量构型气动代理.json")
    (OUT / "训练集模型选择.json").write_text(json.dumps(dict(
        selected=chosen_records, training_samples=len(frame),
        previous_tests="Now development data, not independent validation for this model",
    ), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
