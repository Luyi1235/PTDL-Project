"""Attribute OOF Ew reconstruction errors to lift, gating and wake drag."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import scalar_mean
from VSPAERO气动代理模型 import FEATURE_LABELS
from 研究_约束极曲线代理20261016 import fit, reconstruct, SOURCE, OUT, ANGLES, T
from 训练_主动增量混合代理20261016 import metric


def main():
    frame = pd.read_csv(SOURCE / "增量训练样本.csv")
    frame = frame.loc[frame.wake_quality_pass].copy()
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    y = frame.vspaero_oswald_efficiency.to_numpy(float)
    coefficients = []
    for _, row in frame.iterrows():
        polar = pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv").set_index("alpha_deg").loc[ANGLES]
        coefficients.append(np.r_[np.polynomial.polynomial.polyfit(T, polar.cl, 1),
                                  np.polynomial.polynomial.polyfit(T, polar.cl_wake, 1),
                                  np.polynomial.polynomial.polyfit(T, polar.cd_induced_wake, 2)])
    coefficients = np.array(coefficients)
    selected = json.loads((OUT / "低阶极曲线候选.json").read_text(encoding="utf8"))["selection"]["config"]
    predictions = np.zeros_like(coefficients)
    for a, b in KFold(5, shuffle=True, random_state=2026101506).split(x):
        model = fit(x[a], coefficients[a], *selected)
        converted = np.array([[scalar_mean(m, row[None, :]) for m in model["components"]] for row in x[b]])
        predictions[b] = np.sinh(converted)*model["scales"] if model["transform"] == "asinh" else converted
    names = ["total_lift_offset", "total_lift_slope", "wake_lift_offset", "wake_lift_slope",
             "wake_drag_constant", "wake_drag_linear", "wake_drag_quadratic"]
    component = [dict(component=name, **metric(coefficients[:, i], predictions[:, i]))
                 for i, name in enumerate(names)]
    fallback = pd.read_csv(SOURCE / "选定模型折外预测_1.csv").prediction.to_numpy()
    records = []
    points = frame[["sample"]].copy()
    points["actual"] = y
    for name, indices in (("all_predicted", []), ("exact_total_lift", [0, 1]),
                          ("exact_working_gate", [2, 3]), ("exact_wake_drag", [4, 5, 6]),
                          ("all_exact_curve", list(range(7)))):
        values = predictions.copy()
        values[:, indices] = coefficients[:, indices]
        result = np.array([reconstruct(value, row[1]) for value, row in zip(values, x)])
        failed = ~np.isfinite(result)
        result[failed] = fallback[failed]
        records.append(dict(replacement=name, fallback_count=int(failed.sum()), **metric(y, result)))
        points[name] = result
    pd.DataFrame(component).to_csv(OUT / "极曲线分量折外误差.csv", index=False)
    pd.DataFrame(records).to_csv(OUT / "误差来源替换诊断.csv", index=False)
    points.to_csv(OUT / "误差来源逐点诊断.csv", index=False)
    print(json.dumps(dict(component=component, attribution=records), indent=2), flush=True)


if __name__ == "__main__":
    main()
