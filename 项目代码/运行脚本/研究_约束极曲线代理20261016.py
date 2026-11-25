"""Reconstruct unchanged Ew from low-order lift and induced-drag curves."""

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
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import scalar_mean
from VSPAERO气动代理模型 import FEATURE_LABELS
from 运行_分目标气动代理攻关 import fit_target
from 训练_主动增量混合代理20261016 import metric

SOURCE = ROOT / "运行结果" / "主动增量代理_20261015"
OUT = ROOT / "运行结果" / "约束极曲线代理_20261016"
ANGLES = np.array([-3., -1., 1., 3., 5.])
T = (ANGLES-1)/4


def reconstruct(values, ar):
    cl = values[0]+values[1]*T
    clw = values[2]+values[3]*T
    cdw = values[4]+values[5]*T+values[6]*T**2
    working = abs(clw) >= .15
    if working.sum() < 3 or np.any(cdw[working] <= 0):
        return float("nan")
    return float(np.median(cl[working]**2/(np.pi*ar*cdw[working])))


def fit(x, coefficients, mode, kernel, transform):
    scales = np.maximum(np.median(abs(coefficients), axis=0), 1e-9)
    converted = np.arcsinh(coefficients/scales) if transform == "asinh" else coefficients
    components = [fit_target(x, converted[:, i], mode, kernel)[1] for i in range(7)]
    return dict(components=components, transform=transform, scales=scales.tolist())


def predict(model, x):
    converted = np.array([[scalar_mean(component, row[None, :])
                           for component in model["components"]] for row in x])
    values = np.sinh(converted)*model["scales"] if model["transform"] == "asinh" else converted
    return np.array([reconstruct(value, row[1]) for value, row in zip(values, x)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE / "增量训练样本.csv"
    frame = pd.read_csv(source)
    wake = frame.loc[frame.wake_quality_pass].copy()
    x = wake[list(FEATURE_LABELS)].to_numpy(float)
    y = wake.vspaero_oswald_efficiency.to_numpy(float)
    coefficients, audit = [], []
    for _, row in wake.iterrows():
        path = Path(row.geometry_file).parent / "vspaero_polar.csv"
        polar = pd.read_csv(path).set_index("alpha_deg").loc[ANGLES]
        values = np.r_[np.polynomial.polynomial.polyfit(T, polar.cl, 1),
                       np.polynomial.polynomial.polyfit(T, polar.cl_wake, 1),
                       np.polynomial.polynomial.polyfit(T, polar.cd_induced_wake, 2)]
        coefficients.append(values)
        approximate = reconstruct(values, row.aspect_ratio)
        fitted_drag = values[4]+values[5]*T+values[6]*T**2
        audit.append(dict(sample=row["sample"], actual=row.vspaero_oswald_efficiency,
                          curve_reconstruction=approximate,
                          drag_relative_residual=float(np.linalg.norm(fitted_drag-polar.cd_induced_wake)/
                                                       max(np.linalg.norm(polar.cd_induced_wake), 1e-12))))
    coefficients = np.asarray(coefficients)
    pd.DataFrame(audit).to_csv(OUT / "真实极曲线低阶重构.csv", index=False)
    exact = np.array([r["curve_reconstruction"] for r in audit])
    valid = np.isfinite(exact)
    print("curve representation", json.dumps(dict(count=int(valid.sum()),
          failed=int((~valid).sum()), **metric(y[valid], exact[valid]))), flush=True)
    folds = list(KFold(5, shuffle=True, random_state=2026101506).split(x))
    fallback = pd.read_csv(SOURCE / "选定模型折外预测_1.csv").prediction.to_numpy()
    if len(fallback) != len(y):
        raise ValueError("Fallback folds do not match")
    configs = [(mode, kernel, transform) for mode in ("physics", "dimensionless")
               for kernel in ("matern32", "matern52") for transform in ("identity", "asinh")]
    (OUT / "训练前配置.json").write_text(json.dumps(dict(
        dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), count=len(y),
        folds=5, seed=2026101506, new_test_labels_used=False,
        target_definition="Original median Ew; linear lift, quadratic wake induced drag",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configs=configs), indent=2), encoding="utf8")

    def cv(config):
        prediction = np.zeros(len(y))
        repairs = 0
        for a, b in folds:
            result = predict(fit(x[a], coefficients[a], *config), x[b])
            missing = ~np.isfinite(result)
            repairs += int(missing.sum())
            result[missing] = fallback[b][missing]
            prediction[b] = result
        record = dict(features=config[0], kernel=config[1], transform=config[2],
                      fallback_count=repairs, **metric(y, prediction))
        name = "_".join(config)
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            OUT / f"折外_{name}.csv", index=False)
        print(json.dumps(record), flush=True)
        return record, prediction, config

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(cv, configs))
    records = [r[0] for r in results]
    # Mixtures use matching out-of-fold predictions and retain the same target.
    choices = []
    for record, prediction, config in results:
        for weight in (.25, .5, .75, 1.):
            estimate = weight*prediction+(1-weight)*fallback
            choices.append((dict(config=list(config), weight=weight, **metric(y, estimate)), config))
    selected, config = min(choices, key=lambda item: item[0]["rmse"])
    model = fit(x, coefficients, *config)
    payload = dict(schema="ptdl_research_curve_gp_v1", model=model, selection=selected,
                   fallback_model=str(SOURCE / "增量构型气动代理.json"), count=len(y))
    model_path = OUT / "低阶极曲线候选.json"
    model_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf8")
    pd.DataFrame(records).to_csv(OUT / "极曲线代理五折比较.csv", index=False)
    pd.DataFrame([r[0] for r in choices]).to_csv(OUT / "混合代理五折比较.csv", index=False)
    (OUT / "模型冻结.json").write_text(json.dumps(dict(
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), selected=selected,
        automatic_production_promotion=False), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
