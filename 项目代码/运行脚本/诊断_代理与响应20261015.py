"""Read existing solver labels and compare reproducible training-only models."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO气动代理模型 import FEATURE_LABELS

OUT = ROOT / "运行结果" / "代理与响应攻关_20261015"
TRAIN = ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv"


def physics_features(x):
    area, ar, taper, sweep, twist, length, diameter = np.asarray(x).T
    span = np.sqrt(area * ar)
    root = 2 * area / (span * (1 + taper))
    mac = 2 / 3 * root * (1 + taper + taper**2) / (1 + taper)
    return np.column_stack((ar, taper, sweep, twist, length / span, diameter / mac,
                            diameter / span, mac / length, diameter / length))


def label_audit():
    frame = pd.read_csv(TRAIN)
    records = []
    for _, row in frame.iterrows():
        p = pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv")
        cl = p.cl.to_numpy()
        drag = p.cd_induced_wake.to_numpy()
        coef = np.polyfit(cl, drag, 2)
        k = coef[0]
        # This is a diagnostic alternative, kept separate from legacy Ew.
        e_curvature = 1 / (np.pi * row.aspect_ratio * k) if k > 0 else np.nan
        records.append(dict(
            sample=row["sample"], cl_slope=row.vspaero_cl_alpha_per_rad,
            legacy_ew=row.vspaero_oswald_efficiency, curvature_ew=e_curvature,
            fixed_alpha5_ew=float(p.iloc[-1].e_wake),
            drag_fit_rmse=float(np.sqrt(np.mean((np.polyval(coef, cl) - drag)**2))),
            wake_quality_pass=row.wake_quality_pass,
        ))
    diagnostic = pd.DataFrame(records)
    diagnostic.to_csv(OUT / "标签定义对照.csv", index=False)
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    jobs = []
    for target in ("cl_slope", "legacy_ew", "curvature_ew", "fixed_alpha5_ew"):
        y = diagnostic[target].to_numpy()
        valid = np.isfinite(y) & (y > 0)
        if target != "cl_slope":
            valid &= diagnostic.wake_quality_pass.to_numpy(bool)
        for feature in ("raw", "physics"):
            xx = x[valid] if feature == "raw" else physics_features(x[valid])
            for kind in ("ridge2", "extra", "boost", "gp"):
                jobs.append((target, feature, kind, xx, y[valid]))

    def compare(job):
        target, feature, kind, xx, y = job
        models = {
            "ridge2": make_pipeline(StandardScaler(), PolynomialFeatures(2),
                                   Ridge(alpha=1)),
            "extra": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                        max_features=1., n_jobs=1, random_state=20261015),
            "boost": HistGradientBoostingRegressor(max_iter=200, l2_regularization=1,
                                                   min_samples_leaf=12, random_state=20261015),
            "gp": make_pipeline(StandardScaler(), GaussianProcessRegressor(
                kernel=ConstantKernel(1, (.01, 100)) *
                Matern(np.ones(xx.shape[1]), (.05, 50), nu=1.5) +
                WhiteKernel(.001, (1e-8, 1)), normalize_y=True,
                n_restarts_optimizer=1, random_state=20261015)),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = cross_val_predict(models[kind], xx, y,
                                     cv=KFold(5, shuffle=True, random_state=20261015))
        record = dict(target=target, features=feature, model=kind, n=len(y),
                      r2=float(r2_score(y, pred)),
                      rmse=float(np.sqrt(mean_squared_error(y, pred))))
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(compare, jobs))
    pd.DataFrame(results).to_csv(OUT / "训练集诊断模型比较.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    label_audit()
