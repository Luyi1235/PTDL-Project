"""Compare geometry-only estimators and angle reconstruction using training folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map, scalar_prediction
from VSPAERO气动代理模型 import FEATURE_LABELS
from 运行_分目标气动代理攻关 import fit_target

OUT = ROOT / "运行结果" / "迎角分解非线性代理_20261015"
ANGLES = (-3., -1., 1., 3., 5.)
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def metrics(y, prediction):
    error = prediction - y
    return dict(r2=float(1 - error @ error / np.sum((y-y.mean())**2)),
                mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2))))


def angle_fit(x, polars, mode, transform):
    gates, values = [], []
    for i in range(5):
        gates.append(fit_target(x, polars[:, i, 1], mode, "rbf")[1])
        mask = (abs(polars[:, i, 1]) >= .15) & (polars[:, i, 2] > 0)
        y = polars[mask, i, 0]**2 / (np.pi*x[mask, 1]*polars[mask, i, 2])
        if transform == "log":
            y = np.log(y)
        values.append(fit_target(x[mask], y, mode, "matern32")[1])
    return dict(gates=gates, values=values, transform=transform)


def angle_predict(model, x):
    outputs = []
    for row in x:
        at = row[None, :]
        gate = np.array([scalar_prediction(m, at)[0] for m in model["gates"]])
        values = np.array([scalar_prediction(m, at)[0] for m in model["values"]])
        if model["transform"] == "log":
            values = np.exp(values)
        working = (abs(gate) >= .15) & (values > 0)
        outputs.append(float(np.median(values[working])) if working.sum() >= 3 else np.nan)
    return np.array(outputs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "运行结果" / "增量构型气动代理_20261015" / "增量训练样本.csv")
    tasks = []
    for target in range(2):
        for mode in ("raw", "dimensionless", "physics"):
            for c in (1., 10., 100.):
                for gamma in (.03, .15, .6):
                    tasks.append((target, mode, "svr", c, gamma))
            for leaf in (1, 2, 4):
                tasks.append((target, mode, "trees", leaf, 0.))
            for leaf in (7, 15):
                tasks.append((target, mode, "boosting", leaf, 0.))
    arrays = {}
    for target, name in enumerate(TARGETS):
        mask = np.isfinite(frame[name].to_numpy(float))
        if target:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        subset = frame.loc[mask]
        arrays[target] = (subset[list(FEATURE_LABELS)].to_numpy(float),
                          subset[name].to_numpy(float),
                          list(KFold(5, shuffle=True, random_state=2026101506).split(subset)))
    wake_frame = frame.loc[frame.wake_quality_pass]
    polars = np.array([
        pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv")
        .set_index("alpha_deg").loc[list(ANGLES), ["cl", "cl_wake", "cd_induced_wake"]].to_numpy(float)
        for _, row in wake_frame.iterrows()
    ])

    def run(task):
        target, mode, method, a, b = task
        x, y, folds = arrays[target]
        prediction = np.empty(len(y))
        for train, test in folds:
            if method == "svr":
                estimator = make_pipeline(StandardScaler(), SVR(C=a, gamma=b, epsilon=.02))
            elif method == "trees":
                estimator = ExtraTreesRegressor(n_estimators=240, min_samples_leaf=a,
                                                 max_features=1., random_state=20261015, n_jobs=1)
            else:
                estimator = HistGradientBoostingRegressor(
                    max_iter=180, max_leaf_nodes=a, l2_regularization=1.,
                    learning_rate=.05, min_samples_leaf=10, random_state=20261015)
            z = feature_map(x, mode)
            mean, scale = y[train].mean(), max(y[train].std(), 1e-9)
            estimator.fit(z[train], (y[train]-mean)/scale)
            prediction[test] = mean + scale*estimator.predict(z[test])
        record = dict(target=target, features=mode, method=method, a=a, b=b, **metrics(y, prediction))
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            OUT / f"折外_{target}_{mode}_{method}_{a}_{b}.csv", index=False)
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=10) as pool:
        records = list(pool.map(run, tasks))
    pd.DataFrame(records).to_csv(OUT / "非线性模型五折比较.csv", index=False)
    x, y, folds = arrays[1]

    def angle_cv(config):
        mode, transform = config
        prediction = np.full(len(y), np.nan)
        for train, test in folds:
            model = angle_fit(x[train], polars[train], mode, transform)
            prediction[test] = angle_predict(model, x[test])
        record = dict(target=1, features=mode, method="angle-"+transform,
                      failed_reconstructions=int(np.isnan(prediction).sum()))
        if np.isfinite(prediction).all():
            record.update(metrics(y, prediction))
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            OUT / f"折外_迎角分解_{mode}_{transform}.csv", index=False)
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        angle_records = list(pool.map(angle_cv, [
            ("physics", "identity"), ("physics", "log"),
            ("dimensionless", "identity"), ("dimensionless", "log")]))
    pd.DataFrame(angle_records).to_csv(OUT / "迎角分解五折比较.csv", index=False)
    (OUT / "试验口径.json").write_text(json.dumps(dict(
        training_samples=len(frame), wake_samples=len(y),
        current_fresh_test_used=False,
        target_definition="Original median Ew with predicted abs(CLiw)>=0.15 gate",
        folds=5, fold_seed=2026101506, target_r2=.9,
    ), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
