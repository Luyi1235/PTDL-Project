"""Compare numerical-residual-dependent training noise without removing labels."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map
from VSPAERO气动代理模型 import FEATURE_LABELS
from 研究_分组多尺度Kriging20261016 import make_kernel
from 训练_主动增量混合代理20261016 import metric

OUT = ROOT / "运行结果" / "残差加权Kriging几何索引_20261016"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "运行结果" / "主动增量代理_20261015" / "增量训练样本.csv"
    frame = pd.read_csv(dataset)
    frame = frame.loc[frame.wake_quality_pass].copy()
    residual = pd.read_csv(ROOT / "运行结果" / "原网格训练标签残差核查_20261016" / "逐构型逐迎角最终残差.csv")
    mapping = residual.loc[residual.available].groupby("geometry_file").log10_max_residual.max()
    logs = frame.geometry_file.map(mapping)
    logs = logs.to_numpy(float)
    x, y = frame[list(FEATURE_LABELS)].to_numpy(float), frame.vspaero_oswald_efficiency.to_numpy(float)
    folds = list(KFold(5, shuffle=True, random_state=2026101506).split(x))
    configs = [(kernel, floor, power) for kernel in ("full", "grouped")
               for floor, power in ((1e-8, 0.), (.0001, .25), (.0001, .5), (.001, .25), (.001, .5))]
    (OUT / "训练配置.json").write_text(json.dumps(dict(
        count=len(y), missing_residual_count=int(frame.geometry_file.map(mapping).isna().sum()),
        observation_key="Unique geometry_file, independent of repeated batch sample names",
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        label_exclusions="Only original predefined polar quality flags; no residual-based removal",
        folds=5, fold_seed=2026101506, fresh_test_labels_used=False, configs=configs), indent=2), encoding="utf8")

    def cv(config):
        kernel, floor, power = config
        prediction = np.zeros(len(y))
        for a, b in folds:
            z = feature_map(x, "physics")
            mean, scale = z[a].mean(0), np.maximum(z[a].std(0), 1e-9)
            z = (z-mean)/scale
            train_logs = np.where(np.isfinite(logs[a]), logs[a], np.nanmedian(logs[a]))
            alpha = floor*np.clip(10**(power*(train_logs+3)), 1., 100.)
            estimator = GaussianProcessRegressor(
                kernel=make_kernel(kernel), alpha=alpha, normalize_y=True,
                n_restarts_optimizer=2, random_state=2026101641)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(z[a], y[a])
            prediction[b] = estimator.predict(z[b])
        record = dict(kernel=kernel, noise_floor=floor, residual_power=power, **metric(y, prediction))
        pd.DataFrame(dict(actual=y, prediction=prediction)).to_csv(
            OUT / f"折外_{kernel}_{floor}_{power}.csv", index=False)
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(cv, configs))
    pd.DataFrame(records).to_csv(OUT / "残差加权五折比较.csv", index=False)


if __name__ == "__main__":
    main()
