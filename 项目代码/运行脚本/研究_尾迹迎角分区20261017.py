"""Predict the working-angle set from geometry before local wake regression."""

import argparse
from pathlib import Path
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

import 研究_千构型尾迹分解20261017 as search

ORIGINAL_FIT, ORIGINAL_PREDICT = search.fit, search.predict
CONFIGS = [
    dict(method="regime", mode=mode, expert=expert, blend=blend)
    for mode in ("dimensionless", "physics") for expert in ("gp", "boost")
    for blend in (.5, 1.)
]


def regime(values):
    return (abs(values) >= .15).astype(int) @ (2**np.arange(5))


def fit(config, x, y, polar):
    starter = (dict(method="gp", mode=config["mode"], kernel="matern32", noise=.05)
               if config["expert"] == "gp" else
               dict(method="boost", mode=config["mode"], leaves=7, penalty=1.))
    fallback = ORIGINAL_FIT(starter, x, y, polar)
    z = (search.base.feature_map(x, config["mode"])-fallback["center"])/fallback["scale"]
    gate = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (.01, 100.))*RBF(np.ones(z.shape[1]), (.1, 100.))
        +WhiteKernel(.0001, (1e-7, .01)), normalize_y=True, alpha=1e-9,
        optimizer=search.base.optimize, n_restarts_optimizer=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gate.fit(z, polar[:, :, 1])
    mask = regime(polar[:, :, 1])
    experts = {}
    for key in np.unique(mask):
        selected = mask == key
        if selected.sum() >= 40:
            experts[int(key)] = ORIGINAL_FIT(starter, x[selected], y[selected], polar[selected])
    return dict(config=config, gate=gate, fallback=fallback, experts=experts)


def predict(model, x):
    config, fallback = model["config"], model["fallback"]
    z = (search.base.feature_map(x, config["mode"])-fallback["center"])/fallback["scale"]
    mask = regime(model["gate"].predict(z))
    result = ORIGINAL_PREDICT(fallback, x)
    for key, expert in model["experts"].items():
        selected = mask == key
        if selected.any():
            result[selected] = ((1-config["blend"])*result[selected]
                                +config["blend"]*ORIGINAL_PREDICT(expert, x[selected]))
    return result


search.CONFIGS = CONFIGS
search.fit, search.predict = fit, predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=search.ROOT / "运行结果/尾迹迎角分区_20261017")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    search.run(args.output_dir, args.workers)
