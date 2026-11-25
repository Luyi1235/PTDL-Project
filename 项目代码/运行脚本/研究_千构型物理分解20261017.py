"""Compare polar-coefficient surrogates and multi-start GP on training data."""

import argparse
from pathlib import Path
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

import 研究_千构型尾迹分解20261017 as search

ORIGINAL_FIT, ORIGINAL_PREDICT = search.fit, search.predict
CONFIGS = [
    dict(method="gp_restart", mode=mode, kernel=kernel, noise="learn")
    for mode in ("dimensionless", "physics") for kernel in ("rbf", "matern32", "matern52")
] + [
    dict(method="polar", mode=mode, leaves=leaves, penalty=1., reconstruction=reconstruction)
    for mode in ("dimensionless", "physics") for leaves in (7, 15)
    for reconstruction in ("polynomial", "pointwise")
]


def fit(config, x, y, polar):
    starter = dict(method="boost", mode=config["mode"], leaves=7, penalty=1.)
    model = ORIGINAL_FIT(starter, x, y, polar)
    z = (search.base.feature_map(x, config["mode"])-model["center"])/model["scale"]
    if config["method"] == "gp_restart":
        length = np.ones(z.shape[1])
        kernel = (RBF(length, (.05, 50)) if config["kernel"] == "rbf" else
                  Matern(length, (.05, 50), nu=1.5 if config["kernel"] == "matern32" else 2.5))
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (.01, 100.))*kernel+WhiteKernel(.03, (1e-6, .5)),
            normalize_y=True, alpha=1e-9, optimizer=search.base.optimize,
            n_restarts_optimizer=3, random_state=2026101713)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(z, y)
        model["estimator"] = gp
    else:
        # Fit dimensional force coefficients before division and working-angle selection.
        if config["reconstruction"] == "polynomial":
            values = np.stack([np.polyfit(search.ANGLES, p[:, :3], 2).T for p in polar])
        else:
            values = polar[:, :, :3].transpose(0, 2, 1)
        model["component_shape"] = values.shape[1:]
        values = values.reshape(len(x), -1)
        model["components"] = [search.boost(config).fit(z, values[:, i])
                               for i in range(values.shape[1])]
    model["config"] = config
    return model


def predict(model, x):
    config = model["config"]
    z = (search.base.feature_map(x, config["mode"])-model["center"])/model["scale"]
    result = model["estimator"].predict(z)
    if config["method"] == "gp_restart":
        return result
    values = np.column_stack([m.predict(z) for m in model["components"]])
    values = values.reshape(len(x), *model["component_shape"])
    if config["reconstruction"] == "polynomial":
        values = np.stack([np.stack([np.polyval(component, search.ANGLES) for component in row])
                           for row in values])
    for i, (cl, clw, cdw) in enumerate(values):
        eligible = (abs(clw) >= .15) & (cdw > 0)
        if eligible.sum() >= 3:
            result[i] = np.median(cl[eligible]**2/(np.pi*x[i, 1]*cdw[eligible]))
    return result


search.CONFIGS = CONFIGS
search.fit, search.predict = fit, predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=search.ROOT / "运行结果/千构型物理分解_20261017")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    search.run(args.output_dir, args.workers)
