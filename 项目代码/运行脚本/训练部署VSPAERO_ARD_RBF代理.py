"""Fit the cross-validation-selected ARD-RBF VSPAERO GP and serialize it."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (  # noqa: E402
    FEATURE_LABELS,
    ArdRbfGaussianProcessVSPAeroSurrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402


DEFAULT_DATA = (
    ROOT
    / "气动代理模型数据"
    / "VSPAERO_DOE扩充样本_131点_Ew_物理选点_网格修复_自适应补样_20261011.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "气动代理模型数据"
    / "气动代理模型_扩充DOE131点_ARD_RBF_GP_20261012.json"
)
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def fit_model(frame: pd.DataFrame) -> ArdRbfGaussianProcessVSPAeroSurrogate:
    features = frame.loc[:, FEATURE_LABELS].to_numpy(dtype=float)
    targets = frame.loc[:, TARGETS].to_numpy(dtype=float)
    if len(frame) < 10 or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Training data must contain at least ten finite VSPAERO rows.")

    feature_mean = features.mean(axis=0)
    feature_scale = np.maximum(features.std(axis=0), 1e-9)
    normalized_features = (features - feature_mean) / feature_scale
    target_mean = targets.mean(axis=0)
    target_scale = np.maximum(targets.std(axis=0), 1e-9)
    normalized_targets = (targets - target_mean) / target_scale

    dual = np.empty_like(normalized_targets)
    amplitudes = np.empty(2, dtype=float)
    noise_levels = np.empty(2, dtype=float)
    length_scales = np.empty((2, len(FEATURE_LABELS)), dtype=float)
    inverse_covariances = np.empty(
        (2, len(frame), len(frame)), dtype=float
    )
    for target_index in range(2):
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(
                length_scale=np.ones(len(FEATURE_LABELS)),
                length_scale_bounds=(0.05, 30.0),
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
        )
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=20261012 + target_index,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(normalized_features, targets[:, target_index])
        signal_kernel = estimator.kernel_.k1
        amplitudes[target_index] = float(signal_kernel.k1.constant_value)
        length_scales[target_index] = np.asarray(
            signal_kernel.k2.length_scale, dtype=float
        )
        noise_levels[target_index] = float(estimator.kernel_.k2.noise_level)
        dual[:, target_index] = np.asarray(estimator.alpha_, dtype=float)
        covariance = estimator.kernel_(normalized_features)
        inverse_covariances[target_index] = np.linalg.inv(covariance)

    baseline_features = np.asarray(
        [
            [getattr(AirframeDesign(), name) for name in FEATURE_LABELS]
        ],
        dtype=float,
    )
    baseline_rows = np.flatnonzero(
        np.linalg.norm(features - baseline_features, axis=1) < 1e-8
    )
    if len(baseline_rows) != 1:
        raise ValueError("DOE must contain exactly one declared baseline geometry.")
    baseline_targets = targets[int(baseline_rows[0])]
    return ArdRbfGaussianProcessVSPAeroSurrogate(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        training_features_normalized=normalized_features,
        dual_coefficients=dual,
        target_mean=target_mean,
        target_scale=target_scale,
        kernel_amplitude=amplitudes,
        noise_level=noise_levels,
        length_scales=length_scales,
        inverse_covariances=inverse_covariances,
        baseline_cl_alpha_per_rad=float(baseline_targets[0]),
        baseline_oswald_efficiency=float(baseline_targets[1]),
        training_samples=len(frame),
        oswald_efficiency_bounds=(0.45, 1.00),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    missing = sorted(set(FEATURE_LABELS).union(TARGETS).difference(frame.columns))
    if missing:
        raise ValueError(f"VSPAERO DOE is missing required columns: {missing}")
    model = fit_model(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.to_json(args.output)
    reloaded = ArdRbfGaussianProcessVSPAeroSurrogate.from_json(args.output)
    baseline = reloaded.predict(AirframeDesign())
    if not np.isfinite(baseline).all():
        raise RuntimeError("Serialized ARD-RBF model failed its baseline smoke check.")
    print(
        json.dumps(
            {
                "model": str(args.output.resolve()),
                "training_samples": reloaded.training_samples,
                "baseline_prediction": {
                    TARGETS[0]: baseline[0],
                    TARGETS[1]: baseline[1],
                },
                "length_scales": reloaded.length_scales.tolist(),
                "noise_levels": reloaded.noise_level.tolist(),
                "note": (
                    "Use as an explicit --surrogate option; it remains non-default "
                    "until prediction intervals are calibrated."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
