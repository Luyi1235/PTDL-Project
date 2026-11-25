"""Build and cross-validate the versioned 131-point VSPAERO GP surrogate."""

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
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (  # noqa: E402
    FEATURE_LABELS,
    GaussianProcessVSPAeroSurrogate,
    _basis,
    _quadratic_trend_matern32_kernel,
    design_features,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402


DATA_DIR = ROOT / "气动代理模型数据"
ADAPTIVE_DIR = (
    ROOT / "运行结果" / "VSPAERO自适应补样盲测_20261006_批次03"
)
BASE_DATA = DATA_DIR / "VSPAERO_DOE扩充样本_100点_Ew_物理选点_网格修复_20261006.csv"
ADDED_DATA = ADAPTIVE_DIR / "新增VSPAERO有效标签.csv"
OUTPUT_DATA = DATA_DIR / "VSPAERO_DOE扩充样本_131点_Ew_物理选点_网格修复_自适应补样_20261011.csv"
OUTPUT_MODEL = DATA_DIR / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
OUTPUT_VALIDATION = (
    ROOT
    / "运行结果"
    / "VSPAERO代理模型部署验证_131点_20261011"
)
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
EW_BOUNDS = (0.45, 1.00)


def _make_gp(random_state: int = 20261006) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * (DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) ** 2)
        + ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(
            length_scale=np.ones(len(FEATURE_LABELS)),
            length_scale_bounds=(0.05, 30.0),
            nu=1.5,
        )
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=random_state,
    )


def _fit_gp_state(
    x_train: np.ndarray, y_train: np.ndarray, random_state: int = 20261006
) -> dict[str, object]:
    x_mean = x_train.mean(axis=0)
    x_scale = np.maximum(x_train.std(axis=0), 1e-9)
    normalized_x = (x_train - x_mean) / x_scale
    model = _make_gp(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(normalized_x, y_train)

    fitted = model.kernel_
    trend = fitted.k1.k1
    local = fitted.k1.k2
    trend_amplitude = float(trend.k1.constant_value)
    dot_product_sigma = float(trend.k2.kernel.sigma_0)
    matern_amplitude = float(local.k1.constant_value)
    length_scales = np.asarray(local.k2.length_scale, dtype=float)
    y_mean = float(np.asarray(model._y_train_mean).reshape(-1)[0])
    y_scale = float(np.asarray(model._y_train_std).reshape(-1)[0])
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "normalized_x": model.X_train_.copy(),
        "dual_coefficients": np.asarray(model.alpha_, dtype=float).reshape(-1),
        "y_mean": y_mean,
        "y_scale": y_scale,
        "trend_amplitude": trend_amplitude,
        "dot_product_sigma": dot_product_sigma,
        "matern_amplitude": matern_amplitude,
        "length_scales": length_scales,
        "kernel": str(fitted),
    }


def _train_deployable_model(frame: pd.DataFrame) -> GaussianProcessVSPAeroSurrogate:
    features = frame.loc[:, FEATURE_LABELS].to_numpy(dtype=float)
    targets = frame.loc[:, TARGETS].to_numpy(dtype=float)
    feature_mean = features.mean(axis=0)
    feature_scale = np.maximum(features.std(axis=0), 1e-9)
    normalized_features = (features - feature_mean) / feature_scale
    target_mean = np.empty(2, dtype=float)
    target_scale = np.empty(2, dtype=float)
    trend_amplitude = np.empty(2, dtype=float)
    dot_sigma = np.empty(2, dtype=float)
    matern_amplitude = np.empty(2, dtype=float)
    length_scales = np.empty((2, len(FEATURE_LABELS)), dtype=float)
    dual = np.empty((len(frame), 2), dtype=float)
    fitted_states = []

    for target_index in range(2):
        state = _fit_gp_state(features, targets[:, target_index])
        if not np.allclose(state["x_mean"], feature_mean) or not np.allclose(
            state["x_scale"], feature_scale
        ):
            raise AssertionError("The serialized GP feature normalization drifted.")
        target_mean[target_index] = state["y_mean"]
        target_scale[target_index] = state["y_scale"]
        trend_amplitude[target_index] = state["trend_amplitude"]
        matern_amplitude[target_index] = state["matern_amplitude"]
        length_scales[target_index] = state["length_scales"]
        dual[:, target_index] = state["dual_coefficients"]
        dot_sigma[target_index] = state["dot_product_sigma"]
        fitted_states.append(state)

    baseline_rows = np.flatnonzero(
        np.linalg.norm(features - design_features(AirframeDesign()), axis=1) < 1e-8
    )
    if len(baseline_rows):
        baseline_targets = targets[int(baseline_rows[0])]
    else:
        raise ValueError("The 131-point DOE does not include the declared baseline.")
    baseline_normalized = (
        design_features(AirframeDesign()) - feature_mean
    ) / feature_scale
    baseline_kernel = _quadratic_trend_matern32_kernel(
        baseline_normalized[None, :],
        normalized_features,
        trend_amplitude,
        dot_sigma,
        matern_amplitude,
        length_scales,
    )[0]
    baseline_predictions = target_mean + target_scale * np.sum(
        baseline_kernel * dual, axis=0
    )

    return GaussianProcessVSPAeroSurrogate(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        training_features_normalized=normalized_features,
        dual_coefficients=dual,
        target_mean=target_mean,
        target_scale=target_scale,
        trend_amplitude=trend_amplitude,
        dot_product_sigma=dot_sigma,
        matern_amplitude=matern_amplitude,
        length_scales=length_scales,
        baseline_cl_alpha_per_rad=float(baseline_predictions[0]),
        baseline_oswald_efficiency=float(baseline_predictions[1]),
        training_samples=len(frame),
        oswald_efficiency_bounds=EW_BOUNDS,
    )


def _merge_training_data() -> pd.DataFrame:
    base = pd.read_csv(BASE_DATA)
    added = pd.read_csv(ADDED_DATA)
    if "selection_role" not in added:
        raise ValueError("Adaptive labels are missing the selection_role field.")
    added = added.loc[added["selection_role"] == "adaptive_train"].copy()
    if len(added) != 31:
        raise ValueError(
            f"Expected 31 adaptive training labels after excluding holdout rows, found {len(added)}."
        )
    required = {*FEATURE_LABELS, *TARGETS}
    for name, frame in (("base", base), ("added", added)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} DOE is missing columns: {sorted(missing)}")
        if not np.isfinite(frame.loc[:, list(required)].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} DOE contains non-finite training values.")
    if "oswald_efficiency_definition" in added:
        definitions = set(added["oswald_efficiency_definition"].dropna())
        expected = "median VSPAERO Ew for polar points with abs(CLiw) >= 0.15"
        if definitions != {expected}:
            raise ValueError("Added labels have a mixed or unexpected Ew definition.")
    merged = pd.concat((base, added), ignore_index=True, sort=False)
    feature_rows = np.round(merged.loc[:, FEATURE_LABELS].to_numpy(float), 10)
    if len(np.unique(feature_rows, axis=0)) != len(merged):
        raise ValueError("Duplicate geometries found while assembling the 131-point DOE.")
    if len(merged) != 131:
        raise ValueError(f"Expected 131 valid rows, found {len(merged)}.")
    return merged


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1.0 - np.sum(error**2) / total),
    }


def _cross_validate(frame: pd.DataFrame, repeats: int) -> pd.DataFrame:
    x = frame.loc[:, FEATURE_LABELS].to_numpy(dtype=float)
    y = frame.loc[:, TARGETS].to_numpy(dtype=float)
    predictions: dict[str, np.ndarray] = {
        model: np.full((repeats, len(frame), len(TARGETS)), np.nan)
        for model in ("compact_quadratic_ridge", "quadratic_trend_matern32_gp")
    }
    splitter = RepeatedKFold(
        n_splits=5, n_repeats=repeats, random_state=20261006
    )
    for split_index, (train_index, test_index) in enumerate(splitter.split(x)):
        repeat_index = split_index // 5
        mean = x[train_index].mean(axis=0)
        scale = np.maximum(x[train_index].std(axis=0), 1e-6)
        design_matrix = np.vstack(
            [_basis((row - mean) / scale) for row in x[train_index]]
        )
        penalty = 0.15 * np.eye(design_matrix.shape[1])
        penalty[0, 0] = 0.0
        for target_index in range(2):
            coefficients = np.linalg.solve(
                design_matrix.T @ design_matrix + penalty,
                design_matrix.T @ y[train_index, target_index],
            )
            test_matrix = np.vstack(
                [_basis((row - mean) / scale) for row in x[test_index]]
            )
            predictions["compact_quadratic_ridge"][
                repeat_index, test_index, target_index
            ] = test_matrix @ coefficients

            state = _fit_gp_state(x[train_index], y[train_index, target_index])
            test_normalized = (
                x[test_index] - state["x_mean"]
            ) / state["x_scale"]
            kernel = _quadratic_trend_matern32_kernel(
                test_normalized,
                state["normalized_x"],
                np.asarray([state["trend_amplitude"]]),
                np.asarray([state["dot_product_sigma"]]),
                np.asarray([state["matern_amplitude"]]),
                np.asarray(state["length_scales"], dtype=float),
            )[:, :, 0]
            predictions["quadratic_trend_matern32_gp"][
                repeat_index, test_index, target_index
            ] = state["y_mean"] + state["y_scale"] * (
                kernel @ state["dual_coefficients"]
            )

    rows = []
    for model_name, predicted in predictions.items():
        for target_index, target in enumerate(TARGETS):
            actual = np.tile(y[:, target_index], repeats)
            model_predictions = predicted[:, :, target_index].reshape(-1)
            rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "repeats": repeats,
                    "folds_per_repeat": 5,
                    **_metrics(actual, model_predictions),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive.")

    frame = _merge_training_data()
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DATA, index=False)

    validation = _cross_validate(frame, args.repeats)
    OUTPUT_VALIDATION.mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUTPUT_VALIDATION / "重复五折验证对比.csv", index=False)
    selected_gp = validation.loc[
        validation["model"] == "quadratic_trend_matern32_gp"
    ]
    ridge = validation.loc[
        validation["model"] == "compact_quadratic_ridge"
    ].set_index("target")
    for _, row in selected_gp.iterrows():
        base = ridge.loc[row["target"]]
        if row["R2"] + 1e-12 < base["R2"] or row["RMSE"] > base["RMSE"] + 1e-12:
            raise RuntimeError(
                f"GP did not beat Ridge on both R2 and RMSE for {row['target']}; "
                "model not deployed."
            )

    model = _train_deployable_model(frame)
    model.to_json(OUTPUT_MODEL)
    reloaded = GaussianProcessVSPAeroSurrogate.from_json(OUTPUT_MODEL)
    smoke_design = AirframeDesign()
    before = np.asarray(model.predict(smoke_design))
    after = np.asarray(reloaded.predict(smoke_design))
    if not np.allclose(before, after, rtol=0.0, atol=1e-12):
        raise AssertionError("Serialized GP does not reproduce the fitted predictions.")
    training_predictions = np.asarray(
        [
            reloaded.predict(
                AirframeDesign(
                    **{name: float(row[name]) for name in FEATURE_LABELS}
                )
            )
            for _, row in frame.iterrows()
        ]
    )
    max_training_error = np.max(
        np.abs(training_predictions - frame.loc[:, TARGETS].to_numpy(float)),
        axis=0,
    )
    if np.any(max_training_error > 1e-3):
        raise AssertionError(
            "Serialized GP does not reproduce its training labels within the "
            f"expected low-noise tolerance: {max_training_error.tolist()}"
        )

    metadata = {
        "training_data": str(OUTPUT_DATA.resolve()),
        "training_samples": len(frame),
        "base_samples": len(pd.read_csv(BASE_DATA)),
        "adaptive_added_valid_samples": len(frame) - len(pd.read_csv(BASE_DATA)),
        "model_file": str(OUTPUT_MODEL.resolve()),
        "model_family": "quadratic-trend Matérn-3/2 ARD Gaussian process",
        "cross_validation": "RepeatedKFold 5 folds x configured repeats, deterministic seed 20261006; compared against the prior compact quadratic Ridge",
        "deployment_gate": "GP must improve both R2 and RMSE for each target on repeated CV; otherwise the script refuses deployment.",
        "serialization_training_max_abs_error": {
            target: float(error)
            for target, error in zip(TARGETS, max_training_error)
        },
        "oswald_efficiency_definition": "median VSPAERO Ew for polar points with abs(CLiw) >= 0.15",
        "repeated_five_fold_metrics": validation.to_dict(orient="records"),
        "runtime_dependency": "NumPy only for prediction; scikit-learn is used to train/retune the model.",
        "limitations": [
            "Training labels are VSPAERO VLM results, not wind-tunnel or flight measurements.",
            "The previously generated 8-point holdout is not treated as a new untouched test set.",
            "A higher cross-validation score does not guarantee lower error for every geometry or improve overall aircraft model fidelity.",
        ],
    }
    (OUTPUT_VALIDATION / "部署验证说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(validation.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
