"""Nested leave-one-out comparison for VSPAERO aerodynamic surrogates."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (  # noqa: E402
    FEATURE_LABELS,
    _basis,
    design_features,
    fit_kriging_vsp_surrogate,
    fit_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402


DEFAULT_DATA = (
    ROOT
    / "气动代理模型数据"
    / "VSPAERO_DOE扩充样本_100点_Ew_物理选点_网格修复_20261006.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "运行结果" / "VSPAERO代理模型族嵌套留一比较_20261006"
)
TARGET_COLUMNS = (
    "vspaero_cl_alpha_per_rad",
    "vspaero_oswald_efficiency",
)
EW_BOUNDS = (0.45, 1.00)
GP_LENGTH_SCALE_BOUNDS = (0.05, 30.0)
GP_NOISE_BOUNDS = (1e-8, 1.0)
GP_OPTIMIZER_RESTARTS = 2


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "R2": float(1.0 - np.sum(error**2) / total),
    }


def _features(designs: list[AirframeDesign]) -> np.ndarray:
    return np.vstack([design_features(design) for design in designs])


def _physical_geometry_features(x: np.ndarray) -> np.ndarray:
    area, aspect_ratio, taper, sweep, twist, fuselage_length, fuselage_diameter = x.T
    span = np.sqrt(np.maximum(area * aspect_ratio, 1e-12))
    mean_chord = area / np.maximum(span, 1e-12)
    fineness_ratio = fuselage_length / np.maximum(fuselage_diameter, 1e-12)
    return np.column_stack(
        (span, mean_chord, taper, sweep, twist, fineness_ratio, fuselage_diameter)
    )


def _fit_compact_tuned(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> tuple[float, float]:
    feature_mean = x_train.mean(axis=0)
    feature_scale = np.maximum(x_train.std(axis=0), 1e-6)
    matrix = np.vstack(
        [_basis((row - feature_mean) / feature_scale) for row in x_train]
    )
    test = _basis((x_test[0] - feature_mean) / feature_scale)
    target_mean = float(y_train.mean())
    target_scale = max(float(y_train.std()), 1e-9)
    target = (y_train - target_mean) / target_scale
    penalty_mask = np.eye(matrix.shape[1])
    penalty_mask[0, 0] = 0.0
    best_score = float("inf")
    best_prediction = 0.0
    best_lambda = 0.15
    for ridge_lambda in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0):
        normal = matrix.T @ matrix + ridge_lambda * penalty_mask
        try:
            inverse = np.linalg.inv(normal)
            coefficients = inverse @ matrix.T @ target
            residual = target - matrix @ coefficients
            leverage = np.sum((matrix @ inverse) * matrix, axis=1)
            loo = residual / np.maximum(1.0 - leverage, 1e-10)
        except np.linalg.LinAlgError:
            continue
        score = float(np.mean(loo**2))
        if score < best_score:
            best_score = score
            best_lambda = ridge_lambda
            best_prediction = float(test @ coefficients) * target_scale + target_mean
    return best_prediction, best_lambda


def _fit_spline_additive(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> tuple[float, dict[str, object]]:
    y_mean = float(y_train.mean())
    y_scale = max(float(y_train.std()), 1e-9)
    y = (y_train - y_mean) / y_scale
    best_score = float("inf")
    best_prediction = 0.0
    best_params: dict[str, object] = {}
    for knot_count, degree in ((3, 2), (4, 2), (4, 3), (5, 2), (5, 3), (6, 3)):
        transformer = SplineTransformer(
            n_knots=knot_count,
            degree=degree,
            knots="quantile",
            include_bias=False,
        )
        raw = transformer.fit_transform(x_train)
        test_raw = transformer.transform(x_test)
        basis_mean = raw.mean(axis=0)
        basis_scale = np.maximum(raw.std(axis=0), 1e-8)
        matrix = np.column_stack(
            (np.ones(len(raw)), (raw - basis_mean) / basis_scale)
        )
        test = np.r_[1.0, ((test_raw[0] - basis_mean) / basis_scale)]
        penalty_mask = np.eye(matrix.shape[1])
        penalty_mask[0, 0] = 0.0
        for alpha in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0):
            try:
                inverse = np.linalg.inv(matrix.T @ matrix + alpha * penalty_mask)
                coefficients = inverse @ matrix.T @ y
                residual = y - matrix @ coefficients
                leverage = np.sum((matrix @ inverse) * matrix, axis=1)
                loo = residual / np.maximum(1.0 - leverage, 1e-10)
            except np.linalg.LinAlgError:
                continue
            score = float(np.mean(loo**2))
            if score < best_score:
                best_score = score
                best_prediction = float(test @ coefficients) * y_scale + y_mean
                best_params = {
                    "n_knots": knot_count,
                    "degree": degree,
                    "alpha": alpha,
                }
    return best_prediction, best_params


def _full_quadratic_basis(x: np.ndarray) -> np.ndarray:
    terms = [np.ones(len(x)), *[x[:, i] for i in range(x.shape[1])]]
    terms.extend(
        x[:, i] * x[:, j]
        for i in range(x.shape[1])
        for j in range(i, x.shape[1])
    )
    return np.column_stack(terms)


def _fit_full_quadratic(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> tuple[float, float]:
    mean = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0), 1e-9)
    train = _full_quadratic_basis((x_train - mean) / scale)
    test = _full_quadratic_basis((x_test - mean) / scale)
    y_mean = float(y_train.mean())
    y_scale = max(float(y_train.std()), 1e-9)
    y = (y_train - y_mean) / y_scale
    penalty_mask = np.eye(train.shape[1])
    penalty_mask[0, 0] = 0.0
    best_score = float("inf")
    best_prediction = 0.0
    best_lambda = 0.0
    for ridge_lambda in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0):
        normal = train.T @ train + ridge_lambda * penalty_mask
        try:
            inverse = np.linalg.inv(normal)
            coefficients = inverse @ train.T @ y
            residual = y - train @ coefficients
            leverage = np.sum((train @ inverse) * train, axis=1)
            loo = residual / np.maximum(1.0 - leverage, 1e-10)
        except np.linalg.LinAlgError:
            continue
        score = float(np.mean(loo**2))
        if score < best_score:
            best_score = score
            best_lambda = ridge_lambda
            best_prediction = float((test @ coefficients)[0] * y_scale + y_mean)
    if not np.isfinite(best_prediction):
        raise np.linalg.LinAlgError("Full quadratic ridge fit failed.")
    return best_prediction, best_lambda


def _fit_ard_gp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    family: str,
) -> tuple[float, float, dict[str, object]]:
    if "_geom_" in family:
        x_train = _physical_geometry_features(x_train)
        x_test = _physical_geometry_features(x_test)
    x_mean = x_train.mean(axis=0)
    x_scale = np.maximum(x_train.std(axis=0), 1e-9)
    x = (x_train - x_mean) / x_scale
    test = (x_test - x_mean) / x_scale
    length_scale = np.ones(x.shape[1])
    if family == "gp_rbf_ard":
        base_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
            length_scale=length_scale,
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
        )
    elif family in ("gp_matern32_ard", "gp_geom_matern32_ard"):
        base_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
            nu=1.5,
        )
    elif family == "gp_matern52_ard":
        base_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
            nu=2.5,
        )
    elif family in (
        "gp_quadratic_trend_matern32_ard",
        "gp_geom_quadratic_trend_matern32_ard",
    ):
        global_trend = ConstantKernel(1.0, (1e-2, 1e2)) * (
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) ** 2
        )
        local_residual = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
            nu=1.5,
        )
        base_kernel = global_trend + local_residual
    else:
        raise ValueError(f"Unknown GP family: {family}")
    model = GaussianProcessRegressor(
        kernel=base_kernel + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=GP_NOISE_BOUNDS,
        ),
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=GP_OPTIMIZER_RESTARTS,
        random_state=20261006,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y_train)
    prediction, std = model.predict(test, return_std=True)

    def find_kernel_parameter(kernel, parameter: str):
        if hasattr(kernel, parameter):
            return getattr(kernel, parameter)
        for child_name in ("k1", "k2"):
            child = getattr(kernel, child_name, None)
            if child is not None:
                value = find_kernel_parameter(child, parameter)
                if value is not None:
                    return value
        return None

    fitted_lengths = find_kernel_parameter(model.kernel_, "length_scale")
    fitted_noise = find_kernel_parameter(model.kernel_, "noise_level")
    params = {
        "optimized_kernel": str(model.kernel_),
        "length_scales": (
            np.asarray(fitted_lengths, dtype=float).tolist()
            if fitted_lengths is not None
            else []
        ),
        "noise_level": float(fitted_noise) if fitted_noise is not None else None,
    }
    return np.asarray(prediction, dtype=float), np.asarray(std, dtype=float), params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    frame = pd.read_csv(args.data)
    required = {*FEATURE_LABELS, *TARGET_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"DOE file is missing columns: {missing}")
    designs = [
        AirframeDesign(**{key: float(row[key]) for key in FEATURE_LABELS})
        for _, row in frame.iterrows()
    ]
    x_all = _features(designs)
    y_all = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)
    families = (
        "ridge_compact",
        "ridge_compact_tuned",
        "spline_additive_tuned",
        "ridge_full_quadratic",
        "kriging_rbf_isotropic",
        "gp_rbf_ard",
        "gp_matern32_ard",
        "gp_matern52_ard",
        "gp_quadratic_trend_matern32_ard",
        "gp_geom_matern32_ard",
        "gp_geom_quadratic_trend_matern32_ard",
    )
    predictions = {family: np.empty_like(y_all) for family in families}
    predictive_stds = {
        family: np.empty_like(y_all)
        for family in (
            "gp_rbf_ard",
            "gp_matern32_ard",
            "gp_matern52_ard",
            "gp_quadratic_trend_matern32_ard",
            "gp_geom_matern32_ard",
            "gp_geom_quadratic_trend_matern32_ard",
        )
    }
    detail_rows: list[dict[str, object]] = []
    selected_lambdas: list[float] = []
    compact_selected_lambdas: list[dict[str, object]] = []
    selected_splines: list[dict[str, object]] = []
    selected_gp_params: dict[str, list[dict[str, object]]] = {
        "gp_rbf_ard": [],
        "gp_matern32_ard": [],
        "gp_matern52_ard": [],
        "gp_quadratic_trend_matern32_ard": [],
        "gp_geom_matern32_ard": [],
        "gp_geom_quadratic_trend_matern32_ard": [],
    }

    for held_out in range(len(frame)):
        train_mask = np.arange(len(frame)) != held_out
        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        train_designs = [designs[i] for i in np.flatnonzero(train_mask)]
        compact = fit_vsp_surrogate(
            train_designs, y_train[:, 0], y_train[:, 1]
        )
        compact = replace(compact, oswald_efficiency_bounds=EW_BOUNDS)
        compact_prediction = compact.predict(designs[held_out])
        predictions["ridge_compact"][held_out] = compact_prediction

        for target_index in range(2):
            pred, ridge_lambda = _fit_compact_tuned(
                x_train,
                y_train[:, target_index],
                x_all[held_out : held_out + 1],
            )
            predictions["ridge_compact_tuned"][held_out, target_index] = (
                np.clip(pred, *EW_BOUNDS) if target_index == 1 else pred
            )
            compact_selected_lambdas.append(
                {
                    "fold": held_out,
                    "target": TARGET_COLUMNS[target_index],
                    "ridge_lambda": ridge_lambda,
                }
            )

            pred, spline_params = _fit_spline_additive(
                x_train,
                y_train[:, target_index],
                x_all[held_out : held_out + 1],
            )
            predictions["spline_additive_tuned"][held_out, target_index] = (
                np.clip(pred, *EW_BOUNDS) if target_index == 1 else pred
            )
            selected_splines.append(
                {
                    "fold": held_out,
                    "target": TARGET_COLUMNS[target_index],
                    **spline_params,
                }
            )

        for target_index in range(2):
            pred, ridge_lambda = _fit_full_quadratic(
                x_train,
                y_train[:, target_index],
                x_all[held_out : held_out + 1],
            )
            predictions["ridge_full_quadratic"][held_out, target_index] = pred
            selected_lambdas.append(ridge_lambda)

        kriging = fit_kriging_vsp_surrogate(
            train_designs, y_train[:, 0], y_train[:, 1]
        )
        kriging = replace(kriging, oswald_efficiency_bounds=EW_BOUNDS)
        predictions["kriging_rbf_isotropic"][held_out] = kriging.predict(
            designs[held_out]
        )

        for family in (
            "gp_rbf_ard",
            "gp_matern32_ard",
            "gp_matern52_ard",
            "gp_quadratic_trend_matern32_ard",
            "gp_geom_matern32_ard",
            "gp_geom_quadratic_trend_matern32_ard",
        ):
            for target_index in range(2):
                pred, std, fitted_params = _fit_ard_gp(
                    x_train,
                    y_train[:, target_index],
                    x_all[held_out : held_out + 1],
                    family,
                )
                predictive_stds[family][held_out, target_index] = std[0]
                predictions[family][held_out, target_index] = pred[0]
                if target_index == 1:
                    predictions[family][held_out, target_index] = np.clip(
                        pred[0], *EW_BOUNDS
                    )
                selected_gp_params[family].append(
                    {
                        "fold": held_out,
                        "target": TARGET_COLUMNS[target_index],
                        **fitted_params,
                        "predictive_std": float(std[0]),
                    }
                )

    summaries: list[dict[str, object]] = []
    for family in families:
        for target_index, target in enumerate(TARGET_COLUMNS):
            metrics = _metrics(
                y_all[:, target_index], predictions[family][:, target_index]
            )
            if family in predictive_stds:
                error = predictions[family][:, target_index] - y_all[:, target_index]
                metrics["nominal_90pct_interval_coverage"] = float(
                    np.mean(
                        np.abs(error)
                        <= 1.645 * predictive_stds[family][:, target_index]
                    )
                )
            summaries.append(
                {
                    "model": family,
                    "target": target,
                    **metrics,
                }
            )
            for i in range(len(frame)):
                detail_rows.append(
                    {
                        "fold": i,
                        "sample": frame.iloc[i]["sample"],
                        "selection_role": frame.iloc[i].get("selection_role", ""),
                        "model": family,
                        "target": target,
                        "actual": y_all[i, target_index],
                        "predicted": predictions[family][i, target_index],
                        "residual": predictions[family][i, target_index]
                        - y_all[i, target_index],
                    }
                )

    repeated_models = (
        "ridge_compact",
        "ridge_compact_tuned",
        "spline_additive_tuned",
        "gp_matern32_ard",
        "gp_quadratic_trend_matern32_ard",
        "gp_geom_quadratic_trend_matern32_ard",
    )
    repeated_predictions = {
        family: np.full((args.repeats, len(frame), len(TARGET_COLUMNS)), np.nan)
        for family in repeated_models
    }
    splitter = RepeatedKFold(
        n_splits=5,
        n_repeats=args.repeats,
        random_state=20261006,
    )
    for split_index, (train_index, test_index) in enumerate(splitter.split(x_all)):
        repeat_index = split_index // 5
        train_designs = [designs[i] for i in train_index]
        compact = fit_vsp_surrogate(
            train_designs, y_all[train_index, 0], y_all[train_index, 1]
        )
        compact = replace(compact, oswald_efficiency_bounds=EW_BOUNDS)
        for index in test_index:
            repeated_predictions["ridge_compact"][
                repeat_index, index
            ] = compact.predict(designs[index])

        for target_index in range(len(TARGET_COLUMNS)):
            predicted_compact = []
            selected_fold_lambdas = []
            for index in test_index:
                predicted, ridge_lambda = _fit_compact_tuned(
                    x_all[train_index],
                    y_all[train_index, target_index],
                    x_all[index : index + 1],
                )
                predicted_compact.append(predicted)
                selected_fold_lambdas.append(ridge_lambda)
            repeated_predictions["ridge_compact_tuned"][
                repeat_index, test_index, target_index
            ] = (
                np.clip(predicted_compact, *EW_BOUNDS)
                if target_index == 1
                else predicted_compact
            )
            spline_predictions = []
            for index in test_index:
                predicted, _ = _fit_spline_additive(
                    x_all[train_index],
                    y_all[train_index, target_index],
                    x_all[index : index + 1],
                )
                spline_predictions.append(predicted)
            repeated_predictions["spline_additive_tuned"][
                repeat_index, test_index, target_index
            ] = (
                np.clip(spline_predictions, *EW_BOUNDS)
                if target_index == 1
                else spline_predictions
            )
            for family in repeated_models[3:]:
                predicted, _, _ = _fit_ard_gp(
                    x_all[train_index],
                    y_all[train_index, target_index],
                    x_all[test_index],
                    family,
                )
                if target_index == 1:
                    predicted = np.clip(predicted, *EW_BOUNDS)
                repeated_predictions[family][
                    repeat_index, test_index, target_index
                ] = predicted

    repeated_summaries: list[dict[str, object]] = []
    for family, predictions_by_repeat in repeated_predictions.items():
        for target_index, target in enumerate(TARGET_COLUMNS):
            actual = np.tile(y_all[:, target_index], args.repeats)
            predicted = predictions_by_repeat[:, :, target_index].reshape(-1)
            repeated_summaries.append(
                {
                    "model": family,
                    "target": target,
                    **_metrics(actual, predicted),
                    "repeats": args.repeats,
                    "folds_per_repeat": 5,
                }
            )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "代理模型族留一误差对比.csv", index=False)
    pd.DataFrame(repeated_summaries).to_csv(
        output / "代理模型族重复五折误差对比.csv", index=False
    )
    pd.DataFrame(detail_rows).to_csv(output / "代理模型族留一预测明细.csv", index=False)
    metadata = {
        "training_data": str(args.data.resolve()),
        "sample_count": len(frame),
        "repeated_kfold": {
            "folds": 5,
            "repeats": args.repeats,
            "random_state": 20261006,
            "models": list(repeated_models),
        },
        "validation": "outer leave-one-out; Ridge and spline regularization use analytic inner LOO, isotropic Kriging uses exact inner LOO, and scikit-learn GP kernels optimize marginal likelihood within each outer training fold",
        "models": {
            "ridge_compact": "existing compact quadratic-basis Ridge, lambda 0.15",
            "ridge_compact_tuned": "same compact basis, target-specific ridge lambda selected by analytic LOO within each training fold",
            "spline_additive_tuned": "per-feature quantile B-splines; knot count, degree and Ridge alpha selected by analytic LOO within each training fold",
            "ridge_full_quadratic": "all linear, square and pairwise interaction terms; ridge lambda selected by inner analytic LOO",
            "kriging_rbf_isotropic": "existing isotropic squared-exponential Kriging with inner LOO scale/nugget selection",
            "gp_rbf_ard": "scikit-learn GaussianProcessRegressor with ARD RBF kernel, optimized by marginal likelihood within each outer training fold",
            "gp_matern32_ard": "scikit-learn GaussianProcessRegressor with ARD Matérn-3/2 kernel, optimized by marginal likelihood within each outer training fold",
            "gp_matern52_ard": "scikit-learn GaussianProcessRegressor with ARD Matérn-5/2 kernel, optimized by marginal likelihood within each outer training fold",
            "gp_quadratic_trend_matern32_ard": "Gaussian process with a quadratic polynomial covariance trend plus an ARD Matérn-3/2 local component",
            "gp_geom_matern32_ard": "ARD Matérn-3/2 GP using span, mean chord, taper, sweep, twist, fuselage fineness ratio and diameter features",
            "gp_geom_quadratic_trend_matern32_ard": "quadratic-trend plus ARD Matérn-3/2 GP using physically transformed geometry features",
        },
        "oswald_prediction_bounds": list(EW_BOUNDS),
        "gp_hyperparameter_search": {
            "length_scale_bounds": list(GP_LENGTH_SCALE_BOUNDS),
            "noise_level_bounds": list(GP_NOISE_BOUNDS),
            "optimizer_restarts": GP_OPTIMIZER_RESTARTS,
        },
        "clipping": "Ew predictions clipped to the DOE-supported engineering bounds; CL_alpha left unclipped except by the existing compact Ridge's broad physical bounds.",
        "gp_fold_parameters": selected_gp_params,
        "compact_ridge_fold_parameters": compact_selected_lambdas,
        "spline_fold_parameters": selected_splines,
        "ridge_full_quadratic_selected_lambdas": selected_lambdas,
        "caveat": "VSPAERO DOE-only validation; no independent wind-tunnel or flight validation.",
    }
    (output / "代理模型族留一验证说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nRepeated 5-fold:")
    print(pd.DataFrame(repeated_summaries).to_string(index=False))
    print(f"\n结果目录：{output.resolve()}")


if __name__ == "__main__":
    main()
