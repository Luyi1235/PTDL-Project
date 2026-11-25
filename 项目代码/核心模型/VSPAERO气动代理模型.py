"""VSPAERO-derived aerodynamic correction for the PTDL optimization loop.

VSPAERO is used to learn geometry-dependent lift-slope and induced-drag
trends. The calibrated component-build-up parasite-drag model remains in
place because VSPAERO's VLM analysis is inviscid. Corrections are relative to
the VSPAERO baseline, so the existing 72 h public-data calibration remains the
reference condition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from 机体参数化设计与气动阻力 import BASELINE, AeroCoefficients, AirframeDesign, aero_from_design
from ULTRA基准标定 import UltraCalibration


FEATURE_LABELS = (
    "wing_area_m2",
    "aspect_ratio",
    "taper_ratio",
    "sweep_deg",
    "tip_twist_deg",
    "fuselage_length_m",
    "fuselage_diameter_m",
)
OSWALD_EFFICIENCY_DEFINITION = (
    "median VSPAERO Ew for polar points with abs(CLiw) >= 0.15"
)


def design_features(design: AirframeDesign) -> np.ndarray:
    return np.asarray([getattr(design, label) for label in FEATURE_LABELS], dtype=float)


def _basis(normalized_features: np.ndarray) -> np.ndarray:
    """Use a compact quadratic basis suitable for the limited VSPAERO DOE."""
    x = np.asarray(normalized_features, dtype=float)
    return np.concatenate(
        (
            np.ones(1),
            x,
            np.asarray([x[1] ** 2, x[3] ** 2, x[4] ** 2, x[1] * x[3], x[1] * x[2]]),
        )
    )


@dataclass(frozen=True)
class VSPAeroSurrogate:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    cl_alpha_coefficients: np.ndarray
    oswald_coefficients: np.ndarray
    baseline_cl_alpha_per_rad: float
    baseline_oswald_efficiency: float
    ridge_lambda: float
    training_samples: int
    oswald_efficiency_bounds: tuple[float, float] = (0.55, 0.95)

    def predict(self, design: AirframeDesign) -> tuple[float, float]:
        normalized = (design_features(design) - self.feature_mean) / self.feature_scale
        row = _basis(normalized)
        cl_alpha = float(row @ self.cl_alpha_coefficients)
        oswald = float(row @ self.oswald_coefficients)
        return float(np.clip(cl_alpha, 3.5, 8.5)), float(
            np.clip(oswald, *self.oswald_efficiency_bounds)
        )

    def corrected_aero(self, design: AirframeDesign, calibration: UltraCalibration) -> AeroCoefficients:
        """Return a baseline-calibrated polar with VSPAERO trend corrections."""
        low_order = aero_from_design(design, calibration)
        vsp_cl_alpha, vsp_oswald = self.predict(design)
        # Anchor the correction at the surrogate's own baseline prediction.
        # This preserves the existing 72 h public-data calibration while using
        # VSPAERO to change the aerodynamic trend between geometries.
        baseline_cl_alpha, baseline_oswald = self.predict(BASELINE)
        cl_ratio = vsp_cl_alpha / baseline_cl_alpha
        e_ratio = vsp_oswald / baseline_oswald
        return replace(
            low_order,
            cl_alpha_per_rad=float(np.clip(low_order.cl_alpha_per_rad * cl_ratio, 3.5, 8.5)),
            oswald_efficiency=float(
                np.clip(
                    low_order.oswald_efficiency * e_ratio,
                    *self.oswald_efficiency_bounds,
                )
            ),
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "feature_labels": list(FEATURE_LABELS),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "cl_alpha_coefficients": self.cl_alpha_coefficients.tolist(),
            "oswald_coefficients": self.oswald_coefficients.tolist(),
            "baseline_cl_alpha_per_rad": self.baseline_cl_alpha_per_rad,
            "baseline_oswald_efficiency": self.baseline_oswald_efficiency,
            "ridge_lambda": self.ridge_lambda,
            "training_samples": self.training_samples,
            "oswald_efficiency_bounds": list(self.oswald_efficiency_bounds),
            "oswald_efficiency_definition": OSWALD_EFFICIENCY_DEFINITION,
            "model_note": "VSPAERO VLM corrects CL_alpha and wake-based Ew trends; calibrated component-build-up parasite drag remains separate.",
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "VSPAeroSurrogate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload["feature_labels"]) != FEATURE_LABELS:
            raise ValueError("The VSPAERO surrogate feature schema does not match this optimizer.")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            cl_alpha_coefficients=np.asarray(payload["cl_alpha_coefficients"], dtype=float),
            oswald_coefficients=np.asarray(payload["oswald_coefficients"], dtype=float),
            baseline_cl_alpha_per_rad=float(payload["baseline_cl_alpha_per_rad"]),
            baseline_oswald_efficiency=float(payload["baseline_oswald_efficiency"]),
            ridge_lambda=float(payload["ridge_lambda"]),
            training_samples=int(payload["training_samples"]),
            oswald_efficiency_bounds=tuple(
                map(float, payload.get("oswald_efficiency_bounds", (0.55, 0.95)))
            ),
        )


@dataclass(frozen=True)
class KrigingVSPAeroSurrogate:
    """Small-data Gaussian-process/Kriging model for VSPAERO response surfaces."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_features_normalized: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    target_normalized: np.ndarray
    length_scales: np.ndarray
    nuggets: np.ndarray
    alpha: np.ndarray
    inverse_covariances: np.ndarray
    baseline_cl_alpha_per_rad: float
    baseline_oswald_efficiency: float
    training_samples: int
    oswald_efficiency_bounds: tuple[float, float] = (0.55, 0.95)

    def predict(self, design: AirframeDesign) -> tuple[float, float]:
        cl_alpha, oswald, _, _ = self.predict_with_uncertainty(design)
        return cl_alpha, oswald

    def predict_with_uncertainty(
        self, design: AirframeDesign
    ) -> tuple[float, float, float, float]:
        feature = (design_features(design) - self.feature_mean) / self.feature_scale
        means: list[float] = []
        deviations: list[float] = []
        for index in range(2):
            cross = _rbf_kernel(
                self.training_features_normalized,
                feature[None, :],
                self.length_scales[index],
            )[:, 0]
            normalized_mean = float(cross @ self.alpha[:, index])
            variance = max(
                0.0,
                1.0 - float(cross @ self.inverse_covariances[index] @ cross),
            )
            means.append(
                float(self.target_mean[index] + self.target_scale[index] * normalized_mean)
            )
            deviations.append(float(self.target_scale[index] * np.sqrt(variance)))
        cl_alpha = float(np.clip(means[0], 3.5, 8.5))
        oswald = float(np.clip(means[1], *self.oswald_efficiency_bounds))
        return cl_alpha, oswald, deviations[0], deviations[1]

    def corrected_aero(
        self, design: AirframeDesign, calibration: UltraCalibration
    ) -> AeroCoefficients:
        low_order = aero_from_design(design, calibration)
        cl_alpha, oswald = self.predict(design)
        baseline_cl = max(self.baseline_cl_alpha_per_rad, 1e-9)
        baseline_e = max(self.baseline_oswald_efficiency, 1e-9)
        return replace(
            low_order,
            cl_alpha_per_rad=float(
                np.clip(low_order.cl_alpha_per_rad * cl_alpha / baseline_cl, 3.5, 8.5)
            ),
            oswald_efficiency=float(
                np.clip(
                    low_order.oswald_efficiency * oswald / baseline_e,
                    *self.oswald_efficiency_bounds,
                )
            ),
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "schema": "ptdl_vspaero_kriging_v1",
            "feature_labels": list(FEATURE_LABELS),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "training_features_normalized": self.training_features_normalized.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
            "target_normalized": self.target_normalized.tolist(),
            "length_scales": self.length_scales.tolist(),
            "nuggets": self.nuggets.tolist(),
            "alpha": self.alpha.tolist(),
            "inverse_covariances": self.inverse_covariances.tolist(),
            "baseline_cl_alpha_per_rad": self.baseline_cl_alpha_per_rad,
            "baseline_oswald_efficiency": self.baseline_oswald_efficiency,
            "training_samples": self.training_samples,
            "oswald_efficiency_bounds": list(self.oswald_efficiency_bounds),
            "kernel": "isotropic squared-exponential Gaussian process with fitted nugget",
            "model_note": "VSPAERO VLM response surrogate; estimates and uncertainty are conditional on this DOE and kernel.",
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "KrigingVSPAeroSurrogate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "ptdl_vspaero_kriging_v1":
            raise ValueError("Unsupported Kriging VSPAERO surrogate schema.")
        if tuple(payload["feature_labels"]) != FEATURE_LABELS:
            raise ValueError("The Kriging feature schema does not match this optimizer.")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            training_features_normalized=np.asarray(payload["training_features_normalized"], dtype=float),
            target_mean=np.asarray(payload["target_mean"], dtype=float),
            target_scale=np.asarray(payload["target_scale"], dtype=float),
            target_normalized=np.asarray(payload["target_normalized"], dtype=float),
            length_scales=np.asarray(payload["length_scales"], dtype=float),
            nuggets=np.asarray(payload["nuggets"], dtype=float),
            alpha=np.asarray(payload["alpha"], dtype=float),
            inverse_covariances=np.asarray(payload["inverse_covariances"], dtype=float),
            baseline_cl_alpha_per_rad=float(payload["baseline_cl_alpha_per_rad"]),
            baseline_oswald_efficiency=float(payload["baseline_oswald_efficiency"]),
            training_samples=int(payload["training_samples"]),
            oswald_efficiency_bounds=tuple(
                map(float, payload.get("oswald_efficiency_bounds", (0.55, 0.95)))
            ),
        )


@dataclass(frozen=True)
class GaussianProcessVSPAeroSurrogate:
    """Serialized quadratic-trend Matérn GP for CL-alpha and wake-based Ew."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_features_normalized: np.ndarray
    dual_coefficients: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    trend_amplitude: np.ndarray
    dot_product_sigma: np.ndarray
    matern_amplitude: np.ndarray
    length_scales: np.ndarray
    baseline_cl_alpha_per_rad: float
    baseline_oswald_efficiency: float
    training_samples: int
    oswald_efficiency_bounds: tuple[float, float] = (0.45, 1.00)

    def predict(self, design: AirframeDesign) -> tuple[float, float]:
        feature = (design_features(design) - self.feature_mean) / self.feature_scale
        kernel = _quadratic_trend_matern32_kernel(
            feature[None, :],
            self.training_features_normalized,
            self.trend_amplitude,
            self.dot_product_sigma,
            self.matern_amplitude,
            self.length_scales,
        )[0]
        values = self.target_mean + self.target_scale * np.sum(
            kernel * self.dual_coefficients, axis=0
        )
        return (
            float(np.clip(values[0], 3.5, 8.5)),
            float(np.clip(values[1], *self.oswald_efficiency_bounds)),
        )

    def predict_with_uncertainty(
        self, design: AirframeDesign
    ) -> tuple[float, float]:
        return self.predict(design)

    def corrected_aero(
        self, design: AirframeDesign, calibration: UltraCalibration
    ) -> AeroCoefficients:
        low_order = aero_from_design(design, calibration)
        cl_alpha, oswald = self.predict(design)
        return replace(
            low_order,
            cl_alpha_per_rad=float(
                np.clip(
                    low_order.cl_alpha_per_rad
                    * cl_alpha / max(self.baseline_cl_alpha_per_rad, 1e-9),
                    3.5,
                    8.5,
                )
            ),
            oswald_efficiency=float(
                np.clip(
                    low_order.oswald_efficiency
                    * oswald / max(self.baseline_oswald_efficiency, 1e-9),
                    *self.oswald_efficiency_bounds,
                )
            ),
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "schema": "ptdl_vspaero_quadratic_trend_matern32_gp_v1",
            "feature_labels": list(FEATURE_LABELS),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "training_features_normalized": self.training_features_normalized.tolist(),
            "dual_coefficients": self.dual_coefficients.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
            "trend_amplitude": self.trend_amplitude.tolist(),
            "dot_product_sigma": self.dot_product_sigma.tolist(),
            "matern_amplitude": self.matern_amplitude.tolist(),
            "length_scales": self.length_scales.tolist(),
            "baseline_cl_alpha_per_rad": self.baseline_cl_alpha_per_rad,
            "baseline_oswald_efficiency": self.baseline_oswald_efficiency,
            "training_samples": self.training_samples,
            "oswald_efficiency_bounds": list(self.oswald_efficiency_bounds),
            "oswald_efficiency_definition": OSWALD_EFFICIENCY_DEFINITION,
            "kernel": "Constant*DotProduct^2 + Constant*Matern(nu=1.5, ARD) + WhiteKernel",
            "model_note": "VSPAERO-only DOE model; white-noise term is used during fitting and has zero train-test cross-covariance.",
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "GaussianProcessVSPAeroSurrogate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "ptdl_vspaero_quadratic_trend_matern32_gp_v1":
            raise ValueError("Unsupported quadratic-trend Matérn GP schema.")
        if tuple(payload["feature_labels"]) != FEATURE_LABELS:
            raise ValueError("The GP feature schema does not match this optimizer.")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            training_features_normalized=np.asarray(
                payload["training_features_normalized"], dtype=float
            ),
            dual_coefficients=np.asarray(payload["dual_coefficients"], dtype=float),
            target_mean=np.asarray(payload["target_mean"], dtype=float),
            target_scale=np.asarray(payload["target_scale"], dtype=float),
            trend_amplitude=np.asarray(payload["trend_amplitude"], dtype=float),
            dot_product_sigma=np.asarray(payload["dot_product_sigma"], dtype=float),
            matern_amplitude=np.asarray(payload["matern_amplitude"], dtype=float),
            length_scales=np.asarray(payload["length_scales"], dtype=float),
            baseline_cl_alpha_per_rad=float(payload["baseline_cl_alpha_per_rad"]),
            baseline_oswald_efficiency=float(payload["baseline_oswald_efficiency"]),
            training_samples=int(payload["training_samples"]),
            oswald_efficiency_bounds=tuple(
                map(float, payload.get("oswald_efficiency_bounds", (0.45, 1.00)))
            ),
        )


@dataclass(frozen=True)
class ArdRbfGaussianProcessVSPAeroSurrogate:
    """Serialized ARD-RBF Gaussian process fitted to VSPAERO DOE labels."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_features_normalized: np.ndarray
    dual_coefficients: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    kernel_amplitude: np.ndarray
    noise_level: np.ndarray
    length_scales: np.ndarray
    inverse_covariances: np.ndarray
    baseline_cl_alpha_per_rad: float
    baseline_oswald_efficiency: float
    training_samples: int
    oswald_efficiency_bounds: tuple[float, float] = (0.45, 1.00)

    def predict_with_uncertainty(
        self, design: AirframeDesign
    ) -> tuple[float, float, float, float]:
        feature = (design_features(design) - self.feature_mean) / self.feature_scale
        means = np.empty(2, dtype=float)
        deviations = np.empty(2, dtype=float)
        for index in range(2):
            delta = (
                self.training_features_normalized - feature[None, :]
            ) / self.length_scales[index]
            cross = self.kernel_amplitude[index] * np.exp(
                -0.5 * np.sum(delta**2, axis=1)
            )
            means[index] = self.target_mean[index] + self.target_scale[index] * (
                cross @ self.dual_coefficients[:, index]
            )
            variance = max(
                0.0,
                self.kernel_amplitude[index]
                - cross @ self.inverse_covariances[index] @ cross,
            )
            deviations[index] = self.target_scale[index] * np.sqrt(variance)
        return (
            float(np.clip(means[0], 3.5, 8.5)),
            float(np.clip(means[1], *self.oswald_efficiency_bounds)),
            float(deviations[0]),
            float(deviations[1]),
        )

    def predict(self, design: AirframeDesign) -> tuple[float, float]:
        cl_alpha, oswald, _, _ = self.predict_with_uncertainty(design)
        return cl_alpha, oswald

    def corrected_aero(
        self, design: AirframeDesign, calibration: UltraCalibration
    ) -> AeroCoefficients:
        low_order = aero_from_design(design, calibration)
        cl_alpha, oswald = self.predict(design)
        return replace(
            low_order,
            cl_alpha_per_rad=float(
                np.clip(
                    low_order.cl_alpha_per_rad
                    * cl_alpha / max(self.baseline_cl_alpha_per_rad, 1e-9),
                    3.5,
                    8.5,
                )
            ),
            oswald_efficiency=float(
                np.clip(
                    low_order.oswald_efficiency
                    * oswald / max(self.baseline_oswald_efficiency, 1e-9),
                    *self.oswald_efficiency_bounds,
                )
            ),
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "schema": "ptdl_vspaero_ard_rbf_gp_v1",
            "feature_labels": list(FEATURE_LABELS),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "training_features_normalized": self.training_features_normalized.tolist(),
            "dual_coefficients": self.dual_coefficients.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
            "kernel_amplitude": self.kernel_amplitude.tolist(),
            "noise_level": self.noise_level.tolist(),
            "length_scales": self.length_scales.tolist(),
            "inverse_covariances": self.inverse_covariances.tolist(),
            "baseline_cl_alpha_per_rad": self.baseline_cl_alpha_per_rad,
            "baseline_oswald_efficiency": self.baseline_oswald_efficiency,
            "training_samples": self.training_samples,
            "oswald_efficiency_bounds": list(self.oswald_efficiency_bounds),
            "kernel": "ConstantKernel * ARD RBF + WhiteKernel",
            "prediction_std": "latent posterior standard deviation; excludes observation noise",
            "model_note": (
                "VSPAERO DOE-only model. Cross-validation error and interval "
                "coverage are conditional on the recorded design domain."
            ),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(
        cls, path: str | Path
    ) -> "ArdRbfGaussianProcessVSPAeroSurrogate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "ptdl_vspaero_ard_rbf_gp_v1":
            raise ValueError("Unsupported ARD-RBF Gaussian-process schema.")
        if tuple(payload["feature_labels"]) != FEATURE_LABELS:
            raise ValueError("The ARD-RBF GP feature schema does not match this optimizer.")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            training_features_normalized=np.asarray(
                payload["training_features_normalized"], dtype=float
            ),
            dual_coefficients=np.asarray(payload["dual_coefficients"], dtype=float),
            target_mean=np.asarray(payload["target_mean"], dtype=float),
            target_scale=np.asarray(payload["target_scale"], dtype=float),
            kernel_amplitude=np.asarray(payload["kernel_amplitude"], dtype=float),
            noise_level=np.asarray(payload["noise_level"], dtype=float),
            length_scales=np.asarray(payload["length_scales"], dtype=float),
            inverse_covariances=np.asarray(
                payload["inverse_covariances"], dtype=float
            ),
            baseline_cl_alpha_per_rad=float(payload["baseline_cl_alpha_per_rad"]),
            baseline_oswald_efficiency=float(payload["baseline_oswald_efficiency"]),
            training_samples=int(payload["training_samples"]),
            oswald_efficiency_bounds=tuple(
                map(float, payload.get("oswald_efficiency_bounds", (0.45, 1.00)))
            ),
        )


def _quadratic_trend_matern32_kernel(
    left: np.ndarray,
    right: np.ndarray,
    trend_amplitude: np.ndarray,
    dot_product_sigma: np.ndarray | float,
    matern_amplitude: np.ndarray,
    length_scales: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    amplitudes = np.atleast_1d(np.asarray(trend_amplitude, dtype=float))
    local_amplitudes = np.atleast_1d(np.asarray(matern_amplitude, dtype=float))
    sigmas = np.atleast_1d(np.asarray(dot_product_sigma, dtype=float))
    polynomial = (
        left @ right.T
    )[:, :, None] + sigmas[None, None, :] ** 2
    polynomial = polynomial**2
    scales = np.asarray(length_scales, dtype=float)
    if scales.ndim == 1:
        scales = scales[None, :]
    scaled_delta = (
        left[:, None, None, :] - right[None, :, None, :]
    ) / np.maximum(scales[None, None, :, :], 1e-12)
    radius = np.sqrt(np.sum(scaled_delta**2, axis=3))
    scaled_radius = np.sqrt(3.0) * radius
    matern = (1.0 + scaled_radius) * np.exp(-scaled_radius)
    return (
        amplitudes[None, None, :] * polynomial
        + local_amplitudes[None, None, :] * matern
    )


def load_vsp_surrogate(path: str | Path):
    """Load either the legacy Ridge or the deployable serialized GP model."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") == "ptdl_targetwise_gp_v1":
        from 分目标气动高斯过程 import TargetwiseAeroGP
        return TargetwiseAeroGP.from_json(path)
    if payload.get("schema") == "ptdl_vspaero_ard_rbf_gp_v1":
        return ArdRbfGaussianProcessVSPAeroSurrogate.from_json(path)
    if payload.get("schema") == "ptdl_vspaero_quadratic_trend_matern32_gp_v1":
        return GaussianProcessVSPAeroSurrogate.from_json(path)
    if payload.get("schema") == "ptdl_vspaero_kriging_v1":
        return KrigingVSPAeroSurrogate.from_json(path)
    return VSPAeroSurrogate.from_json(path)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    squared_distance = np.sum(
        (left[:, None, :] - right[None, :, :]) ** 2,
        axis=2,
    )
    return np.exp(-0.5 * squared_distance / max(float(length_scale) ** 2, 1e-12))


def fit_kriging_vsp_surrogate(
    designs: Sequence[AirframeDesign],
    cl_alpha_per_rad: Sequence[float],
    oswald_efficiency: Sequence[float],
) -> KrigingVSPAeroSurrogate:
    """Fit two small-data GPs; choose isotropic scale and nugget by exact LOO."""
    if len(designs) < 10:
        raise ValueError("At least ten VSPAERO designs are required to fit Kriging.")
    features = np.vstack([design_features(design) for design in designs])
    targets = np.column_stack(
        (
            np.asarray(cl_alpha_per_rad, dtype=float),
            np.asarray(oswald_efficiency, dtype=float),
        )
    )
    if targets.shape != (len(designs), 2) or not np.isfinite(targets).all():
        raise ValueError("Kriging targets must be finite vectors matching the design count.")
    feature_mean = features.mean(axis=0)
    feature_scale = np.maximum(features.std(axis=0), 1e-9)
    normalized_features = (features - feature_mean) / feature_scale
    target_mean = targets.mean(axis=0)
    target_scale = np.maximum(targets.std(axis=0), 1e-9)
    normalized_targets = (targets - target_mean) / target_scale
    pairwise_squared = np.sum(
        (normalized_features[:, None, :] - normalized_features[None, :, :]) ** 2,
        axis=2,
    )
    length_grid = (0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
    nugget_grid = (1e-6, 1e-4, 1e-3, 1e-2, 5e-2)
    selected_length_scales = np.empty(2, dtype=float)
    selected_nuggets = np.empty(2, dtype=float)
    alpha = np.empty_like(normalized_targets)
    inverse_covariances = np.empty((2, len(designs), len(designs)), dtype=float)
    identity = np.eye(len(designs))
    for target_index in range(2):
        best_score = float("inf")
        best: tuple[float, float, np.ndarray] | None = None
        values = normalized_targets[:, target_index]
        for length_scale in length_grid:
            kernel = np.exp(-0.5 * pairwise_squared / (length_scale**2))
            for nugget in nugget_grid:
                covariance = kernel + nugget * identity
                try:
                    inverse = np.linalg.inv(covariance)
                    coefficients = inverse @ values
                except np.linalg.LinAlgError:
                    continue
                loo_residual = coefficients / np.maximum(np.diag(inverse), 1e-12)
                score = float(np.mean(loo_residual**2))
                if score < best_score:
                    best_score = score
                    best = (length_scale, nugget, coefficients)
        if best is None:
            raise np.linalg.LinAlgError("Could not fit a stable Kriging covariance matrix.")
        selected_length_scales[target_index] = best[0]
        selected_nuggets[target_index] = best[1]
        alpha[:, target_index] = best[2]
        kernel = _rbf_kernel(
            normalized_features,
            normalized_features,
            selected_length_scales[target_index],
        )
        inverse_covariances[target_index] = np.linalg.inv(
            kernel + selected_nuggets[target_index] * identity
        )
    try:
        baseline_cl, baseline_e = _baseline_targets(
            designs, cl_alpha_per_rad, oswald_efficiency
        )
    except ValueError:
        baseline = (design_features(BASELINE) - feature_mean) / feature_scale
        baseline_cl = float(
            target_mean[0]
            + target_scale[0]
            * (
                _rbf_kernel(
                    normalized_features, baseline[None, :], selected_length_scales[0]
                )[:, 0]
                @ alpha[:, 0]
            )
        )
        baseline_e = float(
            target_mean[1]
            + target_scale[1]
            * (
                _rbf_kernel(
                    normalized_features, baseline[None, :], selected_length_scales[1]
                )[:, 0]
                @ alpha[:, 1]
            )
        )
    return KrigingVSPAeroSurrogate(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        training_features_normalized=normalized_features,
        target_mean=target_mean,
        target_scale=target_scale,
        target_normalized=normalized_targets,
        length_scales=selected_length_scales,
        nuggets=selected_nuggets,
        alpha=alpha,
        inverse_covariances=inverse_covariances,
        baseline_cl_alpha_per_rad=baseline_cl,
        baseline_oswald_efficiency=baseline_e,
        training_samples=len(designs),
    )


def fit_vsp_surrogate(
    designs: Sequence[AirframeDesign],
    cl_alpha_per_rad: Sequence[float],
    oswald_efficiency: Sequence[float],
    ridge_lambda: float = 0.15,
) -> VSPAeroSurrogate:
    if len(designs) < 10:
        raise ValueError("At least ten VSPAERO designs are required to fit the aerodynamic surrogate.")
    features = np.vstack([design_features(design) for design in designs])
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    matrix = np.vstack([_basis((row - mean) / scale) for row in features])
    penalty = ridge_lambda * np.eye(matrix.shape[1])
    penalty[0, 0] = 0.0
    normal = matrix.T @ matrix + penalty
    cl_coefficients = np.linalg.solve(normal, matrix.T @ np.asarray(cl_alpha_per_rad, dtype=float))
    oswald_coefficients = np.linalg.solve(normal, matrix.T @ np.asarray(oswald_efficiency, dtype=float))
    try:
        baseline_cl, baseline_e = _baseline_targets(designs, cl_alpha_per_rad, oswald_efficiency)
    except ValueError:
        baseline_row = _basis((design_features(BASELINE) - mean) / scale)
        baseline_cl = float(baseline_row @ cl_coefficients)
        baseline_e = float(baseline_row @ oswald_coefficients)
    return VSPAeroSurrogate(
        feature_mean=mean,
        feature_scale=scale,
        cl_alpha_coefficients=cl_coefficients,
        oswald_coefficients=oswald_coefficients,
        baseline_cl_alpha_per_rad=baseline_cl,
        baseline_oswald_efficiency=baseline_e,
        ridge_lambda=ridge_lambda,
        training_samples=len(designs),
    )


def _baseline_targets(
    designs: Sequence[AirframeDesign], cl_alpha: Sequence[float], oswald: Sequence[float]
) -> tuple[float, float]:
    distances = [np.linalg.norm(design_features(design) - design_features(BASELINE)) for design in designs]
    index = int(np.argmin(distances))
    if distances[index] > 1e-9:
        raise ValueError("The VSPAERO DOE must include the declared baseline design.")
    return float(cl_alpha[index]), float(oswald[index])
