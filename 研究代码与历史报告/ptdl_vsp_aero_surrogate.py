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

from ptdl_airframe_design import BASELINE, AeroCoefficients, AirframeDesign, aero_from_design
from ptdl_ultra_benchmark import UltraCalibration


FEATURE_LABELS = (
    "wing_area_m2",
    "aspect_ratio",
    "taper_ratio",
    "sweep_deg",
    "tip_twist_deg",
    "fuselage_length_m",
    "fuselage_diameter_m",
)


def design_features(design: AirframeDesign) -> np.ndarray:
    return np.asarray([getattr(design, label) for label in FEATURE_LABELS], dtype=float)


def _basis(normalized_features: np.ndarray) -> np.ndarray:
    """Use a low-variance linear basis suitable for the 15-point VSPAERO DOE.

    The original 13-term quadratic form was over-parameterized for the
    available sample count.  Leave-one-out selection showed that lift-slope
    should use the planform subset and Oswald efficiency can use all geometry
    inputs; unused coefficient rows are set to zero in ``fit_vsp_surrogate``.
    Keeping one shared linear basis preserves the JSON/runtime interface.
    """
    x = np.asarray(normalized_features, dtype=float)
    return np.concatenate((np.ones(1), x))


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

    def predict(self, design: AirframeDesign) -> tuple[float, float]:
        normalized = (design_features(design) - self.feature_mean) / self.feature_scale
        row = _basis(normalized)
        cl_alpha = float(row @ self.cl_alpha_coefficients)
        oswald = float(row @ self.oswald_coefficients)
        return float(np.clip(cl_alpha, 3.5, 8.5)), float(np.clip(oswald, 0.55, 0.95))

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
            oswald_efficiency=float(np.clip(low_order.oswald_efficiency * e_ratio, 0.55, 0.95)),
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
            "regression_spec": {
                "cl_alpha_features": ["aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg"],
                "cl_alpha_ridge_lambda": 1.0,
                "oswald_features": list(FEATURE_LABELS),
                "oswald_ridge_lambda": 0.5,
                "selection_method": "15-point VSPAERO DOE leave-one-out cross-validation",
            },
            "model_note": "VSPAERO VLM corrects CL_alpha and Oswald-efficiency trends; calibrated component-build-up parasite drag remains separate.",
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
        )


def fit_vsp_surrogate(
    designs: Sequence[AirframeDesign],
    cl_alpha_per_rad: Sequence[float],
    oswald_efficiency: Sequence[float],
    ridge_lambda: float | None = None,
    require_baseline: bool = True,
) -> VSPAeroSurrogate:
    if len(designs) < 10:
        raise ValueError("At least ten VSPAERO designs are required to fit the aerodynamic surrogate.")
    features = np.vstack([design_features(design) for design in designs])
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    matrix = np.vstack([_basis((row - mean) / scale) for row in features])
    # Candidate structures and regularization strengths are selected by the
    # companion LOO study, not by in-sample error.  ``ridge_lambda`` remains
    # accepted for backward call compatibility but is deliberately ignored.
    cl_coefficients = _fit_selected_features(
        matrix, np.asarray(cl_alpha_per_rad, dtype=float), feature_indices=(1, 2, 3, 4), ridge_lambda=1.0
    )
    oswald_coefficients = _fit_selected_features(
        matrix, np.asarray(oswald_efficiency, dtype=float), feature_indices=tuple(range(len(FEATURE_LABELS))), ridge_lambda=0.5
    )
    if require_baseline:
        baseline_cl, baseline_e = _baseline_targets(designs, cl_alpha_per_rad, oswald_efficiency)
    else:
        baseline_cl, baseline_e = float("nan"), float("nan")
    return VSPAeroSurrogate(
        feature_mean=mean,
        feature_scale=scale,
        cl_alpha_coefficients=cl_coefficients,
        oswald_coefficients=oswald_coefficients,
        baseline_cl_alpha_per_rad=baseline_cl,
        baseline_oswald_efficiency=baseline_e,
        ridge_lambda=0.5,
        training_samples=len(designs),
    )


def _fit_selected_features(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_indices: tuple[int, ...],
    ridge_lambda: float,
) -> np.ndarray:
    """Fit active features then expand coefficients to the shared basis."""
    active_columns = (0,) + tuple(index + 1 for index in feature_indices)
    active_matrix = matrix[:, active_columns]
    penalty = ridge_lambda * np.eye(active_matrix.shape[1])
    penalty[0, 0] = 0.0
    compact = np.linalg.solve(active_matrix.T @ active_matrix + penalty, active_matrix.T @ target)
    coefficients = np.zeros(matrix.shape[1])
    coefficients[list(active_columns)] = compact
    return coefficients


def _baseline_targets(
    designs: Sequence[AirframeDesign], cl_alpha: Sequence[float], oswald: Sequence[float]
) -> tuple[float, float]:
    distances = [np.linalg.norm(design_features(design) - design_features(BASELINE)) for design in designs]
    index = int(np.argmin(distances))
    if distances[index] > 1e-9:
        raise ValueError("The VSPAERO DOE must include the declared baseline design.")
    return float(cl_alpha[index]), float(oswald[index])
