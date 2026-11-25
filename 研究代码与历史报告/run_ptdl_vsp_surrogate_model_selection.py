"""Select a low-sample VSPAERO surrogate basis using leave-one-out validation.

The original 15-point DOE was paired with a 13-term quadratic basis.  This
script evaluates lower-dimensional physical feature sets before any production
surrogate is changed.  It is a model-development study, not a plotting-only
postprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ptdl_vsp_aero_surrogate import FEATURE_LABELS


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_doe_training.csv"
OUTPUT = ROOT / "ptdl_vsp_aero_surrogate_model_selection_results"


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_indices: tuple[int, ...]
    ridge_lambda: float
    include_squares: bool = False


CANDIDATES = (
    Candidate("constant_mean", (), 0.0),
    Candidate("linear_AR", (1,), 0.15),
    Candidate("linear_AR_sweep_twist", (1, 3, 4), 0.15),
    Candidate("linear_planform", (1, 2, 3, 4), 0.15),
    Candidate("linear_planform_strong_ridge", (1, 2, 3, 4), 1.00),
    Candidate("linear_all_geometry", tuple(range(7)), 0.50),
    Candidate("quadratic_AR_sweep_twist", (1, 3, 4), 1.00, True),
)


def _matrix(features: np.ndarray, candidate: Candidate, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if not candidate.feature_indices:
        return np.ones((features.shape[0], 1))
    x = (features[:, candidate.feature_indices] - mean) / scale
    columns = [np.ones(features.shape[0]), *[x[:, index] for index in range(x.shape[1])]]
    if candidate.include_squares:
        columns.extend(x[:, index] ** 2 for index in range(x.shape[1]))
        for left in range(x.shape[1]):
            for right in range(left + 1, x.shape[1]):
                columns.append(x[:, left] * x[:, right])
    return np.column_stack(columns)


def _loo_predictions(features: np.ndarray, target: np.ndarray, candidate: Candidate, bounds: tuple[float, float]) -> np.ndarray:
    predictions = np.empty(len(target))
    for withheld in range(len(target)):
        keep = np.arange(len(target)) != withheld
        if not candidate.feature_indices:
            predictions[withheld] = float(np.mean(target[keep]))
            continue
        selected = features[:, candidate.feature_indices]
        mean = selected[keep].mean(axis=0)
        scale = np.maximum(selected[keep].std(axis=0), 1e-6)
        matrix = _matrix(features, candidate, mean, scale)
        penalty = candidate.ridge_lambda * np.eye(matrix.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(matrix[keep].T @ matrix[keep] + penalty, matrix[keep].T @ target[keep])
        predictions[withheld] = float(np.clip(matrix[withheld] @ coefficients, *bounds))
    return predictions


def _record(candidate: Candidate, actual: np.ndarray, predicted: np.ndarray, quantity: str) -> dict[str, object]:
    error = predicted - actual
    return {
        "quantity": quantity,
        "candidate": candidate.name,
        "features": ", ".join(FEATURE_LABELS[index] for index in candidate.feature_indices) or "intercept only",
        "quadratic_terms": candidate.include_squares,
        "ridge_lambda": candidate.ridge_lambda,
        "parameter_count": 1 + len(candidate.feature_indices) + (len(candidate.feature_indices) * (len(candidate.feature_indices) + 1) // 2 if candidate.include_squares else 0),
        "loo_mae": float(np.mean(np.abs(error))),
        "loo_rmse": float(np.sqrt(np.mean(error**2))),
        "loo_max_abs_error": float(np.max(np.abs(error))),
        "loo_bias": float(np.mean(error)),
    }


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _plot(summary: pd.DataFrame) -> None:
    image = Image.new("RGB", (1750, 860), "#F8FAFC")
    draw = ImageDraw.Draw(image)
    draw.text((52, 28), "VSPAERO Surrogate Model Selection: LOO RMSE", fill="#0F172A", font=_font(30))
    panels = (("CL_alpha_per_rad", "CL_alpha LOO RMSE", "#2563EB"), ("oswald_efficiency", "Oswald e LOO RMSE", "#D97706"))
    for panel_index, (quantity, title, color) in enumerate(panels):
        left, top, right, bottom = 48, 95 + 365 * panel_index, 1700, 425 + 365 * panel_index
        draw.rounded_rectangle((left, top, right, bottom), radius=8, outline="#CBD5E1", width=2, fill="white")
        frame = summary[summary["quantity"] == quantity].sort_values("loo_rmse").reset_index(drop=True)
        maximum = max(float(frame["loo_rmse"].max()) * 1.18, 1e-4)
        x0, y0, x1, y1 = left + 110, top + 54, right - 35, bottom - 50
        for fraction in np.linspace(0.0, 1.0, 5):
            y = y1 - fraction * (y1 - y0)
            draw.line((x0, y, x1, y), fill="#E5E7EB")
            draw.text((left + 18, y - 8), f"{fraction * maximum:.3f}", fill="#64748B", font=_font(12))
        width = (x1 - x0) / len(frame) * 0.62
        for index, (_, row) in enumerate(frame.iterrows()):
            center = x0 + (index + 0.5) * (x1 - x0) / len(frame)
            top_y = y1 - float(row["loo_rmse"]) / maximum * (y1 - y0)
            fill = "#16A34A" if index == 0 else color
            draw.rectangle((center - width / 2, top_y, center + width / 2, y1), fill=fill)
            draw.text((center - 48, y1 + 8), str(row["candidate"]), fill="#475569", font=_font(11))
        draw.text((left + 18, top + 16), title + " (green = selected)", fill="#0F172A", font=_font(20))
    image.save(OUTPUT / "图3_VSPAERO代理模型候选结构留一误差对比.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    features = frame[list(FEATURE_LABELS)].to_numpy(dtype=float)
    targets = (
        ("CL_alpha_per_rad", frame["vspaero_cl_alpha_per_rad"].to_numpy(dtype=float), (3.5, 8.5)),
        ("oswald_efficiency", frame["vspaero_oswald_efficiency"].to_numpy(dtype=float), (0.55, 0.95)),
    )
    rows: list[dict[str, object]] = []
    for quantity, target, bounds in targets:
        for candidate in CANDIDATES:
            prediction = _loo_predictions(features, target, candidate, bounds)
            rows.append(_record(candidate, target, prediction, quantity))
    summary = pd.DataFrame(rows).sort_values(["quantity", "loo_rmse"]).reset_index(drop=True)
    selected = summary.groupby("quantity", as_index=False).first()
    summary.to_csv(OUTPUT / "VSPAERO代理模型候选结构_留一误差对比.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUTPUT / "VSPAERO代理模型建议结构.csv", index=False, encoding="utf-8-sig")
    _plot(summary)
    note = "# VSPAERO 气动代理模型结构选择\n\n"
    note += "按留一交叉验证 RMSE 最小准则，`CL_alpha` 与 `e` 可分别选择其对应的低样本回归结构。"
    note += " 该选择仅在当前 DOE 覆盖范围内有效，后续补点后必须重新执行本脚本。\n\n"
    note += selected.to_markdown(index=False)
    (OUTPUT / "模型选择结论.md").write_text(note, encoding="utf-8")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
