"""Audit the PTDL VSPAERO aerodynamic surrogate by leave-one-out validation.

This study re-fits the surrogate after withholding one VSPAERO DOE geometry
and predicts that withheld geometry.  It therefore reports generalization
within the current 15-point DOE, rather than the much smaller in-sample fit
error.  The result is intended for the stage-review evidence package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ptdl_airframe_design import AirframeDesign
from ptdl_vsp_aero_surrogate import FEATURE_LABELS, fit_vsp_surrogate


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_doe_training.csv"
OUTPUT = ROOT / "ptdl_vsp_aero_surrogate_audit_results"


def _design_from_row(row: pd.Series) -> AirframeDesign:
    return AirframeDesign(
        wing_area_m2=float(row["wing_area_m2"]),
        aspect_ratio=float(row["aspect_ratio"]),
        taper_ratio=float(row["taper_ratio"]),
        sweep_deg=float(row["sweep_deg"]),
        tip_twist_deg=float(row["tip_twist_deg"]),
        fuselage_length_m=float(row["fuselage_length_m"]),
        fuselage_diameter_m=float(row["fuselage_diameter_m"]),
        propeller_diameter_m=float(row["propeller_diameter_m"]),
        payload_mass_kg=float(row["payload_mass_kg"]),
        hybrid_system_mass_kg=float(row["hybrid_system_mass_kg"]),
    )


def _statistics(actual: np.ndarray, predicted: np.ndarray, target: str) -> dict[str, float | str]:
    error = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 1e-12 else float("nan")
    return {
        "quantity": target,
        "sample_count": int(actual.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_bias": float(np.mean(error)),
        "r2": r2,
    }


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _map(value: float, lower: float, upper: float, pixel_lower: float, pixel_upper: float) -> float:
    return pixel_lower + (value - lower) * (pixel_upper - pixel_lower) / max(upper - lower, 1e-12)


def _scatter_panel(draw: ImageDraw.ImageDraw, frame: pd.DataFrame, box: tuple[int, int, int, int], actual_column: str, predicted_column: str, label: str) -> None:
    left, top, right, bottom = box
    actual = frame[actual_column].to_numpy(dtype=float)
    predicted = frame[predicted_column].to_numpy(dtype=float)
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    margin = 0.07 * max(upper - lower, 1e-3)
    lower -= margin
    upper += margin
    draw.rounded_rectangle(box, radius=8, outline="#CBD5E1", width=2, fill="white")
    plot = (left + 72, top + 50, right - 28, bottom - 60)
    px0, py0, px1, py1 = plot
    for fraction in np.linspace(0.0, 1.0, 5):
        x = px0 + fraction * (px1 - px0)
        y = py1 - fraction * (py1 - py0)
        draw.line((x, py0, x, py1), fill="#E5E7EB")
        draw.line((px0, y, px1, y), fill="#E5E7EB")
    draw.line((px0, py1, px1, py0), fill="#64748B", width=2)
    for _, row in frame.iterrows():
        x = _map(float(row[actual_column]), lower, upper, px0, px1)
        y = _map(float(row[predicted_column]), lower, upper, py1, py0)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#0B5CAD", outline="white", width=1)
        draw.text((x + 6, y - 9), str(row["sample"]), fill="#334155", font=_font(11))
    draw.text((left + 18, top + 16), label, fill="#0F172A", font=_font(20))
    draw.text((px0, bottom - 36), f"VSPAERO: {lower:.3f} to {upper:.3f}", fill="#475569", font=_font(13))
    draw.text((left + 14, py0 - 24), "LOO prediction", fill="#475569", font=_font(13))


def _plot_validation(frame: pd.DataFrame) -> None:
    image = Image.new("RGB", (1760, 780), "#F8FAFC")
    draw = ImageDraw.Draw(image)
    draw.text((55, 28), "VSPAERO Surrogate: Leave-One-Out Cross Validation", fill="#0F172A", font=_font(30))
    _scatter_panel(draw, frame, (45, 90, 865, 735), "vspaero_cl_alpha_per_rad", "loo_cl_alpha_per_rad", "Lift-curve slope CL_alpha (1/rad)")
    _scatter_panel(draw, frame, (895, 90, 1715, 735), "vspaero_oswald_efficiency", "loo_oswald_efficiency", "Oswald efficiency e")
    image.save(OUTPUT / "图1_气动代理模型留一交叉验证.png")


def _bar_panel(draw: ImageDraw.ImageDraw, frame: pd.DataFrame, box: tuple[int, int, int, int], column: str, title: str, color: str) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, outline="#CBD5E1", width=2, fill="white")
    values = frame[column].to_numpy(dtype=float)
    maximum = max(float(values.max()) * 1.20, 1e-4)
    px0, py0, px1, py1 = left + 82, top + 50, right - 30, bottom - 55
    for fraction in np.linspace(0.0, 1.0, 5):
        y = py1 - fraction * (py1 - py0)
        draw.line((px0, y, px1, y), fill="#E5E7EB")
        draw.text((left + 12, y - 8), f"{fraction * maximum:.3f}", fill="#64748B", font=_font(12))
    bar_width = (px1 - px0) / len(frame) * 0.62
    for index, (_, row) in enumerate(frame.iterrows()):
        center = px0 + (index + 0.5) * (px1 - px0) / len(frame)
        bar_top = _map(float(row[column]), 0.0, maximum, py1, py0)
        draw.rectangle((center - bar_width / 2, bar_top, center + bar_width / 2, py1), fill=color)
        draw.text((center - 13, py1 + 8), str(row["sample"]), fill="#475569", font=_font(11))
    draw.text((left + 18, top + 16), title, fill="#0F172A", font=_font(20))


def _plot_errors(frame: pd.DataFrame) -> None:
    ordered = frame.sort_values("sample").reset_index(drop=True)
    image = Image.new("RGB", (1760, 990), "#F8FAFC")
    draw = ImageDraw.Draw(image)
    draw.text((55, 28), "VSPAERO Surrogate: Absolute Error by Withheld Geometry", fill="#0F172A", font=_font(30))
    _bar_panel(draw, ordered, (45, 90, 1715, 510), "cl_alpha_abs_error_per_rad", "Absolute error of CL_alpha (1/rad)", "#2563EB")
    _bar_panel(draw, ordered, (45, 545, 1715, 955), "oswald_abs_error", "Absolute error of Oswald efficiency e", "#D97706")
    image.save(OUTPUT / "图2_气动代理模型留一预测绝对误差.png")


def _write_review_note(statistics: pd.DataFrame) -> None:
    cl = statistics.iloc[0]
    oswald = statistics.iloc[1]
    note = f"""# VSPAERO 气动代理模型阶段审计说明

## 验证方式

本次审计采用 15 个现有 VSPAERO 构型样本进行留一交叉验证（LOO-CV）：每次剔除 1 个构型，以其余 14 个构型重新拟合岭回归代理模型，再预测被剔除构型的 `CL_alpha` 与奥斯瓦尔德效率 `e`。因此，统计值反映代理模型在当前 DOE 参数范围内对未参与当前次拟合构型的预测能力，而不是训练集拟合误差。

## 结果摘要

`CL_alpha` 的 LOO 平均绝对误差为 {cl['mae']:.4f} 1/rad，均方根误差为 {cl['rmse']:.4f} 1/rad，最大绝对误差为 {cl['max_abs_error']:.4f} 1/rad；`e` 的 LOO 平均绝对误差为 {oswald['mae']:.4f}，均方根误差为 {oswald['rmse']:.4f}，最大绝对误差为 {oswald['max_abs_error']:.4f}。

## 使用边界

本结果仅支持当前 15 点 DOE 覆盖的构型区间内进行趋势校正。VSPAERO 为无粘 VLM 分析，当前代理模型只修正 `CL_alpha` 和 `e`；零升阻力 `CD0` 仍由已按 ULTRA 基准标定的部件阻力建立模型在任务积分步内按雷诺数计算。由于运行模型将 `e` 限制在 0.55 至 0.95 的工程可用区间，原始 VSPAERO 样本中高于 0.95 的数值会在实际优化调用中被截断，避免把无粘高估直接引入续航结论。

## 后续建议

在下一轮构型优化前，建议围绕当前 Pareto 候选及误差较大区域补充 VSPAERO 样本；待推进安装构型明确后，再用黏性 CFD 或试验数据替换/校核 `CD0` 与螺旋桨-机翼干扰项。
"""
    (OUTPUT / "审阅说明.md").write_text(note, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(INPUT)
    designs = [_design_from_row(row) for _, row in source.iterrows()]
    cl_targets = source["vspaero_cl_alpha_per_rad"].to_numpy(dtype=float)
    e_targets = source["vspaero_oswald_efficiency"].to_numpy(dtype=float)
    rows: list[dict[str, float | str]] = []
    for withheld in range(len(source)):
        keep = [index for index in range(len(source)) if index != withheld]
        surrogate = fit_vsp_surrogate(
            [designs[index] for index in keep], cl_targets[keep], e_targets[keep], require_baseline=False
        )
        predicted_cl, predicted_e = surrogate.predict(designs[withheld])
        row = {"sample": str(source.iloc[withheld]["sample"])}
        for feature in FEATURE_LABELS:
            row[feature] = float(source.iloc[withheld][feature])
        row.update(
            {
                "vspaero_cl_alpha_per_rad": float(cl_targets[withheld]),
                "loo_cl_alpha_per_rad": predicted_cl,
                "cl_alpha_error_per_rad": predicted_cl - float(cl_targets[withheld]),
                "cl_alpha_abs_error_per_rad": abs(predicted_cl - float(cl_targets[withheld])),
                "vspaero_oswald_efficiency": float(e_targets[withheld]),
                "loo_oswald_efficiency": predicted_e,
                "oswald_error": predicted_e - float(e_targets[withheld]),
                "oswald_abs_error": abs(predicted_e - float(e_targets[withheld])),
            }
        )
        rows.append(row)
    results = pd.DataFrame(rows)
    statistics = pd.DataFrame(
        [
            _statistics(cl_targets, results["loo_cl_alpha_per_rad"].to_numpy(dtype=float), "CL_alpha_per_rad"),
            _statistics(e_targets, results["loo_oswald_efficiency"].to_numpy(dtype=float), "oswald_efficiency"),
        ]
    )
    results.to_csv(OUTPUT / "气动代理模型留一交叉验证_预测明细.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(OUTPUT / "气动代理模型留一交叉验证_汇总统计.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "气动代理模型留一交叉验证_汇总统计.json").write_text(
        json.dumps(statistics.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_validation(results)
    _plot_errors(results)
    _write_review_note(statistics)
    print(statistics.to_string(index=False))
    print(f"Wrote phase-review outputs to: {OUTPUT}")


if __name__ == "__main__":
    main()
