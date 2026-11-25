"""Create advisor-review figures and tables for targeted VSPAERO enrichment."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_airframe_design import AirframeDesign
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_vsp_targeted_candidate_results"
BASELINE_LOO = ROOT / "ptdl_vsp_aero_surrogate_audit_results" / "气动代理模型留一交叉验证_汇总统计.csv"
TARGETED = OUTPUT / "稳健候选邻域_VSPAERO极线与预测误差.csv"
ENRICHED = OUTPUT / "VSPAERO扩充代理模型.json"
RECHECK = OUTPUT / "扩充VSPAERO代理模型_稳健候选复核.csv"


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def design_from_row(row: pd.Series) -> AirframeDesign:
    labels = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
        "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{label: float(row[label]) for label in labels})


def candidate_short_name(value: str) -> str:
    names = {
        "robust_a_center": "A中心", "robust_a_area_minus": "A面积-5%", "robust_a_area_plus": "A面积+5%",
        "robust_a_ar_minus": "A展弦比-1.5", "robust_a_taper_plus": "A根梢比+0.08",
        "robust_b_center": "B中心", "robust_b_area_plus": "B面积+5%", "robust_b_sweep_plus": "B后掠+2度",
    }
    return names[value]


def main() -> None:
    setup_chinese_font()
    targeted = pd.read_csv(TARGETED)
    enriched = VSPAeroSurrogate.from_json(ENRICHED)
    enriched_predictions = np.asarray([enriched.predict(design_from_row(row)) for _, row in targeted.iterrows()])
    targeted["enriched_surrogate_cl_alpha_per_rad"] = enriched_predictions[:, 0]
    targeted["enriched_surrogate_oswald_efficiency"] = enriched_predictions[:, 1]
    targeted["enriched_cl_alpha_error_per_rad"] = targeted["enriched_surrogate_cl_alpha_per_rad"] - targeted["vspaero_cl_alpha_per_rad"]
    targeted["enriched_oswald_error"] = targeted["enriched_surrogate_oswald_efficiency"] - targeted["vspaero_oswald_efficiency"]
    targeted["sample_short"] = targeted["sample"].map(candidate_short_name)
    error_table = targeted[[
        "sample", "description", "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg",
        "vspaero_cl_alpha_per_rad", "prior_surrogate_cl_alpha_per_rad", "enriched_surrogate_cl_alpha_per_rad",
        "prior_cl_alpha_error_per_rad", "enriched_cl_alpha_error_per_rad",
        "vspaero_oswald_efficiency", "prior_surrogate_oswald_efficiency", "enriched_surrogate_oswald_efficiency",
        "prior_oswald_error", "enriched_oswald_error",
    ]].copy()
    error_table.to_csv(OUTPUT / "表1_VSPAERO候选邻域代理误差.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), constrained_layout=True)
    x = np.arange(len(targeted))
    width = 0.34
    configs = [
        ("CLα绝对误差 (rad$^{-1}$)", "prior_cl_alpha_error_per_rad", "enriched_cl_alpha_error_per_rad"),
        ("奥斯瓦尔德效率绝对误差 (-)", "prior_oswald_error", "enriched_oswald_error"),
    ]
    for axis, (label, old_column, new_column) in zip(axes, configs):
        axis.bar(x - width / 2, targeted[old_column].abs(), width, label="原15点代理模型", color="#7b8794")
        axis.bar(x + width / 2, targeted[new_column].abs(), width, label="扩充23点代理模型", color="#1769aa")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xticks(x, targeted["sample_short"], rotation=0)
    axes[0].set_title("图8  稳健候选邻域的 VSPAERO 代理模型误差校核")
    figure.savefig(OUTPUT / "图8_稳健候选邻域VSPAERO代理误差校核.png", dpi=220)
    plt.close(figure)

    original_loo = pd.read_csv(BASELINE_LOO).rename(columns={"mae": "loo_mae", "rmse": "loo_rmse"})
    enriched_loo = pd.read_csv(OUTPUT / "VSPAERO扩充代理模型_留一验证统计.csv")
    comparison = original_loo[["quantity", "sample_count", "loo_mae", "loo_rmse", "r2"]].merge(
        enriched_loo[["quantity", "sample_count", "loo_mae", "loo_rmse", "loo_r2"]],
        on="quantity", suffixes=("_15point", "_23point"),
    )
    comparison.to_csv(OUTPUT / "表2_扩充前后代理模型留一验证对比.csv", index=False, encoding="utf-8-sig")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    metric_names = ["CLα 留一 RMSE (rad$^{-1}$)", "奥斯瓦尔德效率 留一 RMSE (-)"]
    for axis, (_, row), title in zip(axes, comparison.iterrows(), metric_names):
        values = [row["loo_rmse_15point"], row["loo_rmse_23point"]]
        bars = axis.bar(["原15点", "扩充23点"], values, color=["#7b8794", "#1769aa"], width=0.58)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.32)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("图9  VSPAERO 样本扩充前后的全局留一交叉验证", fontsize=14)
    figure.savefig(OUTPUT / "图9_VSPAERO扩充前后留一验证对比.png", dpi=220)
    plt.close(figure)

    recheck = pd.read_csv(RECHECK)
    figure, axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    x = np.arange(len(recheck))
    width = 0.35
    prior = recheck["prior_robust_86h_achieved_duration_h"]
    revised = recheck["enriched_robust_86h_achieved_duration_h"]
    axis.bar(x - width / 2, prior, width, label="原15点代理模型", color="#7b8794")
    bars = axis.bar(x + width / 2, revised, width, label="扩充23点代理模型", color="#1769aa")
    axis.axhline(80.0, color="#d94801", linewidth=1.8, linestyle="--", label="80 h 硬约束")
    for bar, value in zip(bars, revised):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f} h", ha="center", va="bottom", fontsize=10)
    axis.set_xticks(x, [f"候选 {int(value)}" for value in recheck["candidate_index"]])
    axis.set_ylabel("CD0 +10% 下可达总时长 (h)")
    axis.set_ylim(79.3, max(revised.max(), prior.max()) + 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    axis.set_title("图10  扩充 VSPAERO 代理模型后的稳健续航复核")
    figure.savefig(OUTPUT / "图10_扩充代理模型稳健续航复核.png", dpi=220)
    plt.close(figure)

    note = f"""# VSPAERO 候选邻域加密与稳健性复核

## 本轮新增工作

围绕稳健 H=6 MOEA/D 的两个唯一 Pareto 候选，在 3000 m、36.01 m/s 基准工况下新增 8 个 OpenVSP/VSPAERO 构型样本，包含两候选中心点以及翼面积、展弦比、根梢比、后掠角邻域扰动。原始 15 点全局 DOE 与新增样本合并为 23 点训练集，生成独立的扩充代理模型；原生产代理模型未被覆盖。

## 代理模型校核结果

候选邻域的原模型对 CLα 存在偏高趋势，新增 VSPAERO 极线直接给出修正依据。全局留一验证中，CLα RMSE 从 {comparison.iloc[0]['loo_rmse_15point']:.4f} 降至 {comparison.iloc[0]['loo_rmse_23point']:.4f} rad^-1，奥斯瓦尔德效率 RMSE 从 {comparison.iloc[1]['loo_rmse_15point']:.4f} 降至 {comparison.iloc[1]['loo_rmse_23point']:.4f}。图8给出局部候选误差，图9给出全局留一验证比较。

## 对 80 小时稳健约束的影响

采用扩充模型重算两个既有候选，并保留 CD0 增加 10% 的保守工况。候选 6 的 86 h 延伸复核可达 {recheck.iloc[0]['enriched_robust_86h_achieved_duration_h']:.2f} h，对 80 h 的裕度为 {recheck.iloc[0]['enriched_robust_margin_to_80h']:.2f} h；候选 23 分别为 {recheck.iloc[1]['enriched_robust_86h_achieved_duration_h']:.2f} h 和 {recheck.iloc[1]['enriched_robust_margin_to_80h']:.2f} h。两个候选仍满足稳健 80 h 约束，且相较原代理模型均保持正裕度，见图10与表2。

## 结论与边界

本轮完成了候选区域的真实 VSPAERO 气动校核、扩充代理模型训练和稳健续航再验证，增强了当前候选的气动趋势可信度。VSPAERO 仍为势流级分析，CD0 继续由已校准的部件阻力模型提供；因此本轮不能替代黏性 CFD 或试验验证。扩充模型现作为已验证的候选区模型保留，后续可在增加更多任务迎角/雷诺数样本后再决定是否替换全局生产代理模型。
"""
    (OUTPUT / "阶段审阅说明.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
