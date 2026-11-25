"""Create figures and review note for low-speed/high-AR VSPAERO validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_vsp_low_speed_high_ar_validation_results"
RESULTS = OUTPUT / "低速高展弦比VSPAERO验证结果.csv"


def main() -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    frame = pd.read_csv(RESULTS)
    frame["cl_alpha_abs_error_per_rad"] = frame["cl_alpha_error_per_rad"].abs()
    frame["oswald_abs_error"] = frame["oswald_error"].abs()
    frame.to_csv(OUTPUT / "表5_低速高展弦比气动代理验证误差.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for _, row in frame.iterrows():
        polar = pd.read_csv(ROOT / row["geometry_file"].replace("airframe.vsp3", "vspaero_polar.csv"))
        axes[0].plot(polar["alpha_deg"], polar["cl"], marker="o", label=f"{row['description']} ({row['speed_mps']:.0f} m/s)")
        axes[1].plot(polar["alpha_deg"], polar["cd_induced"], marker="o", label=f"{row['description']} ({row['speed_mps']:.0f} m/s)")
    axes[0].set(title="VSPAERO 升力极线", xlabel="迎角 (deg)", ylabel="CL (-)")
    axes[1].set(title="VSPAERO 诱导阻力极线", xlabel="迎角 (deg)", ylabel="CDi (-)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, frameon=False)
    figure.savefig(OUTPUT / "图14_低速高展弦比候选VSPAERO气动极线.png", dpi=220)
    plt.close(figure)

    x = np.arange(len(frame))
    labels = [f"{row['sample']}\n{row['speed_mps']:.0f} m/s" for _, row in frame.iterrows()]
    figure, axes = plt.subplots(2, 1, figsize=(11.8, 7.0), constrained_layout=True)
    axes[0].bar(x, frame["cl_alpha_abs_error_per_rad"], color="#1769aa")
    axes[0].set(ylabel="|CLα 代理误差| (rad$^{-1}$)", title="低速/高展弦比区域的气动代理误差")
    axes[1].bar(x, frame["oswald_abs_error"], color="#0891b2")
    axes[1].set(ylabel="|e 代理误差| (-)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(OUTPUT / "图15_低速高展弦比区域气动代理误差.png", dpi=220)
    plt.close(figure)

    maximum = frame.loc[frame["cl_alpha_abs_error_per_rad"].idxmax()]
    note = f"""# 低速与高展弦比 VSPAERO 定向校核

在 3000 m 下补充了 5 组真实 VSPAERO 极线，覆盖两个稳健候选在 30 m/s 低速 ISR、候选 6 在 42 m/s 的较高巡航速度，以及展弦比 31、翼面积正负 5% 的邻域构型。图14给出升力和诱导阻力极线，图15和表5给出扩充代理模型误差。

最大 CLα 偏差出现在“{maximum['description']}”工况，为 {maximum['cl_alpha_abs_error_per_rad']:.3f} rad^-1；其奥斯瓦尔德效率偏差为 {maximum['oswald_abs_error']:.3f}。因此，当前 23 点代理模型可用于候选区的阶段性筛选，但候选 6 的低速区域仍需要额外样本或更保守的气动裕度。

VSPAERO 使用势流/VLM 形式，所得速度变化主要用于检查几何气动趋势与数值稳定性；黏性 Re 效应和 CD0 速度修正仍由任务积分中的部件阻力模型负责，不能将本图解释为完整黏性 CFD 验证。
"""
    (OUTPUT / "阶段审阅说明.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
