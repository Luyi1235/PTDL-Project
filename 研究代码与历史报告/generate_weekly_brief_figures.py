from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(r"C:\Users\23598\Documents\ChatGPT\LDTP")
OUT = ROOT / "本周汇报图文素材"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"

BLUE = "#4874CB"
NAVY = "#153D78"
RED = "#E62727"
ORANGE = "#EE822F"
GREEN = "#2F8C5B"
GREY = "#64748B"
PALE_BLUE = "#EAF1FF"
PALE_GREEN = "#E8F5EC"


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, title, detail, fill=PALE_BLUE):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2, edgecolor="#9CB7E7", facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=16,
            color=NAVY, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", fontsize=11,
            color="#1F2937", linespacing=1.35)


def figure_1_architecture():
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.93, "本周核心推进：控制策略参数纳入 MOEA/D 联合优化", fontsize=22,
            fontweight="bold", color=NAVY, va="center")
    ax.plot([0.03, 0.97], [0.875, 0.875], color=BLUE, lw=1.7)
    ax.text(0.03, 0.82, "将原先固定的能量管理规则转化为与构型、阶段功率共同寻优的设计变量", fontsize=13,
            color=GREY)

    xs = [0.04, 0.28, 0.52, 0.76]
    titles = ["2.2.1 评价体系", "1.2.1 飞推控闭环", "1.2.2 自适应调度器", "2.2.2 MOEA/D"]
    details = [
        "综合能耗\n飞行稳定性\n任务响应性能",
        "飞行控制\n混合动力系统\n状态实时反馈",
        "风扰补偿\nSOC 自适应\n健康状态估计",
        "权重分解\n候选更新\nPareto 候选",
    ]
    fills = [PALE_BLUE, PALE_BLUE, PALE_GREEN, PALE_BLUE]
    for x, title, detail, fill in zip(xs, titles, details, fills):
        box(ax, x, 0.50, 0.19, 0.20, title, detail, fill)
    for x in [0.235, 0.475, 0.715]:
        ax.annotate("", xy=(x + 0.035, 0.60), xytext=(x, 0.60),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))

    ax.text(0.04, 0.39, "本周新增优化变量", fontsize=18, fontweight="bold", color=RED)
    variables = ["阶段 SOC 目标", "充电与助力偏置", "功率储备", "调度阈值"]
    for i, name in enumerate(variables):
        x = 0.04 + i * 0.235
        patch = FancyBboxPatch((x, 0.21), 0.20, 0.11, boxstyle="round,pad=0.015,rounding_size=0.02",
                               linewidth=1.2, edgecolor="#9CB7E7", facecolor="#F5F7FA")
        ax.add_patch(patch)
        ax.text(x + 0.10, 0.265, name, ha="center", va="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.04, 0.06), 0.92, 0.08, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=1.0, edgecolor="#F5C78B", facecolor="#FFF5E8"))
    ax.text(0.50, 0.10, "核心变化：优化器同时搜索构型、功率分配和能量管理策略", ha="center", va="center",
            fontsize=17, color=NAVY, fontweight="bold")
    save(fig, "图1_控制策略纳入MOEAD联合优化.png")


def figure_2_adaptive_control():
    data = pd.read_csv(ROOT / "ptdl_1_2_3_adaptive_recovery_results" / "表8_组合扰动局部恢复对比.csv")
    focus = data[data["scenario"].isin(["起飞后初始爬升", "低速高升力"])].copy()
    labels = ["起飞后\n初始爬升", "低速\n高升力"]
    baseline = focus[focus["policy"] == "基线模糊调度"].set_index("scenario").loc[
        ["起飞后初始爬升", "低速高升力"], "terminal_altitude_error_m"
    ].to_numpy()
    adaptive = focus[focus["policy"] == "风扰补偿与健康自适应调度"].set_index("scenario").loc[
        ["起飞后初始爬升", "低速高升力"], "terminal_altitude_error_m"
    ].to_numpy()
    reduction = 100 * (1 - adaptive / baseline)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2), gridspec_kw={"width_ratios": [1.25, 0.95]})
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.97, wspace=0.20)
    fig.suptitle("自适应调度器效果：风扰补偿与健康状态估计进入飞推控闭环", x=0.05, y=0.98,
                 ha="left", fontsize=21, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.915, "以组合扰动下的局部工况为例，比较基线模糊调度与自适应调度", fontsize=12, color=GREY)

    x = range(2)
    width = 0.34
    ax1.bar([i - width / 2 for i in x], baseline, width, label="基线模糊调度", color=BLUE)
    ax1.bar([i + width / 2 for i in x], adaptive, width, label="风扰补偿与健康自适应调度", color=ORANGE)
    ax1.set_yscale("log")
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel("终端高度误差 (m，对数坐标)")
    ax1.set_title("局部工况终端高度误差", loc="left", color=NAVY, fontweight="bold", fontsize=16)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="upper left")
    for i, (b, a) in enumerate(zip(baseline, adaptive)):
        ax1.text(i - width / 2, b * 1.2, f"{b:.1f}", ha="center", va="bottom", fontsize=12, color=BLUE)
        ax1.text(i + width / 2, a * 1.25, f"{a:.1f}", ha="center", va="bottom", fontsize=12, color=ORANGE)

    ax2.axis("off")
    ax2.text(0.03, 0.82, "高度误差降幅", fontsize=18, fontweight="bold", color=NAVY)
    for idx, (label, value) in enumerate(zip(["初始爬升", "低速高升力"], reduction)):
        y = 0.62 - idx * 0.24
        ax2.add_patch(FancyBboxPatch((0.03, y), 0.90, 0.16, boxstyle="round,pad=0.02,rounding_size=0.02",
                                     linewidth=1, edgecolor="#D8E1F2", facecolor="#F5F7FA"))
        ax2.text(0.10, y + 0.08, label, va="center", fontsize=15, color="#1F2937")
        ax2.text(0.84, y + 0.08, f"{value:.1f}%", ha="right", va="center", fontsize=25,
                 fontweight="bold", color=RED)
    ax2.text(0.03, 0.13, "控制逻辑：\n垂直风前馈修正航迹角；\n发动机可用功率下降时提高 SOC 储备目标。",
             fontsize=14, color="#1F2937", linespacing=1.6)
    save(fig, "图2_自适应调度器局部恢复效果.png")


def figure_3_moead_convergence():
    history = pd.read_csv(ROOT / "ptdl_moead_h6_combined_degradation_seeded_results" / "组合退化MOEAD收敛历史.csv")
    energy = history["best_J_energy_mj"] / 1000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2))
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.08, right=0.97, wspace=0.20)
    fig.suptitle("MOEA/D 联合优化结果：调度参数参与寻优后形成可行候选集", x=0.05, y=0.98,
                 ha="left", fontsize=21, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.915, "H = 6，8 代迭代；目标函数采用 2.2.1 定义的综合能耗、飞行稳定性和任务响应性能", fontsize=12, color=GREY)

    ax1.plot(history["generation"], history["feasible_ratio"], marker="o", color=BLUE, lw=2.5)
    ax1.axhline(1, color="#D8E1F2", lw=1)
    ax1.set_ylim(0, 1.08)
    ax1.set_xlabel("迭代代数")
    ax1.set_ylabel("可行解比例")
    ax1.set_title("可行率收敛", loc="left", color=NAVY, fontweight="bold", fontsize=16)
    ax1.grid(alpha=0.25)
    ax1.annotate("第 2 代后达到 100%", xy=(2, 1), xytext=(3.0, 0.60),
                 arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, fontsize=13, fontweight="bold")

    ax2.plot(history["generation"], energy, marker="o", color=GREEN, lw=2.5)
    ax2.set_xlabel("迭代代数")
    ax2.set_ylabel("最优综合能耗指标 (GJ)")
    ax2.set_title("最优综合能耗指标下降", loc="left", color=NAVY, fontweight="bold", fontsize=16)
    ax2.grid(alpha=0.25)
    ax2.annotate(f"{energy.iloc[0]:.2f} → {energy.iloc[-1]:.2f} GJ", xy=(8, energy.iloc[-1]), xytext=(3.3, energy.iloc[-1] + 0.18),
                 arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, fontsize=13, fontweight="bold")
    fig.text(0.50, 0.06, "最终得到 7 个可行 Pareto 候选，均满足当前任务剖面的 80 h 总飞行时间硬约束。",
             ha="center", fontsize=14, color=NAVY, fontweight="bold")
    save(fig, "图3_MOEAD联合优化收敛结果.png")


def figure_4_validation():
    data = pd.read_csv(ROOT / "ptdl_combined_degradation_refinement_results" / "表10_组合退化续航与失速约束定向筛选.csv")
    data = data[data["hybrid_system_mass_kg"] == 70.0].sort_values("wing_area_m2")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2), gridspec_kw={"width_ratios": [1.25, 0.95]})
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.97, wspace=0.20)
    fig.suptitle("验证性检查：组合退化用于发现约束薄弱区", x=0.05, y=0.98,
                 ha="left", fontsize=21, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.915, "该检查用于验证硬约束与筛选下一轮优化种子，不作为本周工作的主目标", fontsize=12, color=GREY)

    feasible = data["feasible"]
    ax1.plot(data["wing_area_m2"], data["stall_speed_violation_mps"], color=BLUE, marker="o", lw=2.5)
    ax1.scatter(data.loc[~feasible, "wing_area_m2"], data.loc[~feasible, "stall_speed_violation_mps"], color=RED, s=75, zorder=3, label="失速约束超限")
    ax1.scatter(data.loc[feasible, "wing_area_m2"], data.loc[feasible, "stall_speed_violation_mps"], color=GREEN, s=75, zorder=3, label="满足全部约束")
    ax1.axhline(0, color=RED, lw=1.5, ls="--")
    ax1.axvline(24.58, color=ORANGE, lw=1.5, ls="--")
    ax1.text(24.582, 0.029, "S = 24.58 m²", color=ORANGE, fontsize=13, fontweight="bold")
    ax1.set_xlabel("翼面积 S (m²)")
    ax1.set_ylabel("失速速度约束违反度 (m/s，≤ 0 可行)")
    ax1.set_title("翼面积对失速约束的修复作用", loc="left", color=NAVY, fontweight="bold", fontsize=16)
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="upper right")

    ax2.axis("off")
    ax2.text(0.05, 0.78, "压力测试设置", fontsize=18, fontweight="bold", color=NAVY)
    settings = ["CD0 增加 10%", "发动机可用功率 85%", "电池可用能量 80%", "发动机效率 95%"]
    for i, item in enumerate(settings):
        ax2.text(0.08, 0.65 - i * 0.10, item, fontsize=16, color="#1F2937")
    ax2.add_patch(FancyBboxPatch((0.05, 0.15), 0.88, 0.20, boxstyle="round,pad=0.02,rounding_size=0.02",
                                 linewidth=1, edgecolor="#B9DDBF", facecolor=PALE_GREEN))
    ax2.text(0.49, 0.25, "修复种子\nS ≥ 24.58 m²，混动系统质量 60 至 70 kg\n同时满足 80 h 与失速约束", ha="center", va="center",
             fontsize=15, color=GREEN, fontweight="bold", linespacing=1.45)
    save(fig, "图4_组合退化验证与失速约束修复.png")


if __name__ == "__main__":
    figure_1_architecture()
    figure_2_adaptive_control()
    figure_3_moead_convergence()
    figure_4_validation()
