from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(r"D:\LDTP\LDTP\PTDL飞推控协同优化代码包\运行结果")
DATA = ROOT / "H6四算法同预算_20代_8种子_131GP_20261011"
OUT = Path(r"C:\Users\pc\Desktop")
ALGOS = ["NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO"]
LABELS = ["NSGA-II", "MOEA/D", "MOPSO", "混合算法"]
COLORS = ["#4285A5", "#439A86", "#D47C69", "#947EAF"]

for font_file in ("msyh.ttc", "msyhbd.ttc", "times.ttf"):
    font_manager.fontManager.addfont(str(Path(r"C:\Windows\Fonts") / font_file))
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "Times New Roman"],
    "font.size": 34,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 30,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#777777",
    "axes.linewidth": 1.1,
    "grid.color": "#e6e8ea",
    "grid.linewidth": 0.9,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})

runs = pd.read_csv(DATA / "多随机种子单次运行结果.csv")
history = pd.read_csv(DATA / "多随机种子逐代Pareto超体积.csv")
last = history[history.generation.eq(20)]

fig, axes = plt.subplots(2, 2, figsize=(18, 14.6))
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.11, top=0.88,
                    wspace=0.60, hspace=0.55)

ax = axes[0, 0]
for algo, label, color in zip(ALGOS, LABELS, COLORS):
    groups = history[history.algorithm.eq(algo)].groupby("generation").shared_normalized_hypervolume
    median = groups.median()
    lo, hi = groups.quantile(0.25), groups.quantile(0.75)
    calls = (median.index.to_numpy() + 1) * 28
    ax.fill_between(calls, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(calls, median, color=color, lw=2.8, label=label)
    ax.scatter(calls[-1], median.iloc[-1], s=48, color=color, zorder=4)
ax.set(xlim=(0, 600), ylim=(0, 1.13), xticks=[0, 196, 392, 588])
ax.set_xlabel("方案评价次数")
ax.set_ylabel("归一化超体积")
ax.set_title("(a) 可行解集收敛", loc="left", pad=15)
ax.grid(True)
handles, legend_labels = ax.get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.56, 0.99),
           ncol=4, frameon=False, handlelength=1.15, columnspacing=1.1)

ax = axes[0, 1]
for i, (algo, color) in enumerate(zip(ALGOS, COLORS)):
    rate = runs.loc[runs.algorithm.eq(algo), "final_feasible_count"].sum() / (28 * 8) * 100
    ax.barh(i, 100, height=0.49, color="#E9EDEE")
    ax.barh(i, rate, height=0.49, color=color, alpha=0.86)
    ax.text(rate - 2, i, f"{rate:.1f}%", va="center", ha="right",
            fontsize=32, color="white", fontweight="bold")
ax.set(yticks=range(4), yticklabels=LABELS, ylim=(3.6, -0.6),
       xlim=(0, 105), xticks=[0, 50, 100])
ax.set_xlabel("终代可行率 / %")
ax.set_title("(b) 终代可行率", loc="left", pad=15)
ax.tick_params(axis="y", length=0, pad=8)
ax.spines["left"].set_visible(False)


def raincloud(ax, values_by_algorithm, title, xlabel):
    rng = np.random.default_rng(42)
    for i, (values, color) in enumerate(zip(values_by_algorithm, COLORS)):
        x = np.asarray(values, dtype=float)
        span = max(np.ptp(x), 0.03 * max(np.mean(x), 0.1))
        grid = np.linspace(x.min() - 0.12 * span, x.max() + 0.12 * span, 180)
        density = gaussian_kde(x)(grid)
        density = density / density.max() * 0.30
        ax.fill_between(grid, i - 0.13 - density, i - 0.13, color=color, alpha=0.38, linewidth=0)
        ax.plot(grid, i - 0.13 - density, color=color, lw=1.3)
        ax.boxplot([x], positions=[i + 0.07], vert=False, widths=0.15,
                   patch_artist=True, showfliers=False, manage_ticks=False,
                   boxprops={"facecolor": color, "alpha": 0.8, "edgecolor": "#43515A", "linewidth": 1},
                   medianprops={"color": "white", "linewidth": 2},
                   whiskerprops={"color": "#56616A", "linewidth": 1},
                   capprops={"color": "#56616A", "linewidth": 1})
        ax.scatter(x, i + 0.30 + rng.uniform(-0.045, 0.045, len(x)), s=48,
                   color=color, edgecolors="white", linewidth=0.7, zorder=4)
    ax.set_yticks(range(4), LABELS)
    ax.set_ylim(3.6, -0.65)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=15)


raincloud(axes[1, 0],
          [last.loc[last.algorithm.eq(a), "shared_normalized_hypervolume"] for a in ALGOS],
          "(c) 解集质量分布", "归一化超体积")
raincloud(axes[1, 1],
          [runs.loc[runs.algorithm.eq(a), "runtime_s"] / 60 for a in ALGOS],
          "(d) 计算时间分布", "单次计算时间 / min")

for ext in ("png", "svg"):
    path = OUT / f"PTDL_四算法性能比较_图例调整版.{ext}"
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.14)
    print(path)
plt.close(fig)
