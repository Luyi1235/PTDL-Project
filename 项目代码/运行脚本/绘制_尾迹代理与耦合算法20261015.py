"""Scientific figures from sealed solver validation and counted optimizer runs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "运行结果"
OUT = RESULTS / "尾迹代理与算法图集_20261015"
ALGORITHMS = ["NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO",
              "PTDL-自适应分块", "PTDL-耦合代理辅助", "PTDL-取消在线代理"]
LABELS = ["NSGA-II", "MOEA/D", "MOPSO", "原混合算法", "自适应分块",
          "耦合代理辅助", "取消在线代理"]
COLORS = ["#4B809D", "#549B83", "#D07D64", "#8D7BAA", "#B59746", "#236A73", "#969C9E"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#81908F",
    "axes.labelcolor": "#334B50", "xtick.color": "#5D6F74", "ytick.color": "#5D6F74",
    "grid.color": "#E1E8E7", "grid.linewidth": .7, "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
PAGES = []


def save(fig, name, explanation):
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    PAGES.append((name, explanation))


def validation():
    folder = RESULTS / "尾迹分区趋势代理_20261015"
    frame = pd.read_csv(folder / "新独立验证_逐点.csv")
    decision = json.loads((folder / "新独立验证与部署决策.json").read_text(encoding="utf8"))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
    targets = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
    labels = ("升力斜率 / rad$^{-1}$", "尾迹效率")
    names = ("physics211gp", "incrementalgp")
    model_labels = ("原物理特征模型", "增量构型模型")
    for t, ax in enumerate(axes):
        valid = np.isfinite(frame[targets[t]].to_numpy(float))
        if t:
            valid &= frame.wake_quality_pass.to_numpy(bool)
        actual = frame.loc[valid, targets[t]].to_numpy(float)
        all_values = [actual]
        for i, name in enumerate(names):
            predicted = frame.loc[valid, f"{name}_prediction_{t}"].to_numpy(float)
            all_values.append(predicted)
            metric = decision["metrics"][name][t]
            ax.scatter(actual, predicted, s=48, color=COLORS[0 if i == 0 else 5],
                       marker="o" if i == 0 else "D", alpha=.85, edgecolors="white", linewidth=.5,
                       label=f"{model_labels[i]}  R²={metric['r2']:.2f}  MAE={metric['mae']:.3f}")
        lo, hi = np.min(np.concatenate(all_values)), np.max(np.concatenate(all_values))
        pad = (hi-lo)*.09
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color="#929D9D", lw=1)
        ax.set(xlabel=f"VSPAERO {labels[t]}", ylabel=f"代理预测{labels[t]}",
               title=f"({chr(97+t)}) {labels[t]} · {valid.sum()} 个有效新构型",
               xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=.6)
    save(fig, "01_全新构型代理验证",
         "模型与构型在求解前冻结。图中两种模型接受同一批新构型，旧验证样本已转为增量模型的开发数据。")
    frame.to_csv(OUT / "代理验证绘图数据.csv", index=False)


def labels_and_solver():
    frame = pd.read_csv(RESULTS / "分区尾迹代理_20261015" / "工作迎角与标签诊断.csv")
    probe = pd.read_csv(RESULTS / "尾迹求解收敛_20261015" / "尾迹迭代收敛.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    for position, n in enumerate((3, 4)):
        values = frame.loc[frame.working_count == n, "label"].to_numpy()
        violin = axes[0].violinplot(values, positions=[position], widths=.7, showextrema=False)
        violin["bodies"][0].set_facecolor(COLORS[position])
        violin["bodies"][0].set_alpha(.25)
        axes[0].boxplot(values, positions=[position], widths=.2, patch_artist=True,
                        boxprops=dict(facecolor=COLORS[position], alpha=.65),
                        medianprops=dict(color="white", linewidth=1.5), showfliers=False)
        axes[0].scatter(position+np.random.default_rng(1500+n).uniform(-.16, .16, len(values)), values,
                        s=9, color=COLORS[position], alpha=.45, edgecolors="none")
    axes[0].set(xticks=[0, 1], xticklabels=["3 个工作迎角", "4 个工作迎角"],
                ylabel="训练标签 Ew", title="(a) 迎角筛选规则与标签分布")
    axes[0].grid(axis="y", alpha=.5)
    for i, (sample, group) in enumerate(probe.groupby("sample", sort=False)):
        labels = {"baseline": "基准构型", "angle_sensitive": "另一有效构型",
                  "large_residual": "代理误差最大的构型"}
        axes[1].plot(group.wake_iterations, group.ew, "-o", color=COLORS[i],
                     linewidth=1.8, markersize=6, label=labels[sample])
    axes[1].set(xlabel="尾迹迭代次数", ylabel="VSPAERO Ew",
                title="(b) 相同构型的尾迹迭代复算", xticks=[30, 60, 120])
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=.5)
    save(fig, "02_尾迹标签与求解诊断",
         "迎角筛选会改变中位数所用数据点。三组固定构型的复算分别保存，最大误差构型的 Ew 随尾迹迭代保持一致。")


def algorithms():
    folder = RESULTS / "耦合算法独立八种子H6_20261015"
    hv = pd.read_csv(folder / "共享基准逐代超体积.csv")
    runs = pd.read_csv(folder / "多种子算法结果.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for i, name in enumerate(ALGORITHMS):
        selected = hv[hv.algorithm == name]
        quantile = selected.groupby("generation").hypervolume.quantile([.25, .5, .75]).unstack()
        axes[0].plot(quantile.index, quantile[.5], color=COLORS[i], lw=2 if i == 5 else 1.5,
                     label=LABELS[i], linestyle="--" if i == 6 else "-")
        axes[0].fill_between(quantile.index, quantile[.25], quantile[.75], color=COLORS[i], alpha=.09)
        values = runs.loc[runs.algorithm == name, "hypervolume"].to_numpy()
        axes[1].boxplot(values, positions=[i], widths=.5, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=COLORS[i], alpha=.3),
                       medianprops=dict(color=COLORS[i], linewidth=2))
        axes[1].scatter(i+np.linspace(-.13, .13, len(values)), values, s=26,
                        color=COLORS[i], edgecolors="white", linewidth=.4, zorder=3)
    axes[0].set(xlabel="迭代代数", ylabel="共享归一化超体积", title="(a) 收敛过程与种子间差异")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    axes[0].grid(alpha=.5)
    axes[1].set(ylabel="最终历代档案超体积", title="(b) 八组独立种子的最终结果",
                xticks=range(len(LABELS)), xticklabels=LABELS)
    axes[1].tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[1].grid(axis="y", alpha=.5)
    save(fig, "03_独立八种子算法比较",
         "各算法每次评价 308 个候选。所有已评价可行非支配方案按同一档案口径计分，参考点和目标归一化范围共享。")
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    width = .34
    x = np.arange(len(ALGORITHMS))
    final = [runs.loc[runs.algorithm == a, "feasible_ratio"].mean()*100 for a in ALGORITHMS]
    evaluated = [runs.loc[runs.algorithm == a, "evaluated_feasible_ratio"].mean()*100 for a in ALGORITHMS]
    ax.bar(x-width/2, final, width, color="#4B809D", label="最终种群可行率")
    ax.bar(x+width/2, evaluated, width, color="#549B83", label="全部评价可行率")
    ax.set(ylim=(0, 110), ylabel="可行率 / %", xticks=x, xticklabels=LABELS)
    ax.tick_params(axis="x", labelrotation=25, labelsize=9)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=.5)
    save(fig, "04_两种口径的可行率",
         "最终种群可行率反映输出方案，全部评价可行率反映搜索过程中完整任务评价的可行比例。")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validation()
    labels_and_solver()
    algorithms()
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>尾迹代理与耦合算法</title>',
             '<style>body{margin:0;background:#f4f7f7;color:#294147;font:15px "Microsoft YaHei",sans-serif}'
             'main{max-width:1300px;margin:auto;padding:24px}section{border-top:1px solid #dce5e3;padding:22px 0}'
             'img{width:100%;height:auto;display:block;background:white}p{line-height:1.7}'
             'h1{font-size:25px}h2{font-size:18px}a{color:#236A73;margin-right:20px}</style><main>'
             '<h1>尾迹代理与针对性优化算法</h1><p>2026 年 10 月 15 日 · 实际计算结果</p>']
    for name, explanation in PAGES:
        parts.append(f'<section><h2>{name[3:]}</h2><p>{explanation}</p>'
                     f'<a href="{name}.png">PNG</a><a href="{name}.pdf">PDF</a>'
                     f'<img src="{name}.png" alt="{name[3:]}"></section>')
    parts.append("</main></html>")
    (OUT / "index.html").write_text("".join(parts), encoding="utf8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
