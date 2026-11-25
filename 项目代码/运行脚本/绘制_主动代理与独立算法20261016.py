"""Chinese scientific figures from held-out solver labels and counted archives."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURROGATE = ROOT / "运行结果" / "主动增量代理_20261015"
TRAINED = ROOT / "运行结果" / "主动增量代理_20261015"
BENCH = ROOT / "运行结果" / "九算法独立十六种子_20261016"
OUT = ROOT / "运行结果" / "主动代理与算法图集_20261016"
COLORS = ["#4288a4", "#569e90", "#d5806e", "#9b83b3", "#bd9b45", "#687d92", "#4a9680", "#c56879", "#527b4d"]
NAMES = {
    "NSGA-II": "NSGA-II", "MOEA/D": "MOEA/D", "MOPSO": "MOPSO",
    "Hybrid NSGA-II/MOPSO": "混合算法", "PTDL-自适应分块": "自适应分块",
    "PTDL-取消在线代理": "耦合引导", "PTDL-耦合贡献学习": "贡献学习",
    "PTDL-耦合贡献学习保先验": "贡献学习\n保留先验",
    "PTDL-去重贡献学习": "去重\n贡献学习",
}


def style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "pdf.fonttype": 42,
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#899399", "axes.labelcolor": "#36454d",
        "xtick.color": "#52616a", "ytick.color": "#52616a",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "grid.color": "#e6eaec", "grid.linewidth": .7,
    })


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    test = pd.read_csv(SURROGATE / "合并留出验证.csv")
    metrics = json.loads((SURROGATE / "合并留出评估.json").read_text(encoding="utf8"))["metrics"]
    targets = ["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]
    titles = ["升力斜率", "尾迹效率"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)
    for t, target in enumerate(targets):
        mask = np.isfinite(test[target].to_numpy(float))
        if t:
            mask &= test.wake_quality_pass.to_numpy(bool)
        for c, name in enumerate(("incrementalgp", "activegp")):
            ax = axes[t, c]
            y = test.loc[mask, target]
            prediction = test.loc[mask, f"{name}_prediction_{t}"]
            bounds = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
            padding = max(.02, (bounds[1]-bounds[0])*.12)
            limits = [bounds[0]-padding, bounds[1]+padding]
            ax.plot(limits, limits, color="#899399", lw=1, ls="--")
            ax.scatter(y, prediction, s=57, color=COLORS[c], edgecolors="white", linewidth=.7, zorder=3)
            ax.set(xlim=limits, ylim=limits, xlabel="VSPAERO 计算值", ylabel="代理预测值")
            ax.set_aspect("equal", adjustable="box")
            m = metrics[name][t]
            ax.set_title(f"{'原 258 构型模型' if c == 0 else '主动补样组合模型'} · {titles[t]}", loc="left")
            ax.text(.04, .95, f"R² = {m['r2']:.2f}\nMAE = {m['mae']:.2f}\n有效构型 {m['count']}",
                    transform=ax.transAxes, va="top", linespacing=1.6)
            ax.grid(alpha=.7)
    save(fig, "01_全新构型预测对照")
    cv = pd.read_csv(TRAINED / "训练集五折模型比较.csv")
    selected = json.loads((TRAINED / "训练集模型选择.json").read_text(encoding="utf8"))["selected"]
    candidates = [
        ("gp_physics_rbf", "RBF\n高斯过程"), ("gp_physics_matern32", "Matérn 3/2\n高斯过程"),
        ("gp_physics_matern52", "Matérn 5/2\n高斯过程"),
        ("boost_physics_7", "梯度提升\n7 叶节点"),
        ("boost_physics_15", "梯度提升\n15 叶节点"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for t, ax in enumerate(axes):
        rows = cv[cv.target == t].set_index("name")
        values = [float(rows.loc[k, "r2"]) for k, _ in candidates]+[selected[t]["r2"]]
        bars = ax.bar(np.arange(6), values, color=COLORS[:6], width=.65)
        ax.set_xticks(np.arange(6), [v for _, v in candidates]+["选定\n组合模型"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+.012, f"{value:.2f}", ha="center", fontsize=9)
        ax.axhline(.9, ls="--", color="#aa6a60", lw=1)
        ax.set(title=f"{titles[t]} · 训练集五折交叉验证", ylabel="决定系数 R²", ylim=(0, 1.02))
        ax.grid(axis="y", alpha=.7)
        ax.set_axisbelow(True)
    save(fig, "02_同训练集模型比较")
    runs = pd.read_csv(BENCH / "多种子算法结果.csv")
    history = pd.read_csv(BENCH / "共享基准逐代超体积.csv")
    plan = json.loads((BENCH / "运行前冻结配置.json").read_text(encoding="utf8"))
    algorithms = plan["algorithms"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5), constrained_layout=True)
    focus = ["MOPSO", "Hybrid NSGA-II/MOPSO", "PTDL-取消在线代理",
             "PTDL-耦合贡献学习保先验", "PTDL-去重贡献学习"]
    for name in focus:
        data = history[history.algorithm == name].groupby("generation").hypervolume
        median, q1, q3 = data.median(), data.quantile(.25), data.quantile(.75)
        color = COLORS[algorithms.index(name)]
        axes[0].plot(median.index, median, lw=2, color=color, label=NAMES[name].replace("\n", " "))
        axes[0].fill_between(median.index, q1, q3, color=color, alpha=.09)
    axes[0].set(xlabel="真实任务评价次数", ylabel="共享基准超体积", title="收敛曲线 · 中位数与四分位区间")
    ticks = np.linspace(0, plan["generations"], 5).astype(int)
    axes[0].set_xticks(ticks, [str((t+1)*28) for t in ticks])
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].grid(alpha=.7)
    data = [runs.loc[runs.algorithm == a, "hypervolume"].to_numpy() for a in algorithms]
    violin = axes[1].violinplot(data, showextrema=False, widths=.8)
    for body, color in zip(violin["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_alpha(.17)
    axes[1].boxplot(data, widths=.22, showfliers=False, patch_artist=False,
                    medianprops={"color": "#34444d", "linewidth": 1.5})
    rng = np.random.default_rng(16)
    for i, (values, color) in enumerate(zip(data, COLORS), 1):
        axes[1].scatter(i+rng.uniform(-.11, .11, len(values)), values,
                        color=color, s=19, alpha=.8, edgecolor="white", linewidth=.3)
    axes[1].set_xticks(range(1, len(algorithms)+1), [NAMES[a] for a in algorithms], rotation=35, ha="right")
    axes[1].set(ylabel="最终历史档案超体积", title="16 个独立种子的结果分布")
    axes[1].grid(axis="y", alpha=.7)
    save(fig, "03_独立十六种子算法表现")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
    mopso = runs[runs.algorithm == "MOPSO"].set_index("seed").hypervolume
    for name in ("PTDL-取消在线代理", "PTDL-耦合贡献学习保先验", "PTDL-去重贡献学习"):
        delta = runs[runs.algorithm == name].set_index("seed").hypervolume-mopso
        color = COLORS[algorithms.index(name)]
        axes[0].scatter(delta.index, delta, color=color, s=30, label=NAMES[name].replace("\n", " "))
        axes[0].hlines(delta.mean(), delta.index.min(), delta.index.max(), colors=color, alpha=.4)
    axes[0].axhline(0, color="#8a969c", lw=1)
    axes[0].set(title="相同种子下相对 MOPSO 的变化", xlabel="独立种子序号", ylabel="超体积差值")
    axes[0].set_xticks(mopso.index[::3], [str(i+1) for i in range(0, len(mopso), 3)])
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.7)
    summary = runs.groupby("algorithm")[["feasible_ratio", "evaluated_feasible_ratio"]].mean().loc[algorithms]
    x = np.arange(len(algorithms))
    axes[1].bar(x-.17, summary.feasible_ratio, width=.32, color="#559e92", label="最终种群")
    axes[1].bar(x+.17, summary.evaluated_feasible_ratio, width=.32, color="#8daabc", label="全部真实评价")
    axes[1].set_xticks(x, [NAMES[a] for a in algorithms], rotation=35, ha="right")
    axes[1].set(ylim=(0, 1.08), ylabel="可行方案比例", title="可行率的两个统计口径")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.7)
    save(fig, "04_配对改进与可行率")
    titles = ["全新构型预测对照", "同训练集模型比较", "独立十六种子算法表现", "配对改进与可行率"]
    images = [f"{i+1:02d}_{title}.png" for i, title in enumerate(titles)]
    content = "\n".join(f"<figure><img src='{name}' alt='{title}'><figcaption>{title}</figcaption></figure>"
                        for name, title in zip(images, titles))
    (OUT / "index.html").write_text(
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>主动气动代理与独立算法比较</title>"
        "<style>body{font-family:'Microsoft YaHei',sans-serif;margin:0;background:#f1f5f4;color:#243a40}"
        "main{max-width:1300px;margin:auto;padding:28px}h1{font-size:24px}figure{margin:24px 0;"
        "background:white;padding:18px}img{display:block;width:100%;height:auto}figcaption{"
        "padding-top:12px;font-size:16px}</style><main><h1>主动气动代理与独立算法比较</h1>"
        "<p>代理图汇总两批共 32 个留出构型，有效升力标签 31 个、尾迹标签 30 个。"
        "算法图对应 16 个独立种子、每次 364 次真实评价，使用冻结的 258 构型代理与低阶 80 小时任务。"
        "耦合引导对五个既有强基线的优势通过 Holm 校正；新增变体保留用于方法比较。</p>"
        + content + "</main></html>", encoding="utf8")


if __name__ == "__main__":
    main()
