"""Plot comparable historical predictions and training-only model comparisons."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "运行结果/几何交互代理_20261017"
TARGETS = ["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]


def main():
    font = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")
    plt.rcParams.update({
        "font.family": font.get_name(), "font.size": 12, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
    })
    frame = pd.read_csv(OUT / "later96_回溯逐点.csv")
    metrics = pd.read_csv(OUT / "历史构型回溯评价.csv")
    metrics = metrics.loc[metrics.batch == "later96"]
    models = ["workbench289", "previous1512", "geometry_interaction"]
    names = ["工作台原模型", "上一版扩样模型", "几何特征组合模型"]
    colors = ["#b3a3a9", "#df9963", "#258d90"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6), layout="constrained")
    for t, target in enumerate(TARGETS):
        valid = np.isfinite(frame[target].to_numpy(float))
        if t:
            valid &= frame.wake_quality_pass.to_numpy(bool)
        actual = frame.loc[valid, target].to_numpy(float)
        ax = axes[0, t]
        values = [actual]
        for model, name, color in zip(models, names, colors):
            prediction = frame.loc[valid, f"{model}_prediction_{t}"].to_numpy(float)
            values.append(prediction)
            ax.scatter(actual, prediction, s=32, alpha=.72, c=color, edgecolor="white",
                       linewidth=.4, label=name)
        lo, hi = np.concatenate(values).min(), np.concatenate(values).max()
        margin = .06*(hi-lo)
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], c="#666666", ls="--", lw=1)
        ax.set(xlim=(lo-margin, hi+margin), ylim=(lo-margin, hi+margin),
               xlabel="VSPAERO 计算值", ylabel="代理预测值",
               title=f"({'a' if t == 0 else 'b'}) {'升力斜率' if t == 0 else '尾迹效率'}")
        ax.set_aspect("equal")
        ax.grid(alpha=.14)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        table = metrics.loc[metrics.target == target].set_index("model")
        ax.text(.97, .04, "\n".join(f"{name}  R² {table.loc[key, 'r2']:.2f}"
                                  for key, name in zip(models, names)),
                ha="right", va="bottom", transform=ax.transAxes, fontsize=11)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax = axes[1, 0]
    for i, (model, name, color) in enumerate(zip(models, names, colors)):
        values = [metrics.loc[(metrics.model == model) & (metrics.target == target), "r2"].iloc[0]
                  for target in TARGETS]
        bars = ax.bar(np.arange(2)+(i-1)*.24, values, width=.22, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.axhline(.9, color="#777777", ls="--", lw=1)
    ax.set(xticks=[0, 1], xticklabels=["升力斜率", "尾迹效率"], ylim=(0, 1.05),
           ylabel="决定系数 R²", title="(c) 相同有效构型上的模型比较")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=.14)
    ax = axes[1, 1]
    valid = frame.wake_quality_pass.to_numpy(bool) & np.isfinite(frame[TARGETS[1]].to_numpy(float))
    for model, name, color in zip(models, names, colors):
        error = np.sort(abs(frame.loc[valid, f"{model}_prediction_1"].to_numpy(float)
                            -frame.loc[valid, TARGETS[1]].to_numpy(float)))
        ax.plot(error, np.arange(1, len(error)+1)/len(error), color=color, lw=2, label=name)
    ax.set(xlabel="尾迹效率绝对误差", ylabel="累计构型比例", ylim=(0, 1.03),
           title="(d) 尾迹效率误差分布")
    ax.grid(alpha=.14)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.suptitle("几何特征与组合回归的代理改进", fontsize=17)
    fig.supxlabel("历史 96 构型回溯评价    新增 128 构型独立验证另行计算", fontsize=11)
    fig.savefig(OUT / "几何交互代理精度.png", dpi=210)
    fig.savefig(OUT / "几何交互代理精度.pdf")
    plt.close(fig)

    sources = [
        ("几何特征组合", OUT / "模型选择.json"),
        ("网格计数辅助", ROOT / "运行结果/网格辅助代理_20261017/模型选择.json"),
        ("面元质量辅助", ROOT / "运行结果/网格质量辅助代理_20261017/模型选择.json"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), layout="constrained")
    summary = []
    for name, path in sources:
        selected = json.loads(path.read_text(encoding="utf8"))["selected"]
        for t, entry in enumerate(selected):
            summary.extend(dict(model=name, target=TARGETS[t], **row) for row in entry["oof"])
    pd.DataFrame(summary).to_csv(OUT / "方法交叉验证对比.csv", index=False, encoding="utf-8-sig")
    for t, ax in enumerate(axes):
        data = pd.DataFrame(summary)
        for i, (split, name, color) in enumerate((
            ("random", "随机划分", "#258d90"), ("spatial", "构型区域划分", "#c27c9a"),
        )):
            values = [data.loc[(data.model == model) & (data.target == TARGETS[t])
                               & (data.split == split), "r2"].iloc[0] for model, _ in sources]
            bars = ax.bar(np.arange(3)+(i-.5)*.32, values, width=.30, color=color, label=name)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
        ax.set(xticks=np.arange(3), xticklabels=[name for name, _ in sources], ylim=(0, 1.05),
               ylabel="决定系数 R²",
               title=f"({'a' if t == 0 else 'b'}) {'升力斜率' if t == 0 else '尾迹效率'}")
        ax.axhline(.9, color="#777777", lw=1, ls="--")
        ax.legend(frameon=False, fontsize=10, loc="lower left")
        ax.grid(axis="y", alpha=.14)
    fig.suptitle("几何与网格特征的训练交叉验证", fontsize=17)
    fig.supxlabel("全部迎角按构型分组    模型选择使用训练数据", fontsize=11)
    fig.savefig(OUT / "几何与网格特征比较.png", dpi=210)
    fig.savefig(OUT / "几何与网格特征比较.pdf")
    plt.close(fig)

    refined = pd.read_csv(ROOT / "运行结果/局部网格加密_20261017/局部网格加密.csv")
    original = pd.read_csv(ROOT / "运行结果/微小几何扰动_20261017/局部变化.csv")
    pair = ["local_03_021", "local_03_027"]
    original = original.set_index("sample").loc[pair, "vspaero_oswald_efficiency"].to_numpy(float)
    values = [original]
    for mesh in ("medium", "fine"):
        values.append(refined.loc[refined.mesh == mesh].set_index("sample").loc[
            pair, "vspaero_oswald_efficiency"].to_numpy(float))
    values = np.asarray(values)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    labels = ["原网格", "中等网格", "细网格"]
    for i, (name, color) in enumerate((("原构型", "#258d90"), ("机身直径增加约 0.30 mm", "#c27c9a"))):
        bars = axes[0].bar(np.arange(3)+(i-.5)*.32, values[:, i], width=.30, color=color, label=name)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    axes[0].set(xticks=np.arange(3), xticklabels=labels, ylim=(0, 1.05),
                ylabel="尾迹效率", title="(a) 相同微扰构型对的计算结果")
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    jumps = abs(values[:, 1]-values[:, 0])
    bars = axes[1].bar(np.arange(3), jumps, width=.56, color=["#c27c9a", "#df9963", "#258d90"])
    axes[1].bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
    axes[1].set(xticks=np.arange(3), xticklabels=labels, ylim=(0, .25),
                ylabel="微扰前后尾迹效率绝对差", title="(b) 局部数值敏感性")
    for ax in axes:
        ax.grid(axis="y", alpha=.14)
    fig.suptitle("训练构型的局部网格敏感性诊断", fontsize=17)
    fig.supxlabel("尾迹设置和有效迎角规则保持相同    三种网格的效率绝对值仍有明显差异", fontsize=11)
    fig.savefig(OUT / "局部网格敏感性.png", dpi=210)
    fig.savefig(OUT / "局部网格敏感性.pdf")
    plt.close(fig)
    pd.DataFrame(dict(mesh=labels, original=values[:, 0], perturbed=values[:, 1],
                      absolute_change=jumps)).to_csv(OUT / "局部网格敏感性.csv", index=False)
    print(OUT / "几何交互代理精度.png")


if __name__ == "__main__":
    main()
