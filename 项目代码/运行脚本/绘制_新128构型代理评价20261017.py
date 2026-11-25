"""Chinese figures for complete held-out results, separate from historical comparisons."""

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
OUT = ROOT / "运行结果/密度均衡与残差代理_20261017"
TARGETS = ["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]


def main():
    font = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 12,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    frame = pd.read_csv(OUT / "fresh128_逐点比较.csv")
    report = json.loads((OUT / "留出评价.json").read_text(encoding="utf8"))
    metrics = pd.DataFrame(report["metrics"])
    metrics = metrics.loc[metrics.batch == "fresh128"]
    metrics.to_csv(OUT / "新构型精度对比.csv", index=False, encoding="utf-8-sig")
    keys = ["workbench289", "previous1512", "geometry_interaction", "balanced_candidate"]
    names = ["工作台原模型", "上一版扩样模型", "几何组合模型", "密度均衡组合"]
    colors = ["#b3a3a9", "#dc9d67", "#5199b6", "#258c82"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6), layout="constrained")
    for index, target in enumerate(TARGETS):
        valid = np.isfinite(frame[target].to_numpy(float))
        if index:
            valid &= frame.wake_quality_pass.to_numpy(bool)
        actual = frame.loc[valid, target].to_numpy(float)
        ax = axes[0, index]
        # The two new lift predictors are identical; draw their shared result once.
        shown = [0, 3] if index == 0 else [0, 2, 3]
        values = [actual]
        for i in shown:
            prediction = frame.loc[valid, f"{keys[i]}_prediction_{index}"].to_numpy(float)
            values.append(prediction)
            ax.scatter(actual, prediction, color=colors[i], s=32, alpha=.7, edgecolor="white",
                       linewidth=.4, label=names[i])
        lo, hi = np.concatenate(values).min(), np.concatenate(values).max()
        padding = .06*(hi-lo)
        ax.plot([lo-padding, hi+padding], [lo-padding, hi+padding], c="#666666", ls="--", lw=1)
        ax.set(xlim=(lo-padding, hi+padding), ylim=(lo-padding, hi+padding),
               xlabel="VSPAERO 计算值", ylabel="代理预测值",
               title=f"({'a' if index == 0 else 'b'}) {'升力斜率' if index == 0 else '尾迹效率'}")
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.grid(alpha=.14)
        table = metrics.loc[metrics.target == target].set_index("model")
        ax.text(.97, .04, "\n".join(f"{names[i]}  R² {table.loc[keys[i], 'r2']:.2f}" for i in shown),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax = axes[1, 0]
    for i, (key, name, color) in enumerate(zip(keys, names, colors)):
        values = [metrics.loc[(metrics.model == key) & (metrics.target == t), "r2"].iloc[0]
                  for t in TARGETS]
        bars = ax.bar(np.arange(2)+(i-1.5)*.19, values, width=.175, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set(xticks=[0, 1], xticklabels=["升力斜率", "尾迹效率"], ylim=(0, 1.07),
           ylabel="决定系数 R²", title="(c) 相同有效标签上的预测精度")
    ax.axhline(.9, c="#777777", ls="--", lw=1)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=.14)
    ax = axes[1, 1]
    valid = frame.wake_quality_pass.to_numpy(bool) & np.isfinite(frame[TARGETS[1]].to_numpy(float))
    for key, name, color in zip(keys, names, colors):
        error = np.sort(abs(frame.loc[valid, f"{key}_prediction_1"].to_numpy(float)
                            -frame.loc[valid, TARGETS[1]].to_numpy(float)))
        ax.plot(error, np.arange(1, len(error)+1)/len(error), c=color, lw=2, label=name)
    ax.set(xlabel="尾迹效率绝对误差", ylabel="累计构型比例", ylim=(0, 1.03),
           title="(d) 尾迹效率误差分布")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(alpha=.14)
    fig.suptitle("新构型独立留出评价", fontsize=17)
    fig.supxlabel("128 个新构型    升力斜率有效标签 122 个    尾迹效率有效标签 118 个", fontsize=11)
    fig.savefig(OUT / "新128构型代理精度.png", dpi=210)
    fig.savefig(OUT / "新128构型代理精度.pdf")
    plt.close(fig)
    print(OUT / "新128构型代理精度.png")


if __name__ == "__main__":
    main()
