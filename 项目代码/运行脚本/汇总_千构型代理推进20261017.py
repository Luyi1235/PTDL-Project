"""Save an auditable retrospective comparison and Chinese scientific figures."""

import json
from pathlib import Path
import shutil
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "运行结果/千构型分区主动补样_20261017"


def main():
    font = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 12,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    frame = pd.read_csv(OUT / "原96构型回溯逐点.csv")
    report = json.loads((OUT / "原96构型回溯评价.json").read_text(encoding="utf8"))
    metrics = pd.DataFrame(report["metrics"])
    metrics.to_csv(OUT / "代理精度对比数据.csv", index=False, encoding="utf-8-sig")
    names = ["工作台原模型", "改进组合模型"]
    keys = ["low_reference", "improved"]
    colors = ["#9b707b", "#257f91"]
    targets = ["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]
    titles = ["升力斜率", "尾迹效率"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), layout="constrained")
    for t, (target, title) in enumerate(zip(targets, titles)):
        mask = np.isfinite(frame[target].to_numpy(float))
        if t:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        actual = frame.loc[mask, target].to_numpy(float)
        ax = axes[0, t]
        all_values = [actual]
        for key, name, color in zip(keys, names, colors):
            prediction = frame.loc[mask, f"{key}_prediction_{t}"].to_numpy(float)
            all_values.append(prediction)
            ax.scatter(actual, prediction, s=32, alpha=.7, c=color, edgecolors="white",
                       linewidths=.4, label=name)
        low, high = np.concatenate(all_values).min(), np.concatenate(all_values).max()
        pad = .06*(high-low)
        ax.plot([low-pad, high+pad], [low-pad, high+pad], color="#555555", lw=1., ls="--")
        ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad),
               xlabel="VSPAERO 计算值", ylabel="代理模型预测值",
               title=f"({'a' if t == 0 else 'b'}) {title}  {mask.sum()} 个有效构型")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
        ax.legend(fontsize=10, loc="upper left", frameon=False)
        table = metrics.loc[(metrics.target == target) & metrics.model.isin(keys)].set_index("model")
        ax.text(.97, .04, "\n".join(f"{name}  R² {table.loc[key, 'r2']:.3f}"
                                  for key, name in zip(keys, names)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
    ax = axes[1, 0]
    locations = np.arange(2)
    for i, (key, name, color) in enumerate(zip(keys, names, colors)):
        values = [metrics.loc[(metrics.model == key) & (metrics.target == t), "r2"].iloc[0]
                  for t in targets]
        bars = ax.bar(locations+(i-.5)*.32, values, width=.30, color=color, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
    ax.set(xticks=locations, xticklabels=titles, ylim=(0, 1.04), ylabel="决定系数 R²",
           title="(c) 相同构型上的预测精度")
    ax.axhline(.9, color="#777777", ls="--", lw=1)
    ax.text(1.45, .905, "0.90", ha="right", va="bottom", fontsize=10)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=.15)
    ax = axes[1, 1]
    mask = frame.wake_quality_pass.to_numpy(bool) & np.isfinite(frame[targets[1]].to_numpy(float))
    for key, name, color in zip(keys, names, colors):
        error = np.sort(abs(frame.loc[mask, f"{key}_prediction_1"].to_numpy(float)
                            -frame.loc[mask, targets[1]].to_numpy(float)))
        ax.plot(error, np.arange(1, len(error)+1)/len(error), lw=2, color=color, label=name)
    ax.set(xlabel="尾迹效率绝对误差", ylabel="累计构型比例", ylim=(0, 1.03),
           title="(d) 全部有效构型的尾迹误差分布")
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.suptitle("千构型训练后的代理精度比较", fontsize=17)
    fig.supxlabel("原 96 个留出构型的回溯评价  ·  定向补样与新构型独立验证任务已启动", fontsize=11)
    fig.savefig(OUT / "代理精度对比.png", dpi=210)
    fig.savefig(OUT / "代理精度对比.pdf")
    plt.close(fig)
    # Preserve the exact code used for this development cycle separately from live sources.
    archive = OUT / "源文件快照"
    archive.mkdir(exist_ok=True)
    manifest = {}
    for path in [
        ROOT / "核心模型/分目标气动高斯过程.py",
        *[ROOT / "运行脚本" / name for name in (
            "研究_千构型尾迹分解20261017.py", "研究_千构型物理分解20261017.py",
            "研究_尾迹迎角分区20261017.py", "推进_千构型主动补样20261017.py")],
    ]:
        shutil.copy2(path, archive / path.name)
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (archive / "校验值.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(OUT / "代理精度对比.png")


if __name__ == "__main__":
    main()
