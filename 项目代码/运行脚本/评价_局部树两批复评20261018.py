"""Evaluate training-selected candidates on previously inspected holdouts."""

import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import 研究_局部线性与非对称树20261018 as study
from 局部线性树气动代理 import LocalLinearTreeAero

OUT = study.OUT / "两批历史留出复评"
SOURCE = study.ROOT / "运行结果/精细树与核代理_20261018/留出复评_v2/两批留出逐点.csv"


def main():
    OUT.mkdir(exist_ok=True)
    study.check_hashes(study.batch.previous.read(study.OUT / "独立预测冻结.json")["hashes"])
    selection = study.batch.previous.read(study.OUT / "模型选择.json")
    paths = [SOURCE, Path(__file__), study.OUT / "模型选择.json",
             *map(Path, selection["model_paths"].values())]
    hashes = {str(p): study.digest(p) for p in paths}
    freeze = OUT / "复评配置冻结.json"
    if freeze.exists():
        if study.batch.previous.read(freeze)["hashes"] != hashes:
            raise RuntimeError("Retrospective evaluation inputs changed")
    else:
        study.save(freeze, dict(hashes=hashes, previously_inspected_holdouts=True,
            selected_using_training_oof_only=True, fresh_test_size=192))
    frame = pd.read_csv(SOURCE)
    training = pd.read_csv(study.OUT / "训练标签3840.csv")
    if frame.merge(training[study.FEATURES], on=study.FEATURES).shape[0]:
        raise RuntimeError("Held-out geometry overlaps training")
    x = frame[study.FEATURES].to_numpy(float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for name, path in selection["model_paths"].items():
            p = LocalLinearTreeAero.load(path).predict_many(x)
            for i in range(2):
                frame[f"{name}_prediction_{i}"] = p[:, i]
    models = ["fine3072", "same_data_baseline3840", "local3840"]
    metrics, comparisons = [], []
    for cohort, part in [*frame.groupby("cohort"), ("pooled256", frame)]:
        for i, target in enumerate(study.TARGETS):
            y = part[target].to_numpy(float)
            valid = np.isfinite(y)
            if i:
                valid &= part.wake_quality_pass.fillna(False).to_numpy(bool)
            for name in models:
                p = part[f"{name}_prediction_{i}"].to_numpy(float)
                metrics.append(dict(cohort=cohort, model=name, target=target,
                    count=int(valid.sum()), **study.helpers.direct.base.metric(y[valid], p[valid])))
            for name in models[:2]:
                comparisons.append(dict(cohort=cohort, target=target, reference=name,
                    **study.batch.previous.evaluator.paired_bootstrap(
                        y[valid], part[f"local3840_prediction_{i}"].to_numpy(float)[valid],
                        part[f"{name}_prediction_{i}"].to_numpy(float)[valid])))
    pd.DataFrame(metrics).to_csv(OUT / "两批复评指标.csv", index=False)
    frame.to_csv(OUT / "两批复评逐点.csv", index=False)
    study.save(OUT / "两批复评.json", dict(metrics=metrics, paired_comparisons=comparisons,
        previously_inspected_holdouts=True, selected_using_training_oof_only=True,
        unchanged_quality_rules=True, automatic_production_promotion=False))
    plot(frame, pd.DataFrame(metrics))
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def plot(data, metrics):
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    colors = {"uniform128": "#34859a", "scale128": "#d78164"}
    names = {"uniform128": "第一批留出构型", "scale128": "第二批留出构型"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, target in enumerate(study.TARGETS):
        valid = np.isfinite(data[target].to_numpy(float))
        if i:
            valid &= data.wake_quality_pass.fillna(False).to_numpy(bool)
        d = data.loc[valid]
        for cohort, part in d.groupby("cohort"):
            axes[0, i].scatter(part[target], part[f"local3840_prediction_{i}"],
                s=27, alpha=.7, color=colors[cohort], edgecolor="white", linewidth=.35,
                label=names[cohort])
        bounds = np.quantile(np.r_[d[target], d[f"local3840_prediction_{i}"]], [0., 1.])
        low, high = bounds + np.array([-1., 1.]) * .06 * np.ptp(bounds)
        axes[0, i].plot([low, high], [low, high], "--", color="#6c757b", linewidth=1)
        unit = "（1/rad）" if i == 0 else ""
        title = "升力斜率" if i == 0 else "尾迹效率"
        axes[0, i].set(xlabel="VSPAERO 实算值"+unit, ylabel="代理预测值"+unit,
            title=title+"预测与实算", xlim=(low, high), ylim=(low, high))
        axes[0, i].set_aspect("equal")
        axes[0, i].legend(frameon=False, fontsize=8)
        model_order = ["fine3072", "same_data_baseline3840", "local3840"]
        labels = ["现有模型\n3072 构型", "原结构重训\n3840 构型", "新增树模型组合\n3840 构型"]
        for offset, cohort in ((-.18, "uniform128"), (.18, "scale128")):
            part = metrics[(metrics.target == target) & (metrics.cohort == cohort)].set_index("model")
            values = part.loc[model_order, "r2"].to_numpy()
            bars = axes[1, i].bar(np.arange(3)+offset, values, width=.34,
                                 color=colors[cohort], label=names[cohort])
            for bar, value in zip(bars, values):
                axes[1, i].text(bar.get_x()+bar.get_width()/2, value+.012, f"{value:.2f}",
                               ha="center", fontsize=9)
        axes[1, i].axhline(.9, linestyle=":", color="#818d94", linewidth=1)
        axes[1, i].set(xticks=np.arange(3), xticklabels=labels, ylabel="R²",
                       ylim=(0., 1.06), title=title+"的相同留出构型比较")
        axes[1, i].grid(axis="y", alpha=.16)
        axes[1, i].set_axisbelow(True)
        axes[1, i].tick_params(axis="x", labelsize=9)
    fig.tight_layout(pad=2.)
    fig.savefig(OUT / "新增模型精度对照.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "新增模型精度对照.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
