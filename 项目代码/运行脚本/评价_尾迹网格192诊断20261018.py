"""Reused diagnostic cohort; final generalization is measured on the new 256 cases."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import 研究_尾迹网格与均匀风险20261018 as study
from 均匀风险尾迹代理 import UniformRiskWakeAero
from 尾迹射线网格特征 import FEATURE_NAMES, extract

OUT = study.OUT / "192构型诊断复评"
SOURCE = study.batch.OUT / "独立评价逐点.csv"


def main():
    OUT.mkdir(exist_ok=True)
    selection_path = study.OUT / "模型选择.json"
    selection = study.batch.previous.read(selection_path)
    study.utility.check_hashes(selection["hashes"])
    frame = pd.read_csv(SOURCE)
    x = frame[study.FEATURES].to_numpy(float)
    mesh, missing = [], []
    for _, row in frame.iterrows():
        try:
            value = extract(Path(row.geometry_file).with_suffix(".vspgeom"),
                            row.wing_area_m2, row.aspect_ratio)
            missing.append(False)
        except Exception:
            value = np.zeros(len(FEATURE_NAMES))
            missing.append(True)
        mesh.append(value)
    mesh = np.asarray(mesh)
    models = ["fine3072", "fine4096"]
    for selected in selection["selected"]:
        name = selected["name"]
        p = UniformRiskWakeAero.load(selected["path"]).predict_many(x, mesh)
        for i in range(2):
            frame[f"{name}_prediction_{i}"] = p[:, i]
        models.append(name)
    metrics, intervals = [], []
    for i, target in enumerate(study.batch.TARGETS):
        y = frame[target].to_numpy(float)
        valid = np.isfinite(y)
        if i:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        for name in models:
            p = frame[f"{name}_prediction_{i}"].to_numpy(float)
            metrics.append(dict(model=name, target=target, count=int(valid.sum()),
                **study.helpers.direct.base.metric(y[valid], p[valid])))
            if name not in ("fine3072", "fine4096"):
                intervals.append(dict(model=name, target=target,
                    **study.batch.previous.evaluator.paired_bootstrap(y[valid], p[valid],
                        frame[f"fine4096_prediction_{i}"].to_numpy(float)[valid])))
    pd.DataFrame(metrics).to_csv(OUT / "诊断指标.csv", index=False)
    frame["mesh_feature_missing"] = missing
    frame.to_csv(OUT / "诊断逐点.csv", index=False)
    paths = [SOURCE, selection_path, Path(__file__)] + [
        Path(entry["path"]) for entry in selection["selected"]]
    study.utility.save(OUT / "诊断评价.json", dict(metrics=metrics, comparisons=intervals,
        hashes={str(p): study.utility.digest(p) for p in paths},
        previously_inspected_diagnostic_set=True,
        training_labels_exclude_this_cohort=True, independent_test_size=256,
        target_definition_unchanged=True, quality_rules_unchanged=True,
        missing_geometry_features=int(sum(missing))))
    plot(frame)
    print(json.dumps(metrics), flush=True)


def plot(frame):
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none"})
    valid = frame.wake_quality_pass.fillna(False) & frame[study.TARGET].notna()
    data = frame.loc[valid]
    y = data[study.TARGET].to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    styles = [
        ("fine4096", "4096 构型原模型", "#aab4ba"),
        ("uniform_risk", "均匀区域选型", "#34859a"),
        ("all_risk", "全部区域选型", "#d78164"),
    ]
    bounds = (min(y.min(), .25), max(y.max(), 1.15))
    for axis, name, label, color in (
        (axes[0], "fine4096", "4096 构型原模型", "#87969e"),
        (axes[1], "uniform_risk", "均匀区域选型模型", "#34859a"),
    ):
        p = data[f"{name}_prediction_1"].to_numpy(float)
        axis.scatter(y, p, s=25, alpha=.7, color=color, edgecolors="white", linewidth=.3)
        axis.plot(bounds, bounds, "--", color="#89949b", linewidth=1)
        axis.set(xlim=bounds, ylim=bounds, xlabel="VSPAERO 实算尾迹效率",
                 ylabel="代理预测尾迹效率", title=label)
        axis.set_aspect("equal")
    for name, label, color in styles:
        error = abs(data[f"{name}_prediction_1"].to_numpy(float)-y)
        sorted_error = np.sort(error)
        axes[2].plot(sorted_error, np.arange(1, len(error)+1)/len(error),
                     label=label, color=color, linewidth=1.8)
    axes[2].set(xlabel="尾迹效率绝对误差", ylabel="累计构型比例", title="相同构型的误差分布",
                 ylim=(0., 1.03))
    axes[2].grid(alpha=.15)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=1.7)
    fig.savefig(OUT / "尾迹模型诊断对照.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "尾迹模型诊断对照.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
