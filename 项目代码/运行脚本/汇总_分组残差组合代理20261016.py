"""Freeze a weighted grouped candidate and record retrospective evaluation."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign
from 研究_分组多尺度Kriging20261016 import make_kernel, encode_kernel, predict
from 训练_主动增量混合代理20261016 import metric

RESULTS = ROOT / "运行结果"
PRIOR = RESULTS / "主动增量代理_20261015"
WEIGHTED = RESULTS / "残差加权Kriging几何索引_20261016"
OUT = RESULTS / "分组残差组合代理几何索引_20261016"


def predict_candidate(payload, x):
    path = Path(payload["reference_model"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != payload["reference_sha256"]:
        raise RuntimeError("Frozen reference model changed")
    prior = load_vsp_surrogate(path)
    values = np.array([prior.predict(AirframeDesign(**dict(zip(FEATURE_LABELS, row)))) for row in x])
    weight = payload["wake_weight"]
    values[:, 1] = weight*predict(payload["weighted_wake_model"], x)+(1-weight)*values[:, 1]
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(WEIGHTED / "残差加权五折比较.csv")
    best = rows.loc[rows.rmse.idxmin()]
    filename = f"折外_{best['kernel']}_{best['noise_floor']}_{best['residual_power']}.csv"
    proposed = pd.read_csv(WEIGHTED / filename)
    reference = pd.read_csv(PRIOR / "选定模型折外预测_1.csv")
    np.testing.assert_allclose(proposed.actual, reference.actual, rtol=0, atol=1e-12)
    choices = []
    for weight in (0., .25, .5, .75, 1.):
        estimates = weight*proposed.prediction.to_numpy()+(1-weight)*reference.prediction.to_numpy()
        choices.append(dict(weight=weight, **metric(reference.actual.to_numpy(), estimates)))
    selected = min(choices, key=lambda row: row["rmse"])
    frame = pd.read_csv(PRIOR / "增量训练样本.csv")
    frame = frame.loc[frame.wake_quality_pass]
    x, y = frame[list(FEATURE_LABELS)].to_numpy(float), frame.vspaero_oswald_efficiency.to_numpy(float)
    residual = pd.read_csv(RESULTS / "原网格训练标签残差核查_20261016" / "逐构型逐迎角最终残差.csv")
    logs = frame.geometry_file.map(
        residual.loc[residual.available].groupby("geometry_file").log10_max_residual.max())
    logs = logs.fillna(float(logs.median())).to_numpy(float)
    alpha = best.noise_floor*np.clip(10**(best.residual_power*(logs+3)), 1., 100.)
    features = feature_map(x, "physics")
    mean, scale = features.mean(0), np.maximum(features.std(0), 1e-9)
    z = (features-mean)/scale
    gp = GaussianProcessRegressor(kernel=make_kernel(best["kernel"]), alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=2026101641)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(z, y)
    model = dict(covariance=encode_kernel(gp.kernel_), feature_mean=mean.tolist(),
                 feature_scale=scale.tolist(), train=z.tolist(), dual=gp.alpha_.tolist(),
                 mean=float(gp._y_train_mean), scale=float(gp._y_train_std),
                 transform="identity", alpha=alpha.tolist())
    np.testing.assert_allclose(predict(model, x[:5]), gp.predict(z[:5]), rtol=1e-10, atol=1e-10)
    reference_path = PRIOR / "增量构型气动代理.json"
    payload = dict(schema="ptdl_research_weighted_kernel_blend_v1", training_samples=289,
                   wake_training_samples=len(y), reference_model=str(reference_path),
                   reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                   weighted_wake_model=model, wake_weight=selected["weight"],
                   training_selection=selected, automatic_production_promotion=False)
    path = OUT / "分组残差组合候选.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf8")
    (OUT / "模型冻结.json").write_text(json.dumps(dict(
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), selected=selected), indent=2), encoding="utf8")
    pd.DataFrame(choices).to_csv(OUT / "训练折外组合权重比较.csv", index=False)
    # Both batches were observed in earlier research. This is retrospective.
    test = pd.read_csv(PRIOR / "合并留出验证.csv")
    prediction = predict_candidate(payload, test[list(FEATURE_LABELS)].to_numpy(float))
    metrics = []
    for t, target in enumerate(("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")):
        valid = np.isfinite(test[target].to_numpy(float))
        if t:
            valid &= test.wake_quality_pass.to_numpy(bool)
        actual = test.loc[valid, target].to_numpy(float)
        metrics.append(dict(target=target, count=int(valid.sum()),
                            **metric(actual, prediction[valid, t])))
        test[f"candidate_prediction_{t}"] = prediction[:, t]
    test.to_csv(OUT / "回溯逐点预测.csv", index=False)
    assessment = dict(metrics=metrics, evaluation_role="retrospective; not a new blind test",
                      training_labels_unchanged=True, valid_outliers_retained=True,
                      production_promotion=False)
    (OUT / "32构型回溯评价.json").write_text(json.dumps(assessment, indent=2), encoding="utf8")
    print(json.dumps(dict(training=selected, retrospective=assessment), indent=2), flush=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    grouped = pd.read_csv(RESULTS / "分组多尺度Kriging_20261016" / "分组多尺度五折比较.csv")
    group_r2 = float(grouped.loc[(grouped.target == 1) & (grouped.kernel == "grouped") &
                                (grouped["transform"] == "identity"), "r2"].iloc[0])
    values = [1-np.sum((reference.prediction-reference.actual)**2)/
              np.sum((reference.actual-reference.actual.mean())**2), group_r2,
              float(best.r2), selected["r2"]]
    bars = axes[0].bar(range(4), values, color=["#87aaa9", "#488ba2", "#d48774", "#699a7b"], width=.6)
    axes[0].set_xticks(range(4), ["现有组合", "分组 Kriging", "残差加权", "新的组合"])
    axes[0].set(ylim=(0, 1), ylabel="尾迹效率 R²", title="相同训练集五折验证 · 277 个构型")
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+.02, f"{value:.2f}", ha="center")
    old = json.loads((PRIOR / "合并留出评估.json").read_text(encoding="utf8"))["metrics"]["activegp"][1]["r2"]
    bars = axes[1].bar(range(2), [old, metrics[1]["r2"]], color=["#87aaa9", "#699a7b"], width=.5)
    axes[1].set_xticks(range(2), ["现有组合", "新的组合"])
    axes[1].set(ylim=(min(0., metrics[1]["r2"]-.1), 1), ylabel="尾迹效率 R²",
                title="32 个既有留出构型回溯评价")
    for bar, value in zip(bars, [old, metrics[1]["r2"]]):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.02, f"{value:.2f}", ha="center")
    for ax in axes:
        ax.axhline(.9, color="#a46961", ls="--", lw=1)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "代理结构改进与回溯评价.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "代理结构改进与回溯评价.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
