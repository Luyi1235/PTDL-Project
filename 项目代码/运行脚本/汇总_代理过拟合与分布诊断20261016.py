"""Retrospective model audit after freezing, with independent geometry diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import TargetwiseAeroGP
from 机体参数化设计与气动阻力 import AirframeDesign
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 训练_主动增量混合代理20261016 import metric

TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
SOURCES = {
    "legacy": (
        ROOT / "运行结果" / "百构型加密代理_20261016" / "一致性校核训练" / "low_labels" / "逐构型质量记录.csv",
        ROOT / "运行结果" / "主动增量代理_20261015" / "合并留出验证.csv",
        ROOT / "运行结果" / "主动增量代理_20261015" / "增量构型气动代理.json"),
    "medium": (
        ROOT / "运行结果" / "百构型加密代理_20261016" / "中网格训练实算标签.csv",
        ROOT / "运行结果" / "百构型加密代理_20261016" / "中网格盲测逐点.csv",
        ROOT / "运行结果" / "百构型加密代理_20261016" / "多精度候选代理.json"),
}


def predictions(model, frame):
    return np.array([model.predict(AirframeDesign(**{k: row[k] for k in FEATURE_LABELS}))
                     for _, row in frame.iterrows()])


def main(candidate, out):
    out.mkdir(parents=True, exist_ok=True)
    results, shifts = [], []
    for name, (train_path, test_path, old_path) in SOURCES.items():
        train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        path = candidate / f"{name}_正则化候选.json"
        lock = json.loads((candidate / f"{name}_模型冻结.json").read_text(encoding="utf8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock["sha256"]:
            raise RuntimeError("Frozen regularized model changed")
        models = {"reference": load_vsp_surrogate(old_path), "regularized": TargetwiseAeroGP.from_json(path)}
        for label, model in models.items():
            pred_train, pred_test = predictions(model, train), predictions(model, test)
            for index, target in enumerate(TARGETS):
                for role, frame, pred in (("training_fit", train, pred_train), ("retrospective_test", test, pred_test)):
                    y = frame[target].to_numpy(float)
                    valid = np.isfinite(y)
                    if index:
                        valid &= frame.wake_quality_pass.to_numpy(bool)
                    results.append(dict(dataset=name, model=label, target=target, role=role,
                                        count=int(valid.sum()), **metric(y[valid], pred[valid, index])))
                test[f"{label}_prediction_{index}"] = pred_test[:, index]
        a, b = train[list(FEATURE_LABELS)].to_numpy(float), test[list(FEATURE_LABELS)].to_numpy(float)
        span = np.maximum(a.max(0)-a.min(0), 1e-9)
        test["nearest_training_distance"] = np.linalg.norm((b[:, None]-a[None])/span, axis=2).min(1)
        test.to_csv(out / f"{name}_逐点回溯.csv", index=False)
        labels = np.r_[np.zeros(len(a)), np.ones(len(b))]
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.5, class_weight="balanced", max_iter=1000))
        score = cross_val_predict(estimator, np.vstack((a, b)), labels,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=2026101696),
                                  method="predict_proba")[:, 1]
        midpoint = (a.min(0)+a.max(0))/2
        shifts.append(dict(dataset=name, train_count=len(a), test_count=len(b),
                           geometry_source_classifier_auc=float(roc_auc_score(labels, score)),
                           train_boundary_score=float(np.max(abs((a-midpoint)/(span/2)), axis=1).mean()),
                           test_boundary_score=float(np.max(abs((b-midpoint)/(span/2)), axis=1).mean()),
                           median_test_nearest_distance=float(test.nearest_training_distance.median()),
                           classifier_scope="Feature-only diagnostic; test aerodynamic labels excluded",
                           interpretation="Small test sets; AUC alone does not establish a distribution difference"))
    frame = pd.DataFrame(results)
    frame.to_csv(out / "拟合与回溯比较.csv", index=False)
    (out / "几何分布诊断.json").write_text(json.dumps(shifts, indent=2), encoding="utf8")
    (out / "评价说明.json").write_text(json.dumps(dict(
        new_blind_test=False, candidate_frozen_before_this_scoring=True,
        historical_targets_unchanged=True, outlier_error_filter=False,
        automatic_production_promotion=False, candidate_directory=str(candidate)), indent=2), encoding="utf8")
    draw(frame, candidate, out)


def draw(frame, candidate, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                         "axes.unicode_minus": False, "font.size": 10})
    nested = pd.read_csv(candidate / "嵌套验证汇总.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    for row, dataset in enumerate(("legacy", "medium")):
        for column, target in enumerate(TARGETS):
            ax = axes[row, column]
            selected = frame[(frame.dataset == dataset) & (frame.target == target) & (frame.model == "regularized")]
            original = frame[(frame.dataset == dataset) & (frame.target == target)
                             & (frame.model == "reference") & (frame.role == "retrospective_test")].iloc[0]
            random = nested[(nested.dataset == dataset) & (nested.target == target) & (nested.split == "random")].iloc[0]
            spatial = nested[(nested.dataset == dataset) & (nested.target == target) & (nested.split == "spatial")].iloc[0]
            values = [selected[selected.role == "training_fit"].iloc[0].r2,
                      random.r2, spatial.r2, original.r2,
                      selected[selected.role == "retrospective_test"].iloc[0].r2]
            positions = np.arange(5)
            ax.bar(positions, values, color=["#9db9ac", "#5794ac", "#b79e6d", "#b9b9be", "#ca816d"], width=.65)
            for x, value in zip(positions, values):
                ax.text(x, value+.025 if value >= 0 else value-.035, f"{value:.2f}",
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
            ax.set_xticks(positions, ["训练\n拟合", "随机\n嵌套验证", "区域\n嵌套验证", "原模型\n留出回溯", "新模型\n留出回溯"])
            ax.axhline(.9, color="#995850", linestyle="--", linewidth=1)
            ax.set_ylim(min(-.05, min(values)-.1), 1.10)
            ax.set_title(("原网格" if dataset == "legacy" else "加密网格")
                         + (" · 升力斜率" if column == 0 else " · 尾迹效率"))
            ax.set_ylabel("R²")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
    fig.savefig(out / "拟合与泛化诊断.png", dpi=220)
    fig.savefig(out / "拟合与泛化诊断.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path,
                        default=ROOT / "运行结果" / "正则化与嵌套验证_20261016")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "过拟合诊断_20261016")
    args = parser.parse_args()
    main(args.candidate_dir, args.output_dir)
