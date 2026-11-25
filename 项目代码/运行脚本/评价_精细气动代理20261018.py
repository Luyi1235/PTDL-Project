"""Re-evaluate frozen candidates on both existing excluded cohorts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import 推进_分歧补样4096代理20261018 as study
import 研究_面元邻近代理20261018 as panel
import joblib

ROOT = study.ROOT
OUT = ROOT / "运行结果/精细树与核代理_20261018/留出复评_v2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = dict(
        fine3072=study.fine.OUT / "精细气动候选代理.json",
        original3072=study.BASE / "均匀覆盖候选代理.json",
        canonical3072=study.previous.OUT / "完整训练/尺度与总体加权候选代理.json",
        geometry1512=study.fine.previous.prior.prior.REFERENCE)
    selection = study.previous.read(study.fine.OUT / "模型选择.json")
    if study.previous.expansion.digest(paths["fine3072"]) != selection["model_sha256"]:
        raise RuntimeError("Candidate changed after training-only selection")
    sources = dict(
        uniform128=study.BASE / "独立测试逐点.csv",
        scale128=study.previous.OUT / "3072新增独立测试/新增独立测试逐点.csv")
    panel_selection = panel.OUT / "模型选择.json"
    extra_paths = []
    panel_models = None
    if panel_selection.exists():
        panel_models = study.previous.read(panel_selection)["selected"]
        extra_paths = [panel_selection, Path(panel.__file__),
                       ROOT / "核心模型/气动面元邻近特征.py"]
        extra_paths += [Path(c["path"]) for entry in panel_models for c in entry["components"]]
    frozen = OUT / "评价配置冻结.json"
    hashes = {str(p): study.previous.expansion.digest(p)
              for p in [Path(__file__), *paths.values(), *sources.values(), *extra_paths]}
    if frozen.exists():
        if study.previous.read(frozen)["hashes"] != hashes:
            raise RuntimeError("Retrospective evaluation source changed")
    else:
        study.previous.save(frozen, dict(hashes=hashes, selection_uses_training_oof_only=True,
            test_sets_previously_inspected=True, interpretation="Reused held-out diagnostic",
            new_independent_verification="192 fresh cases in the adaptive 4096 pipeline"))
    models = {}
    for name, path in paths.items():
        schema = study.previous.read(path).get("schema")
        cls = (study.FineAeroSurrogate if schema == "ptdl-fine-aero-v1" else
               study.ScalePopulationAero if schema == "ptdl-scale-population-aero-v1" else
               study.TargetwiseAeroGP)
        models[name] = cls.from_json(path)
    if panel_models is not None:
        models["panel_proximity"] = None
    design_class = study.previous.expansion.balanced.previous.prior.original.prior.AirframeDesign
    all_metrics, all_rows, bootstrap = [], [], []
    for cohort, source in sources.items():
        frame = pd.read_csv(source)
        x = frame[study.FEATURES].to_numpy(float)
        for name, model in models.items():
            if name == "panel_proximity":
                mesh = pd.DataFrame([panel.extract(row) for row in frame.to_dict("records")])[
                    panel.FEATURE_NAMES].to_numpy(float)
                p = np.column_stack([sum(c["weight"]*panel.predict(
                    joblib.load(c["path"]), x, mesh) for c in entry["components"])
                    for entry in panel_models])
            else:
                p = (model.predict_many(x) if hasattr(model, "predict_many")
                     else np.array([model.predict(design_class(
                         **dict(zip(study.FEATURES, row)))) for row in x]))
            for i in range(2):
                frame[f"{name}_prediction_{i}"] = p[:, i]
        for i, target in enumerate(study.TARGETS):
            y = frame[target].to_numpy(float)
            valid = np.isfinite(y)
            if i:
                valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
            for name in models:
                p = frame[f"{name}_prediction_{i}"].to_numpy(float)
                all_metrics.append(dict(cohort=cohort, model=name, target=target,
                    count=int(valid.sum()), **study.previous.expansion.balanced.previous.prior.direct.base.metric(
                        y[valid], p[valid])))
            bootstrap.append(dict(cohort=cohort, target=target,
                **study.previous.evaluator.paired_bootstrap(
                    y[valid], frame[f"fine3072_prediction_{i}"].to_numpy(float)[valid],
                    frame[f"original3072_prediction_{i}"].to_numpy(float)[valid])))
        frame["cohort"] = cohort
        all_rows.append(frame)
    pd.DataFrame(all_metrics).to_csv(OUT / "两批留出指标.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / "两批留出逐点.csv", index=False)
    study.previous.save(OUT / "两批留出评价.json", dict(
        metrics=all_metrics, r2_differences_vs_original3072=bootstrap,
        selection_uses_training_oof_only=True, previously_inspected_test_sets=True,
        quality_rules_unchanged=True, automatic_production_promotion=False))
    plot(pd.concat(all_rows, ignore_index=True), all_metrics)
    print(json.dumps(all_metrics), flush=True)


def plot(data, metrics):
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none"})
    colors = {"uniform128": "#32839b", "scale128": "#df9369"}
    labels = {"uniform128": "第一批留出构型", "scale128": "第二批留出构型"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, target in enumerate(study.TARGETS):
        valid = data[target].notna()
        if i:
            valid &= data.wake_quality_pass.fillna(False).astype(bool)
        d = data.loc[valid]
        low, high = min(d[target].min(), d[f"fine3072_prediction_{i}"].min()), max(
            d[target].max(), d[f"fine3072_prediction_{i}"].max())
        margin = .06*(high-low)
        for cohort, part in d.groupby("cohort"):
            axes[0, i].scatter(part[target], part[f"fine3072_prediction_{i}"],
                s=24, alpha=.65, color=colors[cohort], edgecolor="white", linewidth=.3,
                label=labels[cohort])
        axes[0, i].plot([low-margin, high+margin], [low-margin, high+margin],
                        color="#687279", linewidth=1, linestyle="--")
        axes[0, i].set(xlabel="VSPAERO 实算值（1/rad）" if i == 0 else "VSPAERO 实算值",
            ylabel="代理预测值（1/rad）" if i == 0 else "代理预测值",
            title=("升力斜率" if i == 0 else "尾迹效率") + "预测与实算")
        axes[0, i].set_aspect("equal", adjustable="box")
        axes[0, i].legend(frameon=False, fontsize=8)
        models = ["geometry1512", "original3072", "canonical3072", "fine3072"]
        names = ["1512\n构型模型", "3072\n构型模型", "尺度归一化\n模型", "精细树与核\n模型"]
        if any(r["model"] == "panel_proximity" for r in metrics):
            models.append("panel_proximity")
            names.append("面元邻近\n模型")
        x = np.arange(len(models))
        for offset, cohort in zip((-.18, .18), colors):
            values = [next(r["r2"] for r in metrics if
                (r["cohort"], r["model"], r["target"]) == (cohort, model, target)) for model in models]
            bars = axes[1, i].bar(x+offset, values, width=.34, color=colors[cohort],
                                  label=labels[cohort])
            for bar, value in zip(bars, values):
                axes[1, i].text(bar.get_x()+bar.get_width()/2, value+.012, f"{value:.2f}",
                                ha="center", va="bottom", fontsize=8)
        axes[1, i].axhline(.9, color="#8b969d", linestyle=":", linewidth=1)
        axes[1, i].set(xticks=x, xticklabels=names, ylabel="R²", ylim=(0, 1.06),
                       title="相同留出构型上的模型比较")
        axes[1, i].tick_params(axis="x", labelsize=8)
        axes[1, i].grid(axis="y", alpha=.16)
        axes[1, i].set_axisbelow(True)
    fig.tight_layout(pad=2.)
    fig.savefig(OUT / "代理精度对照.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "代理精度对照.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
