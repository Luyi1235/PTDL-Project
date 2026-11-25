"""Evaluate frozen models after training-only selection; retain all valid holdout labels."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import 研究_密度均衡与残差代理20261017 as study
from 分目标气动高斯过程 import TargetwiseAeroGP


def paired_delta(y, baseline, candidate, seed=2026101731):
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(y), (2000, len(y)))
    actual = y[index]
    variance = np.sum((actual-actual.mean(axis=1, keepdims=True))**2, axis=1)
    delta = (np.sum((baseline[index]-actual)**2, axis=1)
             -np.sum((candidate[index]-actual)**2, axis=1))/variance
    return dict(mean=float(delta.mean()), lower=float(np.quantile(delta, .025)),
                upper=float(np.quantile(delta, .975)), resamples=2000)


def main():
    fresh = study.ROOT / "运行结果/几何交互独立验证_20261017"
    if json.loads((fresh / "进度.json").read_text(encoding="utf8"))["stage"] != "completed":
        raise RuntimeError("Wait for the complete 128-case batch before evaluation")
    paths = {
        "workbench289": study.previous.prior.original.prior.driver.LOW_MODEL,
        "previous1512": study.previous.prior.OUT / "多精度候选代理.json",
        "geometry_interaction": study.REFERENCE,
        "balanced_candidate": study.OUT / "密度均衡候选代理.json",
    }
    selection = study.OUT / "模型选择.json"
    frozen = dict(model_hashes={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
                  selection_hash=hashlib.sha256(selection.read_bytes()).hexdigest(),
                  evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S"), automatic_production_promotion=False)
    (study.OUT / "留出评价模型冻结.json").write_text(json.dumps(frozen, indent=2), encoding="utf8")
    models = {name: TargetwiseAeroGP.from_json(path) for name, path in paths.items()}
    metrics, deltas = [], []
    for batch, path, role in (
        ("earlier96", study.previous.prior.SOURCE / "中网格独立测试标签.csv", "retrospective"),
        ("later96", study.previous.prior.OUT / "中网格独立测试标签.csv", "retrospective"),
        ("fresh128", fresh / "独立测试逐点.csv", "held_out_not_used_for_selection"),
    ):
        frame = pd.read_csv(path)
        for name, model in models.items():
            p = np.asarray([model.predict(study.previous.prior.original.prior.AirframeDesign(
                **{key: float(row[key]) for key in study.FEATURES})) for _, row in frame.iterrows()])
            for index, target in enumerate(study.TARGETS):
                frame[f"{name}_prediction_{index}"] = p[:, index]
                y = frame[target].to_numpy(float)
                valid = np.isfinite(y)
                if index:
                    valid &= frame.wake_quality_pass.to_numpy(bool)
                metrics.append(dict(batch=batch, role=role, model=name, target=target, count=int(valid.sum()),
                                    **study.previous.prior.direct.base.metric(y[valid], p[valid, index])))
        for index, target in enumerate(study.TARGETS):
            y = frame[target].to_numpy(float)
            valid = np.isfinite(y)
            if index:
                valid &= frame.wake_quality_pass.to_numpy(bool)
            baseline = frame[f"geometry_interaction_prediction_{index}"].to_numpy(float)[valid]
            candidate = frame[f"balanced_candidate_prediction_{index}"].to_numpy(float)[valid]
            deltas.append(dict(batch=batch, target=target, compared_to="geometry_interaction",
                               **paired_delta(y[valid], baseline, candidate)))
        frame.to_csv(study.OUT / f"{batch}_逐点比较.csv", index=False)
    pd.DataFrame(metrics).to_csv(study.OUT / "留出评价.csv", index=False)
    (study.OUT / "留出评价.json").write_text(json.dumps(dict(
        **frozen, metrics=metrics, paired_r2_difference=deltas, goal_r2=.9,
        training_labels_only_for_selection=True), indent=2), encoding="utf8")
    print(pd.DataFrame(metrics).loc[lambda d: d.batch == "fresh128",
                                  ["model", "target", "count", "r2", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
