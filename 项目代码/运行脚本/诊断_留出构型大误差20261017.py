"""Attribute squared error and repeat selected native cases without replacing labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import 推进_尺度约束完整训练20261017 as pipeline

ROOT = pipeline.ROOT
OUT = ROOT / "运行结果/尺度约束留出误差诊断_20261017_v2"


def main():
    OUT.mkdir(exist_ok=True)
    source = pipeline.OUT / "2152留出诊断/后续留出诊断逐点.csv"
    frame = pd.read_csv(source)
    contribution = []
    for index, target in enumerate(pipeline.TARGETS):
        y = frame[target].to_numpy(float)
        valid = np.isfinite(y)
        if index:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        sub = frame.loc[valid]
        error = (sub[target].to_numpy(float)
                 - sub[f"scale_candidate_prediction_{index}"].to_numpy(float))**2
        order = np.argsort(-error)
        contribution.append(dict(target=target, valid=int(valid.sum()),
            top1_squared_error_fraction=float(error[order[:1]].sum()/error.sum()),
            top2_squared_error_fraction=float(error[order[:2]].sum()/error.sum()),
            top5_squared_error_fraction=float(error[order[:5]].sum()/error.sum()),
            highest_error_samples=sub.iloc[order[:5]]["sample"].tolist()))
    plan = pipeline.read(pipeline.BATCH / "冻结配置.json")
    samples = contribution[0]["highest_error_samples"][:2]
    jobs = [next(j for j in plan["jobs"] if j["sample"] == name) for name in samples]
    frozen = OUT / "诊断配置.json"
    paths = [Path(__file__), source, pipeline.expansion.balanced.previous.prior.original.WORKER]
    if frozen.exists():
        pipeline.evaluator.check_hashes(pipeline.read(frozen)["hashes"])
    else:
        pipeline.save(frozen, dict(jobs=jobs,
            hashes={str(p): pipeline.expansion.digest(p) for p in paths},
            selection="Largest lift-slope squared errors in the excluded diagnostic cohort",
            used_for_training_or_model_selection=False, original_labels_replaced=False))
    original = pipeline.expansion.balanced.previous.prior.original
    original.controlled.original.WORKER = original.WORKER
    repeats = []
    for job in jobs:
        folder = OUT / job["sample"]
        folder.mkdir(exist_ok=True)
        cache = folder / "record.json"
        if cache.exists():
            row = pipeline.read(cache)
        else:
            try:
                row = original.controlled.run_case(
                    original.prior.solver.DEFAULT_PYTHON, job["design"], job["sample"],
                    folder / "case", 600, "diagnostic", "repeat_largest_excluded_errors",
                    (0., float("inf")), 30, "legacy")
                row.update(original.verify(folder / "case"))
                row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
            except Exception as exc:
                row = dict(solver_failure=repr(exc))
            pipeline.save(cache, row)
        previous = frame.set_index("sample").loc[job["sample"]]
        repeats.append({**row, "sample": job["sample"],
            "original_lift": float(previous[pipeline.TARGETS[0]]),
            "original_wake": float(previous[pipeline.TARGETS[1]])})
        if "solver_failure" not in row:
            repeats[-1]["lift_repeat_difference"] = (
                float(row[pipeline.TARGETS[0]])-float(previous[pipeline.TARGETS[0]]))
            repeats[-1]["wake_repeat_difference"] = (
                float(row[pipeline.TARGETS[1]])-float(previous[pipeline.TARGETS[1]]))
        print("Excluded error repeat:", job["sample"], flush=True)
    result = dict(error_contributions=contribution, repeats=repeats,
                  all_original_labels_and_evaluation_scores_retained=True,
                  diagnostic_labels_used_for_training=False)
    pipeline.save(OUT / "误差集中度与重复实算.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
