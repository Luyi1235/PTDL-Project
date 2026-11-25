"""Measure local solver sensitivity without using any held-out labels."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import 推进_千构型原定义代理20261017 as original

ROOT = original.ROOT
SOURCE = ROOT / "运行结果/千构型分区主动补样_20261017"
OUT = ROOT / "运行结果/微小几何扰动_20261017"


def main():
    OUT.mkdir(exist_ok=True)
    frame = pd.read_csv(SOURCE / "中网格训练实算标签.csv")
    subset = frame.loc[frame.wake_quality_pass.to_numpy(bool)].reset_index(drop=True)
    x = subset[list(original.prior.FEATURE_LABELS)].to_numpy(float)
    records = json.loads((SOURCE / "新增训练交叉验证.json").read_text(encoding="utf8"))
    selected = json.loads((SOURCE / "训练模型选择.json").read_text(encoding="utf8"))["selected"]
    predictions = []
    for split in ("random", "spatial"):
        mix = np.zeros(len(subset))
        for spec in selected:
            p = np.full(len(subset), np.nan)
            for record in records:
                if record["split"] == split and record["spec"]["settings"] == spec["settings"]:
                    p[record["validation"]] = record["prediction"]
            if not np.isfinite(p).all():
                raise RuntimeError("Incomplete training OOF records")
            mix += spec["weight"]*p
        predictions.append(mix)
    error = np.sqrt(np.mean((np.asarray(predictions)-subset.vspaero_oswald_efficiency.to_numpy())**2, axis=0))
    order = np.argsort(error)
    anchors = [0, int(order[len(order)//4]), int(order[3*len(order)//4]), int(order[95*len(order)//100])]
    anchors = list(dict.fromkeys(anchors))
    ranges = frame[list(original.prior.FEATURE_LABELS)].max()-frame[list(original.prior.FEATURE_LABELS)].min()
    tasks = []
    for a, index in enumerate(anchors):
        row = subset.iloc[index]
        design = {k: float(row[k]) for k in (*original.prior.FEATURE_LABELS,
                  "propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
        variations = [("repeat", 0., design.copy())]
        for key in ("wing_area_m2", "tip_twist_deg", "fuselage_diameter_m"):
            for sign in (-1, 1):
                modified = design.copy()
                modified[key] += sign*.001*float(ranges[key])
                violations = original.prior.geometry_constraints(original.prior.AirframeDesign(**modified))
                if max(violations.values()) <= 1e-9:
                    variations.append((key, sign*.001, modified))
        for key, fraction, geometry in variations:
            tasks.append(dict(sample=f"local_{a:02d}_{len(tasks):03d}", design=geometry,
                              anchor_sample=row["sample"], changed=key, range_fraction=fraction,
                              old_lift=float(row.vspaero_cl_alpha_per_rad),
                              old_wake=float(row.vspaero_oswald_efficiency),
                              anchor_oof_error=float(error[index])))
    (OUT / "冻结配置.json").write_text(json.dumps(dict(
        tasks=tasks, target_definition="Unchanged original free-wake labels",
        training_only=True, perturbation_range_fraction=.001), indent=2), encoding="utf8")
    original.prior.solver.WORKER = original.WORKER
    rows = []
    started = time.monotonic()
    for task in tasks:
        folder = OUT / "solver" / task["sample"]
        cache = folder / "record.json"
        if cache.exists():
            row = json.loads(cache.read_text(encoding="utf8"))
        else:
            folder.mkdir(parents=True)
            try:
                row = original.native(
                    original.prior.solver.DEFAULT_PYTHON, task["design"], task["sample"],
                    folder / "case", 3600, "diagnostic", "training_local_sensitivity",
                    (0., float("inf")), 30, "legacy")
                row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
            except Exception as exc:
                row = dict(solver_failure=repr(exc))
            row.update(task)
            cache.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        if "vspaero_oswald_efficiency" in row:
            row["wake_change"] = row["vspaero_oswald_efficiency"]-task["old_wake"]
            row["lift_change"] = row["vspaero_cl_alpha_per_rad"]-task["old_lift"]
        rows.append(row)
        pd.DataFrame(rows).drop(columns=["design"]).to_csv(OUT / "局部变化.csv", index=False)
        (OUT / "进度.json").write_text(json.dumps(dict(
            stage="completed" if len(rows) == len(tasks) else "solver",
            completed=len(rows), requested=len(tasks), elapsed_s=time.monotonic()-started)), encoding="utf8")
        print("Local sensitivity", len(rows), "/", len(tasks), flush=True)


if __name__ == "__main__":
    main()
