"""Check near-field and wake lift consistency before accepting solver labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from 检查_尾迹标签质量 import polar_quality as basic_quality

CRITERION = "abs(CL-CLiw)<=0.05+0.15*max(abs(CL),abs(CLiw)) at all polar angles"


def polar_quality(path):
    result = basic_quality(path)
    polar = pd.read_csv(path)
    difference = abs(polar.cl-polar.cl_wake).to_numpy(float)
    # This provisional discrepancy check flags gross solver disagreement.
    # The allowance is a project QC rule, not a universal solver accuracy bound.
    allowance = .05+.15*np.maximum(abs(polar.cl), abs(polar.cl_wake)).to_numpy(float)
    force_pass = bool(np.isfinite(difference).all() and np.all(difference <= allowance))
    lift = polar.cl.to_numpy(float)
    fitted = (np.polyval(np.polyfit(polar.alpha_deg, lift, 1), polar.alpha_deg)
              if len(lift) >= 2 and np.isfinite(lift).all() else np.full(len(lift), np.nan))
    total = np.sum((lift-lift.mean())**2)
    result.update(
        lift_quality_pass=force_pass,
        wake_quality_pass=bool(result["quality_pass"] and force_pass),
        force_consistency_pass=force_pass,
        max_lift_disagreement=float(np.max(difference)),
        force_inconsistent_points=int(np.sum(difference > allowance)),
        lift_curve_linearity_r2=float(1-np.sum((lift-fitted)**2)/total) if total > 0 else None,
        quality_pass=bool(result["quality_pass"] and force_pass),
    )
    return result


def checked_row(row):
    row = dict(row)
    path = Path(str(row.get("geometry_file", ""))).parent / "vspaero_polar.csv"
    for name in ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"):
        row.setdefault("raw_"+name, row.get(name))
    try:
        if not path.exists():
            raise FileNotFoundError("Missing original polar")
        row.update(polar_quality(path))
    except (OSError, ValueError, KeyError, TypeError) as error:
        row.update(quality_pass=False, lift_quality_pass=False, wake_quality_pass=False,
                   force_consistency_pass=False, quality_error=repr(error))
    row["force_consistency_criterion"] = CRITERION
    if not row["lift_quality_pass"]:
        row["vspaero_cl_alpha_per_rad"] = None
    if not row["wake_quality_pass"]:
        row["vspaero_oswald_efficiency"] = None
    return row


def audit(data, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data)
    result = pd.DataFrame([checked_row(row) for row in frame.to_dict("records")])
    result.to_csv(output / "逐构型质量记录.csv", index=False)
    summary = dict(
        requested=len(frame), valid_lift=int(result.lift_quality_pass.sum()),
        valid_wake=int(result.wake_quality_pass.sum()),
        force_inconsistent=int(((result.force_consistency_pass == False)
                                & result.get("quality_error", pd.Series(index=result.index, dtype=object)).isna()).sum()),
        missing_or_invalid_polars=int(result.get("quality_error", pd.Series(index=result.index, dtype=object)).notna().sum()),
        source_labels_modified=False, criterion=CRITERION,
        exclusions="Solver/force-consistency rules only; no prediction-error or Ew upper-limit filter",
    )
    (output / "质量汇总.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return result, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.output_dir)[1]))
