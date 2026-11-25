"""Read actual final residuals without relabeling or filtering stored DOE."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def final_residuals(path):
    header, rows = None, []
    for line in Path(path).read_text(encoding="utf8", errors="replace").splitlines():
        words = line.replace("L2 Residual", "L2Res").replace("Max Residual", "MaxRes").split()
        if words and words[0] == "Iter" and "MaxRes" in words:
            header = words
            continue
        if header and len(words) == len(header):
            try:
                row = dict(zip(header, map(float, words)))
            except ValueError:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No residual history rows in {path}")
    frame = pd.DataFrame(rows)
    return frame.sort_values("Iter").groupby(["Mach", "AoA", "Beta"], as_index=False).tail(1)


def audit(data, output, threshold_log10=-3.):
    frame = pd.read_csv(data)
    records = []
    for _, row in frame.iterrows():
        files = list(Path(str(row.geometry_file)).parent.glob("*.history"))
        if not files:
            records.append(dict(sample=row["sample"], geometry_file=str(row.geometry_file), available=False))
            continue
        try:
            last = final_residuals(files[0])
            for _, point in last.iterrows():
                records.append(dict(sample=row["sample"], geometry_file=str(row.geometry_file),
                                    available=True, alpha_deg=point.AoA,
                                    final_iteration=int(point.Iter), log10_l2_residual=point.L2Res,
                                    log10_max_residual=point.MaxRes,
                                    threshold_log10=threshold_log10,
                                    passes_reference_residual=bool(point.MaxRes <= threshold_log10)))
        except (OSError, ValueError) as error:
            records.append(dict(sample=row["sample"], geometry_file=str(row.geometry_file),
                                available=False, error=repr(error)))
    result = pd.DataFrame(records)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "逐构型逐迎角最终残差.csv", index=False)
    known = result.loc[result.available]
    summary = dict(
        reference_max_residual=10**threshold_log10, geometries=len(frame),
        histories_available=int(known.geometry_file.nunique()), points=len(known),
        points_passing=int(known.passes_reference_residual.sum()) if len(known) else 0,
        all_point_pass_geometries=int(known.groupby("geometry_file").passes_reference_residual.all().sum())
        if len(known) else 0, labels_modified=False,
    )
    (output / "残差收敛汇总.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.output_dir)))
