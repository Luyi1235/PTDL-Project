"""Paired mesh refinement using unchanged geometries and solver settings."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import pandas as pd

from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON, FEATURE_LABELS

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--levels", nargs="+", default=["legacy", "medium", "fine"])
    args = p.parse_args()
    out = ROOT / "运行结果" / "网格加密对照_20261015"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv")
    selected = frame.iloc[[0, 1, 12]][:args.count]
    rows = []
    def job(row, level):
        name = f"{row['sample']}_{level}"
        folder = out / name
        saved = [folder / "record.json"] + sorted(out.glob(name + "_retry*/record.json"))
        for record_path in saved:
            if record_path.exists():
                return json.loads(record_path.read_text(encoding="utf8"))
        if folder.exists():
            n = 1
            while folder.with_name(name + f"_retry{n}").exists():
                n += 1
            folder = folder.with_name(name + f"_retry{n}")
        design = {k: float(row[k]) for k in (*FEATURE_LABELS, "propeller_diameter_m",
                                             "payload_mass_kg", "hybrid_system_mass_kg")}
        record = _run_case(DEFAULT_PYTHON, design, name, folder, 1200,
                           "mesh_convergence", "same_geometry",
                           (0., float("inf")), 60, level)
        record["mesh_level"] = level
        record["source_sample"] = row["sample"]
        (folder / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                          encoding="utf8")
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(job, row, level): (row["sample"], level)
                for _, row in selected.iterrows() for level in args.levels}
        for f in as_completed(jobs):
            try:
                record = f.result()
            except Exception as e:
                record = dict(sample=jobs[f][0], mesh_level=jobs[f][1], error=repr(e))
            rows.append(record)
            pd.DataFrame(rows).to_csv(out / "网格加密比较.csv", index=False)
            print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
