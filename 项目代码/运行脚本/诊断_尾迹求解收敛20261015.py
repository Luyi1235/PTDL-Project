"""Keep geometry fixed while checking wake iteration and mesh dependence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON, FEATURE_LABELS
from 检查_尾迹标签质量 import polar_quality

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "运行结果" / "尾迹求解收敛_20261015"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv")
    test = pd.read_csv(ROOT / "运行结果" / "物理特征代理_20261015" / "新独立验证_逐点.csv")
    valid = test[test.wake_quality_pass].copy()
    residual = abs(valid.prediction_1 - valid.vspaero_oswald_efficiency)
    selected = [
        ("baseline", train.iloc[0]),
        ("angle_sensitive", train[train["sample"] == "adaptive_train_02"].iloc[0]),
        ("large_residual", valid.loc[residual.idxmax()]),
    ]
    records = []
    for label, row in selected:
        design = {k: float(row[k]) for k in (*FEATURE_LABELS, "propeller_diameter_m",
                                            "payload_mass_kg", "hybrid_system_mass_kg")}
        original = Path(row.geometry_file).parent / "vspaero_polar.csv"
        records.append(dict(sample=label, source_sample=row["sample"], mesh_level="legacy",
                            wake_iterations=30, status="existing label",
                            cl_alpha=float(row.vspaero_cl_alpha_per_rad),
                            ew=float(row.vspaero_oswald_efficiency), **polar_quality(original)))
        for wake in (60, 120):
            folder = OUT / f"{label}_legacy_wake{wake}"
            path = folder / "record.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf8"))
            else:
                if folder.exists():
                    suffix = 1
                    while folder.with_name(folder.name + f"_retry{suffix}").exists():
                        suffix += 1
                    folder = folder.with_name(folder.name + f"_retry{suffix}")
                try:
                    result = _run_case(DEFAULT_PYTHON, design, label, folder, 900,
                                       "convergence_probe", "unchanged_geometry", (0, float("inf")),
                                       wake, "legacy")
                    record = dict(sample=label, source_sample=row["sample"], mesh_level="legacy",
                                  wake_iterations=wake, status="completed",
                                  cl_alpha=result["vspaero_cl_alpha_per_rad"],
                                  ew=result["vspaero_oswald_efficiency"],
                                  runtime_s=result["solver_wall_time_s"],
                                  **polar_quality(folder / "vspaero_polar.csv"))
                except Exception as error:
                    record = dict(sample=label, mesh_level="legacy", wake_iterations=wake,
                                  status="failed", error=repr(error))
                (folder / "record.json").write_text(json.dumps(record, indent=2), encoding="utf8")
            records.append(record)
            pd.DataFrame(records).to_csv(OUT / "尾迹迭代收敛.csv", index=False)
            print(json.dumps(record), flush=True)
    # Reuse the paired mesh measurements without mixing them into training labels.
    mesh = pd.read_csv(ROOT / "运行结果" / "网格加密对照_20261015" / "网格加密比较.csv")
    mesh.to_csv(OUT / "已有网格加密对照.csv", index=False)
    rows = pd.DataFrame(records)
    diagnostic = dict(
        target_definition="Original Ew median with abs(CLiw)>=0.15; unchanged",
        wake_probes=records,
        mesh_note="Existing baseline Ew changes from 0.630 to 0.394 and 0.446 under mesh refinement.",
        acceptance="Inspect wake and grid changes before generating a replacement DOE; no failed point becomes a synthetic label.",
    )
    (OUT / "求解诊断.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
