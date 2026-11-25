"""Re-solve two high-error geometries without replacing held-out legacy labels."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON
from 检查_尾迹标签质量 import polar_quality

SOURCE = ROOT / "运行结果" / "主动气动补样_20261015"
OUT = ROOT / "运行结果" / "主要误差构型网格诊断_20261016"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE / "独立测试逐点.csv")
    frame["squared_lift_error"] = (frame.candidate_prediction_0-frame.vspaero_cl_alpha_per_rad)**2
    selected = frame.nlargest(2, "squared_lift_error")
    plan = json.loads((SOURCE / "采样预案.json").read_text(encoding="utf8"))
    jobs = {j["sample"]: j for j in plan["jobs"]}
    rows = []
    for _, old in selected.iterrows():
        source = jobs[old["sample"]]
        for mesh in ("legacy", "medium"):
            if mesh == "legacy":
                row = dict(old)
                row["mesh"] = mesh
            else:
                folder = OUT / source["sample"] / mesh
                record = folder / "record.json"
                if record.exists():
                    row = json.loads(record.read_text(encoding="utf8"))
                else:
                    if folder.exists():
                        raise RuntimeError(f"Inspect interrupted mesh case: {folder}")
                    try:
                        row = _run_case(DEFAULT_PYTHON, source["design"], source["sample"]+"_"+mesh,
                                        folder, 1800, "diagnostic",
                                        "largest_heldout_lift_errors_no_label_replacement",
                                        (0, float("inf")), 30, mesh)
                        row.update(polar_quality(folder / "vspaero_polar.csv"))
                    except Exception as error:
                        row = dict(solver_failure=repr(error), vspaero_cl_alpha_per_rad=None,
                                   vspaero_oswald_efficiency=None, quality_pass=False)
                    row["source_sample"] = source["sample"]
                    row["mesh"] = mesh
                    record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
            row["source_sample"] = source["sample"]
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "网格对照.csv", index=False)
            print(source["sample"], mesh, row["vspaero_cl_alpha_per_rad"],
                  row["vspaero_oswald_efficiency"], flush=True)
    (OUT / "诊断口径.json").write_text(json.dumps(dict(
        cases=selected["sample"].tolist(),
        squared_error_fraction=float(selected.squared_lift_error.sum()/frame.squared_lift_error.sum()),
        wake_iterations=30, meshes=["legacy", "medium"],
        label_policy="Diagnostic only; original training and held-out labels unchanged",
    ), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
