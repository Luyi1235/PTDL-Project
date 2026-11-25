"""Paired factor/iteration probes; original DOE labels remain unchanged."""

import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
import 重算VSPAERO_Ew_DOE并训练代理 as solver
from 检查_尾迹标签质量 import polar_quality
from 检查_VSPAERO残差收敛20261016 import final_residuals

FIRST = ROOT / "运行结果" / "多精度气动代理_20261016"
OUT = ROOT / "运行结果" / "严格残差对照_20261016"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    solver.WORKER = ROOT / "运行脚本" / "运行_VSPAERO严格残差探针20261016.py"
    rows = []
    for index in (0, 26):
        job = plan["jobs"][index]
        name = job["sample"]
        folder = OUT / name
        record = folder / "record.json"
        if record.exists():
            rows.append(json.loads(record.read_text(encoding="utf8")))
            continue
        try:
            row = solver._run_case(solver.DEFAULT_PYTHON, job["design"], name, folder, 2400,
                                   "convergence_diagnostic", "factor_0.001_wake60",
                                   (0, float("inf")), 60, "medium")
            settings = (folder / "airframe.vspaero").read_text(encoding="utf8")
            factor = float(re.search(r"^ForwardGMRESConvergenceFactor\s*=\s*(\S+)",
                                     settings, flags=re.M).group(1))
            if abs(factor-.001) > 1e-12:
                raise RuntimeError("Requested GMRES factor was not applied")
            last = final_residuals(folder / "airframe.history")
            row.update(polar_quality(folder / "vspaero_polar.csv"),
                       forward_gmres_factor=factor, wake_cap=60,
                       max_final_log10_residual=float(last.MaxRes.max()),
                       converged_points=int((last.MaxRes <= -3).sum()), points=len(last))
            old = json.loads((FIRST / "solver" / name / "record.json").read_text(encoding="utf8"))
            row.update(old_lift=old["vspaero_cl_alpha_per_rad"],
                       old_wake=old["vspaero_oswald_efficiency"])
        except Exception as error:
            row = dict(sample=name, error=repr(error), accepted=False)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "严格求解对照.csv", index=False)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
