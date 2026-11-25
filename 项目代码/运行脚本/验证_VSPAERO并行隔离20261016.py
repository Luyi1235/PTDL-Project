"""Check process startup isolation and repeated-geometry determinism."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON

SOURCE = ROOT / "运行结果" / "主动气动补样_20261015"
OUT = ROOT / "运行结果" / "VSPAERO并行隔离验证_20261016"


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((SOURCE / "采样预案.json").read_text(encoding="utf8"))
    designs = {j["sample"]: j["design"] for j in plan["jobs"]}
    truth = pd.read_csv(SOURCE / "独立测试逐点.csv").set_index("sample")
    samples = ["active_test_004", "active_test_011"]

    def run(sample):
        folder = OUT / sample
        record = folder / "record.json"
        if record.exists():
            return json.loads(record.read_text(encoding="utf8"))
        if folder.exists():
            raise RuntimeError(f"Inspect interrupted process: {folder}")
        row = _run_case(DEFAULT_PYTHON, designs[sample], sample, folder, 900,
                        "parallel_diagnostic", "isolated_startup_and_solver_dirs",
                        (0, float("inf")), 30, "legacy")
        for key in ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"):
            row["repeat_error_"+key] = abs(row[key]-float(truth.loc[sample, key]))
        record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        return row

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run, samples))
    elapsed = time.perf_counter()-started
    passed = all(row[key] < 1e-8 for row in rows for key in row if key.startswith("repeat_error_"))
    result = dict(workers=2, passed=passed, wall_time_s=elapsed,
                  summed_solver_time_s=sum(row["solver_wall_time_s"] for row in rows),
                  isolation="Unique directories and shared inter-process native-solver mutex; BLAS/OpenMP threads 1",
                  rows=rows)
    (OUT / "隔离验证.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not passed:
        raise RuntimeError("Repeated geometry changed under concurrent execution")


if __name__ == "__main__":
    main()
