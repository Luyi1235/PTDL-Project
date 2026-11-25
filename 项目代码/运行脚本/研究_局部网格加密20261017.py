"""Diagnose a training-only local discontinuity across two denser meshes."""

import hashlib
import json
from pathlib import Path
import time

import pandas as pd

import 推进_千构型原定义代理20261017 as original

ROOT = original.ROOT
OUT = ROOT / "运行结果/局部网格加密_20261017"


def main():
    OUT.mkdir(exist_ok=True)
    tasks = json.loads((ROOT / "运行结果/微小几何扰动_20261017/冻结配置.json").read_text(
        encoding="utf8"))["tasks"]
    pair = [t for t in tasks if t["sample"] in ("local_03_021", "local_03_027")]
    if len(pair) != 2 or pair[0]["anchor_sample"] != pair[1]["anchor_sample"]:
        raise ValueError("Training-only diagnostic pair missing")
    worker = Path(__file__).with_name("运行_VSPAERO加密网格诊断20261017.py")
    (OUT / "冻结配置.json").write_text(json.dumps(dict(
        tasks=pair, meshes=["medium", "fine"], training_only=True,
        target_definition="Original lift slope and median Ew",
        wake_nodes=8, wake_iterations=30, cores=8,
        worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest()), indent=2), encoding="utf8")
    original.controlled.original.WORKER = worker
    rows = []
    started = time.monotonic()
    for mesh in ("medium", "fine"):
        for task in pair:
            folder = OUT / "solver" / f"{mesh}_{task['sample']}"
            cache = folder / "record.json"
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf8"))
            else:
                folder.mkdir(parents=True)
                try:
                    row = original.controlled.run_case(
                        original.prior.solver.DEFAULT_PYTHON, task["design"], task["sample"],
                        folder / "case", 240, "diagnostic", "training_local_grid",
                        (0., float("inf")), 30, mesh)
                    row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
                    lines = (folder / "case/airframe.vspgeom").read_text().splitlines()
                    row.update(zip(("nodes", "loops", "surfaces"), map(int, lines[2].split())))
                except Exception as exc:
                    row = dict(solver_failure=repr(exc))
                row.update(mesh=mesh, sample=task["sample"], changed=task["changed"])
                cache.write_text(json.dumps(row, indent=2), encoding="utf8")
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "局部网格加密.csv", index=False)
            (OUT / "进度.json").write_text(json.dumps(dict(
                completed=len(rows), requested=4, elapsed_s=time.monotonic()-started,
                stage="completed" if len(rows) == 4 else "solver")), encoding="utf8")
            print("Grid diagnostic", len(rows), "/ 4", flush=True)


if __name__ == "__main__":
    main()
