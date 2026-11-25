"""Verify cheap topology extraction against archived full-solver meshes."""

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import 推进_千构型原定义代理20261017 as original
from VSPAERO受控子进程20261017 import run_process

ROOT = original.ROOT
OUT = ROOT / "运行结果/几何网格特征_20261017"
WORKER = Path(__file__).with_name("运行_OpenVSP几何预处理20261017.py")


def descriptor(path):
    lines = Path(path).read_text(encoding="utf8").splitlines()
    if not lines[0].startswith("# vspgeom") or int(lines[1]) != 1:
        raise ValueError("Unsupported geometry format")
    return [int(v) for v in lines[2].split()]


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / "运行结果/千构型分区主动补样_20261017/中网格训练实算标签.csv"
    frame = pd.read_csv(source)
    records = []
    for _, row in frame.iterrows():
        if not isinstance(row.geometry_file, str):
            continue
        mesh = Path(row.geometry_file).with_suffix(".vspgeom")
        if not mesh.exists():
            continue
        count = descriptor(mesh)
        records.append(dict(sample=row["sample"], **{k: row[k] for k in original.prior.FEATURE_LABELS},
                            node_count=count[0], loop_count=count[1], surface_count=count[2],
                            wake_quality_pass=bool(row.wake_quality_pass),
                            vspaero_oswald_efficiency=row.vspaero_oswald_efficiency))
    table = pd.DataFrame(records)
    table.to_csv(OUT / "训练网格描述.csv", index=False)
    source_rows = [
        ROOT / "运行结果/微小几何扰动_20261017/solver/local_03_021/record.json",
        ROOT / "运行结果/微小几何扰动_20261017/solver/local_03_027/record.json",
    ]
    checks = []
    for i, source in enumerate(source_rows):
        row = json.loads(source.read_text(encoding="utf8"))
        folder = OUT / f"preflight_{i}"
        folder.mkdir(exist_ok=True)
        result = folder / "result.json"
        if not result.exists():
            run_process([sys.executable, str(WORKER)], dict(design=row["design"], output=str(result)),
                        folder, 180)
        receipt = json.loads(result.read_text(encoding="utf8"))
        actual = descriptor(Path(row["geometry_file"]).with_suffix(".vspgeom"))
        assert [receipt[k] for k in ("node_count", "loop_count", "surface_count")] == actual
        checks.append(receipt)
    (OUT / "预处理一致性.json").write_text(json.dumps(dict(
        passed=True, comparisons=checks, training_geometries=len(table),
        node_count_unique=sorted(table.node_count.unique().tolist()),
        full_force_solver_called=False), indent=2), encoding="utf8")
    print(json.dumps(dict(training=len(table), checks=checks), ensure_ascii=False))


if __name__ == "__main__":
    main()
