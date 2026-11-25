"""Paired numerical wake-core sensitivity checks on an existing training geometry."""

import hashlib
import json
from pathlib import Path
import time

import pandas as pd

import 推进_千构型原定义代理20261017 as original

ROOT = original.ROOT
OUT = ROOT / "运行结果/尾迹涡核一致性_20261017"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / "运行结果/微小几何扰动_20261017/冻结配置.json"
    pair = [t for t in json.loads(source.read_text(encoding="utf8"))["tasks"]
            if t["sample"] in ("local_03_021", "local_03_027")]
    if len(pair) != 2:
        raise RuntimeError("Expected the original training-only perturbation pair")
    worker = Path(__file__).with_name("运行_VSPAERO收敛诊断工作进程20261017.py")
    factors = [1., 2., 4.]
    hashes = {str(p): digest(p) for p in (Path(__file__), source, worker)}
    frozen = OUT / "冻结配置.json"
    manifest = dict(pair=pair, core_size_factors=factors, hashes=hashes,
                    unchanged_definition=True, replace_production_labels=False,
                    training_geometries_only=True, mesh="legacy",
                    wake_nodes=8, wake_iterations=30, cores=8,
                    setting_verification="SetDoubleAnalysisInput and GetDoubleAnalysisInput",
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf8"))
        if old["hashes"] != hashes:
            raise RuntimeError("Wake-core diagnostic sources changed")
    else:
        frozen.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    rows, started = [], time.monotonic()
    for factor in factors:
        for task in pair:
            folder = OUT / "solver" / f"core{int(factor)}_{task['sample']}"
            folder.mkdir(parents=True, exist_ok=True)
            cache = folder / "record.json"
            case = folder / "case"
            case.mkdir(exist_ok=True)
            request = dict(
                design=task["design"], output_dir=str(case),
                protocol=f"wake_core_{factor:g}_diagnostic", mesh="legacy",
                integer_settings=dict(FixedWakeFlag=0, NumWakeNodes=8, NCPU=8,
                                      FarDistToggle=0, WakeNumIter=30),
                double_settings=dict(CoreSizeFactor=factor))
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf8"))
                if row["request"] != request:
                    raise RuntimeError("Cached core-size diagnostic request changed")
            else:
                try:
                    log = original.controlled.run_process(
                        [str(original.prior.solver.DEFAULT_PYTHON), str(worker)], request, case, 300)
                    result = original.controlled.original._extract_worker_result(log)
                    row = original.controlled.original._record_from_result(
                        task["sample"], task["design"], result, case, log,
                        "diagnostic", "training_only_core_size", (0., float("inf")))
                    row.update(original.prior.diagnostic_quality(case / "vspaero_polar.csv"))
                    row["applied"] = result["applied"]
                except Exception as exc:
                    row = dict(solver_failure=repr(exc))
                row.update(core_size_factor=factor, sample=task["sample"], request=request)
                cache.write_text(json.dumps(row, indent=2), encoding="utf8")
            rows.append(row)
            pd.DataFrame(rows).drop(columns=["request"]).to_csv(OUT / "逐构型结果.csv", index=False)
            (OUT / "进度.json").write_text(json.dumps(dict(
                stage="completed" if len(rows) == 6 else "solver", completed=len(rows), requested=6,
                elapsed_s=time.monotonic()-started)), encoding="utf8")
            print("Wake core consistency", len(rows), "/ 6", flush=True)
    summary = []
    for factor in factors:
        values = [row for row in rows if row["core_size_factor"] == factor]
        valid = len(values) == 2 and all(
            "solver_failure" not in row and row.get("quality_pass", False) for row in values)
        summary.append(dict(core_size_factor=factor, both_quality_pass=valid,
                            wake_values=[v.get("vspaero_oswald_efficiency") for v in values],
                            absolute_pair_difference=(abs(values[0]["vspaero_oswald_efficiency"]
                                                          - values[1]["vspaero_oswald_efficiency"])
                                                      if valid else None)))
    (OUT / "配对一致性.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
