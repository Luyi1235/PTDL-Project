"""Training-only paired tests of stricter convergence and implicit free wake."""

import hashlib
import json
from pathlib import Path
import time

import pandas as pd

import 推进_千构型原定义代理20261017 as original

ROOT = original.ROOT
OUT = ROOT / "运行结果/局部收敛与隐式尾迹_20261017"


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / "运行结果/微小几何扰动_20261017/冻结配置.json"
    tasks = json.loads(source.read_text(encoding="utf8"))["tasks"]
    pair = [t for t in tasks if t["sample"] in ("local_03_021", "local_03_027")]
    worker = Path(__file__).with_name("运行_VSPAERO收敛诊断工作进程20261017.py")
    configs = [dict(name="strict_legacy", mesh="legacy", implicit=0),
               dict(name="implicit_legacy", mesh="legacy", implicit=1),
               dict(name="strict_medium", mesh="medium", implicit=0)]
    (OUT / "冻结配置.json").write_text(json.dumps(dict(
        tasks=pair, configs=configs, training_only=True, label_definition_unchanged=True,
        production_labels_replaced=False, worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest()),
        indent=2), encoding="utf8")
    rows = []
    started = time.monotonic()
    for config in configs:
        for task in pair:
            folder = OUT / "solver" / f"{config['name']}_{task['sample']}"
            cache = folder / "record.json"
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf8"))
            else:
                folder.mkdir(parents=True)
                case = folder / "case"
                case.mkdir()
                request = dict(
                    design=task["design"], output_dir=str(case), protocol=config["name"],
                    mesh=config["mesh"],
                    integer_settings=dict(FixedWakeFlag=0, NumWakeNodes=8, NCPU=8,
                                          FarDistToggle=0, WakeNumIter=100,
                                          ImplicitWake=config["implicit"], ImplicitWakeStartIteration=0),
                    double_settings=dict(WakeRelax=.5, ForwardGMRESConvergenceFactor=.01,
                                         NonLinearConvergenceFactor=.01))
                try:
                    log = original.controlled.run_process(
                        [str(original.prior.solver.DEFAULT_PYTHON), str(worker)], request, case, 240)
                    result = original.controlled.original._extract_worker_result(log)
                    row = original.controlled.original._record_from_result(
                        task["sample"], task["design"], result, case, log,
                        "diagnostic", "training_only_convergence", (0., float("inf")))
                    row.update(original.prior.diagnostic_quality(case / "vspaero_polar.csv"))
                    row.update(applied=result["applied"])
                except Exception as exc:
                    row = dict(solver_failure=repr(exc))
                row.update(protocol=config["name"], sample=task["sample"], request=request)
                cache.write_text(json.dumps(row, indent=2), encoding="utf8")
            rows.append(row)
            pd.DataFrame(rows).drop(columns=["request"]).to_csv(OUT / "求解一致性.csv", index=False)
            (OUT / "进度.json").write_text(json.dumps(dict(
                completed=len(rows), requested=6, elapsed_s=time.monotonic()-started,
                stage="completed" if len(rows) == 6 else "solver")), encoding="utf8")
            print("Convergence diagnostic", len(rows), "/ 6", flush=True)


if __name__ == "__main__":
    main()
