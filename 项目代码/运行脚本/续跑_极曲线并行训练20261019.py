"""Resume unchanged fold tasks with a separately recorded process budget."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import traceback

from sklearn.model_selection import GroupKFold, KFold

import 研究_极曲线结构代理20261019 as study


def main():
    out = study.OUT
    data = study.prepare(out)
    original = study.previous.batch.previous.read(out / "配置冻结.json")
    study.previous.utility.check_hashes(original["hashes"])
    if original["threads"] != 4:
        raise RuntimeError("Original task thread budget differs")
    execution_path = out / "并行续跑执行.json"
    execution = dict(workers=4, threads=4, original_execution=str(out / "配置冻结.json"),
        fit_configuration_unchanged=True, fold_assignment_unchanged=True,
        hashes={str(p): study.previous.utility.digest(p) for p in
                (Path(__file__), Path(study.__file__), out / "配置冻结.json")},
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    study.previous.utility.save(execution_path, execution)
    groups = study.previous.helpers.direct.base.geometry_groups(data["x"])
    path = out / "交叉验证逐折.json"
    records = study.previous.batch.previous.read(path) if path.exists() else []
    done = {(r["config"], r["split"], r["fold"]) for r in records if r["mode"] == "half"}
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101803).split(data["x"])),
        ("spatial", GroupKFold(3).split(data["x"], groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, data, 4)
                        for i in range(len(study.CONFIGS)) if (i, split, fold) not in done)
    errors = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(study.task, job): job[:3] for job in jobs}
        for future in as_completed(futures):
            try:
                records.extend(future.result())
                study.previous.utility.save(path, records)
            except Exception:
                errors.append(dict(job=futures[future], traceback=traceback.format_exc()))
                study.previous.utility.save(out / "续跑失败折.json", errors)
            study.previous.utility.save(out / "进度.json", dict(
                stage="training_cv", completed=len(records)//3, requested=48,
                failed=len(errors), workers=4, threads=4,
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            print("Profile CV", len(records)//3, "/ 48; failed", len(errors), flush=True)
    if errors:
        raise RuntimeError("Resumed fold failures recorded")
    study.previous.utility.check_hashes(execution["hashes"])
    study.main(out, original["workers"], original["threads"], prepare_only=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        study.previous.utility.save(study.OUT / "失败.json", dict(error=traceback.format_exc()))
        raise
