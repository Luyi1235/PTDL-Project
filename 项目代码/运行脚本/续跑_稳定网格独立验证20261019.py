"""Resume input-only preprocessing in bounded batches without changing the holdout."""

from pathlib import Path
import subprocess
import sys
import time
import traceback

import 评价_稳定网格尾迹代理20261019 as evaluation


BATCH_SIZE = 8
BATCH_TIMEOUT = 1800
ATTEMPTS = 3


def check_force_labels_absent(jobs):
    root = evaluation.expansion.OUT / "solver"
    if any((root / job["sample"]).exists() for job in jobs):
        raise RuntimeError("Holdout force solving started before prediction registration")


def prepare_batched(out, manifest):
    jobs = manifest["jobs"]
    native = evaluation.expansion.native
    session = time.strftime("%Y%m%d_%H%M%S")
    for start in range(0, len(jobs), BATCH_SIZE):
        check_force_labels_absent(jobs)
        selected = jobs[start:start+BATCH_SIZE]
        for attempt in range(1, ATTEMPTS+1):
            folder = out / "geometry_worker_batches" / session / f"batch_{start:04d}_{attempt}"
            folder.mkdir(parents=True, exist_ok=True)
            request = dict(jobs=selected, output_root=str(out / "geometry"))
            evaluation.utility.save(out / "分批预处理进度.json", dict(
                stage="geometry_preprocessing", completed=start, requested=len(jobs),
                batch_start=start, batch_size=len(selected), attempt=attempt,
                force_solver_called=False, timeout_seconds=BATCH_TIMEOUT,
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            try:
                native.original.controlled.run_process(
                    [sys.executable, "-X", "utf8", str(native.WORKER)],
                    request, folder, BATCH_TIMEOUT)
                break
            except subprocess.TimeoutExpired:
                if attempt == ATTEMPTS:
                    raise
                check_force_labels_absent(jobs)
        evaluation.utility.save(out / "分批预处理进度.json", dict(
            stage="geometry_preprocessing", completed=start+len(selected),
            requested=len(jobs), force_solver_called=False,
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))


def main():
    out = evaluation.OUT
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    old = out / "进度.json"
    if old.exists():
        history = out / "历史执行"
        history.mkdir(exist_ok=True)
        evaluation.utility.save(history / f"恢复前进度_{stamp}.json",
                                evaluation.batch.previous.read(old))
    paths = [Path(__file__), Path(evaluation.__file__),
             evaluation.expansion.native.WORKER,
             evaluation.study.OUT / "模型选择.json",
             evaluation.expansion.OUT / "配置冻结.json"]
    evaluation.utility.save(out / "分批续跑执行.json", dict(
        hashes={str(p): evaluation.utility.digest(p) for p in paths},
        batch_size=BATCH_SIZE, batch_timeout_seconds=BATCH_TIMEOUT,
        cached_geometries_preserved=True, scientific_settings_unchanged=True,
        training_and_test_samples_unchanged=True,
        reason="Whole-cohort geometry preprocessing exceeded its process timeout",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
    evaluation.expansion.native.prepare = prepare_batched
    evaluation.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        evaluation.progress("failed", error=traceback.format_exc())
        raise
