"""Retain scientific settings and reduce the process budget for memory headroom."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
import traceback

import 研究_稳定网格尾迹代理20261019 as study


def bounded_pool(**kwargs):
    kwargs["max_workers"] = 2
    return ProcessPoolExecutor(**kwargs)


def main():
    freeze_path = study.OUT / "配置冻结.json"
    freeze = study.study.batch.previous.read(freeze_path)
    study.utility.check_hashes(freeze["hashes"])
    study.utility.save(study.OUT / "内存受限续跑执行.json", dict(
        workers=2, fit_threads=2, scientific_configuration_unchanged=True,
        existing_folds_retained=True,
        hashes={str(p): study.utility.digest(p) for p in
                (Path(__file__), Path(study.__file__), freeze_path)},
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))
    study.ProcessPoolExecutor = bounded_pool
    study.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        study.utility.save(study.OUT / "失败.json", dict(error=traceback.format_exc()))
        raise
