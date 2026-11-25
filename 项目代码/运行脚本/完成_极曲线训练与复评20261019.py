"""Recover completed fits after a tuple/list JSON round-trip mismatch."""

import argparse
from pathlib import Path
import time
import traceback

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

import 研究_极曲线结构代理20261019 as study
import 验证_极曲线结构代理20261019 as evaluation


def normalized_modes(current, frozen):
    if list(current) != frozen:
        raise RuntimeError("Prediction modes changed")
    return list(current)


def validate_records(data, records):
    groups = study.previous.helpers.direct.base.geometry_groups(data["x"])
    expected = {}
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101803).split(data["x"])),
        ("spatial", GroupKFold(3).split(data["x"], groups=groups)),
    ):
        for fold, (_, b) in enumerate(folds):
            for index in range(len(study.CONFIGS)):
                for mode in study.MODES:
                    expected[index, mode, split, fold] = b
    seen = set()
    for row in records:
        key = row["config"], row["mode"], row["split"], row["fold"]
        if key not in expected or key in seen:
            raise RuntimeError("Duplicate or unexpected fold prediction")
        b = expected[key]
        np.testing.assert_array_equal(b, row["validation"])
        np.testing.assert_allclose(data["y"][b], row["actual"], atol=0, rtol=0)
        prediction = np.asarray(row["prediction"], float)
        if prediction.shape != b.shape or not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete fold predictions")
        seen.add(key)
    if seen != set(expected):
        raise RuntimeError("Completed prediction groups are missing")


def main(training_only):
    out = study.OUT
    frozen = study.previous.batch.previous.read(out / "配置冻结.json")
    study.previous.utility.check_hashes(frozen["hashes"])
    parallel_path = out / "并行续跑执行.json"
    parallel = study.previous.batch.previous.read(parallel_path)
    study.previous.utility.check_hashes(parallel["hashes"])
    data = study.prepare(out)
    records = study.previous.batch.previous.read(out / "交叉验证逐折.json")
    validate_records(data, records)
    study.MODES = normalized_modes(study.MODES, frozen["modes"])
    recovery = dict(reason="JSON tuple/list round-trip normalization",
        original_fit_results_retained=True, fit_parameters_unchanged=True,
        validated_fit_groups=len(records)//len(study.MODES),
        hashes={str(p): study.previous.utility.digest(p) for p in
                (Path(__file__), out / "配置冻结.json", out / "交叉验证逐折.json",
                 parallel_path)}, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    study.previous.utility.save(out / "汇总恢复记录.json", recovery)
    failure = out / "失败.json"
    if failure.exists():
        history = out / "历史故障"
        history.mkdir(exist_ok=True)
        archived = history / ("配置比较故障_"+time.strftime("%Y%m%d_%H%M%S")+".json")
        failure.replace(archived)
    study.previous.utility.save(out / "进度.json", dict(stage="finalizing",
        completed=48, requested=48, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    if not (out / "模型选择.json").exists():
        study.main(out, frozen["workers"], frozen["threads"], prepare_only=False)
    if not training_only:
        evaluation.main(evaluation.OUT, prepare_only=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    try:
        main(args.training_only)
    except Exception:
        study.previous.utility.save(study.OUT / "恢复失败.json",
                                   dict(error=traceback.format_exc()))
        raise
