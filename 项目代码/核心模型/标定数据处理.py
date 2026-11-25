"""Validate future measured component data and split complete runs without leakage."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = (
    "run_id",
    "timestamp_s",
    "subsystem",
    "signal",
    "value",
    "unit",
    "source_type",
    "condition",
    "quality_flag",
)
ALLOWED_SOURCE_TYPES = {"bench", "ground", "flight", "simulation_reference", "synthetic_test"}


def validate_calibration_csv(path: str | Path) -> list[str]:
    """Return data-quality issues; an empty list means schema checks passed."""
    issues: list[str] = []
    rows_by_channel: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            return [f"缺少字段: {', '.join(sorted(missing))}"]
        for line, row in enumerate(reader, start=2):
            if not row["run_id"].strip() or not row["subsystem"].strip() or not row["signal"].strip():
                issues.append(f"第 {line} 行的 run_id/subsystem/signal 不能为空")
                continue
            if row["source_type"] not in ALLOWED_SOURCE_TYPES:
                issues.append(f"第 {line} 行 source_type 不受支持: {row['source_type']}")
            if not row["unit"].strip() or not row["condition"].strip():
                issues.append(f"第 {line} 行缺少 unit 或 condition")
            try:
                timestamp = float(row["timestamp_s"])
                value = float(row["value"])
                if not (timestamp == timestamp and abs(timestamp) != float("inf")):
                    raise ValueError
                if not (value == value and abs(value) != float("inf")):
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(f"第 {line} 行 timestamp_s/value 必须是有限数值")
                continue
            rows_by_channel[(row["run_id"], row["subsystem"], row["signal"])].append((line, timestamp))

    for (run_id, subsystem, signal), samples in rows_by_channel.items():
        times = [timestamp for _, timestamp in samples]
        if any(right <= left for left, right in zip(times, times[1:])):
            issues.append(f"run={run_id}, {subsystem}.{signal} 时间戳必须严格递增")
    return issues


def split_run_ids(run_ids: Iterable[str], validation_fraction: float = 0.2, seed: int = 20261004) -> tuple[list[str], list[str]]:
    """Split by entire test run so samples from one run cannot leak across sets."""
    unique = sorted(set(run_ids))
    if len(unique) < 2:
        raise ValueError("至少需要两个独立 run_id 才能划分训练集和验证集。")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须在 (0, 1) 内。")
    shuffled = unique[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    validation = sorted(shuffled[:validation_count])
    training = sorted(shuffled[validation_count:])
    return training, validation


def write_template(path: str | Path) -> None:
    """Create an empty, explicitly unmeasured CSV template."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerow(REQUIRED_COLUMNS)
