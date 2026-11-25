"""Prepare derived quality-controlled labels without changing sealed solver data."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import time

import pandas as pd

import 运行_多精度气动代理20261016 as driver
from 检查_极曲线力一致性20261016 import audit, checked_row, polar_quality, CRITERION

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LOW = ROOT / "运行结果" / "主动增量代理_20261015" / "增量训练样本.csv"
NATIVE_RUN_CASE = driver._run_case


def install_quality_hooks(data_path):
    driver.LOW_DATA = Path(data_path)

    def checked_native(*args, **kwargs):
        return checked_row(NATIVE_RUN_CASE(*args, **kwargs))

    driver._run_case = checked_native
    driver.polar_quality = polar_quality


def derived_record(source, destination):
    original = json.loads(source.read_text(encoding="utf8"))
    row = checked_row(original)
    row.update(raw_record_source=str(source), raw_record_sha256=driver.digest(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(row, ensure_ascii=False, indent=2)
    if destination.exists() and destination.read_text(encoding="utf8") != text:
        raise RuntimeError("Derived training record changed after preparation")
    destination.write_text(text, encoding="utf8")
    return row


def prepare(first, out):
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((first / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    jobs = [j for j in plan["jobs"] if j["role"] == "train"]
    original_records = [first / "solver" / j["sample"] / "record.json" for j in jobs]
    source_paths = [ORIGINAL_LOW, Path(__file__),
                    Path(__file__).parent / "检查_极曲线力一致性20261016.py", *original_records]
    fingerprints = {str(p): driver.digest(p) for p in source_paths}
    lock = out / "校核输入锁定.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf8")) != fingerprints:
        raise RuntimeError("Quality preparation sources changed; use a new output directory")
    lock.write_text(json.dumps(fingerprints, ensure_ascii=False, indent=2), encoding="utf8")
    _, low_summary = audit(ORIGINAL_LOW, out / "low_labels")
    rows = [derived_record(source, out / "records" / j["sample"] / "record.json")
            for j, source in zip(jobs, original_records)]
    high = pd.DataFrame(rows)
    high.to_csv(out / "中网格训练实算标签.csv", index=False)
    install_quality_hooks(out / "low_labels" / "逐构型质量记录.csv")
    if not (out / "训练模型选择.json").exists():
        driver.fit_models(high, out)
    model_paths = [out / name for name in ("多精度候选代理.json", "仅高精度候选代理.json")]
    model_lock = {str(p): driver.digest(p) for p in model_paths}
    model_lock_path = out / "校核模型锁定.json"
    if model_lock_path.exists() and json.loads(model_lock_path.read_text(encoding="utf8")) != model_lock:
        raise RuntimeError("Prepared model changed")
    model_lock_path.write_text(json.dumps(model_lock, ensure_ascii=False, indent=2), encoding="utf8")
    result = dict(completed_at=time.strftime("%Y-%m-%d %H:%M:%S"), training_count=len(high),
                  valid_lift=int(high.lift_quality_pass.sum()),
                  valid_wake=int(high.wake_quality_pass.sum()),
                  low_quality=low_summary, quality_criterion=CRITERION,
                  first_stage_test_labels_used=False, automatic_production_promotion=False)
    (out / "准备完成.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-stage", type=Path, default=ROOT / "运行结果" / "多精度气动代理_20261016")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "运行结果" / "百构型加密代理_20261016" / "一致性校核训练")
    args = parser.parse_args()
    prepare(args.first_stage, args.output_dir)
