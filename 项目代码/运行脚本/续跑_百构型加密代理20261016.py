"""Continue high-fidelity active sampling after the sealed first-stage run."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
import 运行_多精度气动代理20261016 as driver
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints

FIRST = ROOT / "运行结果" / "多精度气动代理_20261016"
OUT = ROOT / "运行结果" / "百构型加密代理_20261016"


def evaluate_grouped_candidate(first):
    from 研究_分组多尺度Kriging20261016 import predict
    folder = ROOT / "运行结果" / "加密分组Kriging24构型_20261016"
    path, freeze = folder / "分组多尺度候选.json", folder / "模型冻结.json"
    if not path.exists() or not freeze.exists():
        return
    lock = json.loads(freeze.read_text(encoding="utf8"))
    if driver.digest(path) != lock["model_sha256"]:
        raise RuntimeError("Grouped high-fidelity model changed after freezing")
    first_test = first / "solver" / "medium_test_000"
    frozen_at = time.mktime(time.strptime(lock["frozen_at"], "%Y-%m-%d %H:%M:%S"))
    if frozen_at > first_test.stat().st_ctime:
        raise RuntimeError("Grouped model was not frozen before new test calculation")
    frame = pd.read_csv(first / "中网格盲测逐点.csv")
    training = pd.read_csv(folder / "冻结训练数据.csv")
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    a = training[list(FEATURE_LABELS)].to_numpy(float)
    if np.linalg.norm(x[:, None]-a[None, :], axis=2).min() < 1e-7:
        raise RuntimeError("Grouped model test overlaps its training data")
    payload = json.loads(path.read_text(encoding="utf8"))
    predictions = np.column_stack([predict(m, x) for m in payload["targets"]])
    assessment = json.loads((first / "中网格盲测评估.json").read_text(encoding="utf8"))
    records = []
    for t, name in enumerate(driver.TARGETS):
        y = frame[name].to_numpy(float)
        valid = np.isfinite(y)
        if t:
            valid &= frame.wake_quality_pass.to_numpy(bool)
        if valid.sum() < 3 or not np.isfinite(predictions).all():
            raise RuntimeError("Invalid high-fidelity grouped model evaluation")
        records.append(dict(target=name, count=int(valid.sum()),
                            **driver.metric(y[valid], predictions[valid, t])))
        frame[f"grouped_high24_prediction_{t}"] = predictions[:, t]
    assessment["metrics"]["grouped_high24"] = records
    assessment.update(
        grouped_model_sha256=driver.digest(path), grouped_training_geometries=len(training),
        comparison_note="Same sealed medium-mesh test; grouped model uses 24 high-fidelity training geometries, original candidates use 40",
        grouped_goal_pass=[r["r2"] >= .9 for r in records],
        automatic_production_promotion=False,
    )
    frame.to_csv(folder / "加密网格独立预测.csv", index=False)
    (folder / "加密网格独立评价.json").write_text(json.dumps(assessment, indent=2), encoding="utf8")


def write_status(out, stage, **kwargs):
    payload = dict(pid=os.getpid(), stage=stage,
                   updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs)
    temp = out / "续跑状态.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(out / "续跑状态.json")


def plan_expansion(first, total_train=100, test_count=16, fine_count=4, quality_dir=None):
    previous = json.loads((first / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    low = pd.read_csv(driver.LOW_DATA)
    low_x = low[list(FEATURE_LABELS)].to_numpy(float)
    lo, hi = low_x.min(0), low_x.max(0)
    fixed = {k: previous["jobs"][0]["design"][k]
             for k in ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    jobs = []
    for item in (j for j in previous["jobs"] if j["role"] == "train"):
        job = dict(sample=item["sample"], role="train", mesh="medium",
                   design=item["design"], basis="reuse_first_stage_training")
        record = (quality_dir / "records" / item["sample"] / "record.json"
                  if quality_dir else first / "solver" / item["sample"] / "record.json")
        row = json.loads(record.read_text(encoding="utf8"))
        if row.get("quality_pass") is not None and not row.get("solver_failure"):
            job.update(reuse_record=str(record), reuse_sha256=driver.digest(record))
        else:
            job["basis"] = "retry_previously_failed_geometry"
        jobs.append(job)
    if len(jobs) >= total_train:
        raise ValueError("Expansion requires more training geometries")
    prior_test = np.array([[j["design"][k] for k in FEATURE_LABELS]
                           for j in previous["jobs"] if j["role"] == "test"])
    earlier_legacy_test = pd.read_csv(driver.LOW_DIR / "合并留出验证.csv")
    excluded = np.vstack((low_x, prior_test,
                          earlier_legacy_test[list(FEATURE_LABELS)].to_numpy(float)))
    selected = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in jobs])
    rng = np.random.default_rng(2026101644)
    pool = []
    while len(pool) < 5000:
        values = rng.uniform(lo, hi)
        design = {**fixed, **dict(zip(FEATURE_LABELS, values))}
        if (np.linalg.norm((excluded-values)/(hi-lo), axis=1).min() > 1e-5
                and all(v <= 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values())):
            pool.append(values)
    pool = np.array(pool)
    selection_dir = quality_dir or first
    model_path = selection_dir / "多精度候选代理.json"
    model = load_vsp_surrogate(model_path)
    reference = pd.read_csv(selection_dir / "中网格训练实算标签.csv")
    scales = np.array([max(float(reference[name].std()), 1e-8) for name in driver.TARGETS])
    uncertainty = np.array([
        model.predict_with_uncertainty(AirframeDesign(**{**fixed, **dict(zip(FEATURE_LABELS, row))}))[2:]
        for row in pool])
    scores = np.mean(uncertainty/scales, axis=1)
    scores = np.maximum(scores, 1e-12)
    available = np.ones(len(pool), dtype=bool)
    while len(jobs) < total_train:
        distance = np.linalg.norm((pool[:, None, :]-selected[None, :, :])/(hi-lo), axis=2).min(1)
        strategy = "uncertainty_diversity" if len(jobs) % 2 else "global_maximin"
        value = scores*np.sqrt(distance) if strategy == "uncertainty_diversity" else distance
        value[~available] = -1.
        index = int(np.argmax(value))
        available[index] = False
        row = pool[index]
        jobs.append(dict(sample=f"medium_train_{len(jobs):03d}", role="train", mesh="medium",
                         design={**fixed, **dict(zip(FEATURE_LABELS, row))}, basis=strategy))
        selected = np.vstack((selected, row))
    excluded = np.vstack((excluded, selected))
    for i in range(test_count):
        while True:
            values = rng.uniform(lo, hi)
            design = {**fixed, **dict(zip(FEATURE_LABELS, values))}
            if (np.linalg.norm((excluded-values)/(hi-lo), axis=1).min() > 1e-5
                    and all(v <= 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values())):
                break
        jobs.append(dict(sample=f"medium_test_{i:03d}", role="test", mesh="medium",
                         design=design, basis="new_sealed_uniform_feasible"))
        excluded = np.vstack((excluded, values))
    for i, index in enumerate(np.linspace(0, total_train-1, fine_count).astype(int)):
        item = jobs[index]
        job = dict(sample=f"fine_audit_{i:03d}", role="fine_audit", mesh="fine",
                   design=item["design"], medium_source=item["sample"],
                   basis="separate_grid_sensitivity_audit")
        if index == 0:
            old = first / "solver" / "fine_audit_000" / "record.json"
            if quality_dir:
                from 准备_一致性校核加密代理20261016 import derived_record
                derived = quality_dir / "records" / "fine_audit_000" / "record.json"
                derived_record(old, derived)
                old = derived
            row = json.loads(old.read_text(encoding="utf8"))
            if row.get("quality_pass") is not None and not row.get("solver_failure"):
                job.update(reuse_record=str(old), reuse_sha256=driver.digest(old))
        jobs.append(job)
    # Selection consumes only the first-stage training model and geometries.
    # The first-stage blind-test target values are excluded from this plan.
    manifest = driver.sources()
    manifest[str(Path(__file__).relative_to(ROOT))] = driver.digest(Path(__file__))
    if quality_dir:
        for name in ("准备_一致性校核加密代理20261016.py", "检查_极曲线力一致性20261016.py"):
            source = Path(__file__).parent / name
            manifest[str(source.relative_to(ROOT))] = driver.digest(source)
        for source in (model_path, quality_dir / "校核输入锁定.json",
                       *[Path(j["reuse_record"]) for j in jobs if "reuse_record" in j]):
            manifest[str(source.relative_to(ROOT))] = driver.digest(source)
    training = [j for j in jobs if j["role"] == "train"]
    testing = [j for j in jobs if j["role"] == "test"]
    a = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in training])
    b = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in testing])
    if np.min(np.linalg.norm((a[:, None, :]-b[None, :, :])/(hi-lo), axis=2)) < 1e-5:
        raise ValueError("New test overlaps training")
    return dict(jobs=jobs, train_count=total_train, test_count=test_count, fine_count=fine_count,
                source_manifest=manifest, wake_iterations=30, native_workers=1,
                low_data_sha256=driver.digest(driver.LOW_DATA),
                selection_model_sha256=driver.digest(model_path),
                first_stage_manifest_sha256=driver.digest(first / "冻结采样与计算配置.json"),
                old_test_labels_used_for_training=False, old_test_geometries_excluded=True,
                quality_preparation=str(quality_dir) if quality_dir else None,
                invalid_labels_retained_as_failures=True,
                protocol="Reuse training records; alternate uncertainty/diversity and global coverage; fresh sealed test")


def main(first, out, force=False):
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        while True:
            status = json.loads((first / "进度.json").read_text(encoding="utf8"))
            if status["stage"] == "failed":
                raise RuntimeError("First-stage calculation failed; inspect its logs before resuming")
            if status["stage"] == "completed":
                break
            write_status(out, "waiting_first_stage", first_stage=status)
            if time.monotonic()-started > 24*3600:
                raise TimeoutError("First-stage wait exceeded 24 hours")
            time.sleep(20)
        try:
            evaluate_grouped_candidate(first)
        except Exception:
            (out / "附加分组模型评价异常.txt").write_text(traceback.format_exc(), encoding="utf8")
            print("Optional grouped-model assessment failed; continuing primary expansion", flush=True)
        from 准备_一致性校核加密代理20261016 import prepare, install_quality_hooks
        quality_dir = out / "一致性校核训练"
        write_status(out, "preparing_quality_controlled_training")
        prepare(first, quality_dir)
        install_quality_hooks(quality_dir / "low_labels" / "逐构型质量记录.csv")
        # The first-stage score used the earlier QC protocol. Collect a fresh
        # sealed test for the revised protocol regardless of that old score.
        path = out / "冻结采样与计算配置.json"
        if not path.exists():
            write_status(out, "planning_expansion")
            plan = plan_expansion(first, quality_dir=quality_dir)
            path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
        write_status(out, "expansion_running", progress_file=str(out / "进度.json"))
        driver.run(out, (100, 16, 4))
        write_status(out, "completed", assessment=str(out / "中网格盲测评估.json"))
    except Exception:
        write_status(out, "failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-stage", type=Path, default=FIRST)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--force-expand", action="store_true")
    args = parser.parse_args()
    main(args.first_stage, args.output_dir, args.force_expand)
