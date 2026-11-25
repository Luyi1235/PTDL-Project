"""Acquire training labels around OOF errors and sparse regions; seal new tests."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON
from 检查_尾迹标签质量 import polar_quality
from 训练_主动增量混合代理20261016 import portable_target_prediction

OUT = ROOT / "运行结果" / "主动气动补样_20261015"
INCUMBENT = ROOT / "运行结果" / "增量构型气动代理_20261015" / "增量构型气动代理.json"
CANDIDATE = ROOT / "运行结果" / "主动增量代理_20261015" / "增量构型气动代理.json"


def create_plan():
    frame = pd.read_csv(INCUMBENT.parent / "增量训练样本.csv")
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    previous = pd.read_csv(ROOT / "运行结果" / "尾迹分区趋势代理_20261015" / "新独立验证_逐点.csv")
    excluded = np.vstack((x, previous[list(FEATURE_LABELS)].to_numpy(float)))
    original = pd.read_csv(ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv")
    fixed = {k: float(original.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    rng = np.random.default_rng(2026101541)

    def design(values):
        return {**fixed, **dict(zip(FEATURE_LABELS, values))}

    def valid(values):
        return not any(v > 1e-9 for v in geometry_constraints(AirframeDesign(**design(values))).values())

    pool = []
    while len(pool) < 2048:
        values = rng.uniform(lo, hi)
        if valid(values):
            pool.append(values)
    pool = np.array(pool)
    jobs = []
    # OOF errors are on development geometries only; the current 16 tests remain sealed.
    errors = pd.read_csv(ROOT / "运行结果" / "迎角分解非线性代理_20261015" /
                         "折外_1_physics_boosting_7_0.0.csv")
    wake_frame = frame.loc[frame.wake_quality_pass]
    order = np.argsort(abs(errors.prediction-errors.actual).to_numpy())[::-1]
    for idx in order[:12]:
        anchor = wake_frame.iloc[idx][list(FEATURE_LABELS)].to_numpy(float)
        for _ in range(500):
            values = np.clip(anchor+rng.normal(0, .035, 7)*(hi-lo), lo, hi)
            if valid(values) and np.linalg.norm((excluded-values)/(hi-lo), axis=1).min() > 1e-4:
                jobs.append(dict(sample=f"active_train_{len(jobs):03d}", role="train",
                                 basis="training_oof_error_neighborhood", design=design(values)))
                excluded = np.vstack((excluded, values))
                break
    model = load_vsp_surrogate(INCUMBENT)
    uncertainty = np.array([model.predict_with_uncertainty(AirframeDesign(**design(v)))[2:]
                            for v in pool])
    uncertainty = uncertainty / np.maximum(uncertainty.mean(0), 1e-9)
    for index in range(20):
        distances = np.linalg.norm((pool[:, None, :]-excluded[None, :, :])/(hi-lo), axis=2).min(1)
        score = distances if index >= 8 else distances*(1+uncertainty.mean(1))
        chosen = int(np.argmax(score))
        values = pool[chosen]
        jobs.append(dict(sample=f"active_train_{len(jobs):03d}", role="train",
                         basis="space_fill" if index >= 8 else "uncertainty_distance",
                         design=design(values)))
        excluded = np.vstack((excluded, values))
    while sum(j["role"] == "test" for j in jobs) < 16:
        values = rng.uniform(lo, hi)
        if valid(values) and np.linalg.norm((excluded-values)/(hi-lo), axis=1).min() > 1e-4:
            jobs.append(dict(sample=f"active_test_{sum(j['role']=='test' for j in jobs):03d}",
                             role="test", basis="new_uniform_feasible_domain", design=design(values)))
            excluded = np.vstack((excluded, values))
    return dict(jobs=jobs, incumbent_sha256=hashlib.sha256(INCUMBENT.read_bytes()).hexdigest(),
                existing_fresh16_used_for_training=False,
                test_protocol="Test labels collected only after candidate and incumbent hashes are locked")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("train", "test"), default="train")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "采样预案.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(create_plan(), indent=2), encoding="utf8")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if hashlib.sha256(INCUMBENT.read_bytes()).hexdigest() != plan["incumbent_sha256"]:
        raise RuntimeError("Incumbent changed")
    if args.role == "test":
        lock = OUT / "独立测试模型锁定.json"
        hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for name, p in (("incumbent", INCUMBENT), ("candidate", CANDIDATE))}
        if lock.exists() and json.loads(lock.read_text(encoding="utf8")) != hashes:
            raise RuntimeError("Frozen candidate changed")
        lock.write_text(json.dumps(hashes, indent=2), encoding="utf8")
    rows = []
    jobs = [j for j in plan["jobs"] if j["role"] == args.role]
    for job in jobs:
        folder = OUT / "solver" / job["sample"]
        record = folder / "record.json"
        if record.exists():
            row = json.loads(record.read_text(encoding="utf8"))
        else:
            if folder.exists():
                error_path = folder / "case_error.txt"
                if not error_path.exists():
                    raise RuntimeError(f"Inspect incomplete case before resume: {folder}")
                failure = error_path.read_text(encoding="utf8")
                if "PGFace/face errors" not in failure:
                    raise RuntimeError(f"Inspect interrupted case: {failure}")
                row = dict(sample=job["sample"], **job["design"],
                           solver_failure=failure, wake_quality_pass=False,
                           vspaero_cl_alpha_per_rad=None, vspaero_oswald_efficiency=None)
            else:
                try:
                    row = _run_case(DEFAULT_PYTHON, job["design"], job["sample"], folder, 900,
                                    job["role"], job["basis"], (0, float("inf")), 30, "legacy")
                    row.update(polar_quality(folder / "vspaero_polar.csv"))
                    row["wake_quality_pass"] = row["quality_pass"]
                except Exception as error:
                    row = dict(sample=job["sample"], **job["design"],
                               solver_failure=repr(error), wake_quality_pass=False,
                               vspaero_cl_alpha_per_rad=None, vspaero_oswald_efficiency=None)
            record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        rows.append(row)
        attempted = pd.DataFrame(rows)
        attempted.to_csv(OUT / f"{args.role}_全部求解记录.csv", index=False)
        if args.role == "train":
            attempted = attempted[np.isfinite(attempted.vspaero_cl_alpha_per_rad.to_numpy(float))]
        attempted.to_csv(OUT / f"{args.role}_实算标签.csv", index=False)
        print(args.role, len(rows), "/", len(jobs), row["wake_quality_pass"], flush=True)
    if args.role == "test":
        results = {}
        for name, path in (("incumbent", INCUMBENT), ("candidate", CANDIDATE)):
            payload = json.loads(path.read_text(encoding="utf8"))
            frame = pd.DataFrame(rows)
            features = np.array([[j["design"][key] for key in FEATURE_LABELS] for j in jobs])
            predictions = np.column_stack([
                portable_target_prediction(target, features) for target in payload["targets"]])
            results[name] = []
            for t, target in enumerate(("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")):
                y = frame[target].to_numpy(float)
                mask = np.isfinite(y)
                if t:
                    mask &= frame.wake_quality_pass.to_numpy(bool)
                errors = predictions[mask, t]-y[mask]
                results[name].append(dict(
                    target=target, count=int(mask.sum()), mae=float(np.mean(abs(errors))),
                    rmse=float(np.sqrt(np.mean(errors**2))),
                    r2=float(1-np.sum(errors**2)/np.sum((y[mask]-y[mask].mean())**2))))
                frame[f"{name}_prediction_{t}"] = predictions[:, t]
            if name == "incumbent":
                comparison = frame
            else:
                for t in range(2):
                    comparison[f"{name}_prediction_{t}"] = predictions[:, t]
        comparison.to_csv(OUT / "独立测试逐点.csv", index=False)
        (OUT / "独立测试结果.json").write_text(json.dumps(results, indent=2), encoding="utf8")
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
