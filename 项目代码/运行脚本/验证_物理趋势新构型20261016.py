"""Seal another solver batch before testing physical-trend residual models."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO气动代理模型 import FEATURE_LABELS
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON
from 检查_尾迹标签质量 import polar_quality
from 训练_主动增量混合代理20261016 import portable_target_prediction, metric

OUT = ROOT / "运行结果" / "物理趋势残差代理_20261016"
MODELS = {
    "incrementalgp": ROOT / "运行结果" / "增量构型气动代理_20261015" / "增量构型气动代理.json",
    "activegp": ROOT / "运行结果" / "主动增量代理_20261015" / "增量构型气动代理.json",
    "physicaltrendgp": OUT / "物理趋势候选代理.json",
}
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def main():
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in MODELS.items()}
    lock_path = OUT / "全新测试构型与模型锁定.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf8"))
        if lock["models"] != hashes:
            raise RuntimeError("Frozen model changed")
        jobs = lock["jobs"]
    else:
        train = pd.read_csv(ROOT / "运行结果" / "主动增量代理_20261015" / "增量训练样本.csv")
        older = pd.read_csv(ROOT / "运行结果" / "尾迹分区趋势代理_20261015" / "新独立验证_逐点.csv")
        prior_plan = json.loads((ROOT / "运行结果" / "主动气动补样_20261015" /
                                "采样预案.json").read_text(encoding="utf8"))
        excluded = np.vstack((
            train[list(FEATURE_LABELS)].to_numpy(float),
            older[list(FEATURE_LABELS)].to_numpy(float),
            [[job["design"][k] for k in FEATURE_LABELS] for job in prior_plan["jobs"]]))
        lo, hi = train[list(FEATURE_LABELS)].min().to_numpy(), train[list(FEATURE_LABELS)].max().to_numpy()
        fixed = {k: prior_plan["jobs"][0]["design"][k] for k in
                 ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
        rng = np.random.default_rng(2026101621)
        jobs = []
        while len(jobs) < 16:
            values = rng.uniform(lo, hi)
            design = {**fixed, **dict(zip(FEATURE_LABELS, values))}
            if (np.linalg.norm((excluded-values)/(hi-lo), axis=1).min() > 1e-5
                    and not any(v > 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values())):
                jobs.append(dict(sample=f"physics_test_{len(jobs):03d}", design=design))
                excluded = np.vstack((excluded, values))
        lock_path.write_text(json.dumps(dict(
            models=hashes, jobs=jobs, role="Fresh batch after physical-trend model selection",
            settings="legacy mesh, wake iterations 30, five angles",
            previous_solver_batches_used_for_training=False,
        ), indent=2), encoding="utf8")
    rows = []
    for i, job in enumerate(jobs):
        folder = OUT / "fresh_solver" / job["sample"]
        record = folder / "record.json"
        if record.exists():
            row = json.loads(record.read_text(encoding="utf8"))
        else:
            if folder.exists():
                raise RuntimeError(f"Inspect interrupted solver case: {folder}")
            try:
                row = _run_case(DEFAULT_PYTHON, job["design"], job["sample"], folder, 900,
                                "sealed_test", "new_uniform_feasible_domain",
                                (0, float("inf")), 30, "legacy")
                row.update(polar_quality(folder / "vspaero_polar.csv"))
                row["wake_quality_pass"] = row["quality_pass"]
            except Exception as error:
                row = dict(sample=job["sample"], **job["design"], solver_failure=repr(error),
                           wake_quality_pass=False, vspaero_cl_alpha_per_rad=None,
                           vspaero_oswald_efficiency=None)
            record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "全新测试实算标签.csv", index=False)
        print(i+1, "/", len(jobs), row["wake_quality_pass"], flush=True)
    test = pd.DataFrame(rows)
    x = test[list(FEATURE_LABELS)].to_numpy(float)
    results = {}
    for name, path in MODELS.items():
        payload = json.loads(path.read_text(encoding="utf8"))
        results[name] = []
        for t, target in enumerate(TARGETS):
            y = test[target].to_numpy(float)
            mask = np.isfinite(y)
            if t:
                mask &= test.wake_quality_pass.to_numpy(bool)
            prediction = portable_target_prediction(payload["targets"][t], x)
            test[f"{name}_prediction_{t}"] = prediction
            results[name].append(dict(target=target, count=int(mask.sum()), **metric(y[mask], prediction[mask])))
    test.to_csv(OUT / "全新测试逐点.csv", index=False)
    (OUT / "全新测试结果.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
