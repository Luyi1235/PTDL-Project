"""Seal a fresh solver batch and compare the frozen trend GP with its incumbent."""

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

OUT = ROOT / "运行结果" / "尾迹分区趋势代理_20261015"
MODELS = {
    "physics211gp": ROOT / "运行结果" / "物理特征代理_20261015" / "物理特征气动代理.json",
    "trend211gp": OUT / "尾迹分区趋势候选代理.json",
    "incrementalgp": ROOT / "运行结果" / "增量构型气动代理_20261015" / "增量构型气动代理.json",
}


def main():
    lock_path = OUT / "新验证构型与模型锁定.json"
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in MODELS.items()}
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf8"))
        if lock["models"] != hashes:
            raise RuntimeError("Frozen model changed")
        jobs = lock["jobs"]
    else:
        train = pd.read_csv(ROOT / "运行结果" / "分目标代理211点_20261013" / "训练样本211点.csv")
        old_test = pd.read_csv(ROOT / "运行结果" / "物理特征代理_20261015" / "新独立验证_逐点.csv")
        x = train[list(FEATURE_LABELS)].to_numpy(float)
        development = pd.read_csv(ROOT / "运行结果" / "增量构型气动代理_20261015" / "增量训练样本.csv")
        excluded = np.vstack((x, old_test[list(FEATURE_LABELS)].to_numpy(float),
                              development[list(FEATURE_LABELS)].to_numpy(float)))
        lo, hi = x.min(0), x.max(0)
        rng = np.random.default_rng(2026101505)
        fixed = {k: float(train.iloc[0][k]) for k in
                 ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
        jobs = []
        while len(jobs) < 16:
            value = rng.uniform(lo, hi)
            if np.linalg.norm((excluded - value) / (hi - lo), axis=1).min() < 1e-5:
                continue
            design = {**fixed, **dict(zip(FEATURE_LABELS, value))}
            if any(v > 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values()):
                continue
            jobs.append(dict(sample=f"trend_fresh_{len(jobs):03d}", design=design))
        lock_path.write_text(json.dumps(dict(
            models=hashes, jobs=jobs, protocol="Unchanged legacy mesh, 30 wake iterations, five angles",
            role="New geometries after method development; no training or model selection on this batch",
        ), indent=2), encoding="utf8")
    rows = []
    for job in jobs:
        folder = OUT / "fresh_solver" / job["sample"]
        record = folder / "record.json"
        if record.exists():
            row = json.loads(record.read_text(encoding="utf8"))
        else:
            if folder.exists():
                raise RuntimeError(f"Unfinished solver folder must be inspected before resume: {folder}")
            try:
                row = _run_case(DEFAULT_PYTHON, job["design"], job["sample"], folder, 900,
                                "sealed_test", "uniform_feasible_domain", (0, float("inf")), 30, "legacy")
                row.update(polar_quality(folder / "vspaero_polar.csv"))
                row["wake_quality_pass"] = row["quality_pass"]
            except Exception as error:
                row = dict(sample=job["sample"], **job["design"], solver_failure=repr(error),
                           wake_quality_pass=False, vspaero_cl_alpha_per_rad=None,
                           vspaero_oswald_efficiency=None)
            record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "新验证实算标签.csv", index=False)
        print("solver", len(rows), "/", len(jobs), row["wake_quality_pass"], flush=True)
    test = pd.DataFrame(rows)
    results = {}
    for name, path in MODELS.items():
        model = load_vsp_surrogate(path)
        predictions = np.array([
            model.predict(AirframeDesign(**{k: float(row[k]) for k in FEATURE_LABELS}))
            for _, row in test.iterrows()
        ])
        metrics = []
        for i, target in enumerate(("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")):
            y = test[target].to_numpy(float)
            valid = np.isfinite(y)
            if i:
                valid &= test.wake_quality_pass.to_numpy(bool)
            error = predictions[valid, i] - y[valid]
            variance = np.sum((y[valid] - y[valid].mean())**2)
            metrics.append(dict(target=target, count=int(valid.sum()),
                                r2=float(1 - np.sum(error**2) / variance),
                                mae=float(np.mean(abs(error))),
                                rmse=float(np.sqrt(np.mean(error**2)))))
            test[f"{name}_prediction_{i}"] = predictions[:, i]
        results[name] = metrics
        print(name, json.dumps(metrics), flush=True)
    test.to_csv(OUT / "新独立验证_逐点.csv", index=False)
    promotions = {
        name: all(a["rmse"] <= b["rmse"] + 1e-12
                  for a, b in zip(results[name], results["physics211gp"]))
        for name in ("trend211gp", "incrementalgp")
    }
    (OUT / "新独立验证与部署决策.json").write_text(json.dumps(dict(
        models=hashes, metrics=results, promote=promotions,
        goal_r2=.9, target_pass={name: [v["r2"] >= .9 for v in metrics] for name, metrics in results.items()},
        invalid_solver_cases=[r["sample"] for r in rows if not r["wake_quality_pass"]],
    ), ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
