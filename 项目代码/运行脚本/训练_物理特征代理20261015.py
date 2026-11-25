"""Train on development data, select by CV, and score an untouched solver batch."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import TargetwiseAeroGP, feature_map
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints
from 运行_分目标气动代理攻关 import fit_target, TARGETS
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON
from 检查_尾迹标签质量 import polar_quality

OUT = ROOT / "运行结果" / "物理特征代理_20261015"
PRIOR = ROOT / "运行结果" / "分目标代理211点_20261013"


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PRIOR / "训练样本211点.csv")
    x = frame[list(FEATURE_LABELS)].to_numpy(float)
    candidates = [(t, f, k) for t in range(2) for f in ("physics", "dimensionless")
                  for k in ("matern32", "matern52", "rbf")]

    def fit_job(job):
        t, feature, kernel = job
        valid = np.ones(len(frame), bool) if t == 0 else frame.wake_quality_pass.to_numpy(bool)
        xx, y = x[valid], frame.loc[valid, TARGETS[t]].to_numpy(float)
        pred, std = np.zeros(len(y)), np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=20261015).split(xx):
            gp, params = fit_target(xx[a], y[a], feature, kernel)
            z = (feature_map(xx[b], feature) - params["feature_mean"]) / params["feature_scale"]
            pred[b], std[b] = gp.predict(z, return_std=True)
        err = pred - y
        record = dict(target=t, features=feature, kernel=kernel,
                      r2=float(r2_score(y, pred)), rmse=float(np.sqrt(np.mean(err**2))),
                      interval_scale=max(1., float(np.quantile(abs(err)/np.maximum(std, 1e-9),
                                                              .9, method="higher")/1.645)))
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fit_job, candidates))
    pd.DataFrame(records).to_csv(OUT / "训练集五折模型比较.csv", index=False)
    selected = []
    for target in range(2):
        chosen = min((r for r in records if r["target"] == target), key=lambda r: r["rmse"])
        valid = np.ones(len(frame), bool) if target == 0 else frame.wake_quality_pass.to_numpy(bool)
        _, params = fit_target(x[valid], frame.loc[valid, TARGETS[target]].to_numpy(float),
                              chosen["features"], chosen["kernel"])
        params["interval_scale"] = chosen["interval_scale"]
        selected.append(params)
    baseline = np.asarray([getattr(AirframeDesign(), k) for k in FEATURE_LABELS])
    index = int(np.argmin(np.linalg.norm(x - baseline, axis=1)))
    payload = dict(schema="ptdl_targetwise_gp_v1", feature_labels=list(FEATURE_LABELS),
                   training_samples=len(frame), baseline_targets=frame.iloc[index][list(TARGETS)].tolist(),
                   targets=selected, selection="training-only five-fold minimum RMSE",
                   label_definition="legacy median Ew, unchanged",
                   feature_note="span, mean aerodynamic chord and body ratios; no solver response as input")
    model = TargetwiseAeroGP(payload)
    model.to_json(OUT / "物理特征气动代理.json")
    # Previous holdout is now development evidence, never reported as a fresh test.
    test = pd.read_csv(PRIOR / "独立验证逐点预测.csv")
    score(model, test, "历史验证开发对照")
    return model


def score(model, test, name):
    estimates = np.asarray([model.predict(AirframeDesign(**{k: float(r[k]) for k in FEATURE_LABELS}))
                            for _, r in test.iterrows()])
    results = []
    for target in range(2):
        key = f"scorable_{TARGETS[target]}"
        valid = test[key].to_numpy(bool, copy=True) if key in test else np.ones(len(test), bool)
        if target == 1 and "wake_quality_pass" in test:
            valid &= test.wake_quality_pass.to_numpy(bool)
        y = test[TARGETS[target]].to_numpy(float)
        valid &= np.isfinite(y)
        err = estimates[valid, target] - y[valid]
        results.append(dict(target=TARGETS[target], count=int(valid.sum()),
                            r2=float(r2_score(y[valid], estimates[valid, target])),
                            rmse=float(np.sqrt(np.mean(err**2))),
                            mae=float(np.mean(abs(err)))))
        test[f"prediction_{target}"] = estimates[:, target]
    test.to_csv(OUT / f"{name}_逐点.csv", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
    print(name, json.dumps(results), flush=True)
    return results


def fresh_test(count, workers):
    path = OUT / "独立验证构型锁定.json"
    frame = pd.read_csv(PRIOR / "训练样本211点.csv")
    if path.exists():
        jobs = json.loads(path.read_text(encoding="utf8"))["jobs"]
        if len(jobs) != count:
            raise ValueError(f"This sealed test contains {len(jobs)} geometries; use --test-count {len(jobs)}")
    else:
        x = frame[list(FEATURE_LABELS)].to_numpy(float)
        lo, hi = x.min(0), x.max(0)
        rng = np.random.default_rng(2026101501)
        fixed = {k: float(frame.iloc[0][k]) for k in
                 ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
        jobs = []
        while len(jobs) < count:
            design = AirframeDesign(**dict(zip(FEATURE_LABELS, rng.uniform(lo, hi))))
            if any(v > 1e-9 for v in geometry_constraints(design).values()):
                continue
            values = np.array([getattr(design, k) for k in FEATURE_LABELS])
            if np.linalg.norm((x - values) / (hi - lo), axis=1).min() < 1e-6:
                continue
            jobs.append(dict(name=f"fresh_{len(jobs):03d}",
                             design={**fixed, **dict(zip(FEATURE_LABELS, values))}))
        model_path = OUT / "物理特征气动代理.json"
        path.write_text(json.dumps(dict(jobs=jobs, frozen_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                        use="test only; no selection/refit after test"), indent=2), encoding="utf8")
    frozen = json.loads(path.read_text(encoding="utf8"))["frozen_sha256"]
    model_path = OUT / "物理特征气动代理.json"
    if frozen != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise RuntimeError("Frozen validation model changed")
    def run(job):
        folder = OUT / "fresh_solver" / job["name"]
        record = folder / "record.json"
        if record.exists():
            return json.loads(record.read_text(encoding="utf8"))
        polars = list(folder.glob("*.polar")) if folder.exists() else []
        if polars and (folder / "case_error.txt").exists():
            reason = (folder / "case_error.txt").read_text(encoding="utf8")
            if "no valid wake-efficiency" in reason:
                lines = polars[0].read_text(encoding="utf8").splitlines()
                header = next(i for i, line in enumerate(lines) if "AoA" in line.split())
                raw = pd.read_csv(polars[0], sep=r"\s+", skiprows=header+1,
                                  names=lines[header].split())
                angles = np.deg2rad(raw.AoA.to_numpy(float))
                lift = raw.CLtot.to_numpy(float)
                slope = float(np.polyfit(angles, lift, 1)[0])
                row = dict(sample=job["name"], **job["design"],
                           vspaero_cl_alpha_per_rad=slope, vspaero_oswald_efficiency=None,
                           wake_quality_pass=False, quality_pass=False,
                           geometry_file=str(folder / "airframe.vsp3"),
                           solver_failure="No usable Ew; retained as an unscorable wake label",
                           **{"scorable_"+TARGETS[0]: bool(np.isfinite(slope) and 3.5 <= slope <= 8.5),
                              "scorable_"+TARGETS[1]: False})
                record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
                return row
        if folder.exists():
            n = 1
            while folder.with_name(folder.name + f"_retry{n}").exists():
                n += 1
            folder = folder.with_name(folder.name + f"_retry{n}")
        row = _run_case(DEFAULT_PYTHON, job["design"], job["name"], folder, 600,
                        "sealed_test", "uniform_feasible_domain", (0., float("inf")))
        row.update(polar_quality(folder / "vspaero_polar.csv"))
        row["wake_quality_pass"] = row["quality_pass"]
        row["scorable_" + TARGETS[0]] = True
        row["scorable_" + TARGETS[1]] = row["quality_pass"]
        (folder / "record.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        return row
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, j): j for j in jobs}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
                print("fresh", len(rows), "/", len(jobs), flush=True)
            except Exception as e:
                errors.append(dict(name=futures[future]["name"], error=repr(e)))
            pd.DataFrame(rows).to_csv(OUT / "独立验证实算标签.csv", index=False)
            (OUT / "独立验证进度.json").write_text(json.dumps(dict(completed=len(rows), total=len(jobs),
                                                                 errors=errors), indent=2), encoding="utf8")
    if len(rows) != len(jobs):
        raise RuntimeError("Incomplete sealed solver batch; rerun --test-only to resume")
    model = load_vsp_surrogate(model_path)
    results = score(model, pd.DataFrame(rows), "新独立验证")
    (OUT / "精度目标验收.json").write_text(json.dumps({
        "goal_r2": .9, "per_target_pass": [r["r2"] >= .9 for r in results],
        "metrics": results, "model_sha256": frozen, "test_role": "independent frozen solver batch",
    }, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--test-count", type=int, default=24)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-test", action="store_true")
    args = p.parse_args()
    if not args.test_only:
        train()
    if not args.skip_test:
        fresh_test(args.test_count, args.workers)
