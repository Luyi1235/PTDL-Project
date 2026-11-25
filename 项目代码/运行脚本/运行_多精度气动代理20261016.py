"""Resumable medium-mesh DOE, conditional co-Kriging and sealed validation."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import TargetwiseAeroGP, target_prediction, target_mean
from 多精度气动高斯过程 import fit_autoregressive, autoregressive_prediction
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints
from 重算VSPAERO_Ew_DOE并训练代理 import _run_case, DEFAULT_PYTHON
from 检查_尾迹标签质量 import polar_quality
from 运行_分目标气动代理攻关 import fit_target
from 训练_主动增量混合代理20261016 import metric

LOW_DIR = ROOT / "运行结果" / "主动增量代理_20261015"
LOW_DATA = LOW_DIR / "增量训练样本.csv"
LOW_MODEL = LOW_DIR / "增量构型气动代理.json"
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = [p for p in (ROOT / "核心模型").rglob("*")
             if p.suffix in (".py", ".json", ".csv")]
    paths += [LOW_DATA, LOW_MODEL, Path(__file__),
              Path(__file__).parent / "运行_VSPAERO单构型评估.py",
              Path(__file__).parent / "重算VSPAERO_Ew_DOE并训练代理.py",
              Path(__file__).parent / "检查_尾迹标签质量.py",
              Path(__file__).parent / "运行_分目标气动代理攻关.py",
              Path(__file__).parent / "训练_主动增量混合代理20261016.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def design_plan(train_count, test_count, fine_count):
    train = pd.read_csv(LOW_DATA)
    x = train[list(FEATURE_LABELS)].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    heldout = pd.read_csv(LOW_DIR / "合并留出验证.csv")
    excluded = np.vstack((x, heldout[list(FEATURE_LABELS)].to_numpy(float)))
    prior = json.loads((ROOT / "运行结果" / "主动气动补样_20261015" /
                        "采样预案.json").read_text(encoding="utf8"))
    known = {j["sample"]: j["design"] for j in prior["jobs"]}
    fixed = {k: prior["jobs"][0]["design"][k] for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    jobs = [dict(sample="medium_train_000", role="train", mesh="medium",
                 design={**fixed, **{k: getattr(AirframeDesign(), k) for k in FEATURE_LABELS}},
                 basis="baseline")]
    for name in ("active_test_004", "active_test_011"):
        reuse = ROOT / "运行结果" / "主要误差构型网格诊断_20261016" / name / "medium" / "record.json"
        jobs.append(dict(sample=f"medium_train_{len(jobs):03d}", role="train", mesh="medium",
                         design=known[name], basis="mesh_sensitive_development",
                         reuse_record=str(reuse), reuse_sha256=digest(reuse)))
    rng = np.random.default_rng(2026101631)
    pool = []
    while len(pool) < 6000:
        value = rng.uniform(lo, hi)
        design = {**fixed, **dict(zip(FEATURE_LABELS, value))}
        if all(v <= 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values()):
            pool.append(value)
    pool = np.array(pool)
    selected = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in jobs])
    while len(jobs) < train_count:
        distance = np.linalg.norm((pool[:, None, :]-selected[None, :, :])/(hi-lo), axis=2).min(1)
        avoid = np.linalg.norm((pool[:, None, :]-excluded[None, :, :])/(hi-lo), axis=2).min(1)
        distance[avoid < 1e-5] = -1
        value = pool[int(np.argmax(distance))]
        jobs.append(dict(sample=f"medium_train_{len(jobs):03d}", role="train", mesh="medium",
                         design={**fixed, **dict(zip(FEATURE_LABELS, value))}, basis="high_fidelity_maximin"))
        selected = np.vstack((selected, value))
    excluded = np.vstack((excluded, selected))
    for i in range(test_count):
        while True:
            value = rng.uniform(lo, hi)
            design = {**fixed, **dict(zip(FEATURE_LABELS, value))}
            if (np.linalg.norm((excluded-value)/(hi-lo), axis=1).min() > 1e-5
                    and all(v <= 1e-9 for v in geometry_constraints(AirframeDesign(**design)).values())):
                break
        jobs.append(dict(sample=f"medium_test_{i:03d}", role="test", mesh="medium",
                         design=design, basis="sealed_uniform_feasible"))
        excluded = np.vstack((excluded, value))
    for i, index in enumerate(np.linspace(0, train_count-1, fine_count).astype(int)):
        source = jobs[index]
        jobs.append(dict(sample=f"fine_audit_{i:03d}", role="fine_audit", mesh="fine",
                         design=source["design"], medium_source=source["sample"],
                         basis="separate_grid_sensitivity_audit"))
    return dict(jobs=jobs, source_manifest=sources(), train_count=train_count,
                test_count=test_count, fine_count=fine_count, wake_iterations=30,
                native_workers=1, low_data_sha256=digest(LOW_DATA),
                protocol="Medium labels isolated from legacy labels; test solver calls follow model freezing")


def fit_models(frame, out):
    low_frame = pd.read_csv(LOW_DATA)
    base = json.loads(LOW_MODEL.read_text(encoding="utf8"))
    candidate, direct = dict(base), dict(base)
    candidate["targets"], direct["targets"] = [], []
    records, chosen = [], []
    low_used, used_low_frames = [], []
    for t, name in enumerate(TARGETS):
        low_mask = np.isfinite(low_frame[name].to_numpy(float))
        high_mask = np.isfinite(frame[name].to_numpy(float))
        if t:
            low_mask &= low_frame.wake_quality_pass.to_numpy(bool)
            high_mask &= frame.wake_quality_pass.to_numpy(bool)
        low_data, high = low_frame.loc[low_mask], frame.loc[high_mask]
        if len(high) < 15:
            raise RuntimeError("Fewer than 15 usable medium-mesh labels; inspect solver failures")
        low = fit_target(low_data[list(FEATURE_LABELS)].to_numpy(float),
                         low_data[name].to_numpy(float), "physics", "matern52" if t == 0 else "matern32")[1]
        x, y = high[list(FEATURE_LABELS)].to_numpy(float), high[name].to_numpy(float)
        folds = list(KFold(5, shuffle=True, random_state=2026101632).split(x))
        choices = []
        for method in ("direct-high", "autoregressive"):
            for kernel in ("matern32", "matern52"):
                pred, std = np.zeros(len(y)), np.zeros(len(y))
                for a, b in folds:
                    model = (fit_autoregressive(low, x[a], y[a], kernel) if method == "autoregressive"
                             else fit_target(x[a], y[a], "physics", kernel)[1])
                    for index in b:
                        pred[index], std[index] = target_prediction(model, x[index:index+1])
                scale = max(1., float(np.quantile(abs(pred-y)/np.maximum(std, 1e-9),
                                                  .9, method="higher")/1.645))
                record = dict(target=t, method=method, kernel=kernel, count=len(y),
                              interval_scale=scale, **metric(y, pred))
                records.append(record)
                choices.append(record)
                print(json.dumps(record), flush=True)
                pd.DataFrame(records).to_csv(out / "高精度训练五折比较.csv", index=False)
        best = min(choices, key=lambda r: r["rmse"])
        best_direct = min((r for r in choices if r["method"] == "direct-high"), key=lambda r: r["rmse"])
        selected = (fit_autoregressive(low, x, y, best["kernel"]) if best["method"] == "autoregressive"
                    else fit_target(x, y, "physics", best["kernel"])[1])
        selected["interval_scale"] = best["interval_scale"]
        reference = fit_target(x, y, "physics", best_direct["kernel"])[1]
        reference["interval_scale"] = best_direct["interval_scale"]
        candidate["targets"].append(selected)
        direct["targets"].append(reference)
        chosen.append(best)
        low_used.append(best["method"] == "autoregressive")
        if low_used[-1]:
            used_low_frames.append(low_data)
    baseline = frame.loc[frame["sample"] == "medium_train_000"].iloc[0]
    if not baseline.wake_quality_pass or not np.isfinite(baseline[list(TARGETS)].to_numpy(float)).all():
        raise RuntimeError("High-fidelity baseline is invalid")
    high_valid = frame.loc[np.isfinite(frame[TARGETS[0]].to_numpy(float))]
    used = pd.concat(used_low_frames+[high_valid])
    unique = used[list(FEATURE_LABELS)].drop_duplicates()
    for payload, samples in ((candidate, len(unique)), (direct, len(high_valid))):
        payload.update(
            baseline_targets=baseline[list(TARGETS)].to_numpy(float).tolist(),
            training_samples=samples, low_available_samples=len(low_frame),
            high_training_samples=len(high_valid), mesh_reference="medium",
            selection="Training-only five-fold RMSE; direct high-fidelity GP versus conditional autoregressive co-Kriging")
    candidate["low_fidelity_used_by_target"] = low_used
    paths = {"high_candidate": out / "多精度候选代理.json",
             "direct_high": out / "仅高精度候选代理.json", "low_reference": LOW_MODEL}
    for name, payload in (("high_candidate", candidate), ("direct_high", direct)):
        TargetwiseAeroGP(payload).to_json(paths[name])
        TargetwiseAeroGP.from_json(paths[name])
    (out / "训练模型选择.json").write_text(json.dumps(chosen, indent=2), encoding="utf8")
    return paths


def run(out, counts):
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "冻结采样与计算配置.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf8"))
        if (plan["train_count"], plan["test_count"], plan["fine_count"]) != counts:
            raise RuntimeError("Resume budget changed")
    else:
        plan = design_plan(*counts)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf8")
    started = time.time()
    rows = {role: [] for role in ("train", "test", "fine_audit")}

    def check_sources():
        for relative, sha in plan["source_manifest"].items():
            if digest(ROOT / relative) != sha:
                raise RuntimeError(f"Frozen calculation source changed: {relative}")

    def progress(stage, error=None):
        record = dict(pid=os.getpid(), stage=stage, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                      elapsed_s=time.time()-started, error=error,
                      completed={role: len(value) for role, value in rows.items()},
                      requested=dict(train=counts[0], test=counts[1], fine_audit=counts[2]))
        tmp = out / "进度.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
        tmp.replace(out / "进度.json")

    def solve(job):
        check_sources()
        folder = out / "solver" / job["sample"]
        record = folder / "record.json"
        if record.exists():
            return json.loads(record.read_text(encoding="utf8"))
        if folder.exists():
            raise RuntimeError(f"Inspect interrupted solver folder before resume: {folder}")
        folder.mkdir(parents=True)
        if "reuse_record" in job:
            source = Path(job["reuse_record"])
            if digest(source) != job["reuse_sha256"]:
                raise RuntimeError("Reused diagnostic label changed")
            row = json.loads(source.read_text(encoding="utf8"))
            for name in FEATURE_LABELS:
                if abs(row[name]-job["design"][name]) > 1e-8:
                    raise RuntimeError("Reused geometry mismatch")
            row.update(sample=job["sample"], reuse_source=str(source),
                       wake_quality_pass=bool(row["quality_pass"]))
        else:
            # _run_case owns its new case directory; the outer folder holds the journal.
            try:
                row = _run_case(DEFAULT_PYTHON, job["design"], job["sample"], folder / "case",
                                1800, job["role"], job["basis"], (0, float("inf")), 30, job["mesh"])
                row.update(polar_quality(folder / "case" / "vspaero_polar.csv"))
                row["wake_quality_pass"] = row["quality_pass"]
            except Exception as error:
                row = dict(sample=job["sample"], **job["design"], solver_failure=repr(error),
                           wake_quality_pass=False, vspaero_cl_alpha_per_rad=None,
                           vspaero_oswald_efficiency=None)
        row["mesh_level"] = job["mesh"]
        record.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
        return row

    try:
        for job in (j for j in plan["jobs"] if j["role"] == "train"):
            rows["train"].append(solve(job))
            pd.DataFrame(rows["train"]).to_csv(out / "中网格训练实算标签.csv", index=False)
            progress("medium_training")
            print("medium training", len(rows["train"]), "/", counts[0], flush=True)
        check_sources()
        train = pd.DataFrame(rows["train"])
        selection = out / "训练模型选择.json"
        if not selection.exists():
            progress("training_models")
            paths = fit_models(train, out)
        else:
            paths = {"high_candidate": out / "多精度候选代理.json",
                     "direct_high": out / "仅高精度候选代理.json", "low_reference": LOW_MODEL}
        hashes = {name: digest(path) for name, path in paths.items()}
        lock_path = out / "盲测前模型锁定.json"
        if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf8")) != hashes:
            raise RuntimeError("Frozen model changed after test collection")
        lock_path.write_text(json.dumps(hashes, indent=2), encoding="utf8")
        for job in (j for j in plan["jobs"] if j["role"] == "test"):
            rows["test"].append(solve(job))
            pd.DataFrame(rows["test"]).to_csv(out / "中网格独立测试标签.csv", index=False)
            progress("medium_blind_test")
            print("medium test", len(rows["test"]), "/", counts[1], flush=True)
        test = pd.DataFrame(rows["test"])
        results = {}
        for name, path in paths.items():
            model = load_vsp_surrogate(path)
            pred = np.array([model.predict(AirframeDesign(**{k: row[k] for k in FEATURE_LABELS}))
                             for _, row in test.iterrows()])
            results[name] = []
            for t, target in enumerate(TARGETS):
                y = test[target].to_numpy(float)
                mask = np.isfinite(y)
                if t:
                    mask &= test.wake_quality_pass.to_numpy(bool)
                if mask.sum() < 3:
                    raise RuntimeError("Insufficient valid independent test labels")
                results[name].append(dict(target=target, count=int(mask.sum()), **metric(y[mask], pred[mask, t])))
                test[f"{name}_prediction_{t}"] = pred[:, t]
        test.to_csv(out / "中网格盲测逐点.csv", index=False)
        (out / "中网格盲测评估.json").write_text(json.dumps(dict(
            metrics=results, goal_r2=.9,
            target_pass=[r["r2"] >= .9 for r in results["high_candidate"]],
            automatic_production_promotion=False, mesh_reference="medium",
        ), indent=2), encoding="utf8")
        for job in (j for j in plan["jobs"] if j["role"] == "fine_audit"):
            row = solve(job)
            row["medium_source"] = job["medium_source"]
            rows["fine_audit"].append(row)
            pd.DataFrame(rows["fine_audit"]).to_csv(out / "细网格独立诊断.csv", index=False)
            progress("fine_mesh_audit")
        progress("completed")
    except Exception:
        progress("failed", traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "多精度气动代理_20261016")
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--fine-count", type=int, default=4)
    args = parser.parse_args()
    if args.train_count < 15 or args.test_count < 3 or not 0 <= args.fine_count <= args.train_count:
        parser.error("Require at least 15 training and three test cases")
    run(args.output_dir, (args.train_count, args.test_count, args.fine_count))
