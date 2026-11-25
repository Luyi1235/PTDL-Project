"""Freeze training-only selection, acquire new geometries, and evaluate fresh labels."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.model_selection import KFold, GroupKFold

import 研究_千构型尾迹分解20261017 as direct

DIRECT_FIT, DIRECT_PREDICT, DIRECT_CONFIGS = direct.fit, direct.predict, direct.CONFIGS
physical = importlib.import_module("研究_千构型物理分解20261017")
direct.fit, direct.predict, direct.CONFIGS = DIRECT_FIT, DIRECT_PREDICT, DIRECT_CONFIGS
regime = importlib.import_module("研究_尾迹迎角分区20261017")
direct.fit, direct.predict, direct.CONFIGS = DIRECT_FIT, DIRECT_PREDICT, DIRECT_CONFIGS
import 推进_千构型原定义代理20261017 as original

ROOT, SOURCE = direct.ROOT, direct.SOURCE
OUT = ROOT / "运行结果/千构型分区主动补样_20261017"
FOLDERS = {
    "direct": ROOT / "运行结果/千构型尾迹分解_20261017",
    "physical": ROOT / "运行结果/千构型物理分解_20261017",
    "regime": ROOT / "运行结果/尾迹迎角分区_20261017",
}


def fit(spec, x, y, polar):
    return {"direct": DIRECT_FIT, "physical": physical.fit, "regime": regime.fit}[
        spec["family"]](spec["settings"], x, y, polar)


def predict(spec, model, x):
    return {"direct": DIRECT_PREDICT, "physical": physical.predict, "regime": regime.predict}[
        spec["family"]](model, x)


def data(frame):
    mask = frame.wake_quality_pass.to_numpy(bool).copy()
    mask &= np.isfinite(frame[direct.TARGET].to_numpy(float))
    subset = frame.loc[mask].reset_index(drop=True)
    polars = [pd.read_csv(Path(row.geometry_file).parent / "vspaero_polar.csv").sort_values(
        "alpha_deg")[["cl", "cl_wake", "cd_induced_wake", "e_wake"]].to_numpy(float)
        for _, row in subset.iterrows()]
    return subset, subset[direct.FEATURES].to_numpy(float), subset[direct.TARGET].to_numpy(float), np.asarray(polars)


def records_matrix(records, n, config):
    matrix = []
    for split in ("random", "spatial"):
        prediction = np.full(n, np.nan)
        count = np.zeros(n, int)
        for row in records:
            if row["config"] == config and row["split"] == split:
                prediction[row["validation"]] = row["prediction"]
                count[row["validation"]] += 1
        if not np.all(count == 1) or not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete or repeated OOF coverage")
        matrix.append(prediction)
    return np.asarray(matrix)


def select_mixture(predictions, y):
    def score(p):
        return float(np.sqrt(np.mean((p-y[None, :])**2, axis=1)).mean())
    choices = [(score(p), [(i, 1.)], p) for i, p in enumerate(predictions)]
    for left in range(len(predictions)):
        for right in range(left+1, len(predictions)):
            for weight in (.25, .5, .75):
                p = weight*predictions[left]+(1-weight)*predictions[right]
                choices.append((score(p), [(left, weight), (right, 1-weight)], p))
    return min(choices, key=lambda row: row[0])


def portable_scalar(model, interval):
    config = model["config"]
    if config["method"] == "regime":
        estimator, fallback = model["gate"], model["fallback"]
        gate = dict(features=config["mode"], kernel="rbf",
                    feature_mean=fallback["center"].tolist(), feature_scale=fallback["scale"].tolist(),
                    train=estimator.X_train_.tolist(), mean=estimator._y_train_mean.tolist(),
                    scale=estimator._y_train_std.tolist(),
                    amplitude=float(estimator.kernel_.k1.k1.constant_value),
                    length=np.asarray(estimator.kernel_.k1.k2.length_scale).tolist(),
                    dual=estimator.alpha_.tolist())
        return dict(kind="angle-regime-ensemble", gate=gate, blend=config["blend"],
                    fallback=portable_scalar(fallback, interval), interval_std=interval,
                    experts={str(k): portable_scalar(v, interval) for k, v in model["experts"].items()})
    if config["method"] == "polar":
        starter = dict(method="boost", mode=config["mode"], leaves=7, penalty=1.)
        fallback = {**model, "config": starter}
        components = [portable_scalar({**model, "config": starter, "estimator": estimator}, interval)
                      for estimator in model["components"]]
        return dict(kind="force-polar-ensemble", components=components,
                    reconstruction=config["reconstruction"], interval_std=interval,
                    fallback=portable_scalar(fallback, interval))
    estimator = model["estimator"]
    if config["method"] == "boost":
        value = direct.base.boosting_payload(estimator, config["mode"])
        value.update(feature_mean=model["center"].tolist(), feature_scale=model["scale"].tolist())
    elif config["method"] in ("gp", "gp_restart"):
        z = estimator.X_train_
        value = dict(features=config["mode"], kernel=config["kernel"],
                     feature_mean=model["center"].tolist(), feature_scale=model["scale"].tolist(),
                     train=z.tolist(), mean=float(estimator._y_train_mean),
                     scale=float(estimator._y_train_std),
                     amplitude=float(estimator.kernel_.k1.k1.constant_value),
                     length=np.atleast_1d(estimator.kernel_.k1.k2.length_scale).tolist(),
                     noise=float(estimator.kernel_.k2.noise_level), dual=estimator.alpha_.tolist(),
                     inverse=cho_solve((estimator.L_, True), np.eye(len(z))).tolist(), interval_scale=1.)
    else:
        raise ValueError("Research-only model requires a separate production adapter")
    return dict(kind="weighted-ensemble", models=[value], weights=[1.], interval_std=interval)


def initial_selection(out):
    frame, x, y, polar = data(pd.read_csv(SOURCE / "原网格训练数据.csv"))
    specs, predictions = [], []
    for family, folder in FOLDERS.items():
        if json.loads((folder / "进度.json").read_text(encoding="utf8"))["stage"] != "completed":
            raise RuntimeError(f"Comparison incomplete: {family}")
        configurations = json.loads((folder / "配置冻结.json").read_text(encoding="utf8"))["configurations"]
        records = json.loads((folder / "交叉验证逐折.json").read_text(encoding="utf8"))
        for index, config in enumerate(configurations):
            specs.append(dict(family=family, config=index, settings=config))
            predictions.append(records_matrix(records, len(y), index))
    # Every method remains in the comparison; production compatibility is recorded separately.
    score, components, oof = select_mixture(predictions, y)
    compatible = [i for i, s in enumerate(specs)
                  if s["settings"]["method"] in ("gp", "gp_restart", "boost", "polar", "regime")]
    prod_score, prod_components, prod_oof = select_mixture([predictions[i] for i in compatible], y)
    prod_components = [(compatible[i], w) for i, w in prod_components]
    order = sorted(compatible, key=lambda i: np.sqrt(np.mean((predictions[i]-y)**2, axis=1)).mean())
    shortlist = list(dict.fromkeys([i for i, _ in prod_components]+order[:4]))[:5]
    interval = max(1e-9, float(np.quantile(abs(prod_oof-y), .9)/1.645))
    payload = json.loads((SOURCE / "多精度候选代理.json").read_text(encoding="utf8"))
    models = []
    for i, weight in prod_components:
        fitted = fit(specs[i], x, y, polar)
        native = portable_scalar(fitted, interval)
        probe = x[::max(1, len(x)//30)]
        np.testing.assert_allclose(
            [direct.base.target_mean(native, row[None, :]) for row in probe],
            predict(specs[i], fitted, probe), rtol=1e-9, atol=1e-9)
        models.append(native)
    payload["targets"][1] = dict(kind="weighted-ensemble", models=models,
                                 weights=[w for _, w in prod_components], interval_std=interval)
    payload.update(selection="Training-only random/spatial OOF convex mixture",
                   research_source=str(out), automatic_production_promotion=False)
    path = out / "千构型改进候选.json"
    direct.base.TargetwiseAeroGP(payload).to_json(path)
    direct.base.TargetwiseAeroGP.from_json(path)
    result = dict(
        all_method_best=dict(score=score, components=[dict(**specs[i], weight=w) for i, w in components]),
        portable_selected=dict(score=prod_score,
                               components=[dict(**specs[i], weight=w) for i, w in prod_components]),
        shortlist=[specs[i] for i in shortlist], external_test_used_for_selection=False,
        training_oof=[dict(split=split, **direct.base.metric(y, p))
                      for split, p in zip(("random", "spatial"), prod_oof)],
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out / "千构型模型选择.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    frame["oof_error"] = np.sqrt(np.mean((prod_oof-y)**2, axis=0))
    frame.to_csv(out / "训练折外误差.csv", index=False)
    return result


def make_plan(out, extra, tests):
    if extra % 4:
        raise ValueError("Acquisition count must be divisible by four")
    training = pd.read_csv(SOURCE / "原网格训练数据.csv")
    x = training[direct.FEATURES].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    span = hi-lo
    prior_plan = json.loads((SOURCE / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    fixed = {k: prior_plan["jobs"][0]["design"][k] for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    forbidden = [np.asarray([[j["design"][k] for k in direct.FEATURES] for j in prior_plan["jobs"]
                            if j["role"] == "test"])]
    for source in (original.prior.FIRST, ROOT / "运行结果/多精度气动代理_20261016"):
        old = json.loads((source / "冻结采样与计算配置.json").read_text(encoding="utf8"))
        forbidden.append(np.asarray([[j["design"][k] for k in direct.FEATURES] for j in old["jobs"]
                                     if j["role"] == "test"]))
    forbidden.append(pd.read_csv(original.prior.driver.LOW_DIR / "合并留出验证.csv")[
        direct.FEATURES].to_numpy(float))
    avoid = np.vstack([x, *forbidden])
    def feasible(row):
        return all(v <= 1e-9 for v in original.prior.geometry_constraints(
            original.prior.AirframeDesign(**fixed, **dict(zip(direct.FEATURES, row)))).values())
    pool = qmc.scale(qmc.Sobol(7, scramble=True, seed=2026101715).random_base2(14), lo, hi)
    pool = np.asarray([r for r in pool if feasible(r)])
    distance = cKDTree((avoid-lo)/span).query((pool-lo)/span)[0]
    pool, distance = pool[distance > 1e-5], distance[distance > 1e-5]
    jobs = []
    reuse_dir = out / "复用记录"
    reuse_dir.mkdir(exist_ok=True)
    for job in prior_plan["jobs"]:
        if job["role"] != "train":
            continue
        original_record = SOURCE / "solver" / job["sample"] / "record.json"
        row = json.loads(original_record.read_text(encoding="utf8"))
        row["quality_pass"] = bool(row.get("wake_quality_pass", False))
        row["original_record_sha256"] = original.prior.driver.digest(original_record)
        source = reuse_dir / f"{job['sample']}.json"
        source.write_text(json.dumps(row, ensure_ascii=False), encoding="utf8")
        jobs.append(dict(job, reuse_record=str(source), reuse_sha256=original.prior.driver.digest(source)))
    picked = []
    for _ in range(extra//2):
        index = int(np.argmax(distance))
        row = pool[index]
        picked.append((row.copy(), "space_filling"))
        distance = np.minimum(distance, np.linalg.norm((pool-row)/span, axis=1))
    hard = pd.read_csv(out / "训练折外误差.csv").sort_values("oof_error", ascending=False)
    anchors = hard.iloc[:max(24, len(hard)//10)][direct.FEATURES].to_numpy(float)
    near_error = cKDTree((anchors-lo)/span).query((pool-lo)/span)[0]
    # Moderate proximity to difficult training regions, balanced by distance from existing labels.
    priority = distance/(.08+near_error)
    for _ in range(extra//4):
        index = int(np.argmax(priority))
        row = pool[index]
        picked.append((row.copy(), "training_oof_error_neighborhood"))
        distance = np.minimum(distance, np.linalg.norm((pool-row)/span, axis=1))
        priority = distance/(.08+near_error)
    model = direct.base.TargetwiseAeroGP.from_json(out / "千构型改进候选.json")
    # The selected GP components expose predictive variance independently of test labels.
    def scalar_gps(target):
        if target.get("kind") == "weighted-ensemble":
            for part in target["models"]:
                yield from scalar_gps(part)
        elif target.get("kind") == "angle-regime-ensemble":
            yield from scalar_gps(target["fallback"])
        elif target.get("kind") == "force-polar-ensemble":
            yield from scalar_gps(target["fallback"])
        elif "inverse" in target:
            yield target
    gp = next(scalar_gps(model.payload["targets"][1]), None)
    if gp is None:
        raise RuntimeError("Acquisition requires a selected GP uncertainty component")
    z = (direct.base.feature_map(pool, gp["features"])-gp["feature_mean"])/gp["feature_scale"]
    from 分目标气动高斯过程 import kernel_cross
    uncertainty = []
    inverse = np.asarray(gp["inverse"])
    for start in range(0, len(z), 256):
        cross = kernel_cross(z[start:start+256], np.asarray(gp["train"]),
                             np.asarray(gp["length"]), gp["amplitude"], gp["kernel"])
        variance = gp["amplitude"]+gp["noise"]-np.sum((cross@inverse)*cross, axis=1)
        uncertainty.extend(np.maximum(variance, 0))
    uncertainty = np.sqrt(np.asarray(uncertainty))
    for _ in range(extra//4):
        index = int(np.argmax(uncertainty*np.minimum(distance/.12, 1.)))
        row = pool[index]
        picked.append((row.copy(), "gp_uncertainty"))
        distance = np.minimum(distance, np.linalg.norm((pool-row)/span, axis=1))
    for row, basis in picked:
        jobs.append(dict(sample=f"active_train_{len(jobs):04d}", role="train", mesh="legacy",
                         design={**fixed, **dict(zip(direct.FEATURES, row))}, basis=basis))
    avoid = np.vstack((avoid, [row for row, _ in picked]))
    rng = np.random.default_rng(2026101716)
    while sum(j["role"] == "test" for j in jobs) < tests:
        row = rng.uniform(lo, hi)
        if feasible(row) and np.linalg.norm((avoid-row)/span, axis=1).min() > 1e-5:
            jobs.append(dict(sample=f"active_test_{len(jobs)-len(x)-extra:04d}", role="test",
                             mesh="legacy", design={**fixed, **dict(zip(direct.FEATURES, row))},
                             basis="fresh_sealed_uniform_feasible"))
            avoid = np.vstack((avoid, row))
    hashes = original.prior.driver.sources()
    for module in (direct, physical, regime, original):
        path = Path(module.__file__)
        hashes[str(path.relative_to(ROOT))] = original.prior.driver.digest(path)
    for path in (Path(__file__), original.WORKER, Path(original.controlled.__file__),
                 out / "千构型模型选择.json", out / "千构型改进候选.json"):
        hashes[str(path.relative_to(ROOT))] = original.prior.driver.digest(path)
    return dict(jobs=jobs, source_manifest=hashes, train_count=len(x)+extra, test_count=tests,
                fine_count=0, wake_protocol="free-wake-legacy-v1", mesh_reference="legacy",
                reused_same_protocol_count=len(x), new_training_count=extra,
                automatic_production_promotion=False, label_definition_unchanged=True)


def cv_task(args):
    spec, x, y, polar, split, fold, a, b = args
    fitted = fit(spec, x[a], y[a], polar[a])
    return dict(spec=spec, split=split, fold=fold, validation=b.tolist(),
                prediction=predict(spec, fitted, x[b]).tolist(), actual=y[b].tolist())


def fit_models(frame, out):
    selection = json.loads((out / "千构型模型选择.json").read_text(encoding="utf8"))
    specs = selection["shortlist"]
    subset, x, y, polar = data(frame)
    groups = direct.base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101712).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((spec, x, y, polar, split, fold, a, b) for spec in specs)
    records = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for future in as_completed([pool.submit(cv_task, j) for j in jobs]):
            row = future.result()
            row["config"] = specs.index(row["spec"])
            records.append(row)
            (out / "新增训练交叉验证.json").write_text(json.dumps(records), encoding="utf8")
            print("Active training CV", len(records), "/", len(jobs), flush=True)
    predictions = [records_matrix(records, len(y), i) for i in range(len(specs))]
    score, chosen, oof = select_mixture(predictions, y)
    interval = max(1e-9, float(np.quantile(abs(oof-y), .9)/1.645))
    payload = json.loads((out / "千构型改进候选.json").read_text(encoding="utf8"))
    components = [portable_scalar(fit(specs[i], x, y, polar), interval) for i, _ in chosen]
    payload["targets"][1] = dict(kind="weighted-ensemble", models=components,
                                 weights=[w for _, w in chosen], interval_std=interval)
    lift_mask = np.isfinite(frame[direct.base.TARGETS[0]].to_numpy(float))
    lift_x = frame.loc[lift_mask, direct.FEATURES].to_numpy(float)
    lift_y = frame.loc[lift_mask, direct.base.TARGETS[0]].to_numpy(float)
    lift_config = dict(method="boost", mode="physics", penalty=5.)
    fitted = direct.base.fit(lift_config, lift_x, lift_y)
    # The existing lift architecture is fixed; its accuracy is measured on the new holdout.
    old_interval = payload["targets"][0]["interval_std"]
    payload["targets"][0] = portable_scalar(fitted, old_interval)
    payload.update(training_samples=len(frame), high_training_samples=int(lift_mask.sum()),
                   selection="Active expansion, training-only random/spatial CV mixture",
                   uncertainty="Wake OOF error band; lift band retained from 1000-case training audit")
    candidate = out / "多精度候选代理.json"
    direct.base.TargetwiseAeroGP(payload).to_json(candidate)
    direct.base.TargetwiseAeroGP.from_json(candidate)
    before = out / "仅高精度候选代理.json"
    shutil.copy2(out / "千构型改进候选.json", before)
    (out / "训练模型选择.json").write_text(json.dumps(dict(
        selected=[dict(**specs[i], weight=w) for i, w in chosen], score=score,
        valid_counts=dict(lift=int(lift_mask.sum()), wake=len(y)),
        external_labels_used=False)), encoding="utf8")
    return dict(high_candidate=candidate, direct_high=before, low_reference=original.prior.driver.LOW_MODEL)


def retrospective(out):
    model = direct.base.TargetwiseAeroGP.from_json(out / "千构型改进候选.json")
    frame = pd.read_csv(SOURCE / "中网格盲测逐点.csv")
    x = frame[direct.FEATURES].to_numpy(float)
    prediction = np.array([[direct.base.target_mean(t, row[None, :]) for t in model.payload["targets"]]
                           for row in x])
    rows = []
    for t, target in enumerate(direct.base.TARGETS):
        y = frame[target].to_numpy(float)
        mask = np.isfinite(y)
        if t:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        frame[f"improved_prediction_{t}"] = prediction[:, t]
        for name in ("high_candidate", "direct_high", "low_reference", "improved"):
            rows.append(dict(model=name, target=target, count=int(mask.sum()),
                             **direct.base.metric(y[mask], frame[f"{name}_prediction_{t}"].to_numpy()[mask])))
    frame.to_csv(out / "原96构型回溯逐点.csv", index=False)
    (out / "原96构型回溯评价.json").write_text(json.dumps(dict(
        role="Retrospective comparison after diagnosing this batch; not a new blind test",
        metrics=rows), indent=2), encoding="utf8")


def run(out, extra, tests, prepare_only):
    out.mkdir(parents=True, exist_ok=True)
    selection_path = out / "千构型模型选择.json"
    if not selection_path.exists():
        initial_selection(out)
    plan_path = out / "冻结采样与计算配置.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(make_plan(out, extra, tests), indent=2), encoding="utf8")
    retrospective(out)
    if prepare_only:
        return
    original.prior.solver.WORKER = original.WORKER
    original.prior.driver._run_case = original.native
    original.prior.driver.polar_quality = original.prior.diagnostic_quality
    original.prior.driver.fit_models = fit_models
    original.prior.driver.run(out, (1000+extra, tests, 0))
    result = json.loads((out / "中网格盲测评估.json").read_text(encoding="utf8"))
    result.update(mesh_reference="legacy", wake_protocol="free-wake-legacy-v1",
                  labels_unchanged=True, grid_convergence_claimed=False,
                  comparison=dict(high_candidate="expanded training",
                                  direct_high="improved 1000-case model",
                                  low_reference="workbench 289-case model"))
    (out / "主动补样独立评价.json").write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--extra", type=int, default=512)
    parser.add_argument("--tests", type=int, default=96)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.extra, args.tests, args.prepare_only)
