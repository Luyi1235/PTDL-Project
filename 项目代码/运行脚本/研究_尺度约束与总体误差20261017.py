"""Compare scale normalization and sampling weights on a frozen uniform expansion."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

import 研究_局部细分尾迹代理20261017 as prior
from 尺度约束气动代理 import ScalePopulationAero, geometry_transform
from 局部几何气动代理 import numeric_tree_predict
from 导出_几何交互代理20261017 import portable as historical_portable

ROOT = prior.ROOT
OUT = ROOT / "运行结果/尺度约束与总体误差_20261017"
BATCH = ROOT / "运行结果/均匀覆盖3072代理_20261017"
FEATURES, TARGET = prior.prior.FEATURES, prior.TARGET
CONFIGS = [
    dict(family="tree", transform=transform, weighting=weighting, depth=depth,
         penalty=penalty, iterations=iterations, ordered=ordered)
    for transform in ("identity", "canonical-area")
    for weighting in ("equal", "uniform4")
    for depth, penalty, iterations, ordered in (
        (6, 20., 1500, False), (6, 80., 900, True), (4, 40., 900, True))
] + [dict(family="historical-architecture", transform="identity", weighting="equal")]


def snapshot(out):
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "数据冻结.json"
    if frozen.exists():
        record = json.loads(frozen.read_text(encoding="utf8"))
        for path, value in record["hashes"].items():
            if prior.digest(path) != value:
                raise RuntimeError("Uniform snapshot source changed")
        return out / "训练标签冻结.csv"
    plan_path = BATCH / "冻结配置.json"
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    source = Path(plan["source_data"])
    rows, paths = [], []
    for job in [j for j in plan["jobs"] if j["role"] == "train"][:640]:
        path = BATCH / "solver" / job["sample"] / "record.json"
        row = json.loads(path.read_text(encoding="utf8"))
        if row["sample"] != job["sample"] or row["role"] != "train":
            raise RuntimeError("Incorrect uniform expansion record")
        np.testing.assert_allclose([row[k] for k in FEATURES],
                                   [job["design"][k] for k in FEATURES], atol=1e-10, rtol=1e-12)
        rows.append(row)
        paths.append(path)
    frame = pd.concat((pd.read_csv(source), pd.DataFrame(rows)), ignore_index=True)
    if frame.duplicated(FEATURES).any():
        raise RuntimeError("Duplicate snapshot geometry")
    data = out / "训练标签冻结.csv"
    frame.to_csv(data, index=False)
    frozen.write_text(json.dumps(dict(
        hashes={str(p): prior.digest(p) for p in [source, plan_path, data, *paths]},
        original_samples=len(frame)-len(rows), new_samples=len(rows),
        all_failed_records_retained=True, sampled_by_external_errors=False,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    return data


def weights(acquisition, mode):
    acquisition = np.asarray(acquisition, str)
    if mode == "equal":
        return np.ones(len(acquisition))
    if mode != "uniform4":
        raise ValueError("Unknown population weighting")
    values = np.where(acquisition == "uniform_feasible_coverage", 4., 1.)
    return values / values.mean()


def fit(config, x, y, acquisition):
    if config["family"] == "historical-architecture":
        settings = [
            dict(method="cat", mode="interference", depth=6, penalty=20.,
                 transform="identity", iterations=1500),
            dict(method="boost", mode="interference", leaves=4, penalty=5.,
                 transform="identity", iterations=600)]
        return dict(config=config, models=[prior.prior.previous.fit(s, x, y) for s in settings])
    transformed = geometry_transform(x, config["transform"])
    f = prior.prior.previous.features(transformed, "interference")
    estimator = CatBoostRegressor(
        loss_function="RMSE", iterations=config["iterations"], depth=config["depth"],
        learning_rate=.035, l2_leaf_reg=config["penalty"],
        boosting_type="Ordered" if config["ordered"] else "Plain",
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        random_seed=2026101751, thread_count=1, verbose=False, allow_writing_files=False)
    estimator.fit(f, y, sample_weight=weights(acquisition, config["weighting"]))
    return dict(config=config, estimator=estimator)


def predict(fitted, x):
    config = fitted["config"]
    if config["family"] == "historical-architecture":
        return sum(w*prior.prior.previous.predict(m, x)
                   for w, m in zip((.75, .25), fitted["models"]))
    transformed = geometry_transform(x, config["transform"])
    return fitted["estimator"].predict(prior.prior.previous.features(transformed, "interference"))


def task(args):
    index, split, fold, a, b, x, y, acquisition = args
    started = time.monotonic()
    fitted = fit(CONFIGS[index], x[a], y[a], acquisition[a])
    p = predict(fitted, x[b])
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=p.tolist(),
                train_r2=prior.prior.previous.prior.direct.base.metric(y[a], predict(fitted, x[a]))["r2"],
                seconds=time.monotonic()-started,
                **prior.prior.previous.prior.direct.base.metric(y[b], p))


def select(predictions, y, population):
    population = np.asarray(population, bool)
    if population.sum() < 30:
        raise ValueError("Insufficient held-out uniform training geometries")

    def score(p):
        return float(np.sqrt(np.mean((p[:, population]-y[None, population])**2, axis=1)).mean())
    choices = [(score(p), [(i, 1.)], p) for i, p in enumerate(predictions)]
    for i in range(len(predictions)):
        for j in range(i+1, len(predictions)):
            for weight in (.25, .5, .75):
                p = weight*predictions[i]+(1-weight)*predictions[j]
                choices.append((score(p), [(i, weight), (j, 1-weight)], p))
    return min(choices, key=lambda row: row[0])


def run(out, workers):
    data = snapshot(out)
    frame = pd.read_csv(data)
    valid = np.isfinite(frame[TARGET].to_numpy(float))
    valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
    sub = frame.loc[valid]
    x, y = sub[FEATURES].to_numpy(float), sub[TARGET].to_numpy(float)
    acquisition = sub.selection_basis.to_numpy(str)
    population = acquisition == "uniform_feasible_coverage"
    sources = [Path(__file__), ROOT / "核心模型/尺度约束气动代理.py",
               ROOT / "核心模型/局部几何气动代理.py", data, prior.prior.REFERENCE]
    manifest = dict(configs=CONFIGS, hashes={str(p): prior.digest(p) for p in sources},
                    raw_count=len(frame), wake_valid_count=len(y),
                    uniform_validation_count=int(population.sum()), external_labels_used=False,
                    scoring="Mean random/spatial OOF RMSE on the uniform training population",
                    other_population_metrics_also_retained=True,
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    frozen = out / "训练配置冻结.json"
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf8"))
        if old["configs"] != CONFIGS or old["hashes"] != manifest["hashes"]:
            raise RuntimeError("Scale-population experiment changed")
    else:
        frozen.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    groups = prior.prior.previous.prior.direct.base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101752).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, x, y, acquisition) for i in range(len(CONFIGS)))
    records_path = out / "交叉验证逐折.json"
    records = json.loads(records_path.read_text(encoding="utf8")) if records_path.exists() else []
    completed = {(r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:3] not in completed]
    requested, started = len(jobs)+len(records), time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            records_path.write_text(json.dumps(records), encoding="utf8")
            (out / "进度.json").write_text(json.dumps(dict(
                stage="training_cv", completed=len(records), requested=requested,
                elapsed_s=time.monotonic()-started)), encoding="utf8")
            print("Scale-population CV", len(records), "/", requested, flush=True)
    predictions = [prior.prior.previous.prior.records_matrix(records, len(y), i)
                   for i in range(len(CONFIGS))]
    score, chosen, oof = select(predictions, y, population)
    summaries = []
    for index, values in enumerate(predictions):
        for split, p in zip(("random", "spatial"), values):
            for cohort, mask in (("all_training", np.ones(len(y), bool)),
                                 ("uniform_training", population), ("other_training", ~population)):
                summaries.append(dict(config=index, split=split, cohort=cohort, count=int(mask.sum()),
                    **CONFIGS[index], **prior.prior.previous.prior.direct.base.metric(y[mask], p[mask])))
    pd.DataFrame(summaries).to_csv(out / "训练交叉验证汇总.csv", index=False)
    interval = max(1e-9, float(np.quantile(abs(oof[:, population]-y[None, population]), .9)/1.645))
    probe = x[np.linspace(0, len(x)-1, 40).astype(int)]
    probe = np.vstack((probe, (probe+np.roll(probe, 1, axis=0))/2))
    components, expected, checks = [], np.zeros(len(probe)), []
    for i, weight in chosen:
        config = CONFIGS[i]
        fitted = fit(config, x, y, acquisition)
        if config["family"] == "historical-architecture":
            models = [historical_portable(m, interval) for m in fitted["models"]]
            component = dict(kind="target", transform="identity", weight=weight,
                             model=dict(kind="weighted-ensemble", weights=[.75, .25],
                                        models=models, interval_std=interval))
            actual = np.array([prior.prior.previous.prior.direct.base.target_mean(
                component["model"], row[None]) for row in probe])
        else:
            fitted["estimator"].save_model(str(out / f"wake_config_{i}.cbm"))
            model = prior.export(fitted["estimator"], "interference")
            actual = numeric_tree_predict(model, geometry_transform(probe, config["transform"]))
            component = dict(kind="tree", transform=config["transform"], weight=weight, model=model)
        native = predict(fitted, probe)
        np.testing.assert_allclose(actual, native, rtol=1e-9, atol=1e-9)
        expected += weight*native
        checks.append(dict(config=i, max_abs_difference=float(np.max(abs(native-actual)))))
        component["config"] = i
        components.append(component)
    baseline = json.loads(prior.prior.REFERENCE.read_text(encoding="utf8"))
    payload = dict(schema="ptdl-scale-population-aero-v1", baseline=baseline,
                   training_samples=len(frame), wake_valid_samples=len(y), wake_components=components,
                   wake_interval_std=interval, canonical_area_m2=30.,
                   automatic_production_promotion=False,
                   interval_scope="Uniform training OOF errors; not calibrated posterior probability")
    path = out / "尺度与总体加权候选代理.json"
    ScalePopulationAero(payload).to_json(path)
    restored = ScalePopulationAero.from_json(path)
    np.testing.assert_allclose(restored.predict_many(probe)[:, 1], expected, rtol=1e-9, atol=1e-9)
    result = dict(selected=[dict(config=i, weight=w, settings=CONFIGS[i]) for i, w in chosen],
                  score=score, raw_training_count=len(frame), valid_count=len(y),
                  uniform_valid_count=int(population.sum()),
                  oof=[dict(split=s, cohort=c, **prior.prior.previous.prior.direct.base.metric(y[m], p[m]))
                       for s, p in zip(("random", "spatial"), oof)
                       for c, m in (("all_training", np.ones(len(y), bool)),
                                    ("uniform_training", population))],
                  model_sha256=prior.digest(path), export_checks=checks, external_labels_used=False,
                  selected_at=time.strftime("%Y-%m-%d %H:%M:%S"), automatic_production_promotion=False)
    (out / "模型选择.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    (out / "进度.json").write_text(json.dumps(dict(stage="completed", completed=len(records),
        requested=requested, elapsed_s=time.monotonic()-started)), encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.workers)
