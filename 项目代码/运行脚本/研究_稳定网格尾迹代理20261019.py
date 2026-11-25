"""Compare numerical deadbands using training-only random and spatial validation."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import traceback

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

import 研究_极曲线结构代理20261019 as profile
from 稳定网格尾迹代理 import StableMeshWakeAero, features
from 精细气动代理 import FineAeroSurrogate

study, utility = profile.previous, profile.previous.utility
OUT = study.ROOT / "运行结果/稳定网格尾迹代理_20261019"
CONFIGS = [
    dict(mesh=True, transform=transform, policy=policy, depth=depth, penalty=penalty,
         iterations=iterations, weighting=weighting, zero_tolerance=tolerance)
    for tolerance in (1e-8, 1e-10)
    for transform, policy, depth, penalty, iterations, weighting in (
        ("identity", "Ordered", 6, 40., 1300, "equal"),
        ("canonical-area", "Plain", 5, 20., 1800, "equal"),
        ("canonical-area", "Depthwise", 6, 60., 1300, "uniform-priority"),
    )
]


def fit(config, x, y, mesh, uniform, threads=2):
    options = dict(loss_function="RMSE", iterations=config["iterations"],
        depth=config["depth"], learning_rate=.035, l2_leaf_reg=config["penalty"],
        rsm=.5, bootstrap_type="Bayesian", bagging_temperature=.5,
        random_strength=.5, random_seed=2026101826, thread_count=threads,
        verbose=False, allow_writing_files=False)
    if config["policy"] == "Depthwise":
        options.update(grow_policy="Depthwise", min_data_in_leaf=30)
    else:
        options["boosting_type"] = config["policy"]
    weights = np.ones(len(y))
    if config["weighting"] == "uniform-priority":
        weights[~uniform] = .35
    model = CatBoostRegressor(**options)
    model.fit(features(x, mesh, config), y, sample_weight=weights/weights.mean())
    return model


def task(args):
    index, split, fold, a, b, data, uniform = args
    started = time.monotonic()
    x, y, mesh = (data[k] for k in ("x", "y", "mesh"))
    model = fit(CONFIGS[index], x[a], y[a], mesh[a], uniform[a])
    prediction = model.predict(features(x[b], mesh[b], CONFIGS[index]), thread_count=2)
    trained = model.predict(features(x[a], mesh[a], CONFIGS[index]), thread_count=2)
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
        actual=y[b].tolist(), prediction=prediction.tolist(),
        train_r2=study.helpers.direct.base.metric(y[a], trained)["r2"],
        seconds=time.monotonic()-started, **study.helpers.direct.base.metric(y[b], prediction))


def main():
    OUT.mkdir(exist_ok=True)
    data = profile.prepare(profile.OUT)
    frame = pd.read_csv(study.DATA)
    uniform = frame.selection_basis.to_numpy(str)[data["raw_indices"]] == "uniform_feasible_coverage"
    paths = [Path(__file__), study.ROOT / "核心模型/稳定网格尾迹代理.py",
        profile.OUT / "极曲线标签审计.json", study.OUT / "模型选择.json"]
    frozen = dict(configs=CONFIGS, hashes={str(p): utility.digest(p) for p in paths},
        label_definition_unchanged=True, scientific_features_from_geometry_only=True)
    freeze_path = OUT / "配置冻结.json"
    if freeze_path.exists():
        if study.batch.previous.read(freeze_path) != frozen:
            raise RuntimeError("Stable descriptor configuration changed")
    else:
        utility.save(freeze_path, frozen)
    utility.check_hashes(frozen["hashes"])
    groups = study.helpers.direct.base.geometry_groups(data["x"])
    records_path = OUT / "交叉验证逐折.json"
    records = study.batch.previous.read(records_path) if records_path.exists() else []
    completed = {(r["config"], r["split"], r["fold"]) for r in records}
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101803).split(data["x"])),
        ("spatial", GroupKFold(3).split(data["x"], groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, data, uniform)
                        for i in range(len(CONFIGS)) if (i, split, fold) not in completed)
    errors = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(task, job): job[:3] for job in jobs}
        for future in as_completed(futures):
            try:
                records.append(future.result())
                utility.save(records_path, records)
            except Exception:
                errors.append(dict(job=futures[future], error=traceback.format_exc()))
                utility.save(OUT / "失败折.json", errors)
            utility.save(OUT / "进度.json", dict(stage="training_cv", completed=len(records),
                requested=36, failed=len(errors), updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            print("Stable mesh CV", len(records), "/ 36; failed", len(errors), flush=True)
    if errors:
        raise RuntimeError("Stable descriptor fold failed")
    y = data["y"]
    predictions = [study.helpers.records_matrix(records, len(y), i) for i in range(len(CONFIGS))]
    references = profile.reference_oof(len(y))
    predictions.extend(references)
    score, chosen, oof = study.helpers.select_mixture(predictions, y)
    baseline = study.batch.previous.read(study.BASELINE)
    components = []
    for index, weight in chosen:
        if index < len(CONFIGS):
            estimator = fit(CONFIGS[index], data["x"], y, data["mesh"], uniform, threads=8)
            components.append(dict(kind="tree", estimator=estimator, config=CONFIGS[index],
                                   weight=weight))
        elif index == len(CONFIGS):
            components.append(dict(kind="fine", payload=baseline, weight=weight))
        else:
            name = ("uniform_risk", "all_risk")[index-len(CONFIGS)-1]
            components.append(dict(kind="prior", weight=weight,
                                   payload=joblib.load(study.OUT / (name+".joblib"))))
    payload = dict(schema="ptdl-stable-mesh-wake-v1", baseline=baseline,
                   wake_components=components, training_samples=4096)
    model_path = OUT / "稳定网格候选.joblib"
    joblib.dump(payload, model_path, compress=3)
    model = StableMeshWakeAero.load(model_path)
    probe = model.predict_many(data["x"][:20], data["mesh"][:20])
    np.testing.assert_allclose(probe[:, 0],
        FineAeroSurrogate(baseline).predict_many(data["x"][:20])[:, 0], atol=1e-12, rtol=0)
    summaries = []
    for index, values in enumerate(predictions):
        for split, prediction in zip(("random", "spatial"), values):
            summaries.append(dict(config=index, split=split,
                **study.helpers.direct.base.metric(y, prediction)))
    pd.DataFrame(summaries).to_csv(OUT / "模型比较.csv", index=False)
    utility.check_hashes(frozen["hashes"])
    utility.save(OUT / "模型选择.json", dict(score=score, selected=[
        dict(config=i, weight=w) for i, w in chosen], path=str(model_path),
        hashes={str(p): utility.digest(p) for p in (model_path, freeze_path,
            study.ROOT / "核心模型/稳定网格尾迹代理.py")},
        oof=[dict(split=s, **study.helpers.direct.base.metric(y, p))
             for s, p in zip(("random", "spatial"), oof)],
        test_labels_used_for_selection=False, automatic_production_promotion=False))
    utility.save(OUT / "进度.json", dict(stage="completed", completed=36, requested=36,
        updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        utility.save(OUT / "失败.json", dict(error=traceback.format_exc()))
        raise
