"""Test whether solver-free mesh topology features reduce aerodynamic surrogate error."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

import 研究_几何交互代理20261017 as direct

OUT = direct.ROOT / "运行结果/网格辅助代理_20261017"
MESH = direct.ROOT / "运行结果/几何网格特征_20261017/训练网格描述.csv"
COLS = ["node_count", "loop_count", "surface_count"]
CONFIGS = [dict(mode=mode, depth=depth) for mode in ("physics", "interference") for depth in (4, 6)]


def fit(config, x, y, mesh):
    f = np.column_stack((direct.features(x, config["mode"]), mesh))
    estimator = CatBoostRegressor(
        loss_function="RMSE", iterations=1500, depth=config["depth"], learning_rate=.035,
        l2_leaf_reg=10., random_strength=.5, bootstrap_type="Bayesian", bagging_temperature=.5,
        random_seed=2026101725, thread_count=1, verbose=False, allow_writing_files=False).fit(f, y)
    return dict(config=config, estimator=estimator)


def predict(model, x, mesh):
    return model["estimator"].predict(np.column_stack((direct.features(x, model["config"]["mode"]), mesh)))


def task(args):
    target, config, split, fold, a, b, x, y, mesh = args
    model = fit(CONFIGS[config], x[a], y[a], mesh[a])
    prediction = predict(model, x[b], mesh[b])
    return dict(target=target, config=config, split=split, fold=fold, validation=b.tolist(),
                prediction=prediction.tolist(), actual=y[b].tolist(),
                **direct.prior.direct.base.metric(y[b], prediction))


def main():
    OUT.mkdir(exist_ok=True)
    frame = pd.read_csv(direct.DATA).merge(pd.read_csv(MESH)[["sample", *COLS]], on="sample",
                                         how="left", validate="one_to_one")
    arrays, jobs = {}, []
    for target in direct.TARGETS:
        mask = np.isfinite(frame[target].to_numpy(float))
        if target == direct.TARGETS[1]:
            mask &= frame.wake_quality_pass.to_numpy(bool)
        sub = frame.loc[mask]
        x, y, mesh = sub[direct.FEATURES].to_numpy(float), sub[target].to_numpy(float), sub[COLS].to_numpy(float)
        if not np.isfinite(mesh).all():
            raise ValueError("Missing geometry descriptor")
        arrays[target] = (x, y, mesh)
        groups = direct.prior.direct.base.geometry_groups(x)
        for split, folds in (
            ("random", KFold(3, shuffle=True, random_state=2026101722).split(x)),
            ("spatial", GroupKFold(3).split(x, groups=groups)),
        ):
            for fold, (a, b) in enumerate(folds):
                jobs.extend((target, i, split, fold, a, b, x, y, mesh) for i in range(len(CONFIGS)))
    (OUT / "配置冻结.json").write_text(json.dumps(dict(
        configs=CONFIGS, external_labels_used=False, inputs="Geometry + geometry-only mesh counts",
        full_force_solver_required_at_prediction=False, geometry_preprocessing_required=True,
        hashes={str(p.relative_to(direct.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), MESH, direct.DATA)}), indent=2), encoding="utf8")
    start = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(task, args) for args in jobs]):
            records.append(future.result())
            (OUT / "交叉验证逐折.json").write_text(json.dumps(records), encoding="utf8")
            rows = []
            for target in direct.TARGETS:
                for i in range(len(CONFIGS)):
                    for split in ("random", "spatial"):
                        parts = [r for r in records if (r["target"], r["config"], r["split"]) == (target, i, split)]
                        if len(parts) == 3:
                            rows.append(dict(target=target, config=i, split=split, **CONFIGS[i],
                                **direct.prior.direct.base.metric(
                                    np.concatenate([r["actual"] for r in parts]),
                                    np.concatenate([r["prediction"] for r in parts]))))
            pd.DataFrame(rows).to_csv(OUT / "训练交叉验证汇总.csv", index=False)
            (OUT / "进度.json").write_text(json.dumps(dict(stage="training_cv",
                completed=len(records), requested=len(jobs), elapsed_s=time.monotonic()-start)), encoding="utf8")
            print("Mesh assisted CV", len(records), "/", len(jobs), flush=True)
    selected = []
    for target in direct.TARGETS:
        x, y, mesh = arrays[target]
        results = [r for r in records if r["target"] == target]
        predictions = [direct.prior.records_matrix(results, len(y), i) for i in range(len(CONFIGS))]
        score, components, oof = direct.prior.select_mixture(predictions, y)
        chosen = []
        for i, weight in components:
            model = fit(CONFIGS[i], x, y, mesh)
            path = OUT / f"target_{direct.TARGETS.index(target)}_config_{i}.joblib"
            joblib.dump(model, path)
            chosen.append(dict(config=i, settings=CONFIGS[i], weight=weight, path=str(path),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        selected.append(dict(target=target, score=score, components=chosen,
                             oof=[dict(split=s, **direct.prior.direct.base.metric(y, p))
                                  for s, p in zip(("random", "spatial"), oof)]))
    (OUT / "模型选择.json").write_text(json.dumps(dict(selected=selected,
        external_labels_used=False, geometry_preprocessing_required=True), indent=2), encoding="utf8")
    (OUT / "进度.json").write_text(json.dumps(dict(stage="completed", completed=len(records),
        requested=len(jobs), elapsed_s=time.monotonic()-start)), encoding="utf8")


if __name__ == "__main__":
    main()
