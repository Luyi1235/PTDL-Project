"""Test cross-surface mesh proximity with unchanged labels and geometry-level folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

import 研究_精细树与核代理20261018 as previous
from 气动面元邻近特征 import panel_proximity, FEATURE_NAMES

ROOT = previous.ROOT
OUT = ROOT / "运行结果/面元邻近代理_20261018"
CONFIGS = [dict(mesh=mesh, depth=depth, penalty=penalty)
           for mesh in (False, True) for depth, penalty in ((4, 10.), (6, 20.), (6, 80.))]


def extract(row):
    path = Path(row["geometry_file"]).with_suffix(".vspgeom") if isinstance(
        row.get("geometry_file"), str) else None
    try:
        values = panel_proximity(path, row["wing_area_m2"], row["aspect_ratio"])
        return dict(sample=row["sample"], **dict(zip(FEATURE_NAMES, values)),
                    extraction_error=None)
    except Exception as exc:
        return dict(sample=row["sample"], **dict(zip(FEATURE_NAMES, np.zeros(len(FEATURE_NAMES)))),
                    extraction_error=repr(exc))


def fit(config, x, y, mesh):
    features = previous.previous.prior.prior.previous.features(x, "interference")
    if config["mesh"]:
        features = np.column_stack((features, mesh))
    model = CatBoostRegressor(iterations=1500, depth=config["depth"], learning_rate=.035,
        l2_leaf_reg=config["penalty"], loss_function="RMSE", rsm=.8,
        bootstrap_type="Bayesian", bagging_temperature=.5, random_strength=.5,
        random_seed=2026101816, thread_count=1, verbose=False, allow_writing_files=False)
    model.fit(features, y)
    return dict(config=config, estimator=model)


def predict(fitted, x, mesh):
    features = previous.previous.prior.prior.previous.features(x, "interference")
    if fitted["config"]["mesh"]:
        features = np.column_stack((features, mesh))
    return fitted["estimator"].predict(features)


def task(args):
    target, i, split, fold, a, b, x, y, mesh = args
    model = fit(CONFIGS[i], x[a], y[a], mesh[a])
    return dict(target=target, config=i, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=predict(model, x[b], mesh[b]).tolist())


def main(workers):
    OUT.mkdir(exist_ok=True)
    frame = pd.read_csv(previous.DATA)
    hashes = {str(p): previous.previous.prior.digest(p) for p in
              (Path(__file__), ROOT / "核心模型/气动面元邻近特征.py", previous.DATA)}
    frozen = OUT / "配置冻结.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf8"))["hashes"] != hashes:
            raise RuntimeError("Panel proximity sources changed")
    else:
        frozen.write_text(json.dumps(dict(hashes=hashes, configs=CONFIGS,
            descriptors_use_only_mesh_geometry=True, external_labels_used=False,
            unavailable_features_retained_as_zero_with_availability_flag=True)), encoding="utf8")
    features_path = OUT / "训练面元邻近特征.csv"
    if features_path.exists():
        features = pd.read_csv(features_path).set_index("sample").loc[frame["sample"]].reset_index()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            features = pd.DataFrame(pool.map(extract, frame.to_dict("records"), chunksize=12))
        features.to_csv(features_path, index=False)
    arrays, jobs = {}, []
    for target in previous.TARGETS:
        valid = np.isfinite(frame[target].to_numpy(float))
        if target == previous.TARGETS[1]:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        x, y = frame.loc[valid, previous.FEATURES].to_numpy(float), frame.loc[valid, target].to_numpy(float)
        mesh = features.loc[valid, FEATURE_NAMES].to_numpy(float)
        arrays[target] = x, y, mesh
        groups = previous.previous.prior.prior.previous.prior.direct.base.geometry_groups(x)
        for split, folds in (
            ("random", KFold(3, shuffle=True, random_state=2026101803).split(x)),
            ("spatial", GroupKFold(3).split(x, groups=groups))):
            for fold, (a, b) in enumerate(folds):
                jobs.extend((target, i, split, fold, a, b, x, y, mesh) for i in range(len(CONFIGS)))
    records_path = OUT / "交叉验证逐折.json"
    records = json.loads(records_path.read_text(encoding="utf8")) if records_path.exists() else []
    done = {(r["target"], r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:4] not in done]
    total, start = len(jobs)+len(records), time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            records_path.write_text(json.dumps(records), encoding="utf8")
            (OUT / "进度.json").write_text(json.dumps(dict(stage="training",
                completed=len(records), requested=total)), encoding="utf8")
    selected, summaries = [], []
    base = previous.previous.prior.prior.previous.prior.direct.base
    for target, (x, y, mesh) in arrays.items():
        sub = [r for r in records if r["target"] == target]
        p = [previous.previous.prior.prior.previous.prior.records_matrix(sub, len(y), i)
             for i in range(len(CONFIGS))]
        score, chosen, oof = previous.previous.prior.prior.previous.prior.select_mixture(p, y)
        components = []
        for i, weight in chosen:
            path = OUT / f"target_{previous.TARGETS.index(target)}_config_{i}.joblib"
            joblib.dump(fit(CONFIGS[i], x, y, mesh), path)
            components.append(dict(config=i, weight=weight, path=str(path)))
        selected.append(dict(target=target, score=score, components=components,
            oof=[dict(split=s, **base.metric(y, values))
                 for s, values in zip(("random", "spatial"), oof)]))
        summaries.extend(dict(target=target, config=i, split=s, **CONFIGS[i], **base.metric(y, values))
            for i, pair in enumerate(p) for s, values in zip(("random", "spatial"), pair))
    pd.DataFrame(summaries).to_csv(OUT / "训练比较.csv", index=False)
    (OUT / "模型选择.json").write_text(json.dumps(dict(selected=selected,
        requires_geometric_preprocessing=True, external_labels_used=False,
        missing_geometry_count=int(features.extraction_error.notna().sum()))), encoding="utf8")
    (OUT / "进度.json").write_text(json.dumps(dict(stage="completed",
        completed=total, requested=total, elapsed_s=time.monotonic()-start)), encoding="utf8")
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    main(args.workers)
