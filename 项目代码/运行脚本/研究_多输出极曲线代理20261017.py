"""Learn correlated force-coefficient profiles before reconstructing wake efficiency."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
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

ROOT = direct.ROOT
OUT = ROOT / "运行结果/多输出极曲线代理_20261017"
ANGLES = np.array([-3., -1., 1., 3., 5.])
CONFIGS = [dict(mode=mode, depth=depth, reconstruction=kind)
           for mode in ("physics", "interference") for depth in (4, 6)
           for kind in ("polynomial", "pointwise")]


def reconstruct(values, ar, kind, fallback):
    if kind == "polynomial":
        curves = np.array([[np.polyval(c, ANGLES) for c in row.reshape(3, 3)] for row in values])
    else:
        curves = values.reshape(len(values), 3, 5)
    result = fallback.copy()
    for i, (cl, clw, cdw) in enumerate(curves):
        eligible = (abs(clw) >= .15) & (cdw > 0)
        if eligible.sum() >= 3:
            result[i] = np.median(cl[eligible]**2/(np.pi*ar[i]*cdw[eligible]))
    return result


def force_values(polar, kind):
    if kind == "polynomial":
        return np.array([np.polyfit(ANGLES, row[:, :3], 2).T.reshape(-1) for row in polar])
    return polar[:, :, :3].transpose(0, 2, 1).reshape(len(polar), -1)


def estimator(depth, multi=False):
    return CatBoostRegressor(
        loss_function="MultiRMSE" if multi else "RMSE", iterations=1200,
        depth=depth, learning_rate=.035, l2_leaf_reg=10., random_strength=.5,
        bootstrap_type="Bayesian", bagging_temperature=.5, random_seed=2026101724,
        thread_count=1, verbose=False, allow_writing_files=False)


def fit(config, x, y, polar):
    f = direct.features(x, config["mode"])
    values = force_values(polar, config["reconstruction"])
    center, scale = values.mean(0), np.maximum(values.std(0), 1e-10)
    model = estimator(config["depth"], True).fit(f, (values-center)/scale)
    fallback = estimator(4).fit(f, y)
    return dict(config=config, estimator=model, center=center, scale=scale, fallback=fallback)


def predict(model, x):
    f = direct.features(x, model["config"]["mode"])
    values = model["center"]+model["scale"]*model["estimator"].predict(f)
    return reconstruct(values, x[:, 1], model["config"]["reconstruction"], model["fallback"].predict(f))


def task(args):
    index, split, fold, a, b, x, y, polar = args
    fitted = fit(CONFIGS[index], x[a], y[a], polar[a])
    prediction = predict(fitted, x[b])
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
                prediction=prediction.tolist(), actual=y[b].tolist(),
                **direct.prior.direct.base.metric(y[b], prediction))


def run(workers):
    OUT.mkdir(exist_ok=True)
    frame, x, y, polar = direct.prior.data(pd.read_csv(direct.DATA))
    reconstruction = []
    for kind in ("polynomial", "pointwise"):
        values = force_values(polar, kind)
        prediction = reconstruct(values, x[:, 1], kind, y)
        reconstruction.append(dict(kind=kind, **direct.prior.direct.base.metric(y, prediction)))
    (OUT / "重构误差.json").write_text(json.dumps(reconstruction, indent=2), encoding="utf8")
    groups = direct.prior.direct.base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101722).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, x, y, polar) for i in range(len(CONFIGS)))
    (OUT / "配置冻结.json").write_text(json.dumps(dict(
        configs=CONFIGS, count=len(y), external_labels_used=False,
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(direct.__file__), direct.DATA,
                          ROOT / "核心模型/气动几何特征.py")},
        selection="Training-only mean random/spatial OOF RMSE"), indent=2), encoding="utf8")
    start = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, args) for args in jobs]):
            records.append(future.result())
            (OUT / "交叉验证逐折.json").write_text(json.dumps(records), encoding="utf8")
            rows = []
            for i in range(len(CONFIGS)):
                for split in ("random", "spatial"):
                    parts = [r for r in records if r["config"] == i and r["split"] == split]
                    if len(parts) == 3:
                        rows.append(dict(config=i, split=split, **CONFIGS[i],
                            **direct.prior.direct.base.metric(
                                np.concatenate([r["actual"] for r in parts]),
                                np.concatenate([r["prediction"] for r in parts]))))
            pd.DataFrame(rows).to_csv(OUT / "训练交叉验证汇总.csv", index=False)
            (OUT / "进度.json").write_text(json.dumps(dict(
                stage="training_cv", completed=len(records), requested=len(jobs),
                elapsed_s=time.monotonic()-start)), encoding="utf8")
            print("Multioutput polar CV", len(records), "/", len(jobs), flush=True)
    predictions = [direct.prior.records_matrix(records, len(y), i) for i in range(len(CONFIGS))]
    score, components, oof = direct.prior.select_mixture(predictions, y)
    chosen = []
    for i, weight in components:
        model = fit(CONFIGS[i], x, y, polar)
        path = OUT / f"polar_config_{i}.joblib"
        joblib.dump(model, path)
        chosen.append(dict(config=i, settings=CONFIGS[i], weight=weight, path=str(path),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT / "模型选择.json").write_text(json.dumps(dict(
        selected=chosen, score=score, external_labels_used=False,
        oof=[dict(split=s, **direct.prior.direct.base.metric(y, p))
             for s, p in zip(("random", "spatial"), oof)]), indent=2), encoding="utf8")
    (OUT / "进度.json").write_text(json.dumps(dict(stage="completed", completed=len(records),
        requested=len(jobs), elapsed_s=time.monotonic()-start)), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.workers)
