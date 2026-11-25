"""Compare linear and constant tree leaves without using external test labels."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

import 研究_局部细分尾迹代理20261017 as prior
from 线性叶节点气动代理 import LinearLeafAero, features, predict_trees

ROOT = prior.ROOT
OUT = ROOT / "运行结果/线性叶节点代理_20261017_v2"
CONFIGS = [
    dict(mode=mode, linear=linear, leaves=leaves, min_data=min_data,
         penalty=5., linear_penalty=20., iterations=700)
    for mode in ("raw", "interference") for linear in (False, True)
    for leaves, min_data in ((4, 30), (8, 40), (16, 60))
]


def fit(config, x, y):
    f = features(x, config["mode"])
    center, scale = f.mean(0), np.maximum(f.std(0), 1e-9)
    estimator = lgb.LGBMRegressor(
        n_estimators=config["iterations"], learning_rate=.035,
        num_leaves=config["leaves"], min_child_samples=config["min_data"],
        reg_lambda=config["penalty"], linear_tree=config["linear"],
        linear_lambda=config["linear_penalty"], verbosity=-1, n_jobs=1,
        deterministic=True, force_col_wise=True, random_state=2026101742)
    estimator.fit((f-center)/scale, y)
    return dict(estimator=estimator, center=center, scale=scale, config=config)


def predict(model, x):
    z = (features(x, model["config"]["mode"])-model["center"])/model["scale"]
    return model["estimator"].booster_.predict(z, num_threads=1)


def portable(model):
    dump = model["estimator"].booster_.dump_model()
    if dump["num_tree_per_iteration"] != 1 or dump["average_output"]:
        raise ValueError("Only additive scalar tree models are supported")
    return dict(features=model["config"]["mode"], center=model["center"].tolist(),
                scale=model["scale"].tolist(),
                trees=[tree["tree_structure"] for tree in dump["tree_info"]])


def task(args):
    index, split, fold, a, b, x, y = args
    start = time.monotonic()
    model = fit(CONFIGS[index], x[a], y[a])
    p = predict(model, x[b])
    return dict(config=index, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=p.tolist(),
                train_r2=prior.prior.previous.prior.direct.base.metric(y[a], predict(model, x[a]))["r2"],
                seconds=time.monotonic()-start,
                **prior.prior.previous.prior.direct.base.metric(y[b], p))


def main(out, workers=12):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(prior.prior.previous.DATA)
    valid = frame.wake_quality_pass.to_numpy(bool).copy()
    valid &= np.isfinite(frame[prior.TARGET].to_numpy(float))
    sub = frame.loc[valid]
    x, y = sub[prior.prior.FEATURES].to_numpy(float), sub[prior.TARGET].to_numpy(float)
    source_paths = [Path(__file__), ROOT / "核心模型/线性叶节点气动代理.py",
                    prior.prior.previous.DATA, prior.OUT / "模型选择.json",
                    prior.OUT / "交叉验证逐折.json", prior.OUT / "局部几何候选代理.json"]
    manifest = dict(configs=CONFIGS, external_labels_used=False, lightgbm_version=lgb.__version__,
                    hashes={str(p): prior.digest(p) for p in source_paths},
                    selection="Training random/spatial OOF RMSE; includes frozen references",
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    frozen = out / "配置冻结.json"
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf8"))
        if old["configs"] != CONFIGS or old["hashes"] != manifest["hashes"]:
            raise RuntimeError("Linear-leaf sources changed")
    else:
        frozen.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    groups = prior.prior.previous.prior.direct.base.geometry_groups(x)
    jobs = []
    for split, folds in (
        ("random", KFold(3, shuffle=True, random_state=2026101722).split(x)),
        ("spatial", GroupKFold(3).split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            jobs.extend((i, split, fold, a, b, x, y) for i in range(len(CONFIGS)))
    records_path = out / "交叉验证逐折.json"
    records = json.loads(records_path.read_text(encoding="utf8")) if records_path.exists() else []
    done = {(r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:3] not in done]
    total = len(jobs)+len(records)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            records_path.write_text(json.dumps(records), encoding="utf8")
            (out / "进度.json").write_text(json.dumps(dict(stage="training_cv", completed=len(records),
                requested=total, elapsed_s=time.monotonic()-started)), encoding="utf8")
            print("Linear leaf CV", len(records), "/", total, flush=True)
    n = len(CONFIGS)
    predictions = [prior.prior.previous.prior.records_matrix(records, len(y), i) for i in range(n)]
    predictions.append(prior.reference_oof(len(y)))
    local_selection = json.loads((prior.OUT / "模型选择.json").read_text(encoding="utf8"))
    local_records = json.loads((prior.OUT / "交叉验证逐折.json").read_text(encoding="utf8"))
    local_oof = np.zeros((2, len(y)))
    for component in local_selection["components"]:
        i, weight = component["config"], component["weight"]
        local_oof += weight * (predictions[n] if i == len(prior.CONFIGS)
                              else prior.prior.previous.prior.records_matrix(local_records, len(y), i))
    predictions.append(local_oof)
    score, chosen, oof = prior.prior.previous.prior.select_mixture(predictions, y)
    local_payload = json.loads((prior.OUT / "局部几何候选代理.json").read_text(encoding="utf8"))
    balanced_payload = dict(local_payload, wake_components=[
        dict(kind="reference", weight=1., model=json.loads(
            (prior.prior.OUT / "密度均衡候选代理.json").read_text(encoding="utf8"))["targets"][1])])
    probe = x[np.linspace(0, len(x)-1, 40).astype(int)]
    probe = np.vstack((probe, (probe + np.roll(probe, 1, axis=0))/2))
    components, checks = [], []
    for i, weight in chosen:
        if i >= n:
            components.append(dict(kind="reference", weight=weight,
                                   model=balanced_payload if i == n else local_payload))
        else:
            model = fit(CONFIGS[i], x, y)
            (out / f"wake_config_{i}.txt").write_text(
                model["estimator"].booster_.model_to_string(), encoding="utf8")
            exported = portable(model)
            expected, actual = predict(model, probe), predict_trees(exported, probe)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
            checks.append(dict(config=i, max_abs_difference=float(np.max(abs(actual-expected)))))
            components.append(dict(kind="linear-tree", weight=weight, model=exported, config=i))
    payload = dict(schema="ptdl-linear-leaf-aero-v1", lift_reference=local_payload,
                   training_samples=len(frame), wake_valid_samples=len(y),
                   wake_components=components, wake_interval_std=max(1e-9, float(
                       np.quantile(abs(oof-y), .9)/1.645)), automatic_production_promotion=False)
    path = out / "线性叶节点候选代理.json"
    LinearLeafAero(payload).to_json(path)
    restored = LinearLeafAero.from_json(path)
    np.testing.assert_allclose(restored.predict_many(probe), LinearLeafAero(payload).predict_many(probe),
                               rtol=1e-9, atol=1e-9)
    summary = []
    for i, values in enumerate(predictions):
        for split, p in zip(("random", "spatial"), values):
            summary.append(dict(config=i, split=split,
                **prior.prior.previous.prior.direct.base.metric(y, p),
                **(CONFIGS[i] if i < n else {"mode": "reference"})))
    pd.DataFrame(summary).to_csv(out / "训练交叉验证汇总.csv", index=False)
    selection = dict(components=[dict(config=i, weight=w) for i, w in chosen], score=score,
                     oof=[dict(split=s, **prior.prior.previous.prior.direct.base.metric(y, p))
                          for s, p in zip(("random", "spatial"), oof)],
                     model_path=str(path), model_sha256=prior.digest(path), export_checks=checks,
                     frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                     external_labels_used=False, automatic_production_promotion=False)
    (out / "模型选择.json").write_text(json.dumps(selection, indent=2), encoding="utf8")
    (out / "进度.json").write_text(json.dumps(dict(stage="completed", completed=len(records),
        requested=total, elapsed_s=time.monotonic()-started)), encoding="utf8")
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.workers)
