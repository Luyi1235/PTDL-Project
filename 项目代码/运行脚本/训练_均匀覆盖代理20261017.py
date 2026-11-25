"""Retrain portable tree ensembles on an expanded, frozen geometry dataset."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

import 研究_密度均衡与残差代理20261017 as balanced
from 导出_密度均衡与残差代理20261017 import portable as portable_balanced
from 导出_几何交互代理20261017 import portable as portable_previous
from 分目标气动高斯过程 import TargetwiseAeroGP, target_mean

ROOT = balanced.ROOT
FEATURES, TARGETS = balanced.FEATURES, balanced.TARGETS
CONFIGS = [
    dict(method="tree", mode=mode, depth=depth, penalty=penalty, weighting=weighting,
         loss="RMSE", ordered=ordered, iterations=1500)
    for mode, depth, penalty in (("physics", 4, 3.), ("interference", 6, 20.))
    for weighting in ("none", "density") for ordered in (False, True)
] + [
    dict(method="boost", mode=mode, leaves=leaves, penalty=penalty,
         transform="identity", iterations=600)
    for mode, leaves, penalty in (("physics", 7, 1.), ("interference", 4, 5.))
]


def fit(config, x, y, acquisition):
    if config["method"] == "tree":
        return balanced.fit_tree(config, x, y, acquisition)
    return balanced.previous.fit(config, x, y)


def predict(model, x):
    return (balanced.tree_predict(model, x) if model["config"]["method"] == "tree" else
            balanced.previous.predict(model, x))


def portable(model, interval):
    return (portable_balanced(model, interval) if model["config"]["method"] == "tree" else
            portable_previous(model, interval))


def task(args):
    target, index, split, fold, a, b, x, y, acquisition = args
    model = fit(CONFIGS[index], x[a], y[a], acquisition[a])
    p = predict(model, x[b])
    return dict(target=target, config=index, split=split, fold=fold, validation=b.tolist(),
                actual=y[b].tolist(), prediction=p.tolist(),
                **balanced.previous.prior.direct.base.metric(y[b], p))


def main(data, out, workers=16):
    out.mkdir(parents=True, exist_ok=True)
    paths = (Path(__file__), Path(balanced.__file__), Path(balanced.previous.__file__),
             data, balanced.REFERENCE)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frozen = dict(configs=CONFIGS, hashes=manifest, test_labels_used=False,
                  selection="Training random/spatial OOF RMSE; convex pair or single",
                  started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    frozen_path = out / "训练配置冻结.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text(encoding="utf8"))
        if existing["hashes"] != manifest or existing["configs"] != CONFIGS:
            raise RuntimeError("Expanded training sources changed")
    else:
        frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf8")
    frame = pd.read_csv(data)
    arrays, jobs = {}, []
    for target in TARGETS:
        valid = np.isfinite(frame[target].to_numpy(float))
        if target == TARGETS[1]:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        sub = frame.loc[valid]
        x, y = sub[FEATURES].to_numpy(float), sub[target].to_numpy(float)
        acquisition = sub.selection_basis.to_numpy(str)
        arrays[target] = x, y, acquisition
        groups = balanced.previous.prior.direct.base.geometry_groups(x)
        for split, folds in (
            ("random", KFold(3, shuffle=True, random_state=2026101732).split(x)),
            ("spatial", GroupKFold(3).split(x, groups=groups)),
        ):
            for fold, (a, b) in enumerate(folds):
                jobs.extend((target, i, split, fold, a, b, x, y, acquisition)
                            for i in range(len(CONFIGS)))
    records_path = out / "训练交叉验证逐折.json"
    records = json.loads(records_path.read_text(encoding="utf8")) if records_path.exists() else []
    done = {(r["target"], r["config"], r["split"], r["fold"]) for r in records}
    jobs = [j for j in jobs if j[:4] not in done]
    requested = len(records)+len(jobs)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(task, j) for j in jobs]):
            records.append(future.result())
            records_path.write_text(json.dumps(records), encoding="utf8")
            (out / "训练进度.json").write_text(json.dumps(dict(
                completed=len(records), requested=requested, elapsed_s=time.monotonic()-started)),
                encoding="utf8")
            print("Uniform expansion CV", len(records), "/", requested, flush=True)
    payload = copy.deepcopy(json.loads(balanced.REFERENCE.read_text(encoding="utf8")))
    payload["targets"] = []
    selected, summaries = [], []
    for target in TARGETS:
        x, y, acquisition = arrays[target]
        subset = [r for r in records if r["target"] == target]
        predictions = [balanced.previous.prior.records_matrix(subset, len(y), i)
                       for i in range(len(CONFIGS))]
        score, chosen, oof = balanced.previous.prior.select_mixture(predictions, y)
        interval = max(1e-9, float(np.quantile(abs(oof-y), .9)/1.645))
        models, components = [], []
        probe = x[::max(1, len(x)//40)]
        probe = np.vstack((probe, (probe[1:]+probe[:-1])/2))
        for index, weight in chosen:
            model = fit(CONFIGS[index], x, y, acquisition)
            path = out / f"target_{TARGETS.index(target)}_config_{index}.joblib"
            joblib.dump(model, path)
            exported = portable(model, interval)
            np.testing.assert_allclose(
                [target_mean(exported, row[None, :]) for row in probe],
                predict(model, probe), rtol=1e-9, atol=1e-9)
            models.append(exported)
            components.append(dict(config=index, settings=CONFIGS[index], weight=weight,
                                   path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        payload["targets"].append(dict(
            kind="weighted-ensemble", models=models, weights=[w for _, w in chosen], interval_std=interval))
        selected.append(dict(target=target, count=len(y), score=score, components=components))
        summaries.extend(dict(target=target, split=split, **balanced.previous.prior.direct.base.metric(y, p))
                         for split, p in zip(("random", "spatial"), oof))
    payload.update(
        training_samples=len(frame), high_training_samples=len(arrays[TARGETS[0]][1]),
        valid_training_counts={target: len(values[1]) for target, values in arrays.items()},
        selection="Uniform expansion, training-only random/spatial CV",
        automatic_production_promotion=False,
        uncertainty="Fixed absolute error bands from expanded training OOF predictions")
    path = out / "均匀覆盖候选代理.json"
    TargetwiseAeroGP(payload).to_json(path)
    TargetwiseAeroGP.from_json(path)
    pd.DataFrame(summaries).to_csv(out / "训练交叉验证汇总.csv", index=False)
    (out / "模型选择.json").write_text(json.dumps(dict(
        selected=selected, oof=summaries, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        test_labels_used=False, selected_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    main(args.data.resolve(), args.output_dir.resolve(), args.workers)
