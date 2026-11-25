"""Share aerodynamic observations across angles with geometry-level nested validation."""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
from 分目标气动高斯过程 import feature_map, kernel_cross
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign
from 研究_代理正则化与嵌套验证20261016 import geometry_groups, selection_splits
from 训练_主动增量混合代理20261016 import metric

DATA = ROOT / "运行结果" / "百构型加密代理_20261016"
ANGLES = np.array([-3., -1., 1., 3., 5.])
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
CONFIGS = [
    dict(method="direct", mode=mode, noise=noise, transform="identity")
    for mode in ("raw", "physics") for noise in (.05, .2)
] + [
    dict(method="joint-angle", mode=mode, noise=noise, transform="log")
    for mode in ("raw", "dimensionless") for noise in (.05, .2)
] + [dict(method="joint-angle", mode="dimensionless", noise=.1, transform="identity")]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_training(path):
    frame = pd.read_csv(path)
    mask = frame.wake_quality_pass.to_numpy(bool).copy()
    mask &= np.isfinite(frame[list(TARGETS)].to_numpy(float)).all(1)
    frame = frame.loc[mask].reset_index(drop=True)
    if frame.duplicated(list(FEATURE_LABELS)).any():
        raise ValueError("Duplicated training geometry")
    polars, sources = [], {}
    for _, row in frame.iterrows():
        source = Path(row.geometry_file).parent / "vspaero_polar.csv"
        polar = pd.read_csv(source).set_index("alpha_deg").loc[ANGLES]
        values = polar[["cl", "cl_wake", "e_wake"]].to_numpy(float)
        working = (abs(values[:, 1]) >= .15) & (values[:, 2] > 0)
        original = float(np.median(values[working, 2]))
        if not np.isclose(original, row[TARGETS[1]], atol=1e-8, rtol=0):
            raise RuntimeError("Polar does not reconstruct the original Ew target")
        polars.append(values)
        sources[str(source)] = digest(source)
    return frame, np.array(polars), sources


def features(x, mode, angles=False):
    value = feature_map(x, mode)
    if not angles:
        return value
    return np.column_stack((np.repeat(value, len(ANGLES), axis=0),
                            np.tile(ANGLES, len(value))))


def optimize(objective, initial, bounds):
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 50, "ftol": 1e-7})
    return result.x, result.fun


def fit_gp(x, y, noise):
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-9)
    z = (x-center)/scale
    estimator = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (.02, 30.))*Matern(np.ones(x.shape[1]), (.15, 30.), nu=1.5)
        + WhiteKernel(noise, "fixed"), alpha=1e-9, normalize_y=True, optimizer=optimize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(z, y)
    pack = dict(center=center.tolist(), scale=scale.tolist(), train=z.tolist(),
                length=np.asarray(estimator.kernel_.k1.k2.length_scale).tolist(),
                amplitude=float(estimator.kernel_.k1.k1.constant_value),
                mean=np.atleast_1d(estimator._y_train_mean).tolist(),
                output_scale=np.atleast_1d(estimator._y_train_std).tolist(),
                dual=np.atleast_2d(estimator.alpha_).tolist())
    if np.asarray(estimator.alpha_).ndim == 1:
        pack["dual"] = estimator.alpha_[:, None].tolist()
    np.testing.assert_allclose(gp_predict(pack, x), np.asarray(estimator.predict(z)).reshape(len(x), -1),
                               rtol=1e-9, atol=1e-9)
    return pack


def gp_predict(model, x):
    z = (x-model["center"])/model["scale"]
    cross = kernel_cross(z, np.asarray(model["train"]), np.asarray(model["length"]),
                         model["amplitude"], "matern32")
    return np.asarray(model["mean"])+np.asarray(model["output_scale"])*(cross @ np.asarray(model["dual"]))


def fit(config, x, polars, y):
    direct = fit_gp(features(x, config["mode"]), y, config["noise"])
    pack = dict(config=config, direct=direct)
    if config["method"] == "direct":
        return pack
    z = features(x, config["mode"], angles=True)
    observations = polars.reshape(-1, 3)
    pack["lift"] = fit_gp(z, observations[:, :2], config["noise"])
    mask = (abs(observations[:, 1]) >= .15) & (observations[:, 2] > 0)
    wake = observations[mask, 2]
    if config["transform"] == "log":
        wake = np.log(wake)
    pack["wake"] = fit_gp(z[mask], wake, config["noise"])
    return pack


def predict(model, x):
    config = model["config"]
    result = gp_predict(model["direct"], features(x, config["mode"]))
    if config["method"] == "direct":
        return result
    z = features(x, config["mode"], angles=True)
    lift = gp_predict(model["lift"], z).reshape(len(x), len(ANGLES), 2)
    wake = gp_predict(model["wake"], z).reshape(len(x), len(ANGLES))
    if config["transform"] == "log":
        wake = np.exp(wake)
    angle = np.deg2rad(ANGLES-ANGLES.mean())
    result[:, 0] = lift[:, :, 0] @ angle / (angle @ angle)
    for i in range(len(x)):
        working = (abs(lift[i, :, 1]) >= .15) & (wake[i] > 0) & np.isfinite(wake[i])
        if working.sum() >= 3:
            result[i, 1] = np.median(wake[i, working])
    return result


def select(x, polars, y, groups):
    scores = []
    for index, config in enumerate(CONFIGS):
        errors = []
        for a, b in selection_splits(x, groups):
            predicted = predict(fit(config, x[a], polars[a], y[a]), x[b])
            errors.append(np.sqrt(np.mean((predicted-y[b])**2, axis=0)))
        scores.append(dict(config=index, mean_rmse=np.mean(errors, axis=0).tolist()))
    selected = [min(scores, key=lambda r: r["mean_rmse"][t])["config"] for t in range(2)]
    return selected, scores


def outer_task(task):
    x, polar, y, groups, a, b, metadata = task
    selected, scores = select(x[a], polar[a], y[a], groups[a])
    predictions = [predict(fit(config, x[a], polar[a], y[a]), x[b]) for config in CONFIGS]
    chosen = np.column_stack([predictions[index][:, t] for t, index in enumerate(selected)])
    return dict(**metadata, validation=b.tolist(), selected_config=selected, inner_scores=scores,
                actual=y[b].tolist(), prediction=chosen.tolist(),
                candidates=[p.tolist() for p in predictions])


def summarize(records, out):
    results = []
    for split in ("random", "spatial"):
        values = [r for r in records if r["split"] == split]
        if len(values) != 5:
            continue
        actual = np.vstack([r["actual"] for r in values])
        for method in [-1]+list(range(len(CONFIGS))):
            predicted = np.vstack([r["prediction"] if method == -1 else r["candidates"][method] for r in values])
            for t, target in enumerate(TARGETS):
                results.append(dict(split=split, config=method,
                                    method="nested-selected" if method == -1 else CONFIGS[method]["method"],
                                    target=target, **metric(actual[:, t], predicted[:, t])))
    pd.DataFrame(results).to_csv(out / "联合迎角嵌套验证.csv", index=False)


def composite_predict(payload, x):
    predictions = [predict(model, x) for model in payload["models"]]
    return np.column_stack([predictions[index][:, t]
                            for t, index in enumerate(payload["component_indices"])])


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    training = DATA / "中网格训练实算标签.csv"
    frame, polar, polar_hashes = read_training(training)
    x, y = frame[list(FEATURE_LABELS)].to_numpy(float), frame[list(TARGETS)].to_numpy(float)
    groups = geometry_groups(x)
    (out / "训练前配置.json").write_text(json.dumps(dict(
        train_sha256=digest(training), source_sha256=digest(__file__), polar_sources=polar_hashes,
        configs=CONFIGS, training_geometry_count=len(frame), angle_observation_count=len(frame)*5,
        all_angles_of_each_geometry_stay_in_one_fold=True,
        target_definition="Original CL slope and eligible VSPAERO Ew median",
        external_labels_used_for_selection=False), indent=2), encoding="utf8")
    tasks = []
    for split, folds in (
        ("random", KFold(5, shuffle=True, random_state=2026101701).split(x)),
        ("spatial", LeaveOneGroupOut().split(x, groups=groups)),
    ):
        for fold, (a, b) in enumerate(folds):
            tasks.append((x, polar, y, groups, a, b, dict(split=split, fold=fold)))
    records, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(outer_task, task) for task in tasks]):
            records.append(future.result())
            (out / "逐折嵌套记录.json").write_text(json.dumps(records), encoding="utf8")
            summarize(records, out)
            (out / "进度.json").write_text(json.dumps(dict(stage="nested_validation",
                completed=len(records), requested=len(tasks), elapsed_s=time.monotonic()-started)), encoding="utf8")
            print("nested", len(records), "/", len(tasks), flush=True)
    selected, scores = select(x, polar, y, groups)
    unique = list(dict.fromkeys(selected))
    payload = dict(schema="ptdl_joint_angle_research_v1", configs=CONFIGS,
                   models=[fit(CONFIGS[i], x, polar, y) for i in unique],
                   component_indices=[unique.index(i) for i in selected],
                   selected_configs=selected, inner_scores=scores,
                   training_geometry_count=len(frame), production_promotion=False)
    model_path = out / "联合迎角候选.json"
    model_path.write_text(json.dumps(payload), encoding="utf8")
    (out / "模型冻结.json").write_text(json.dumps(dict(sha256=digest(model_path),
        external_labels_used_for_selection=False, retrospective_test=True)), encoding="utf8")
    test = pd.read_csv(DATA / "中网格盲测逐点.csv")
    valid = test.wake_quality_pass.to_numpy(bool) & np.isfinite(test[list(TARGETS)].to_numpy(float)).all(1)
    test = test.loc[valid].reset_index(drop=True)
    tx, ty = test[list(FEATURE_LABELS)].to_numpy(float), test[list(TARGETS)].to_numpy(float)
    candidate = composite_predict(json.loads(model_path.read_text(encoding="utf8")), tx)
    reference = load_vsp_surrogate(DATA / "多精度候选代理.json")
    old = np.array([reference.predict(AirframeDesign(**dict(zip(FEATURE_LABELS, row)))) for row in tx])
    assessment = []
    for label, predicted in (("reference", old), ("joint-selected", candidate)):
        for t, target in enumerate(TARGETS):
            assessment.append(dict(model=label, target=target, count=len(test),
                                   **metric(ty[:, t], predicted[:, t])))
            test[f"{label}_prediction_{t}"] = predicted[:, t]
    test.to_csv(out / "留出回溯逐点.csv", index=False)
    pd.DataFrame(assessment).to_csv(out / "留出回溯比较.csv", index=False)
    (out / "进度.json").write_text(json.dumps(dict(stage="completed",
        completed=len(records), requested=len(tasks), elapsed_s=time.monotonic()-started)), encoding="utf8")
    print(json.dumps(assessment), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "联合迎角代理_20261017")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.output_dir, args.workers)
