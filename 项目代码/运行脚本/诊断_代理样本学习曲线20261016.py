"""Training-only learning curves with nested subsets and fixed validation folds."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 分目标气动高斯过程 import feature_map
from VSPAERO气动代理模型 import FEATURE_LABELS
from 训练_主动增量混合代理20261016 import metric

TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
DEFAULT_DATA = {
    "legacy": ROOT / "运行结果" / "百构型加密代理_20261016" / "一致性校核训练" / "low_labels" / "逐构型质量记录.csv",
    "medium": ROOT / "运行结果" / "百构型加密代理_20261016" / "中网格训练实算标签.csv",
}


def optimizer(objective, initial, bounds):
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": 100, "ftol": 1e-8})
    return result.x, result.fun


def run_task(task):
    x, y, indices, validation, meta = task
    features = feature_map(x, "physics")
    a, b = features[indices], features[validation]
    center, scale = a.mean(0), np.maximum(a.std(0), 1e-9)
    if meta["model"] == "Kriging":
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1., (.01, 100.))*Matern(
                np.ones(a.shape[1]), (.05, 50.), nu=1.5)+WhiteKernel(.001, (1e-8, 1.)),
            normalize_y=True, alpha=1e-8, optimizer=optimizer,
            n_restarts_optimizer=0, random_state=meta["seed"])
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=.06, max_iter=180, max_leaf_nodes=8, min_samples_leaf=7,
            l2_regularization=.1, early_stopping=False, random_state=meta["seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit((a-center)/scale, y[indices])
    prediction = model.predict((b-center)/scale)
    return dict(**meta, validation=validation.tolist(), actual=y[validation].tolist(),
                prediction=prediction.tolist(), **metric(y[validation], prediction))


def summarize(records, out):
    rows = []
    groups = {}
    for record in records:
        key = tuple(record[k] for k in ("dataset", "target", "model", "size", "repeat"))
        groups.setdefault(key, []).append(record)
    for key, items in groups.items():
        if len(items) != 5:
            continue
        y = np.concatenate([i["actual"] for i in items])
        p = np.concatenate([i["prediction"] for i in items])
        rows.append(dict(zip(("dataset", "target", "model", "size", "repeat"), key),
                         **metric(y, p)))
    repeated = pd.DataFrame(rows)
    if repeated.empty:
        return repeated
    repeated.to_csv(out / "重复划分学习曲线.csv", index=False)
    summary = repeated.groupby(["dataset", "target", "model", "size"], as_index=False).agg(
        r2=("r2", "mean"), r2_std=("r2", "std"), rmse=("rmse", "mean"),
        rmse_std=("rmse", "std"), mae=("mae", "mean"), repeats=("repeat", "count"))
    summary.to_csv(out / "学习曲线汇总.csv", index=False)
    return summary


def figures(frame, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=True)
    colors = {"Kriging": "#397f9c", "Boosting": "#cb7a64"}
    for row, dataset in enumerate(DEFAULT_DATA):
        for column, target in enumerate(TARGETS):
            ax = axes[row, column]
            for model, group in frame[(frame.dataset == dataset) & (frame.target == target)].groupby("model"):
                group = group.sort_values("size")
                x, y = group["size"].to_numpy(), group.r2.to_numpy()
                error = group.r2_std.fillna(0).to_numpy()
                ax.plot(x, y, "o-", color=colors[model], label="Kriging" if model == "Kriging" else "梯度提升")
                ax.fill_between(x, y-error, y+error, color=colors[model], alpha=.13)
            ax.axhline(.9, ls="--", lw=1, color="#a25c57")
            ax.set(title=("原网格" if dataset == "legacy" else "加密网格")
                   + (" · 升力斜率" if column == 0 else " · 尾迹效率"),
                   xlabel="每折实际训练构型数", ylabel="训练集折外 R²")
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "样本学习曲线.png", dpi=220)
    fig.savefig(out / "样本学习曲线.pdf")
    plt.close(fig)


def main(out, workers, repeats):
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    fingerprints = {}
    for dataset, path in DEFAULT_DATA.items():
        fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        if frame.duplicated(list(FEATURE_LABELS)).any():
            raise ValueError("Repeated geometry in training data")
        for target in TARGETS:
            valid = np.isfinite(frame[target].to_numpy(float))
            if target == TARGETS[1]:
                valid &= frame.wake_quality_pass.to_numpy(bool)
            subset = frame.loc[valid]
            x, y = subset[list(FEATURE_LABELS)].to_numpy(float), subset[target].to_numpy(float)
            maximum = len(y)-int(np.ceil(len(y)/5))
            sizes = sorted({n for n in (20, 40, 60, 80, 120, 160, 220, maximum) if n <= maximum})
            for repeat in range(repeats):
                seed = 2026101660+repeat
                for fold, (train, validation) in enumerate(KFold(5, shuffle=True, random_state=seed).split(x)):
                    ordered = np.random.default_rng(seed+1000*fold).permutation(train)
                    for size in sizes:
                        for model in ("Kriging", "Boosting"):
                            meta = dict(dataset=dataset, target=target, model=model, size=size,
                                        repeat=repeat, fold=fold, seed=seed, valid_count=len(y))
                            tasks.append((x, y, ordered[:size], validation, meta))
    manifest = dict(sources=fingerprints, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    external_test_labels_used=False, fold_count=5, repeats=repeats,
                    nested_training_subsets=True, tasks=len(tasks),
                    purpose="Fixed-model training-only learning curve; not a production-model score")
    (out / "诊断配置.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    records = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 20 == 0 or len(records) == len(tasks):
                (out / "进度.json").write_text(json.dumps(dict(
                    completed=len(records), requested=len(tasks), elapsed_s=time.monotonic()-started,
                    stage="running" if len(records) < len(tasks) else "completed"), indent=2), encoding="utf8")
                (out / "逐折预测.json").write_text(json.dumps(records), encoding="utf8")
                summarize(records, out)
                print("Learning curve", len(records), "/", len(tasks), flush=True)
    frame = summarize(records, out)
    figures(frame, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "样本学习曲线_20261016")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    main(args.output_dir, args.workers, args.repeats)
