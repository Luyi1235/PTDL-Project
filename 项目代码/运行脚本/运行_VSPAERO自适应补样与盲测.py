"""Select, run, and preserve an adaptive VSPAERO DOE with a sealed holdout set."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (  # noqa: E402
    FEATURE_LABELS,
    OSWALD_EFFICIENCY_DEFINITION,
    _basis,
    design_features,
)
from 机体参数化设计与气动阻力 import AirframeDesign, geometry_constraints  # noqa: E402


SOURCE = (
    ROOT
    / "气动代理模型数据"
    / "VSPAERO_DOE扩充样本_100点_Ew_物理选点_网格修复_20261006.csv"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "VSPAERO自适应补样盲测_20261006_批次02"
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
TRAIN_COUNT = 32
HOLDOUT_COUNT = 8
POOL_SIZE = 30000
SEED = 20261007


def _fit_compact_predict(x: np.ndarray, y: np.ndarray, pool: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-9)
    train_basis = np.vstack([_basis((row - mean) / scale) for row in x])
    pool_basis = np.vstack([_basis((row - mean) / scale) for row in pool])
    y_mean = float(y.mean())
    y_scale = max(float(y.std()), 1e-9)
    target = (y - y_mean) / y_scale
    penalty = np.eye(train_basis.shape[1])
    penalty[0, 0] = 0.0
    matrix = train_basis.T @ train_basis + 0.1 * penalty
    coefficients = np.linalg.solve(matrix, train_basis.T @ target)
    return pool_basis @ coefficients * y_scale + y_mean


def _fit_gp_pair(
    x: np.ndarray, y: np.ndarray, pool: np.ndarray, with_trend: bool
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0), 1e-9)
    x_norm = (x - x_mean) / x_scale
    pool_norm = (pool - x_mean) / x_scale
    local = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(x.shape[1]),
        length_scale_bounds=(0.05, 30.0),
        nu=1.5,
    )
    if with_trend:
        trend = ConstantKernel(1.0, (1e-2, 1e2)) * (
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) ** 2
        )
        kernel = trend + local
    else:
        kernel = local
    model = GaussianProcessRegressor(
        kernel=kernel + WhiteKernel(1e-3, noise_level_bounds=(1e-8, 1.0)),
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_norm, y)
    prediction, uncertainty = model.predict(pool_norm, return_std=True)
    return np.asarray(prediction), np.asarray(uncertainty)


def _maximin_indices(
    pool_norm: np.ndarray,
    reference_norm: np.ndarray,
    count: int,
    blocked: np.ndarray | None = None,
) -> list[int]:
    nearest = np.linalg.norm(
        pool_norm[:, None, :] - reference_norm[None, :, :], axis=2
    ).min(axis=1)
    if blocked is not None and len(blocked):
        blocked_distance = np.linalg.norm(
            pool_norm[:, None, :] - blocked[None, :, :], axis=2
        ).min(axis=1)
        nearest[blocked_distance < 0.08] = -1.0
    selected: list[int] = []
    for _ in range(count):
        index = int(np.argmax(nearest))
        if nearest[index] < 0:
            break
        selected.append(index)
        distance = np.linalg.norm(pool_norm - pool_norm[index], axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[index] = -1.0
    return selected


def _select_points(
    data: pd.DataFrame, pool_size: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    x = data.loc[:, FEATURE_LABELS].to_numpy(float)
    lower = x.min(axis=0)
    upper = x.max(axis=0)
    span = np.maximum(upper - lower, 1e-12)
    normalized_x = (x - lower) / span
    pool: list[np.ndarray] = []
    baseline = data.iloc[0]
    fixed_context = {
        key: float(baseline[key])
        for key in (
            "propeller_diameter_m",
            "payload_mass_kg",
            "hybrid_system_mass_kg",
        )
    }
    while len(pool) < pool_size:
        batch = rng.uniform(lower, upper, size=(max(2000, pool_size), len(FEATURE_LABELS)))
        for values in batch:
            design = AirframeDesign(
                **fixed_context,
                **dict(zip(FEATURE_LABELS, values.tolist())),
            )
            if all(value <= 1e-9 for value in geometry_constraints(design).values()):
                pool.append(values)
                if len(pool) == pool_size:
                    break
    pool_array = np.asarray(pool, dtype=float)
    pool_norm = (pool_array - lower) / span

    # Seal the holdout first using only feature-space coverage, before fitting acquisition models.
    holdout_idx = _maximin_indices(pool_norm, normalized_x, HOLDOUT_COUNT)
    holdout_norm = pool_norm[holdout_idx]
    holdout_indices = set(holdout_idx)

    predictions: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        compact = _fit_compact_predict(x, y, pool_array)
        gp_local, std_local = _fit_gp_pair(x, y, pool_array, with_trend=False)
        gp_trend, std_trend = _fit_gp_pair(x, y, pool_array, with_trend=True)
        predictions.append(np.column_stack((compact, gp_local, gp_trend)))
        uncertainties.append(np.column_stack((std_local, std_trend)))

    acquisition = np.zeros(pool_size, dtype=float)
    for target_index in range(len(TARGETS)):
        pred = predictions[target_index]
        target_scale = max(float(data[TARGETS[target_index]].std()), 1e-9)
        disagreement = np.std(pred, axis=1) / target_scale
        uncertainty = np.mean(uncertainties[target_index], axis=1) / target_scale
        acquisition += disagreement + uncertainty
    acquisition /= len(TARGETS)

    available = np.ones(pool_size, dtype=bool)
    available[list(holdout_indices)] = False
    distance_to_existing = np.linalg.norm(
        pool_norm[:, None, :] - normalized_x[None, :, :], axis=2
    ).min(axis=1)
    distance_to_holdout = np.linalg.norm(
        pool_norm[:, None, :] - holdout_norm[None, :, :], axis=2
    ).min(axis=1)
    available &= distance_to_holdout >= 0.08
    # Half the batch follows model disagreement/uncertainty; half adds broad coverage.
    selected_adaptive: list[int] = []
    for _ in range(TRAIN_COUNT // 2):
        chosen_so_far = (
            pool_norm[selected_adaptive]
            if selected_adaptive
            else np.empty((0, pool_norm.shape[1]))
        )
        references = [normalized_x, holdout_norm]
        if len(chosen_so_far):
            references.append(chosen_so_far)
        nearest = np.linalg.norm(
            pool_norm[:, None, :] - np.vstack(references)[None, :, :], axis=2
        ).min(axis=1)
        score = acquisition * (0.35 + 0.65 * np.minimum(nearest / 0.7, 1.0))
        score[~available] = -np.inf
        index = int(np.argmax(score))
        selected_adaptive.append(index)
        available[index] = False

    selected_coverage = _maximin_indices(
        pool_norm[available],
        np.vstack((normalized_x, holdout_norm, pool_norm[selected_adaptive])),
        TRAIN_COUNT - len(selected_adaptive),
    )
    remaining_indices = np.flatnonzero(available)
    selected_coverage = [int(remaining_indices[index]) for index in selected_coverage]
    selected_train = selected_adaptive + selected_coverage
    if len(selected_train) != TRAIN_COUNT:
        raise RuntimeError("Could not select the requested number of distinct training points.")

    fixed = fixed_context
    def records(indices: list[int], role: str, basis: str) -> pd.DataFrame:
        rows = []
        for order, index in enumerate(indices, start=1):
            rows.append(
                {
                    "sample": f"{role}_{order:02d}",
                    **fixed,
                    **dict(zip(FEATURE_LABELS, pool_array[index].tolist())),
                    "selection_role": role,
                    "selection_basis": basis,
                    "vspaero_label_status": "pending_solver_run",
                }
            )
        return pd.DataFrame(rows)

    train_frame = records(
        selected_train,
        "adaptive_train",
        "16个模型分歧/GP不确定性与空间距离联合选点 + 16个可行域maximin覆盖点",
    )
    holdout_frame = records(
        holdout_idx,
        "blind_holdout",
        "先于自适应选点锁定的可行域maximin独立盲测；不参与代理拟合或选点",
    )
    diagnostics = {
        "source_rows": len(data),
        "candidate_pool_size": pool_size,
        "requested_training_additions": TRAIN_COUNT,
        "requested_blind_holdout": HOLDOUT_COUNT,
        "training_adaptive_count": len(selected_adaptive),
        "training_coverage_count": len(selected_coverage),
        "featurewise_bounds": {
            name: [float(a), float(b)] for name, a, b in zip(FEATURE_LABELS, lower, upper)
        },
        "fixed_powertrain_context": fixed_context,
        "min_normalized_distance_train_to_existing": float(
            np.linalg.norm(
                ((pool_array[selected_train] - lower) / span)[:, None, :]
                - normalized_x[None, :, :],
                axis=2,
            ).min()
        ),
        "holdout_sealed_before_model_acquisition": True,
    }
    return train_frame, holdout_frame, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--vsp-python",
        type=Path,
        default=ROOT.parent.parent / "tools" / "Python311" / "python.exe",
    )
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not args.select_only and not args.vsp_python.is_file():
        raise FileNotFoundError(f"OpenVSP Python runtime not found: {args.vsp_python}")

    source = pd.read_csv(args.data)
    if len(source) != 100:
        raise ValueError(f"Expected the current 100-point DOE, found {len(source)} rows.")
    if set(source["oswald_efficiency_definition"].dropna()) != {
        OSWALD_EFFICIENCY_DEFINITION
    }:
        raise ValueError("Source DOE does not use the current wake-based Ew definition.")
    if not np.isfinite(source.loc[:, [*FEATURE_LABELS, *TARGETS]].to_numpy(float)).all():
        raise ValueError("Source DOE contains non-finite features or labels.")

    output_dir.mkdir(parents=True)
    train, holdout, diagnostics = _select_points(source, args.pool_size, args.seed)
    train_path = output_dir / "新增训练DOE_32点.csv"
    holdout_path = output_dir / "独立盲测DOE_8点.csv"
    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    holdout.to_csv(holdout_path, index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_data": str(args.data.resolve()),
        "source_label_definition": OSWALD_EFFICIENCY_DEFINITION,
        "seed": args.seed,
        "diagnostics": diagnostics,
        "status": "selected_only" if args.select_only else "running",
        "warning": "盲测点在首轮对比前不参与训练；看过盲测分数后，该集合不再是后续迭代的全新盲测集。",
    }
    metadata_path = output_dir / "补样元数据.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.select_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    runner = importlib.import_module("重算VSPAERO_Ew_DOE并训练代理")
    status_path = output_dir / "运行进度.json"
    attempts_path = output_dir / "求解尝试记录.csv"
    successes: list[dict] = []
    failures: list[dict] = []
    cases = [
        (train, "adaptive_train"),
        (holdout, "blind_holdout"),
    ]
    total = len(train) + len(holdout)
    started = time.perf_counter()

    def persist(status: str, completed: int) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "completed": completed,
                    "total": total,
                    "valid_training_labels": sum(r["selection_role"] == "adaptive_train" for r in successes),
                    "valid_holdout_labels": sum(r["selection_role"] == "blind_holdout" for r in successes),
                    "failed": len(failures),
                    "elapsed_s": time.perf_counter() - started,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(successes).to_csv(
            output_dir / "新增VSPAERO有效标签.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(failures).to_csv(attempts_path, index=False, encoding="utf-8-sig")

    completed = 0
    persist("running", completed)
    for frame, role in cases:
        for _, selected in frame.iterrows():
            sample = str(selected["sample"])
            design = {
                key: float(selected[key])
                for key in (*FEATURE_LABELS, "propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")
            }
            case_dir = output_dir / "cases" / sample
            basis = str(selected["selection_basis"])
            try:
                row = runner._run_case(
                    args.vsp_python,
                    design,
                    sample,
                    case_dir,
                    args.timeout_s,
                    role,
                    basis,
                )
                successes.append(row)
                print(
                    f"[{completed + 1:02d}/{total}] {sample}: "
                    f"CLalpha={row[TARGETS[0]]:.5f}, Ew={row[TARGETS[1]]:.5f}",
                    flush=True,
                )
            except Exception as error:
                failures.append({"sample": sample, "selection_role": role, "error": str(error)})
                print(f"[{completed + 1:02d}/{total}] {sample}: REJECTED {error}", flush=True)
            completed += 1
            persist("running", completed)

    labels = pd.DataFrame(successes)
    train_count = int((labels["selection_role"] == "adaptive_train").sum()) if len(labels) else 0
    holdout_count = int((labels["selection_role"] == "blind_holdout").sum()) if len(labels) else 0
    if train_count < 30 or holdout_count != HOLDOUT_COUNT:
        metadata["status"] = "incomplete"
        metadata["failure_note"] = (
            f"Need at least 30 valid training labels and exactly {HOLDOUT_COUNT} holdout labels; "
            f"got {train_count} and {holdout_count}."
        )
        metadata["valid_counts"] = {
            "adaptive_train": train_count,
            "blind_holdout": holdout_count,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        persist("incomplete", completed)
        raise RuntimeError(metadata["failure_note"])

    labels["oswald_efficiency_definition"] = OSWALD_EFFICIENCY_DEFINITION
    labels.to_csv(output_dir / "新增VSPAERO有效标签.csv", index=False, encoding="utf-8-sig")
    train_labels = labels.loc[labels["selection_role"] == "adaptive_train"].copy()
    holdout_labels = labels.loc[labels["selection_role"] == "blind_holdout"].copy()
    train_labels.to_csv(output_dir / f"新增训练DOE_{len(train_labels)}点_含标签.csv", index=False, encoding="utf-8-sig")
    holdout_labels.to_csv(output_dir / f"独立盲测DOE_{len(holdout_labels)}点_含标签.csv", index=False, encoding="utf-8-sig")
    metadata["status"] = (
        "complete" if train_count == TRAIN_COUNT else "complete_with_rejected_candidate"
    )
    metadata["valid_counts"] = {
        "adaptive_train": len(train_labels),
        "blind_holdout": len(holdout_labels),
    }
    metadata["rejected_candidates"] = failures
    metadata["runtime_s"] = time.perf_counter() - started
    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    persist("complete", completed)
    print(f"完成，结果目录：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
