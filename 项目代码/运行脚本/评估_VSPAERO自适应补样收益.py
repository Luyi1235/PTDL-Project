"""Evaluate an adaptive VSPAERO DOE against a sealed holdout and repeated CV."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
SCRIPTS = ROOT / "运行脚本"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from VSPAERO气动代理模型 import (  # noqa: E402
    FEATURE_LABELS,
    fit_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402
from 比较VSPAERO气动代理模型族 import (  # noqa: E402
    _features,
    _fit_ard_gp,
    _fit_compact_tuned,
)


SOURCE = (
    ROOT
    / "气动代理模型数据"
    / "VSPAERO_DOE扩充样本_100点_Ew_物理选点_网格修复_20261006.csv"
)
RUN_DIR = ROOT / "运行结果" / "VSPAERO自适应补样盲测_20261006_批次03"
TRAIN_LABELS = RUN_DIR / "新增训练DOE_32点_含标签.csv"
HOLDOUT_LABELS = RUN_DIR / "独立盲测DOE_8点_含标签.csv"
ALL_LABELS = RUN_DIR / "新增VSPAERO有效标签.csv"
OUTPUT = RUN_DIR / "代理模型补样收益评估"
TARGETS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "R2": float(1.0 - np.sum(error**2) / total) if total > 1e-15 else float("nan"),
    }


def _predict(
    train: pd.DataFrame, holdout: pd.DataFrame
) -> dict[str, np.ndarray]:
    train_designs = [
        AirframeDesign(**{name: float(row[name]) for name in FEATURE_LABELS})
        for _, row in train.iterrows()
    ]
    holdout_designs = [
        AirframeDesign(**{name: float(row[name]) for name in FEATURE_LABELS})
        for _, row in holdout.iterrows()
    ]
    x_train = _features(train_designs)
    x_test = _features(holdout_designs)
    y = train.loc[:, TARGETS].to_numpy(float)
    predictions = {
        model: np.zeros((len(holdout), len(TARGETS)), dtype=float)
        for model in ("production_compact_ridge", "tuned_compact_ridge", "quadratic_trend_matern32_gp")
    }
    compact = replace(
        fit_vsp_surrogate(train_designs, y[:, 0], y[:, 1]),
        oswald_efficiency_bounds=(0.45, 1.00),
    )
    for i, design in enumerate(holdout_designs):
        predictions["production_compact_ridge"][i] = compact.predict(design)
    for target_index in range(len(TARGETS)):
        for row_index, test_x in enumerate(x_test):
            prediction, _ = _fit_compact_tuned(
                x_train, y[:, target_index], test_x[None, :]
            )
            predictions["tuned_compact_ridge"][row_index, target_index] = prediction
        prediction, _, _ = _fit_ard_gp(
            x_train,
            y[:, target_index],
            x_test,
            "gp_quadratic_trend_matern32_ard",
        )
        predictions["quadratic_trend_matern32_gp"][:, target_index] = prediction
    predictions["production_compact_ridge"][:, 1] = np.clip(
        predictions["production_compact_ridge"][:, 1], 0.45, 1.0
    )
    predictions["tuned_compact_ridge"][:, 1] = np.clip(
        predictions["tuned_compact_ridge"][:, 1], 0.45, 1.0
    )
    predictions["quadratic_trend_matern32_gp"][:, 1] = np.clip(
        predictions["quadratic_trend_matern32_gp"][:, 1], 0.45, 1.0
    )
    return predictions


def _assemble_training(source: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat((source, additions), ignore_index=True)
    key = combined.loc[:, FEATURE_LABELS].round(9).astype(str).agg("|".join, axis=1)
    if key.duplicated().any():
        raise ValueError("Duplicate geometry found across the source DOE and additions.")
    if not np.isfinite(combined.loc[:, [*FEATURE_LABELS, *TARGETS]].to_numpy(float)).all():
        raise ValueError("Training data contains non-finite features or labels.")
    return combined


def _cross_validate(
    frame: pd.DataFrame, name: str, repeats: int = 5
) -> list[dict[str, object]]:
    designs = [
        AirframeDesign(**{key: float(row[key]) for key in FEATURE_LABELS})
        for _, row in frame.iterrows()
    ]
    x = _features(designs)
    y = frame.loc[:, TARGETS].to_numpy(float)
    records: list[dict[str, object]] = []
    for repeat in range(repeats):
        splitter = KFold(n_splits=5, shuffle=True, random_state=20261020 + repeat)
        predictions = {
            model: np.full_like(y, np.nan)
            for model in ("production_compact_ridge", "quadratic_trend_matern32_gp")
        }
        for train_index, test_index in splitter.split(x):
            train_designs = [designs[i] for i in train_index]
            compact = fit_vsp_surrogate(
                train_designs, y[train_index, 0], y[train_index, 1]
            )
            compact = replace(compact, oswald_efficiency_bounds=(0.45, 1.00))
            for index in test_index:
                predictions["production_compact_ridge"][index] = compact.predict(designs[index])
            for target_index in range(len(TARGETS)):
                gp_prediction, _, _ = _fit_ard_gp(
                    x[train_index],
                    y[train_index, target_index],
                    x[test_index],
                    "gp_quadratic_trend_matern32_ard",
                )
                predictions["quadratic_trend_matern32_gp"][
                    test_index, target_index
                ] = gp_prediction
        for model_name, predicted in predictions.items():
            predicted[:, 1] = np.clip(predicted[:, 1], 0.45, 1.0)
            for target_index, target in enumerate(TARGETS):
                records.append(
                    {
                        "dataset": name,
                        "repeat": repeat + 1,
                        "sample_count": len(frame),
                        "model": model_name,
                        "target": target,
                        **_score(y[:, target_index], predicted[:, target_index]),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--labels", type=Path, default=ALL_LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--cv-repeats", type=int, default=5)
    args = parser.parse_args()
    source = pd.read_csv(args.source)
    labels = pd.read_csv(args.labels)
    additions = labels.loc[labels["selection_role"] == "adaptive_train"].copy()
    holdout = labels.loc[labels["selection_role"] == "blind_holdout"].copy()
    if len(source) != 100 or len(additions) < 30 or len(holdout) != 8:
        raise ValueError(
            f"Expected 100 baseline, >=30 added training, and 8 holdout rows; got "
            f"{len(source)}, {len(additions)}, {len(holdout)}."
        )
    base_train = source.loc[:, [*FEATURE_LABELS, *TARGETS]].copy()
    expanded_train = _assemble_training(base_train, additions)
    blind = holdout.loc[:, [*FEATURE_LABELS, *TARGETS]].copy()

    detail: list[dict[str, object]] = []
    holdout_summary: list[dict[str, object]] = []
    for dataset_name, train in (
        ("baseline_100", base_train),
        (f"expanded_{len(expanded_train)}", expanded_train),
    ):
        predictions = _predict(train, blind)
        for model_name, predicted in predictions.items():
            for target_index, target in enumerate(TARGETS):
                metrics = _score(
                    blind[target].to_numpy(float), predicted[:, target_index]
                )
                holdout_summary.append(
                    {
                        "training_dataset": dataset_name,
                        "training_samples": len(train),
                        "blind_holdout_samples": len(blind),
                        "model": model_name,
                        "target": target,
                        **metrics,
                    }
                )
                for i in range(len(blind)):
                    detail.append(
                        {
                            "training_dataset": dataset_name,
                            "model": model_name,
                            "target": target,
                            "sample": holdout.iloc[i]["sample"],
                            "actual": blind.iloc[i][target],
                            "predicted": predicted[i, target_index],
                            "residual": predicted[i, target_index] - blind.iloc[i][target],
                        }
                    )

    cv_rows: list[dict[str, object]] = []
    cv_rows.extend(_cross_validate(base_train, "baseline_100", args.cv_repeats))
    cv_rows.extend(_cross_validate(expanded_train, f"expanded_{len(expanded_train)}", args.cv_repeats))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    holdout_summary_frame = pd.DataFrame(holdout_summary)
    detail_frame = pd.DataFrame(detail)
    cv_frame = pd.DataFrame(cv_rows)
    holdout_summary_frame.to_csv(output / "独立盲测指标.csv", index=False, encoding="utf-8-sig")
    detail_frame.to_csv(output / "独立盲测逐点预测.csv", index=False, encoding="utf-8-sig")
    cv_frame.to_csv(output / "重复五折交叉验证.csv", index=False, encoding="utf-8-sig")
    cv_aggregate = (
        cv_frame.groupby(["dataset", "sample_count", "model", "target"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
        )
    )
    cv_aggregate.to_csv(output / "重复五折汇总.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_training_rows": len(base_train),
        "added_training_rows": len(additions),
        "expanded_training_rows": len(expanded_train),
        "sealed_holdout_rows": len(blind),
        "holdout_participated_in_training": False,
        "holdout_participated_in_selection": False,
        "cross_validation_repeats": args.cv_repeats,
        "production_model_file_modified": False,
        "limitations": [
            "8个盲测点对R2非常敏感，盲测指标仅作小样本独立检查。",
            "重复五折交叉验证用于同一设计域内部比较，不能替代风洞/飞行验证。",
            "标签为指定网格策略和工况下的VSPAERO计算结果，不是实测气动数据。",
        ],
        "outputs": {
            "holdout_summary": str(output / "独立盲测指标.csv"),
            "holdout_detail": str(output / "独立盲测逐点预测.csv"),
            "cv_aggregate": str(output / "重复五折汇总.csv"),
        },
    }
    (output / "评估元数据.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("独立盲测：")
    print(holdout_summary_frame.to_string(index=False))
    print("\n重复五折均值：")
    print(cv_aggregate.to_string(index=False))
    print(f"\n评估文件：{output}", flush=True)


if __name__ == "__main__":
    main()
