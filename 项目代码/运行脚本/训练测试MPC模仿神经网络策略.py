"""Train and group-holdout-test a safety-projected neural MPC imitation policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 智能能量管理与容错调度 import (  # noqa: E402
    FUZZY_SOC_TARGETS,
    NEURAL_PHASES,
    neural_imitation_feature_vector,
)
from 八十小时续航任务模型 import load_endurance_profile  # noqa: E402


DEFAULT_SOURCE = ROOT / "运行结果" / "六项推进_v5_多工况控制器对比_20261003"
DEFAULT_CANDIDATE = ROOT / "运行结果" / "H6_三项风险闭环复验" / "NSGA-II与MOEA-D择优可行候选.json"
DEFAULT_OUTPUT = ROOT / "运行结果" / "神经网络MPC模仿策略训练测试_20261011"
HOLDOUT_SCENARIO = "长航时迎风阶跃"
MPC_SUFFIX = "_MPC滚动调度_时序.csv"
REQUIRED_COLUMNS = (
    "time_s",
    "phase",
    "soc",
    "e_h_m",
    "e_v_mps",
    "p_propulsion_request_kw",
    "p_engine_target_kw",
)


def _load_cases(source_dir: Path, candidate_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    engine_caps = candidate.get("engine_caps_kw")
    if not isinstance(engine_caps, list) or len(engine_caps) != len(NEURAL_PHASES):
        raise ValueError("Candidate is missing six phase engine-cap values.")
    cap_by_phase = dict(zip(NEURAL_PHASES, map(float, engine_caps)))
    durations = {
        segment["label"]: float(segment.get("duration_s", 76.0 * 3600.0))
        for segment in load_endurance_profile()["segments"]
    }

    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    group_rows: list[str] = []
    scenario_counts: dict[str, int] = {}
    for path in sorted(source_dir.glob(f"*{MPC_SUFFIX}")):
        if path.name.startswith("零幅扰动同采样基线_"):
            continue
        scenario = path.name.removesuffix(MPC_SUFFIX)
        frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
        frame["phase"] = frame["phase"].map(
            lambda phase: "long_descent"
            if str(phase).startswith("long_descent")
            else str(phase)
        )
        missing_phase = set(frame["phase"].astype(str)).difference(NEURAL_PHASES)
        if missing_phase:
            raise ValueError(f"{path.name} has unsupported phases: {sorted(missing_phase)}")
        frame["dt_s"] = frame["time_s"].diff().clip(lower=1e-3)
        first_dt = (
            frame.groupby("phase", sort=False)["dt_s"]
            .transform(lambda values: values.dropna().median())
        )
        frame["dt_s"] = frame["dt_s"].fillna(first_dt).fillna(1.0)
        frame["phase_elapsed_s"] = frame.groupby("phase", sort=False)["time_s"].transform(
            lambda values: values - values.min()
        )
        frame["previous_engine_target_kw"] = frame["p_engine_target_kw"].shift(1).fillna(0.0)
        for row in frame.itertuples(index=False):
            phase = str(row.phase)
            cap = max(cap_by_phase[phase], 1e-6)
            features = neural_imitation_feature_vector(
                phase,
                row.dt_s,
                row.soc,
                row.e_h_m,
                row.e_v_mps,
                row.p_propulsion_request_kw,
                cap,
                row.previous_engine_target_kw,
                row.phase_elapsed_s,
                durations[phase],
            )
            feature_rows.append(features)
            target_rows.append(float(np.clip(row.p_engine_target_kw / cap, 0.0, 1.0)))
            group_rows.append(scenario)
        scenario_counts[scenario] = len(frame)
    if not feature_rows or HOLDOUT_SCENARIO not in scenario_counts:
        raise FileNotFoundError(
            f"Need MPC time-series including the held-out scenario '{HOLDOUT_SCENARIO}'."
        )
    return (
        np.vstack(feature_rows),
        np.asarray(target_rows, dtype=float),
        np.asarray(group_rows, dtype=object),
        np.asarray([scenario_counts[name] for name in sorted(scenario_counts)], dtype=int),
    )


def _scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE_engine_cap_fraction": float(np.mean(np.abs(error))),
        "RMSE_engine_cap_fraction": float(np.sqrt(np.mean(error**2))),
        "R2": float(1.0 - np.sum(error**2) / total) if total > 1e-12 else 0.0,
        "max_abs_error_engine_cap_fraction": float(np.max(np.abs(error))),
    }


def _build_model(random_state: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    x, y, groups, scenario_counts = _load_cases(args.source_dir, args.candidate)
    held_out = groups == HOLDOUT_SCENARIO
    train = ~held_out
    feature_mean = x[train].mean(axis=0)
    feature_scale = np.maximum(x[train].std(axis=0), 1e-8)
    x_scaled = (x - feature_mean) / feature_scale
    model = _build_model(20261011)
    model.fit(x_scaled[train], y[train])
    predicted = np.clip(model.predict(x_scaled[held_out]), 0.0, 1.0)
    baseline_prediction = np.full(
        int(np.sum(held_out)), float(np.mean(y[train])), dtype=float
    )
    nn_metrics = _scores(y[held_out], predicted)
    baseline_metrics = _scores(y[held_out], baseline_prediction)
    grouped_scores: dict[str, dict[str, float | int]] = {}
    for fold_index, scenario in enumerate(sorted(set(groups.tolist()))):
        fold_test = groups == scenario
        fold_train = ~fold_test
        fold_mean = x[fold_train].mean(axis=0)
        fold_scale = np.maximum(x[fold_train].std(axis=0), 1e-8)
        fold_model = _build_model(20261011 + fold_index)
        fold_model.fit((x[fold_train] - fold_mean) / fold_scale, y[fold_train])
        fold_prediction = np.clip(
            fold_model.predict((x[fold_test] - fold_mean) / fold_scale), 0.0, 1.0
        )
        fold_baseline = np.full(
            int(np.sum(fold_test)), float(np.mean(y[fold_train])), dtype=float
        )
        grouped_scores[scenario] = {
            "sample_count": int(np.sum(fold_test)),
            "neural_policy": _scores(y[fold_test], fold_prediction),
            "training_mean_baseline": _scores(y[fold_test], fold_baseline),
            "iterations": int(fold_model.n_iter_),
        }
    grouped_rmse = [
        float(value["neural_policy"]["RMSE_engine_cap_fraction"])
        for value in grouped_scores.values()
    ]
    grouped_baseline_rmse = [
        float(value["training_mean_baseline"]["RMSE_engine_cap_fraction"])
        for value in grouped_scores.values()
    ]

    profile = load_endurance_profile()
    phase_durations = {
        segment["label"]: float(segment.get("duration_s", 76.0 * 3600.0))
        for segment in profile["segments"]
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "study": "offline MPC behavior cloning for engine-generator target fraction",
        "training_source": str(args.source_dir.resolve()),
        "candidate_anchor": str(args.candidate.resolve()),
        "total_samples": int(len(y)),
        "training_samples": int(np.sum(train)),
        "held_out_samples": int(np.sum(held_out)),
        "held_out_scenario": HOLDOUT_SCENARIO,
        "scenario_sample_counts": {
            name: int(count)
            for name, count in zip(sorted(set(groups.tolist())), scenario_counts)
        },
        "model": {
            "family": "scikit-learn MLPRegressor",
            "hidden_layers": [32, 16],
            "activation": "tanh",
            "iterations": int(model.n_iter_),
            "converged": bool(model.n_iter_ < model.max_iter),
        },
        "held_out_metrics": {
            "neural_policy": nn_metrics,
            "training_mean_baseline": baseline_metrics,
        },
        "grouped_scenario_cross_validation": {
            "fold_count": len(grouped_scores),
            "folds": grouped_scores,
            "mean_neural_RMSE_engine_cap_fraction": float(np.mean(grouped_rmse)),
            "max_neural_RMSE_engine_cap_fraction": float(np.max(grouped_rmse)),
            "mean_baseline_RMSE_engine_cap_fraction": float(
                np.mean(grouped_baseline_rmse)
            ),
            "all_folds_better_than_training_mean": bool(
                all(
                    float(value["neural_policy"]["RMSE_engine_cap_fraction"])
                    < float(
                        value["training_mean_baseline"][
                            "RMSE_engine_cap_fraction"
                        ]
                    )
                    for value in grouped_scores.values()
                )
            ),
        },
        "deployment_gate_passed": bool(
            nn_metrics["RMSE_engine_cap_fraction"]
            < baseline_metrics["RMSE_engine_cap_fraction"]
        ),
        "safety_supervisor": (
            "runtime dispatch must clip engine request to propulsion-feasible lower "
            "bound and engine cap, enforce engine slew, and derive motor balance"
        ),
        "limitations": [
            "Labels come from one estimated-model candidate and one MPC configuration.",
            "The held-out case is one named scenario, not an independent flight test.",
            "The network predicts an MPC action fraction, not an optimal policy guarantee.",
        ],
    }
    (output_dir / "神经网络模仿策略评估.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_frame = pd.DataFrame(
        {
            "scenario": groups[held_out],
            "actual_engine_cap_fraction": y[held_out],
            "predicted_engine_cap_fraction": predicted,
            "residual": predicted - y[held_out],
        }
    )
    prediction_frame.to_csv(output_dir / "迎风阶跃留出预测.csv", index=False)

    if report["deployment_gate_passed"]:
        payload = {
            "schema": "ptdl_mpc_imitation_mlp_v1",
            "phase_labels": list(NEURAL_PHASES),
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "target_mean": 0.0,
            "target_scale": 1.0,
            "coefs": [layer.tolist() for layer in model.coefs_],
            "intercepts": [layer.tolist() for layer in model.intercepts_],
            "phase_durations_s": phase_durations,
            "phase_soc_targets": FUZZY_SOC_TARGETS,
            "training_samples": int(np.sum(train)),
            "held_out_scenario": HOLDOUT_SCENARIO,
            "held_out_metrics": nn_metrics,
            "source_model_status": "estimated-model MPC imitation prototype; not an independently optimized controller",
        }
        model_path = output_dir / "神经网络MPC模仿策略.json"
        model_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["deployable_model_file"] = str(model_path)
        (output_dir / "神经网络模仿策略评估.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
