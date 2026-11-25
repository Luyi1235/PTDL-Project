"""One-at-a-time sensitivity study around a selected PTDL feasible candidate."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from MOEAD多目标联合优化 import OBJECTIVE_LABELS, PTDLEnduranceJointProblem
from 八十小时续航任务模型 import calibrate_endurance_task_baseline
from 飞推控闭环物理模型 import LongitudinalDragDisturbance
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from VSPAERO气动代理模型 import load_vsp_surrogate


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "H6_三项风险闭环复验"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "H6_非标定扩展验证" / "方案参数敏感性"
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
DISTURBANCES = (
    LongitudinalDragDisturbance(
        phase_name="endurance_loiter",
        start_time_s=3600.0,
        duration_s=600.0,
        additional_drag_n=100.0,
        response_step_s=5.0,
        recovery_window_s=900.0,
        recovery_dwell_s=30.0,
    ),
)


def _evaluation_row(
    evaluation: object,
    label: str,
    level: float,
    direction: str,
    base_value: float,
    perturbed_value: float,
    bound_width: float,
    baseline: object,
) -> dict[str, object]:
    result = evaluation.result
    base_result = baseline.result
    row: dict[str, object] = {
        "decision": label,
        "perturbation_fraction_of_range": level,
        "direction": direction,
        "baseline_value": base_value,
        "perturbed_value": perturbed_value,
        "actual_step_fraction_of_range": (perturbed_value - base_value) / bound_width,
        "feasible": evaluation.feasible,
        "baseline_feasible": baseline.feasible,
        "total_violation": evaluation.violation,
        "feasibility_changed": evaluation.feasible != baseline.feasible,
    }
    for objective in OBJECTIVE_LABELS:
        baseline_value = float(base_result.objectives[objective])
        current_value = float(result.objectives[objective])
        row[objective] = current_value
        row[f"delta_{objective}"] = current_value - baseline_value
        row[f"relative_delta_{objective}"] = (
            current_value - baseline_value
        ) / max(abs(baseline_value), 1e-12)
    row.update({f"g_{name}": value for name, value in result.constraints.items()})
    row.update({f"diagnostic_{name}": value for name, value in result.diagnostics.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="围绕 H=6 择优方案进行设计变量和阶段功率决策的单因素敏感性分析。"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surrogate", type=Path, default=SURROGATE_PATH)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=(0.05, 0.10),
        help="相对每个变量设计范围的扰动比例。",
    )
    args = parser.parse_args()
    if any(level <= 0.0 or level > 0.25 for level in args.levels):
        raise SystemExit("扰动比例须在 (0, 0.25] 内。")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = np.asarray(candidate["decision"], dtype=float)
    calibration = calibrate_endurance_task_baseline()
    surrogate = load_vsp_surrogate(args.surrogate)
    aero_provider = lambda design, active_calibration: surrogate.corrected_aero(
        design, active_calibration
    )
    powertrain_factory = partial(
        HybridSystemPowertrain, config=HybridSystemAdapterConfig()
    )
    problem = PTDLEnduranceJointProblem(
        aero_provider=aero_provider,
        powertrain_factory=powertrain_factory,
        dispatch_policy_factory=MPCEnergyManagementCoordinator,
        disturbances=DISTURBANCES,
    )
    if decision.shape != problem.lower_bounds.shape:
        raise ValueError(
            f"候选决策维数 {decision.size} 与当前优化问题维数 {problem.lower_bounds.size} 不一致。"
        )
    decision = np.clip(decision, problem.lower_bounds, problem.upper_bounds)
    baseline = problem.evaluate(decision)
    rows: list[dict[str, object]] = []
    widths = problem.upper_bounds - problem.lower_bounds
    labels = problem.decision_labels
    if args.levels == sorted(set(args.levels)):
        levels = tuple(args.levels)
    else:
        levels = tuple(sorted(set(args.levels)))

    for decision_index, label in enumerate(labels):
        base_value = float(decision[decision_index])
        width = float(widths[decision_index])
        for level in levels:
            delta = level * width
            for direction, sign in (("minus", -1.0), ("plus", 1.0)):
                perturbed = decision.copy()
                perturbed[decision_index] = np.clip(
                    base_value + sign * delta,
                    problem.lower_bounds[decision_index],
                    problem.upper_bounds[decision_index],
                )
                if abs(perturbed[decision_index] - base_value) <= 1e-12:
                    continue
                evaluation = problem.evaluate(perturbed)
                rows.append(
                    _evaluation_row(
                        evaluation,
                        label,
                        level,
                        direction,
                        base_value,
                        float(perturbed[decision_index]),
                        width,
                        baseline,
                    )
                )
        pd.DataFrame(rows).to_csv(output_dir / "单因素扰动明细.csv", index=False)

    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for label in labels:
        variable_rows = detail.loc[detail["decision"] == label]
        feasible_rows = variable_rows.loc[variable_rows["feasible"]]
        summary: dict[str, object] = {
            "decision": label,
            "baseline_value": float(decision[labels.index(label)]),
            "perturbation_case_count": len(variable_rows),
            "feasibility_change_count": int(variable_rows["feasibility_changed"].sum()),
            "non_feasible_perturbation_count": int((~variable_rows["feasible"]).sum()),
            "worst_total_violation": float(variable_rows["total_violation"].max())
            if not variable_rows.empty
            else float(baseline.violation),
        }
        for objective in OBJECTIVE_LABELS:
            response_column = f"relative_delta_{objective}"
            normalized_effect = (
                feasible_rows[response_column].abs()
                / feasible_rows["actual_step_fraction_of_range"].abs()
            )
            summary[f"max_normalized_sensitivity_{objective}"] = (
                float(normalized_effect.max()) if not normalized_effect.empty else 0.0
            )
            summary[f"max_abs_relative_change_{objective}"] = (
                float(feasible_rows[response_column].abs().max())
                if not feasible_rows.empty
                else 0.0
            )
        summary["sensitivity_rank"] = 0
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    sensitivity_columns = [
        f"max_normalized_sensitivity_{objective}" for objective in OBJECTIVE_LABELS
    ]
    summary_frame["overall_sensitivity_score"] = summary_frame[sensitivity_columns].max(axis=1)
    summary_frame["sensitivity_rank"] = (
        summary_frame["overall_sensitivity_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    risk_order = summary_frame.sort_values(
        ["feasibility_change_count", "worst_total_violation"],
        ascending=[False, False],
    ).index
    risk_rank = {index: rank for rank, index in enumerate(risk_order, start=1)}
    summary_frame["feasibility_risk_rank"] = summary_frame.index.map(risk_rank)
    summary_frame.sort_values(
        ["sensitivity_rank", "decision"], inplace=True, ignore_index=True
    )
    summary_frame.to_csv(output_dir / "决策变量敏感性排序.csv", index=False)
    baseline_record = {
        "candidate": str(args.candidate.resolve()),
        "optimizer": candidate.get("optimizer"),
        "baseline_feasible": baseline.feasible,
        "baseline_total_violation": baseline.violation,
        "baseline_objectives": baseline.result.objectives,
        "baseline_constraints": baseline.result.constraints,
        "baseline_diagnostics": baseline.result.diagnostics,
        "perturbation_levels_of_full_design_range": list(levels),
        "method": "one-at-a-time bounded decision perturbation, re-evaluated with the same component-backed H=6 problem",
        "case_count": len(detail),
        "model_boundary": "component ratings and efficiency assumptions are unchanged; this is decision sensitivity, not model-parameter uncertainty",
        "interpretation": (
            "Local sensitivity around one selected candidate; do not interpret as global sensitivity or statistical confidence. "
            "Objective sensitivity ranking excludes infeasible/early-terminated runs so shorter failed missions cannot appear to improve energy."
        ),
    }
    (output_dir / "敏感性分析说明.json").write_text(
        json.dumps(baseline_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(f"\n扰动计算数：{len(detail)}；结果目录：{output_dir}")


if __name__ == "__main__":
    main()
