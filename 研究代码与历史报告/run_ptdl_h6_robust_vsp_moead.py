"""Run a stage-level robust H=6 MOEA/D optimization for the PTDL model.

Each candidate first passes the nominal 2.2.1 evaluation.  Nominally feasible
candidates are then re-evaluated with a 10% parasite-drag increase.  The
second condition contributes endurance, SOC, and flight/power constraint
violations to the same feasibility decision, so the Pareto set cannot contain
a design that only reaches 80 h under its nominal aerodynamic assumption.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_2_2_2_optimizer import CandidateEvaluation, MOEAD, OBJECTIVE_LABELS, PTDLEnduranceJointProblem, pareto_indices
from ptdl_80h_endurance_profile import evaluate_endurance_task
from ptdl_airframe_design import AirframeDesign
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_moead_h6_robust_vsp_results"
SURROGATE_PATH = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
H = 6
GENERATIONS = 25
SEED = 20260971
ROBUST_CD0_FACTOR = 1.10


class RobustVSPAeroJointProblem(PTDLEnduranceJointProblem):
    """Joint geometry-control problem with a mandatory conservative aero case."""

    def __init__(self, surrogate: VSPAeroSurrogate) -> None:
        self.surrogate = surrogate
        super().__init__(aero_provider=lambda design, calibration: surrogate.corrected_aero(design, calibration))
        # A nonzero hybrid package is retained for this early control-study
        # stage.  Its mass, power, and energy correlations remain provisional,
        # but allowing zero mass would remove the 1.2 energy-management path.
        self.lower_bounds[9] = 60.0
        self.upper_bounds[9] = 140.0

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        samples = super().sample(count, rng)
        if count:
            # Transparent feasible-seeking anchor; all other members retain
            # random MOEA/D initialization inside the declared bounds.
            samples[0, :10] = np.asarray((25.18, 30.0, 0.45, 0.30, -0.63, 8.25, 0.667, 2.29, 110.0, 80.0))
            samples[0, 10] = 30.1
            phase_count = len(self.phase_labels)
            samples[0, 11 : 11 + phase_count] = np.asarray((145.0, 145.0, 115.0, 65.0, 80.0, 55.0))
            samples[0, 11 + phase_count :] = 0.0
        return np.clip(samples, self.lower_bounds, self.upper_bounds)

    def _mission_inputs(self, x: np.ndarray) -> tuple[AirframeDesign, np.ndarray, np.ndarray, list[float]]:
        design = self.design_from_decision(x)
        phase_count = len(self.phase_labels)
        engine_caps = x[11 : 11 + phase_count]
        motor_assist = x[11 + phase_count :]
        speeds = [phase.speed_target_mps for phase in self.base_mission.phases]
        speeds[self.phase_labels.index("endurance_loiter")] = float(x[10])
        return design, engine_caps, motor_assist, speeds

    def evaluate(self, decision: Sequence[float]) -> CandidateEvaluation:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        design, engine_caps, motor_assist, speeds = self._mission_inputs(x)
        nominal_aero = self.surrogate.corrected_aero(design, self.calibration)
        nominal = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            profile_path=self.profile_path,
            calibration=self.calibration,
            assumptions=self.task_calibration.assumptions,
            aero_override=nominal_aero,
        )
        result = nominal.result
        result.diagnostics.update(
            {
                "nominal_achieved_total_duration_h": nominal.achieved_duration_h,
                "nominal_cd0_reference": nominal_aero.cd0,
            }
        )
        # There is no value in paying for the conservative mission when the
        # nominal candidate already violates constraints.  Its robust status is
        # unambiguously infeasible through the nominal constraint set.
        if not result.feasible:
            result.constraints["robust_cd0_case_not_evaluated"] = 1.0
            return CandidateEvaluation(x, result)

        conservative_aero = replace(nominal_aero, cd0=nominal_aero.cd0 * ROBUST_CD0_FACTOR)
        conservative = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            profile_path=self.profile_path,
            calibration=self.calibration,
            assumptions=self.task_calibration.assumptions,
            aero_override=conservative_aero,
        )
        # Physical and mission constraints must hold in the conservative case;
        # geometry constraints are unchanged and already present nominally.
        for name, value in conservative.result.constraints.items():
            if name not in {"span_limit", "stall_speed", "fuel_volume", "fuel_mass_min", "payload_min", "hybrid_mass_max", "mtow_closure", "fuselage_fineness_min"}:
                result.constraints[f"robust_cd0p10_{name}"] = value
        result.diagnostics.update(
            {
                "robust_cd0_factor": ROBUST_CD0_FACTOR,
                "robust_achieved_total_duration_h": conservative.achieved_duration_h,
                "robust_terminal_soc": float(conservative.simulation.profile["soc"].iloc[-1]),
                "robust_minimum_soc": float(conservative.simulation.profile["soc"].min()),
                "robust_cd0_reference": conservative_aero.cd0,
            }
        )
        return CandidateEvaluation(x, result)


def _records(problem: RobustVSPAeroJointProblem, run: object) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(run.evaluations):
        row: dict[str, float | bool | int] = {"candidate_index": index}
        row.update({label: float(value) for label, value in zip(problem.decision_labels, item.decision)})
        row.update(item.result.objectives)
        row.update({f"g_{name}": value for name, value in item.result.constraints.items()})
        row.update(item.result.diagnostics)
        row["feasible"] = item.feasible
        row["total_violation"] = item.violation
        rows.append(row)
    return pd.DataFrame(rows)


def _write_figure(pareto: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(10.5, 7.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    if pareto.empty:
        axis.text2D(0.18, 0.50, "No robust feasible solution at this stage", transform=axis.transAxes, fontsize=15)
    else:
        point = axis.scatter(
            pareto["J_energy_mj"], pareto["J_stability"], pareto["J_response_s"],
            c=pareto["robust_extension_achieved_duration_h"], cmap="viridis", s=70, edgecolor="black", linewidth=0.4,
        )
        figure.colorbar(point, ax=axis, shrink=0.72, pad=0.12, label="extension duration under CD0 +10% (h)")
    axis.set_xlabel("J_energy (MJ)")
    axis.set_ylabel("J_stability (-)")
    axis.set_zlabel("J_response (s)")
    axis.set_title("Robust H=6 MOEA/D feasible Pareto set")
    figure.savefig(OUTPUT / "图7_稳健H6_MOEAD三目标Pareto解集.png", dpi=220)
    plt.close(figure)


def _write_note(population: pd.DataFrame, pareto: pd.DataFrame, raw_pareto_count: int) -> None:
    if pareto.empty:
        result = "本轮 H=6 阶段寻优未找到同时满足名义与 `CD0 +10%` 保守工况的可行解，应将最小违反度候选作为下一轮模型/边界修订依据。"
    else:
        low = float(pareto["robust_extension_achieved_duration_h"].min())
        high = float(pareto["robust_extension_achieved_duration_h"].max())
        result = (
            f"原始非支配记录为 {raw_pareto_count} 个，按目标向量去重后得到 {len(pareto)} 个稳健可行候选；"
            f"所有候选均完成 `CD0 +10%` 下的 80 h 约束，86 h 延伸复核的可达总时长为 {low:.2f} 至 {high:.2f} h。"
        )
    note = f"""# 稳健 H=6 MOEA/D 阶段说明

## 优化定义

外层采用 MOEA/D，决策变量包括机翼/机身/螺旋桨尺寸、有效载荷与混动系统质量分配、ISR 巡航速度，以及六个任务阶段的发动机功率上限与电机辅助功率。目标仍采用 2.2.1 定义的综合能耗、飞行稳定性和任务响应性能；除名义任务约束外，增加 `CD0 +10%` 保守气动工况下的同类任务、SOC、功率和飞行包线约束。

## 本轮结果

{result}

## 适用边界

混动系统质量下限暂设为 60 kg，以保证本阶段仍检验 1.2.2 能量管理接口；该范围、发动机效率图和电池比能量均为待真实部件数据替换的初步假设。本轮用于证明“稳健约束已进入 MOEA/D 可行性判定”，并筛选后续 VSPAERO 加密样本与部件模型校核的重点，不等同于最终设计定型。
"""
    (OUTPUT / "阶段审阅说明.md").write_text(note, encoding="utf-8")


def _extension_record(problem: RobustVSPAeroJointProblem, row: pd.Series) -> dict[str, float | str]:
    decision = row[list(problem.decision_labels)].to_numpy(dtype=float)
    design, engine_caps, motor_assist, speeds = problem._mission_inputs(decision)
    nominal_aero = problem.surrogate.corrected_aero(design, problem.calibration)
    conservative = evaluate_endurance_task(
        design,
        engine_caps_kw=engine_caps,
        motor_assist_kw=motor_assist,
        speeds_mps=speeds,
        target_total_duration_h=86.0,
        profile_path=problem.profile_path,
        calibration=problem.calibration,
        assumptions=problem.task_calibration.assumptions,
        aero_override=replace(nominal_aero, cd0=nominal_aero.cd0 * ROBUST_CD0_FACTOR),
    )
    return {
        "robust_extension_achieved_duration_h": conservative.achieved_duration_h,
        "robust_extension_margin_to_80h": conservative.achieved_duration_h - 80.0,
        "robust_extension_termination_reason": conservative.simulation.termination_reason,
    }


def _unique_objective_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates(subset=list(OBJECTIVE_LABELS), keep="first").copy().reset_index(drop=True)


def _non_dominated_records_from_population(population: pd.DataFrame) -> pd.DataFrame:
    feasible = population.loc[population["feasible"].astype(bool)].copy()
    values = feasible[list(OBJECTIVE_LABELS)].to_numpy(dtype=float)
    keep = []
    for index in range(len(feasible)):
        dominated = np.any(np.all(values <= values[index], axis=1) & np.any(values < values[index], axis=1))
        if not dominated:
            keep.append(index)
    return feasible.iloc[keep].copy().reset_index(drop=True)


def _export_postprocessed_results(problem: RobustVSPAeroJointProblem, population: pd.DataFrame, raw_pareto: pd.DataFrame) -> pd.DataFrame:
    pareto = _unique_objective_candidates(raw_pareto)
    # Reprocessing an already exported file must replace, not duplicate, its
    # extension-validation fields.
    existing_extension_columns = [column for column in pareto.columns if column.startswith("robust_extension_")]
    if existing_extension_columns:
        pareto = pareto.drop(columns=existing_extension_columns)
    extensions = pd.DataFrame([_extension_record(problem, row) for _, row in pareto.iterrows()]) if not pareto.empty else pd.DataFrame()
    if not extensions.empty:
        pareto = pd.concat((pareto, extensions), axis=1)
    raw_pareto.to_csv(OUTPUT / "稳健H6_MOEAD非支配候选_含重复记录.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(OUTPUT / "稳健H6_MOEAD可行Pareto候选.csv", index=False, encoding="utf-8-sig")
    _write_figure(pareto)
    _write_note(population, pareto, len(raw_pareto))
    return pareto


def postprocess_existing_results() -> None:
    """Deduplicate an existing run and add 86-hour robust-margin checks."""
    population = pd.read_csv(OUTPUT / "稳健H6_MOEAD最终种群.csv")
    raw_pareto = _non_dominated_records_from_population(population)
    problem = RobustVSPAeroJointProblem(VSPAeroSurrogate.from_json(SURROGATE_PATH))
    pareto = _export_postprocessed_results(problem, population, raw_pareto)
    metadata_path = OUTPUT / "稳健H6_MOEAD运行说明.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("final_robust_feasible_count", None)
    metadata["final_robust_feasible_records"] = len(raw_pareto)
    metadata["raw_non_dominated_count"] = len(raw_pareto)
    metadata["unique_robust_pareto_count"] = len(pareto)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pareto[["candidate_index", "robust_extension_achieved_duration_h", "robust_extension_margin_to_80h"]].to_string(index=False))


def main() -> None:
    if not SURROGATE_PATH.exists():
        raise FileNotFoundError("Run build_ptdl_vsp_aero_surrogate.py before robust optimization.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    problem = RobustVSPAeroJointProblem(VSPAeroSurrogate.from_json(SURROGATE_PATH))
    run = MOEAD(resolution=H, generations=GENERATIONS, neighborhood_size=5, seed=SEED).run(problem)
    population = _records(problem, run)
    raw_pareto = population.iloc[pareto_indices(run.evaluations)].copy()
    run.history.to_csv(OUTPUT / "稳健H6_MOEAD收敛历史.csv", index=False, encoding="utf-8-sig")
    population.to_csv(OUTPUT / "稳健H6_MOEAD最终种群.csv", index=False, encoding="utf-8-sig")
    pareto = _export_postprocessed_results(problem, population, raw_pareto)
    metadata = {
        "study_type": "stage-level robust VSPAERO-corrected joint PTDL MOEA/D",
        "objectives": list(OBJECTIVE_LABELS),
        "nominal_hard_constraint": "T_endurance >= 80 h",
        "robust_case": {"cd0_factor": ROBUST_CD0_FACTOR, "constraint_rule": "all mission, power, SOC and flight-envelope constraints must also satisfy g <= 0"},
        "moead": {"resolution_H": H, "weights": 28, "generations": GENERATIONS, "seed": SEED},
        "hybrid_mass_study_range_kg": [60.0, 140.0],
        "final_population": len(population),
        "final_robust_feasible_records": len(raw_pareto),
        "raw_non_dominated_count": len(raw_pareto),
        "unique_robust_pareto_count": len(pareto),
    }
    (OUTPUT / "稳健H6_MOEAD运行说明.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not pareto.empty:
        print(pareto[["candidate_index", "J_energy_mj", "J_stability", "J_response_s", "robust_achieved_total_duration_h"]].to_string(index=False))


if __name__ == "__main__":
    postprocess_existing_results() if "--postprocess" in sys.argv else main()
