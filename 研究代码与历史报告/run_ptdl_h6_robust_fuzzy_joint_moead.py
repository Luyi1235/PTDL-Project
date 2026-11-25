"""Robust H=6 MOEA/D with joint airframe, stage-control and fuzzy EMS tuning.

The fuzzy parameters alter the physical motor reserve and engine transient
command.  Their effect is therefore exposed when power capability is tight,
instead of being represented by an artificial objective penalty.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_2_2_2_optimizer import CandidateEvaluation, MOEAD, OBJECTIVE_LABELS, pareto_indices
from ptdl_80h_endurance_profile import evaluate_endurance_task
from ptdl_intelligent_energy_management import FUZZY_SOC_TARGETS, FuzzyAdaptiveCoordinator, FuzzyEnergyManagementParameters
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from run_ptdl_h6_robust_vsp_moead import ROBUST_CD0_FACTOR, RobustVSPAeroJointProblem


ROOT = Path(__file__).resolve().parent
# Candidate-region VSPAERO enrichment is now available.  Keep the original
# 15-point model intact, but use the independently validated 23-point model
# for new joint optimization runs in the robust-candidate region.
SURROGATE_PATH = ROOT / "ptdl_vsp_targeted_candidate_results" / "VSPAERO扩充代理模型.json"
FUZZY_LABELS = (
    "fuzzy_soc_target_bias",
    "fuzzy_maximum_charge_bias_kw",
    "fuzzy_maximum_assist_bias_kw",
    "fuzzy_tracking_reserve_limit_kw",
    "fuzzy_engine_target_rate_limit_kwps",
)
FUZZY_LOWER = np.asarray((-0.06, 4.0, 0.0, 0.0, 1.0), dtype=float)
FUZZY_UPPER = np.asarray((0.06, 22.0, 16.0, 20.0, 5.0), dtype=float)


class RobustFuzzyJointProblem(RobustVSPAeroJointProblem):
    """Extend the robust joint problem with physically active EMS parameters."""

    def __init__(self, surrogate: VSPAeroSurrogate) -> None:
        super().__init__(surrogate)
        self.base_decision_count = len(self.decision_labels)
        self.decision_labels = self.decision_labels + FUZZY_LABELS
        self.lower_bounds = np.concatenate((self.lower_bounds, FUZZY_LOWER))
        self.upper_bounds = np.concatenate((self.upper_bounds, FUZZY_UPPER))

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        samples = super().sample(count, rng)
        if count:
            samples[0, self.base_decision_count :] = np.asarray((0.0, 18.0, 12.0, 18.0, 3.0))
        return samples

    def _mission_inputs(self, x: np.ndarray):
        """Keep the six motor-stage variables separate from fuzzy parameters."""
        design = self.design_from_decision(x)
        phase_count = len(self.phase_labels)
        engine_start = 11
        motor_start = engine_start + phase_count
        engine_caps = x[engine_start:motor_start]
        motor_assist = x[motor_start:motor_start + phase_count]
        speeds = [phase.speed_target_mps for phase in self.base_mission.phases]
        speeds[self.phase_labels.index("endurance_loiter")] = float(x[10])
        return design, engine_caps, motor_assist, speeds

    def _policy_from_decision(self, x: np.ndarray) -> FuzzyAdaptiveCoordinator:
        bias, charge, assist, reserve, rate = x[self.base_decision_count :]
        targets = {
            name: float(np.clip(target + bias, 0.25, 0.90))
            for name, target in FUZZY_SOC_TARGETS.items()
        }
        return FuzzyAdaptiveCoordinator(FuzzyEnergyManagementParameters(
            soc_targets=targets,
            maximum_charge_bias_kw=float(charge),
            maximum_assist_bias_kw=float(assist),
            tracking_reserve_limit_kw=float(reserve),
            engine_target_rate_limit_kwps=float(rate),
        ))

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
            dispatch_policy=self._policy_from_decision(x),
        )
        result = nominal.result
        result.diagnostics.update({
            "nominal_achieved_total_duration_h": nominal.achieved_duration_h,
            "nominal_cd0_reference": nominal_aero.cd0,
            **{label: float(value) for label, value in zip(FUZZY_LABELS, x[self.base_decision_count :])},
        })
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
            dispatch_policy=self._policy_from_decision(x),
        )
        for name, value in conservative.result.constraints.items():
            if name not in {
                "span_limit", "stall_speed", "fuel_volume", "fuel_mass_min", "payload_min",
                "hybrid_mass_max", "mtow_closure", "fuselage_fineness_min",
            }:
                result.constraints[f"robust_cd0p10_{name}"] = value
        result.diagnostics.update({
            "robust_cd0_factor": ROBUST_CD0_FACTOR,
            "robust_achieved_total_duration_h": conservative.achieved_duration_h,
            "robust_terminal_soc": float(conservative.simulation.profile["soc"].iloc[-1]),
            "robust_minimum_soc": float(conservative.simulation.profile["soc"].min()),
            "robust_cd0_reference": conservative_aero.cd0,
        })
        return CandidateEvaluation(x, result)


def records(problem: RobustFuzzyJointProblem, run: object) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(run.evaluations):
        row: dict[str, float | bool | int] = {"candidate_index": index}
        row.update({label: float(value) for label, value in zip(problem.decision_labels, item.decision)})
        row.update(item.result.objectives)
        row.update({f"g_{name}": float(value) for name, value in item.result.constraints.items()})
        row.update(item.result.diagnostics)
        row["feasible"] = item.feasible
        row["total_violation"] = item.violation
        rows.append(row)
    return pd.DataFrame(rows)


def export_figures(output: Path, history: pd.DataFrame, pareto: pd.DataFrame) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    axes[0].plot(history["generation"], history["feasible_ratio"], marker="o", ms=3, color="#1769aa")
    axes[0].set(title="可行解比例收敛", xlabel="迭代代数", ylabel="可行解比例", ylim=(-0.03, 1.03))
    axes[0].grid(alpha=0.25)
    if pareto.empty:
        axes[1].text(0.16, 0.5, "未获得稳健可行解", transform=axes[1].transAxes, fontsize=14)
    else:
        points = axes[1].scatter(
            pareto["J_energy_mj"], pareto["J_stability"], c=pareto["fuzzy_tracking_reserve_limit_kw"],
            cmap="viridis", s=68, edgecolor="#1f2937", linewidth=0.4,
        )
        figure.colorbar(points, ax=axes[1], label="功率储备参数 (kW)")
    axes[1].set(title="联合优化的可行 Pareto 投影", xlabel="综合能耗 J_energy (MJ)", ylabel="飞行稳定性 J_stability (-)")
    axes[1].grid(alpha=0.25)
    figure.savefig(output / "图11_模糊调度参数与构型控制联合MOEAD结果.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="ptdl_moead_h6_robust_fuzzy_joint_results")
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260980)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    problem = RobustFuzzyJointProblem(VSPAeroSurrogate.from_json(SURROGATE_PATH))
    run = MOEAD(resolution=args.resolution, generations=args.generations, neighborhood_size=5, seed=args.seed).run(problem)
    population = records(problem, run)
    pareto = population.iloc[pareto_indices(run.evaluations)].copy()
    pareto = pareto.drop_duplicates(subset=list(OBJECTIVE_LABELS)).reset_index(drop=True)
    population.to_csv(output / "稳健模糊联合MOEAD最终种群.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(output / "稳健模糊联合MOEAD可行Pareto候选.csv", index=False, encoding="utf-8-sig")
    run.history.to_csv(output / "稳健模糊联合MOEAD收敛历史.csv", index=False, encoding="utf-8-sig")
    export_figures(output, run.history, pareto)
    metadata = {
        "scope": "robust joint optimization of geometry, phase power variables and five fuzzy EMS parameters",
        "hard_constraint": "T_endurance >= 80 h under nominal and CD0 +10% cases",
        "moead": {"resolution_H": args.resolution, "generations": args.generations, "seed": args.seed},
        "fuzzy_parameters": list(FUZZY_LABELS),
        "final_feasible_records": int(population["feasible"].sum()),
        "unique_pareto_count": len(pareto),
        "physical_interpretation": "The power-reserve parameter limits available motor support during low-error transient operation and releases it as tracking error grows.",
    }
    (output / "运行说明.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not pareto.empty:
        print(pareto[["candidate_index", "J_energy_mj", "J_stability", "J_response_s", *FUZZY_LABELS]].to_string(index=False))


if __name__ == "__main__":
    main()
