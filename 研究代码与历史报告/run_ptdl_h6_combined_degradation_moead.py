"""Optimize PTDL geometry, phase controls and EMS under combined degradation.

The outer MOEA/D problem retains the nominal and ``CD0 +10%`` cases, then
adds a mandatory 80-hour case with simultaneous drag growth, engine derating,
battery energy reduction and lower engine efficiency.  It is an engineering
stress-test interface: the provisional fractions can be replaced by measured
component maps without changing the optimization structure.
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
from ptdl_intelligent_energy_management import FaultTolerantFuzzyCoordinator, HealthState
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from run_ptdl_h6_robust_fuzzy_joint_moead import (
    FUZZY_LABELS,
    SURROGATE_PATH,
    RobustFuzzyJointProblem,
    records,
)


ROOT = Path(__file__).resolve().parent
REFERENCE_PARETO_PATH = ROOT / "ptdl_moead_h6_robust_vsp_results" / "稳健H6_MOEAD可行Pareto候选.csv"
CONSTRUCTIVE_SEED_PATH = ROOT / "ptdl_combined_degradation_refinement_results" / "组合退化可行修复种子.csv"
COMBINED_CD0_FACTOR = 1.10
COMBINED_ENGINE_AVAILABLE_FRACTION = 0.85
COMBINED_BATTERY_ENERGY_FRACTION = 0.80
COMBINED_ENGINE_EFFICIENCY_FACTOR = 0.95
GEOMETRY_ONLY_CONSTRAINTS = {
    "span_limit", "stall_speed", "fuel_volume", "fuel_mass_min", "payload_min",
    "hybrid_mass_max", "mtow_closure", "fuselage_fineness_min",
}


class CombinedDegradationRobustProblem(RobustFuzzyJointProblem):
    """Require all nominal, aero-robust and combined-degradation constraints."""

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        samples = super().sample(count, rng)
        # Seed verified aero-robust candidates rather than relying only on a
        # nominal feasibility anchor.  Their fuzzy parameters remain at the
        # declared default until the outer loop adapts them.
        if count and REFERENCE_PARETO_PATH.exists():
            references = pd.read_csv(REFERENCE_PARETO_PATH).sort_values("candidate_index")
            for sample_index, (_, row) in enumerate(references.head(count).iterrows()):
                for decision_index, label in enumerate(self.decision_labels[:self.base_decision_count]):
                    if label in row.index:
                        samples[sample_index, decision_index] = float(row[label])
                samples[sample_index, self.base_decision_count:] = np.asarray((0.0, 18.0, 12.0, 18.0, 3.0))
        if count >= 3 and CONSTRUCTIVE_SEED_PATH.exists():
            seed = pd.read_csv(CONSTRUCTIVE_SEED_PATH).iloc[0]
            for decision_index, label in enumerate(self.decision_labels):
                if label in seed.index:
                    samples[2, decision_index] = float(seed[label])
        return np.clip(samples, self.lower_bounds, self.upper_bounds)

    def _combined_policy(self, x: np.ndarray) -> FaultTolerantFuzzyCoordinator:
        return FaultTolerantFuzzyCoordinator(
            self._policy_from_decision(x),
            HealthState(
                engine_available_fraction=COMBINED_ENGINE_AVAILABLE_FRACTION,
                battery_available_energy_fraction=COMBINED_BATTERY_ENERGY_FRACTION,
            ),
        )

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
            **{label: float(value) for label, value in zip(FUZZY_LABELS, x[self.base_decision_count:])},
        })
        aero_robust = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            profile_path=self.profile_path,
            calibration=self.calibration,
            assumptions=self.task_calibration.assumptions,
            aero_override=replace(nominal_aero, cd0=nominal_aero.cd0 * COMBINED_CD0_FACTOR),
            dispatch_policy=self._policy_from_decision(x),
        )
        for name, value in aero_robust.result.constraints.items():
            if name not in GEOMETRY_ONLY_CONSTRAINTS:
                result.constraints[f"robust_cd0p10_{name}"] = float(value)
        result.diagnostics.update({
            "robust_achieved_total_duration_h": aero_robust.achieved_duration_h,
            "robust_terminal_soc": float(aero_robust.simulation.profile["soc"].iloc[-1]),
            "robust_minimum_soc": float(aero_robust.simulation.profile["soc"].min()),
        })

        combined_aero = replace(nominal_aero, cd0=nominal_aero.cd0 * COMBINED_CD0_FACTOR)
        combined_assumptions = replace(
            self.task_calibration.assumptions,
            engine_efficiency_values=tuple(
                value * COMBINED_ENGINE_EFFICIENCY_FACTOR
                for value in self.task_calibration.assumptions.engine_efficiency_values
            ),
            hybrid_system_specific_energy_wh_per_kg=(
                self.task_calibration.assumptions.hybrid_system_specific_energy_wh_per_kg
                * COMBINED_BATTERY_ENERGY_FRACTION
            ),
        )
        combined = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            profile_path=self.profile_path,
            target_total_duration_h=80.0,
            calibration=self.calibration,
            assumptions=combined_assumptions,
            aero_override=combined_aero,
            dispatch_policy=self._combined_policy(x),
        )
        for name, value in combined.result.constraints.items():
            if name not in GEOMETRY_ONLY_CONSTRAINTS:
                result.constraints[f"combined_{name}"] = float(value)
        achieved_h = float(combined.achieved_duration_h)
        # Keep the 80 h requirement explicit even if a future evaluator changes
        # its internal mission-duration constraint representation.
        result.constraints["combined_endurance_80h"] = 80.0 - achieved_h
        profile = combined.simulation.profile
        result.diagnostics.update(
            {
                "combined_cd0_factor": COMBINED_CD0_FACTOR,
                "combined_engine_available_fraction": COMBINED_ENGINE_AVAILABLE_FRACTION,
                "combined_battery_energy_fraction": COMBINED_BATTERY_ENERGY_FRACTION,
                "combined_engine_efficiency_factor": COMBINED_ENGINE_EFFICIENCY_FACTOR,
                "combined_achieved_total_duration_h": achieved_h,
                "combined_margin_to_80h": achieved_h - 80.0,
                "combined_terminal_soc": float(profile["soc"].iloc[-1]),
                "combined_minimum_soc": float(profile["soc"].min()),
                "combined_cd0_reference": combined_aero.cd0,
            }
        )
        return CandidateEvaluation(x, result)


def unique_feasible_pareto(population: pd.DataFrame) -> pd.DataFrame:
    feasible = population[population["feasible"]].copy()
    if feasible.empty:
        return feasible
    values = feasible[list(OBJECTIVE_LABELS)].to_numpy(dtype=float)
    keep = []
    for index in range(len(feasible)):
        dominated = np.any(np.all(values <= values[index], axis=1) & np.any(values < values[index], axis=1))
        if not dominated:
            keep.append(index)
    return feasible.iloc[keep].drop_duplicates(list(OBJECTIVE_LABELS)).reset_index(drop=True)


def write_figures(output: Path, history: pd.DataFrame, pareto: pd.DataFrame) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    axes[0].plot(history["generation"], history["feasible_ratio"], marker="o", ms=3, color="#1769aa")
    axes[0].set(title="组合退化约束下的可行率收敛", xlabel="迭代代数", ylabel="可行解比例", ylim=(-0.03, 1.03))
    axes[0].grid(alpha=0.25)
    if pareto.empty:
        axes[1].text(0.20, 0.50, "未获得组合退化可行解", transform=axes[1].transAxes, fontsize=13)
    else:
        order = pareto.sort_values("combined_achieved_total_duration_h")
        colors = ["#1769aa" if value >= 80.0 else "#c2410c" for value in order["combined_achieved_total_duration_h"]]
        axes[1].bar(range(len(order)), order["combined_achieved_total_duration_h"], color=colors)
        axes[1].axhline(80.0, color="#c2410c", linestyle="--", label="80 h 硬约束")
        axes[1].set(title="组合退化下的可行 Pareto 候选", xlabel="候选排序", ylabel="可达总时长 (h)")
        axes[1].legend(frameon=False)
    figure.savefig(output / "图18_组合退化约束MOEAD收敛与80小时复核.png", dpi=220)
    plt.close(figure)

    figure = plt.figure(figsize=(9.3, 7.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    if pareto.empty:
        axis.text2D(0.18, 0.50, "未获得组合退化可行解", transform=axis.transAxes, fontsize=14)
    else:
        points = axis.scatter(
            pareto["J_energy_mj"], pareto["J_stability"], pareto["J_response_s"],
            c=pareto["combined_margin_to_80h"], cmap="viridis", s=72, edgecolor="#1f2937", linewidth=0.4,
        )
        figure.colorbar(points, ax=axis, shrink=0.72, pad=0.12, label="组合退化续航裕度 (h)")
    axis.set(xlabel="J_energy (MJ)", ylabel="J_stability (-)", zlabel="J_response (s)")
    axis.set_title("组合退化约束下的三目标 Pareto 解集")
    figure.savefig(output / "图19_组合退化约束三目标Pareto解集.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="ptdl_moead_h6_combined_degradation_results")
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--generations", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260990)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    problem = CombinedDegradationRobustProblem(VSPAeroSurrogate.from_json(SURROGATE_PATH))
    run = MOEAD(resolution=args.resolution, generations=args.generations, neighborhood_size=5, seed=args.seed).run(problem)
    population = records(problem, run)
    raw_pareto = population.iloc[pareto_indices(run.evaluations)].copy()
    pareto = unique_feasible_pareto(population)
    run.history.to_csv(output / "组合退化MOEAD收敛历史.csv", index=False, encoding="utf-8-sig")
    population.to_csv(output / "组合退化MOEAD最终种群.csv", index=False, encoding="utf-8-sig")
    raw_pareto.to_csv(output / "组合退化MOEAD原始非支配记录.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(output / "组合退化MOEAD可行Pareto候选.csv", index=False, encoding="utf-8-sig")
    write_figures(output, run.history, pareto)
    metadata = {
        "study_type": "H=6 combined-degradation robust fuzzy joint MOEA/D",
        "objectives": list(OBJECTIVE_LABELS),
        "hard_constraint": "combined scenario T_endurance >= 80 h and all evaluator constraints g <= 0",
        "combined_scenario": {
            "cd0_factor": COMBINED_CD0_FACTOR,
            "engine_available_fraction": COMBINED_ENGINE_AVAILABLE_FRACTION,
            "battery_energy_fraction": COMBINED_BATTERY_ENERGY_FRACTION,
            "engine_efficiency_factor": COMBINED_ENGINE_EFFICIENCY_FACTOR,
        },
        "moead": {"resolution_H": args.resolution, "weights": int((args.resolution + 1) * (args.resolution + 2) / 2), "generations": args.generations, "seed": args.seed},
        "final_population": len(population),
        "raw_non_dominated_records": len(raw_pareto),
        "unique_feasible_pareto_candidates": len(pareto),
    }
    (output / "组合退化MOEAD运行说明.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    note = (
        "# 组合退化约束 MOEA/D 阶段说明\n\n"
        "外层 MOEA/D 在名义、CD0 +10% 和组合退化条件下评价每个构型、阶段功率与模糊调度参数。"
        "组合退化包括 CD0 +10%、发动机可用功率 85%、电池可用能量 80% 与发动机效率 95%，"
        "并要求 80 h 任务及全部能源、功率和飞行约束同时满足。"
        "垂直风扰属于短时飞行控制扰动，在独立的 1.2.3 闭环恢复试验中验证，未被错误施加为 80 h 持续风场。\n"
    )
    (output / "阶段审阅说明.md").write_text(note, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not pareto.empty:
        print(pareto[["candidate_index", "J_energy_mj", "J_stability", "J_response_s", "combined_achieved_total_duration_h", "combined_margin_to_80h"]].to_string(index=False))


if __name__ == "__main__":
    main()
