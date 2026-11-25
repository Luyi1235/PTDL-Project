"""Tune PTDL 1.2.2 fuzzy-dispatch parameters on a fixed hybrid demonstrator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from MOEAD多目标联合优化 import CandidateEvaluation, MOEAD, pareto_indices
from 八十小时续航任务模型 import calibrate_endurance_task_baseline, evaluate_endurance_task
from 机体参数化设计与气动阻力 import AirframeDesign
from 智能能量管理与容错调度 import FUZZY_SOC_TARGETS, FuzzyAdaptiveCoordinator, FuzzyEnergyManagementParameters
from VSPAERO气动代理模型 import VSPAeroSurrogate


OUTPUT = ROOT / "运行结果" / "模糊调度参数MOEAD预寻优"
SURROGATE = ROOT / "气动代理模型数据" / "气动代理模型.json"
PHASES = (
    "main_climb_to_leveloff", "short_leveloff_to_isr_altitude", "endurance_loiter",
    "long_descent", "approach_descent", "flare",
)


class FuzzyTuningProblem:
    """MOEA/D adapter for online fuzzy-control parameters, with a nonzero hybrid package."""

    decision_labels = (
        "soc_target_bias",
        "maximum_charge_bias_kw",
        "maximum_assist_bias_kw",
        "tracking_reserve_limit_kw",
        "engine_target_rate_limit_kwps",
    )
    lower_bounds = np.asarray((-0.06, 4.0, 0.0, 0.0, 1.0), dtype=float)
    upper_bounds = np.asarray((0.06, 22.0, 16.0, 20.0, 5.0), dtype=float)

    def __init__(self) -> None:
        calibration = calibrate_endurance_task_baseline()
        self.calibration = calibration.dimension_calibration
        self.assumptions = calibration.assumptions
        self.design = AirframeDesign(
            wing_area_m2=25.18,
            aspect_ratio=30.0,
            taper_ratio=0.45,
            sweep_deg=0.30,
            tip_twist_deg=-0.63,
            fuselage_length_m=8.25,
            fuselage_diameter_m=0.667,
            propeller_diameter_m=2.29,
            payload_mass_kg=110.0,
            hybrid_system_mass_kg=80.0,
        )
        self.engine_caps_kw = (145.0, 145.0, 115.0, 65.0, 80.0, 55.0)
        self.motor_assist_kw = (0.0,) * len(PHASES)
        self.speeds_mps = (52.0, 36.0, 30.1, 42.0, 30.0, 21.0)
        self.aero = VSPAeroSurrogate.from_json(SURROGATE).corrected_aero(self.design, self.calibration)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        samples = rng.uniform(self.lower_bounds, self.upper_bounds, size=(count, len(self.decision_labels)))
        if count:
            samples[0] = np.asarray((0.0, 18.0, 12.0, 18.0, 3.0))
        return samples

    def evaluate(self, decision: Sequence[float]) -> CandidateEvaluation:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        target_bias, charge_bias, assist_bias, tracking_reserve, rate_limit = x
        targets = {name: float(np.clip(value + target_bias, 0.25, 0.90)) for name, value in FUZZY_SOC_TARGETS.items()}
        parameters = FuzzyEnergyManagementParameters(
            soc_targets=targets,
            maximum_charge_bias_kw=float(charge_bias),
            maximum_assist_bias_kw=float(assist_bias),
            tracking_reserve_limit_kw=float(tracking_reserve),
            engine_target_rate_limit_kwps=float(rate_limit),
        )
        evaluation = evaluate_endurance_task(
            self.design,
            engine_caps_kw=self.engine_caps_kw,
            motor_assist_kw=self.motor_assist_kw,
            speeds_mps=self.speeds_mps,
            calibration=self.calibration,
            assumptions=self.assumptions,
            aero_override=self.aero,
            dispatch_policy=FuzzyAdaptiveCoordinator(parameters),
        )
        return CandidateEvaluation(x, evaluation.result)


def records(run: object, indices: list[int]) -> pd.DataFrame:
    rows = []
    for index in indices:
        item = run.evaluations[index]
        row = {"candidate_index": index, **dict(zip(FuzzyTuningProblem.decision_labels, item.decision))}
        row.update(item.result.objectives)
        row.update({f"g_{name}": value for name, value in item.result.constraints.items()})
        row["total_violation"] = item.violation
        row["feasible"] = item.feasible
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    problem = FuzzyTuningProblem()
    run = MOEAD(resolution=3, generations=16, neighborhood_size=5, seed=20260970).run(problem)
    run.history.to_csv(OUTPUT / "fuzzy_tuning_convergence.csv", index=False)
    population = records(run, list(range(len(run.evaluations))))
    # Different weight subproblems can converge to numerically identical
    # objective vectors.  Keep one representative so the report does not
    # mistake repeated solutions for a Pareto trade-off.
    pareto = records(run, pareto_indices(run.evaluations)).drop_duplicates(
        subset=["J_energy_mj", "J_stability", "J_response_s"]
    )
    population.to_csv(OUTPUT / "fuzzy_tuning_final_population.csv", index=False)
    pareto.to_csv(OUTPUT / "fuzzy_tuning_feasible_pareto.csv", index=False)
    metadata = {
        "scope": "preliminary fuzzy-parameter MOEA/D tuning with a fixed 80 kg hybrid package",
        "hard_constraint": "T_endurance >= 80 h",
        "moead": {"resolution_H": 3, "weights": 10, "generations": 16, "seed": 20260970},
        "fixed_design_note": "The hybrid mass and component capabilities are provisional placeholders pending propulsion-system data.",
        "final_feasible_count": int(population["feasible"].sum()),
        "pareto_count": len(pareto),
        "interpretation": (
            "Under the fixed 80-hour reference profile, all three current objectives converge to one energy-management solution. "
            "This validates the tuning interface but does not yet establish a control-parameter Pareto trade-off."
        ),
    }
    (OUTPUT / "fuzzy_tuning_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(pareto.to_string(index=False))


if __name__ == "__main__":
    main()
