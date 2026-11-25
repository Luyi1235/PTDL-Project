"""Recheck robust H=6 candidates using the separately enriched VSPAERO model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from ptdl_80h_endurance_profile import evaluate_endurance_task
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from run_ptdl_h6_robust_vsp_moead import ROBUST_CD0_FACTOR, RobustVSPAeroJointProblem


ROOT = Path(__file__).resolve().parent
PARETO = ROOT / "ptdl_moead_h6_robust_vsp_results" / "稳健H6_MOEAD可行Pareto候选.csv"
ORIGINAL = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
ENRICHED = ROOT / "ptdl_vsp_targeted_candidate_results" / "VSPAERO扩充代理模型.json"
OUTPUT = ROOT / "ptdl_vsp_targeted_candidate_results"


def extension_check(problem: RobustVSPAeroJointProblem, decision) -> tuple[float, float, str]:
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
    return conservative.achieved_duration_h, nominal_aero.cd0, conservative.simulation.termination_reason


def main() -> None:
    candidates = pd.read_csv(PARETO)
    original_problem = RobustVSPAeroJointProblem(VSPAeroSurrogate.from_json(ORIGINAL))
    enriched_problem = RobustVSPAeroJointProblem(VSPAeroSurrogate.from_json(ENRICHED))
    rows = []
    for _, candidate in candidates.iterrows():
        decision = candidate[list(original_problem.decision_labels)].to_numpy(dtype=float)
        original_result = original_problem.evaluate(decision).result
        enriched_result = enriched_problem.evaluate(decision).result
        original_extension, original_cd0, original_reason = extension_check(original_problem, decision)
        enriched_extension, enriched_cd0, enriched_reason = extension_check(enriched_problem, decision)
        rows.append({
            "candidate_index": int(candidate["candidate_index"]),
            "prior_CL_alpha_per_rad": original_result.diagnostics["aero_cl_alpha_per_rad"],
            "enriched_CL_alpha_per_rad": enriched_result.diagnostics["aero_cl_alpha_per_rad"],
            "prior_oswald_efficiency": original_result.diagnostics["aero_oswald_efficiency"],
            "enriched_oswald_efficiency": enriched_result.diagnostics["aero_oswald_efficiency"],
            "prior_cd0": original_cd0,
            "enriched_cd0": enriched_cd0,
            "prior_robust_80h_feasible": original_result.feasible,
            "enriched_robust_80h_feasible": enriched_result.feasible,
            "prior_robust_86h_achieved_duration_h": original_extension,
            "enriched_robust_86h_achieved_duration_h": enriched_extension,
            "prior_robust_margin_to_80h": original_extension - 80.0,
            "enriched_robust_margin_to_80h": enriched_extension - 80.0,
            "prior_termination_reason": original_reason,
            "enriched_termination_reason": enriched_reason,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "扩充VSPAERO代理模型_稳健候选复核.csv", index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
