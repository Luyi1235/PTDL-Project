"""Run the first 1.2.1 hybrid flight-control coordination comparison case."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptdl_80h_endurance_profile import calibrate_endurance_task_baseline, evaluate_endurance_task
from ptdl_airframe_design import AirframeDesign
from ptdl_hybrid_coordination import HybridCoordinationParameters, RuleBasedHybridCoordinator
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "ptdl_moead_h6_vsp_coupled_results" / "h6_vsp_feasible_pareto_with_aero.csv"
SURROGATE = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
OUTPUT = ROOT / "ptdl_1_2_1_coordination_results"


def design_from_row(row: pd.Series) -> AirframeDesign:
    labels = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
        "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{label: float(row[label]) for label in labels})


def summary(label: str, evaluation: object) -> dict[str, float | str]:
    profile = evaluation.simulation.profile
    mode_changes = int((profile["power_mode"] != profile["power_mode"].shift()).sum() - 1) if "power_mode" in profile else 0
    return {
        "case": label,
        "termination_reason": evaluation.simulation.termination_reason,
        "achieved_duration_h": evaluation.achieved_duration_h,
        "fuel_consumed_kg": float(np.trapezoid(profile["fuel_flow_kgps"], profile["time_s"])),
        "terminal_fuel_kg": evaluation.simulation.terminal_fuel_kg,
        "minimum_soc": float(profile["soc"].min()),
        "terminal_soc": float(profile["soc"].iloc[-1]),
        "mean_engine_power_kw": float(profile["p_engine_kw"].mean()),
        "mean_motor_power_kw": float(profile["p_motor_kw"].mean()),
        "power_mode_changes": mode_changes,
        "J_energy_mj": float(evaluation.result.objectives["J_energy_mj"]),
        "J_stability": float(evaluation.result.objectives["J_stability"]),
        "J_response_s": float(evaluation.result.objectives["J_response_s"]),
    }


def plot_comparison(static: object, coordinated: object) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.5), sharex=True, constrained_layout=True)
    for label, evaluation, color in (
        ("Stage-fixed schedule", static, "#6B7280"),
        ("SOC-feedback coordination", coordinated, "#0072B2"),
    ):
        profile = evaluation.simulation.profile
        time_h = profile["time_s"] / 3600.0
        axes[0].plot(time_h, profile["soc"], label=label, color=color, linewidth=1.6)
        axes[1].plot(time_h, profile["p_engine_kw"], label=f"{label}: engine", color=color, linewidth=1.4)
        axes[1].plot(time_h, profile["p_motor_kw"], label=f"{label}: motor", color=color, linewidth=1.1, linestyle="--")
    axes[0].set(ylabel="SOC", title="PTDL 1.2.1 hybrid-flight-control coordination comparison")
    axes[1].set(xlabel="Mission time (h)", ylabel="Power (kW)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.savefig(OUTPUT / "coordination_soc_power_comparison.png", dpi=260)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(INPUT).loc[lambda frame: frame["J_energy_mj"].idxmin()]
    design = design_from_row(selected)
    phase_names = (
        "main_climb_to_leveloff", "short_leveloff_to_isr_altitude", "endurance_loiter",
        "long_descent", "approach_descent", "flare",
    )
    engine_caps = [float(selected[f"engine_cap_{name}_kw"]) for name in phase_names]
    motor_assist = [float(selected[f"motor_assist_{name}_kw"]) for name in phase_names]
    speeds = [52.0, 36.01, float(selected["isr_cruise_speed_mps"]), 42.0, 30.0, 21.0]
    calibration = calibrate_endurance_task_baseline()
    surrogate = VSPAeroSurrogate.from_json(SURROGATE)
    aero = surrogate.corrected_aero(design, calibration.dimension_calibration)

    static = evaluate_endurance_task(
        design, engine_caps_kw=engine_caps, motor_assist_kw=motor_assist, speeds_mps=speeds,
        calibration=calibration.dimension_calibration, assumptions=calibration.assumptions, aero_override=aero,
    )
    coordinator = RuleBasedHybridCoordinator(HybridCoordinationParameters())
    coordinated = evaluate_endurance_task(
        design, engine_caps_kw=engine_caps, motor_assist_kw=motor_assist, speeds_mps=speeds,
        calibration=calibration.dimension_calibration, assumptions=calibration.assumptions, aero_override=aero,
        dispatch_policy=coordinator,
    )
    static.simulation.profile.to_csv(OUTPUT / "stage_fixed_time_history.csv", index=False)
    coordinated.simulation.profile.to_csv(OUTPUT / "soc_feedback_coordination_time_history.csv", index=False)
    summaries = pd.DataFrame((summary("stage_fixed", static), summary("soc_feedback_coordination", coordinated)))
    summaries.to_csv(OUTPUT / "coordination_comparison_summary.csv", index=False)
    plot_comparison(static, coordinated)
    metadata = {
        "study": "PTDL 1.2.1 first hybrid-flight-control coordination comparison",
        "candidate_source": "H=6 VSPAERO-corrected feasible Pareto set, minimum-energy candidate",
        "candidate_index": int(selected["candidate_index"]),
        "aerodynamics": "VSPAERO DOE surrogate relative correction for CL_alpha and Oswald efficiency; calibrated low-order CD0",
        "coordination_policy": "phase-dependent SOC target, SOC feedback, tracking reserve, generator slew limiting",
        "parameters": HybridCoordinationParameters().__dict__,
        "assumption_status": "preliminary powertrain values pending engine, motor, battery, and controller test data",
    }
    (OUTPUT / "coordination_case_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
