"""Compare stage-fixed, rule-based, and fuzzy-adaptive PTDL energy management."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ptdl_80h_endurance_profile import calibrate_endurance_task_baseline, evaluate_endurance_task
from ptdl_airframe_design import AirframeDesign
from ptdl_hybrid_coordination import HybridCoordinationParameters, RuleBasedHybridCoordinator
from ptdl_intelligent_energy_management import FuzzyAdaptiveCoordinator
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "ptdl_moead_h6_vsp_coupled_results" / "h6_vsp_feasible_pareto_with_aero.csv"
SURROGATE = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
OUTPUT = ROOT / "ptdl_1_2_2_intelligent_energy_management_results"
PHASES = (
    "main_climb_to_leveloff", "short_leveloff_to_isr_altitude", "endurance_loiter",
    "long_descent", "approach_descent", "flare",
)


def design_from_row(row: pd.Series) -> AirframeDesign:
    labels = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
        "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{label: float(row[label]) for label in labels})


def summarize(label: str, evaluation: object) -> dict[str, float | str]:
    profile = evaluation.simulation.profile
    return {
        "algorithm": label,
        "termination_reason": evaluation.simulation.termination_reason,
        "achieved_duration_h": evaluation.achieved_duration_h,
        "fuel_consumed_kg": float(np.trapezoid(profile["fuel_flow_kgps"], profile["time_s"])),
        "terminal_fuel_kg": evaluation.simulation.terminal_fuel_kg,
        "minimum_soc": float(profile["soc"].min()),
        "terminal_soc": float(profile["soc"].iloc[-1]),
        "mean_engine_power_kw": float(profile["p_engine_kw"].mean()),
        "mean_motor_power_kw": float(profile["p_motor_kw"].mean()),
        "power_mode_changes": int((profile["power_mode"] != profile["power_mode"].shift()).sum() - 1),
        "J_energy_mj": float(evaluation.result.objectives["J_energy_mj"]),
        "J_stability": float(evaluation.result.objectives["J_stability"]),
        "J_response_s": float(evaluation.result.objectives["J_response_s"]),
    }


def plot(cases: dict[str, object]) -> None:
    width, height = 1800, 1450
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        label_font = ImageFont.truetype("arial.ttf", 25)
        legend_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        title_font = label_font = legend_font = ImageFont.load_default()
    left, right = 165, width - 80
    top_soc, bottom_soc = 110, 650
    top_power, bottom_power = 805, 1330
    time_max_h = max(float(evaluation.simulation.profile["time_s"].iloc[-1] / 3600.0) for evaluation in cases.values())
    all_power = np.concatenate([
        evaluation.simulation.profile.loc[:, ["p_engine_kw", "p_motor_kw"]].to_numpy(dtype=float).ravel()
        for evaluation in cases.values()
    ])
    power_min = min(-10.0, float(np.nanmin(all_power)) - 5.0)
    power_max = max(10.0, float(np.nanmax(all_power)) + 5.0)

    def x_coord(value: float) -> int:
        return int(left + (right - left) * value / max(time_max_h, 1e-6))

    def y_coord(value: float, lower: float, upper: float, top: int, bottom: int) -> int:
        return int(bottom - (bottom - top) * (value - lower) / max(upper - lower, 1e-6))

    def draw_axes(top: int, bottom: int, lower: float, upper: float, y_label: str) -> None:
        draw.rectangle((left, top, right, bottom), outline="#3F3F46", width=2)
        for fraction in np.linspace(0.0, 1.0, 5):
            y = int(bottom - (bottom - top) * fraction)
            draw.line((left, y, right, y), fill="#E5E7EB", width=1)
            value = lower + (upper - lower) * fraction
            draw.text((25, y - 12), f"{value:.1f}", fill="#374151", font=legend_font)
        for fraction in np.linspace(0.0, 1.0, 5):
            x = int(left + (right - left) * fraction)
            draw.line((x, top, x, bottom), fill="#F3F4F6", width=1)
            draw.text((x - 16, bottom + 12), f"{time_max_h * fraction:.0f}", fill="#374151", font=legend_font)
        draw.text((left, top - 48), y_label, fill="#111827", font=label_font)

    draw.text((left, 28), "PTDL 1.2.2 closed-loop energy-management comparison", fill="#111827", font=title_font)
    draw_axes(top_soc, bottom_soc, 0.0, 1.0, "SOC")
    draw_axes(top_power, bottom_power, power_min, power_max, "Power (kW)")
    draw.text(((left + right) // 2 - 70, height - 65), "Mission time (h)", fill="#111827", font=label_font)
    colors = {"stage_fixed": "#6B7280", "rule_soc_feedback": "#0072B2", "fuzzy_adaptive": "#D55E00"}
    labels = {"stage_fixed": "Stage-fixed", "rule_soc_feedback": "Rule SOC-feedback", "fuzzy_adaptive": "Fuzzy adaptive"}
    for key, evaluation in cases.items():
        profile = evaluation.simulation.profile
        time_h = profile["time_s"].to_numpy(dtype=float) / 3600.0
        soc = profile["soc"].to_numpy(dtype=float)
        engine = profile["p_engine_kw"].to_numpy(dtype=float)
        motor = profile["p_motor_kw"].to_numpy(dtype=float)
        draw.line([(x_coord(t), y_coord(v, 0.0, 1.0, top_soc, bottom_soc)) for t, v in zip(time_h, soc)], fill=colors[key], width=3)
        draw.line([(x_coord(t), y_coord(v, power_min, power_max, top_power, bottom_power)) for t, v in zip(time_h, engine)], fill=colors[key], width=3)
        draw.line([(x_coord(t), y_coord(v, power_min, power_max, top_power, bottom_power)) for t, v in zip(time_h, motor)], fill=colors[key], width=2)
    legend_y = top_soc + 20
    for key in ("stage_fixed", "rule_soc_feedback", "fuzzy_adaptive"):
        draw.line((right - 330, legend_y + 14, right - 270, legend_y + 14), fill=colors[key], width=4)
        draw.text((right - 255, legend_y), labels[key], fill="#111827", font=legend_font)
        legend_y += 34
    image.save(OUTPUT / "fuzzy_adaptive_soc_power_comparison.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(INPUT).loc[lambda frame: frame["J_energy_mj"].idxmin()]
    design = design_from_row(selected)
    engine_caps = [float(selected[f"engine_cap_{phase}_kw"]) for phase in PHASES]
    motor_assist = [float(selected[f"motor_assist_{phase}_kw"]) for phase in PHASES]
    speeds = [52.0, 36.01, float(selected["isr_cruise_speed_mps"]), 42.0, 30.0, 21.0]
    calibration = calibrate_endurance_task_baseline()
    aero = VSPAeroSurrogate.from_json(SURROGATE).corrected_aero(design, calibration.dimension_calibration)
    common = dict(
        design=design, engine_caps_kw=engine_caps, motor_assist_kw=motor_assist, speeds_mps=speeds,
        calibration=calibration.dimension_calibration, assumptions=calibration.assumptions, aero_override=aero,
    )
    cases = {
        "stage_fixed": evaluate_endurance_task(**common),
        "rule_soc_feedback": evaluate_endurance_task(
            **common, dispatch_policy=RuleBasedHybridCoordinator(HybridCoordinationParameters())
        ),
        "fuzzy_adaptive": evaluate_endurance_task(**common, dispatch_policy=FuzzyAdaptiveCoordinator()),
    }
    for label, evaluation in cases.items():
        evaluation.simulation.profile.to_csv(OUTPUT / f"{label}_time_history.csv", index=False)
    summary = pd.DataFrame([summarize(label, evaluation) for label, evaluation in cases.items()])
    summary.to_csv(OUTPUT / "intelligent_energy_management_comparison.csv", index=False)
    plot(cases)
    metadata = {
        "study": "PTDL 1.2.2 initial algorithm comparison",
        "candidate_source": "H=6 VSPAERO-corrected Pareto set, minimum-energy candidate",
        "candidate_index": int(selected["candidate_index"]),
        "algorithms": {
            "stage_fixed": "phase-fixed engine and motor schedule",
            "rule_soc_feedback": "SOC feedback, tracking reserve, and generator slew control",
            "fuzzy_adaptive": "fuzzy SOC-demand-tracking power split with safety limits",
        },
        "scope_limit": "The neural SOC predictor and component-fault plant models are interfaces pending high-fidelity component data and labeled scenario data.",
    }
    (OUTPUT / "intelligent_energy_management_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
