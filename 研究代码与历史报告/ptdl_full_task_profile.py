"""Adapter from the supplied ULTRA checkout JSON to the PTDL full-task model.

The source JSON is a 5,000 km checkout profile rather than an 80 h mission.
This adapter retains its airborne phase structure and boundary targets, skips
the ground-roll phases that the point-mass flight model cannot represent, and
extends only the ISR cruise phase when an 80 h total-duration study is needed.
All preliminary propulsion values in this module are declared assumptions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from ptdl_2_2_1_evaluator import EvaluationResult
from ptdl_airframe_design import AeroCoefficients, AirframeDesign, aircraft_from_design, design_metrics, geometry_constraints
from ptdl_fixed_mtow_sizing import calibrate_dimension_baseline
from ptdl_flight_propulsion_control import (
    AircraftParameters,
    EngineGeneratorModel,
    HybridPowertrainModel,
    MissionDefinition,
    MissionSimulation,
    PhaseCommand,
    evaluate_coupled_mission,
)
from ptdl_ultra_benchmark import UltraCalibration


DEFAULT_PROFILE_PATH = Path("飞推控耦合参考") / "参考" / "flight_profile_ultra_70h.json"


@dataclass(frozen=True)
class PreliminaryPropulsionAssumptions:
    """Replaceable estimates used until supplier or test data are available."""

    baseline_engine_power_kw: float = 80.0
    engine_upgrade_kw_per_hybrid_kg: float = 0.25
    engine_power_ceiling_kw: float = 120.0
    engine_efficiency_power_fractions: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    engine_efficiency_values: tuple[float, ...] = (0.22, 0.28, 0.315, 0.31, 0.29)
    hybrid_system_specific_energy_wh_per_kg: float = 220.0
    motor_power_kw_per_hybrid_kg: float = 1.0


@dataclass(frozen=True)
class FullTaskEvaluation:
    design: AirframeDesign
    mission: MissionDefinition
    aircraft: AircraftParameters
    simulation: MissionSimulation
    result: EvaluationResult
    source_profile: dict[str, Any]
    requested_total_duration_h: float

    @property
    def achieved_duration_h(self) -> float:
        return float(self.simulation.profile["time_s"].iloc[-1] / 3600.0)


def load_source_profile(path: str | Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    """Load and minimally validate the supplied checkout-profile JSON."""
    source_path = Path(path)
    profile = json.loads(source_path.read_text(encoding="utf-8"))
    if "segments" not in profile or not isinstance(profile["segments"], list):
        raise ValueError("Source profile must contain a list named 'segments'.")
    labels = {segment.get("label") for segment in profile["segments"]}
    required = {
        "takeoff_roll",
        "main_climb_to_leveloff",
        "short_leveloff_to_isr_altitude",
        "isr_cruise_alpha0_5000km",
        "long_descent",
        "approach_descent",
        "flare",
        "landing_roll",
    }
    missing = sorted(required.difference(labels))
    if missing:
        raise ValueError(f"Source profile is missing required segments: {', '.join(missing)}")
    return profile


def source_profile_duration_h(profile: dict[str, Any]) -> float:
    """Return the sum of source initial durations, including ground rolls."""
    return float(sum(float(segment["duration_initial"]) for segment in profile["segments"]) / 3600.0)


def build_full_task_mission(
    profile: dict[str, Any],
    target_total_duration_h: float = 80.0,
) -> MissionDefinition:
    """Create an airborne mission from the supplied JSON phase definitions."""
    if target_total_duration_h <= 1.0:
        raise ValueError("target_total_duration_h must exceed one hour.")
    segments = {segment["label"]: segment for segment in profile["segments"]}
    takeoff = segments["takeoff_roll"]
    airborne_labels = (
        "main_climb_to_leveloff",
        "short_leveloff_to_isr_altitude",
        "isr_cruise_alpha0_5000km",
        "long_descent",
        "approach_descent",
        "flare",
    )
    non_cruise_duration_s = sum(
        float(segments[label]["duration_initial"])
        for label in airborne_labels
        if label != "isr_cruise_alpha0_5000km"
    )
    cruise_duration_s = target_total_duration_h * 3600.0 - non_cruise_duration_s
    if cruise_duration_s <= 0.0:
        raise ValueError("Target duration is shorter than the non-cruise airborne phases.")

    phase_specs = (
        ("main_climb_to_leveloff", 120.0, 10.0, 8.0, 60.0),
        ("short_leveloff_to_isr_altitude", 110.0, 5.0, 4.0, 60.0),
        ("isr_cruise_alpha0_5000km", 100.0, 0.0, 2.0, 300.0),
        ("long_descent", 45.0, 0.0, 6.0, 90.0),
        ("approach_descent", 70.0, 8.0, 6.0, 30.0),
        ("flare", 55.0, 8.0, 5.0, 20.0),
    )
    phases: list[PhaseCommand] = []
    for label, engine_limit_kw, motor_assist_kw, gamma_limit_deg, timestep_s in phase_specs:
        segment = segments[label]
        duration_s = cruise_duration_s if label == "isr_cruise_alpha0_5000km" else float(segment["duration_initial"])
        altitude_target_m = float(segment["end_y"])
        speed_target_mps = _target_speed(segment, profile, label)
        phases.append(
            PhaseCommand(
                name=label,
                duration_s=duration_s,
                altitude_target_m=altitude_target_m,
                speed_target_mps=speed_target_mps,
                engine_power_kw=engine_limit_kw,
                motor_power_kw=motor_assist_kw,
                max_flight_path_angle_deg=gamma_limit_deg,
                time_step_s=timestep_s,
            )
        )
    takeoff_speed_mps = _midpoint(takeoff["end_x_dot"])
    return MissionDefinition(
        phases=tuple(phases),
        initial_altitude_m=0.0,
        initial_speed_mps=takeoff_speed_mps,
        required_range_m=None,
        required_loiter_s=None,
        # The phase terminal targets already provide altitude tracking.  A
        # separate terminal-position constraint is introduced only after a
        # candidate survives the complete task; applying it to an early fuel
        # or SOC termination would reward an earlier failure numerically.
        terminal_position_error_max_m=None,
    )


def preliminary_hybrid_aircraft(
    design: AirframeDesign,
    calibration: UltraCalibration,
    assumptions: PreliminaryPropulsionAssumptions = PreliminaryPropulsionAssumptions(),
    aero_override: AeroCoefficients | None = None,
) -> AircraftParameters:
    """Map fixed-MTOW hybrid-system mass into declared preliminary capabilities."""
    base = aircraft_from_design(design, calibration, aero_override=aero_override)
    hybrid_mass = design.hybrid_system_mass_kg
    engine_power_kw = min(
        assumptions.engine_power_ceiling_kw,
        assumptions.baseline_engine_power_kw + assumptions.engine_upgrade_kw_per_hybrid_kg * hybrid_mass,
    )
    return replace(
        base,
        engine_electrical_power_max_kw=engine_power_kw,
        engine_efficiency_power_fractions=assumptions.engine_efficiency_power_fractions,
        engine_efficiency_values=assumptions.engine_efficiency_values,
        motor_power_min_kw=-0.25 * assumptions.motor_power_kw_per_hybrid_kg * hybrid_mass,
        motor_power_max_kw=assumptions.motor_power_kw_per_hybrid_kg * hybrid_mass,
        battery_capacity_mj=max(1.0, hybrid_mass * assumptions.hybrid_system_specific_energy_wh_per_kg * 3.6e-3),
        initial_soc=0.85 if hybrid_mass > 0.0 else 0.50,
        soc_safe=0.15 if hybrid_mass > 0.0 else 0.0,
        soc_reserve=0.20 if hybrid_mass > 0.0 else 0.0,
    )


def evaluate_full_task_profile(
    design: AirframeDesign,
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    target_total_duration_h: float = 80.0,
    calibration: UltraCalibration | None = None,
    assumptions: PreliminaryPropulsionAssumptions = PreliminaryPropulsionAssumptions(),
    engine_model: EngineGeneratorModel | None = None,
    powertrain: HybridPowertrainModel | None = None,
) -> FullTaskEvaluation:
    """Run the source-profile-derived task through the common 2.2.1 evaluator."""
    source_profile = load_source_profile(profile_path)
    active_calibration = calibration or calibrate_dimension_baseline()
    mission = build_full_task_mission(source_profile, target_total_duration_h)
    aircraft = preliminary_hybrid_aircraft(design, active_calibration, assumptions)
    simulation, result = evaluate_coupled_mission(
        mission,
        aircraft,
        engine_model=engine_model,
        powertrain=powertrain,
    )
    metrics = design_metrics(design, active_calibration.reference)
    result.constraints.update(geometry_constraints(design, active_calibration.reference))
    result.diagnostics.update(
        {
            "source_profile_duration_h": source_profile_duration_h(source_profile),
            "requested_total_duration_h": target_total_duration_h,
            "achieved_total_duration_h": float(simulation.profile["time_s"].iloc[-1] / 3600.0),
            "mission_range_m": simulation.range_m,
            "wing_span_m": metrics.wing_span_m,
            "fuel_mass_kg": metrics.fuel_mass_kg,
            "payload_mass_kg": metrics.payload_mass_kg,
            "hybrid_system_mass_kg": metrics.hybrid_system_mass_kg,
            "engine_power_max_kw": aircraft.engine_electrical_power_max_kw,
            "battery_capacity_mj": aircraft.battery_capacity_mj,
        }
    )
    return FullTaskEvaluation(design, mission, aircraft, simulation, result, source_profile, target_total_duration_h)


def export_full_task_baseline(
    output_dir: str | Path,
    design: AirframeDesign,
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    target_total_duration_h: float = 80.0,
) -> FullTaskEvaluation:
    """Export a traceable baseline before MOEA/D varies its control schedules."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_full_task_profile(design, profile_path, target_total_duration_h)
    evaluation.simulation.profile.to_csv(output_path / "full_task_baseline_time_history.csv", index=False)
    pd.DataFrame([evaluation.result.to_record()]).to_csv(output_path / "full_task_baseline_evaluation.csv", index=False)
    metadata = {
        "model_status": "source-JSON-derived full-task baseline with preliminary propulsion assumptions; not validated PTDL aircraft data",
        "source_profile_name": evaluation.source_profile["profile_name"],
        "source_profile_initial_duration_h": source_profile_duration_h(evaluation.source_profile),
        "source_profile_target_range_m": evaluation.source_profile["reference"]["target_total_range_m"],
        "target_total_duration_h": target_total_duration_h,
        "ground_segments_not_simulated": ["takeoff_roll", "landing_roll"],
        "design": asdict(design),
        "preliminary_propulsion_assumptions": asdict(PreliminaryPropulsionAssumptions()),
        "termination_reason": evaluation.simulation.termination_reason,
    }
    (output_path / "full_task_baseline_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return evaluation


def _target_speed(segment: dict[str, Any], profile: dict[str, Any], label: str) -> float:
    if label == "isr_cruise_alpha0_5000km":
        return float(profile["reference"]["cruise_speed_mps"])
    return _midpoint(segment["end_x_dot"])


def _midpoint(value: float | list[float]) -> float:
    if isinstance(value, list):
        return 0.5 * (float(value[0]) + float(value[1]))
    return float(value)


if __name__ == "__main__":
    # A nonzero hybrid mass is required to honor the source profile's high
    # transient-power limits; it is an initial architecture assumption only.
    baseline_design = AirframeDesign(payload_mass_kg=100.0, hybrid_system_mass_kg=120.0)
    result = export_full_task_baseline("ptdl_full_task_baseline_results", baseline_design)
    print(f"Termination: {result.simulation.termination_reason}")
    print(f"Achieved duration: {result.achieved_duration_h:.3f} h")
