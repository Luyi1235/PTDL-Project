"""PTDL 80 h endurance-task definition and evaluation adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ptdl_2_2_1_evaluator import EvaluationResult
from ptdl_airframe_design import BASELINE, AeroCoefficients, AirframeDesign, geometry_constraints
from ptdl_calibration_cache import load_dimension_calibration
from ptdl_flight_propulsion_control import (
    EngineGeneratorModel,
    HybridDispatchPolicy,
    HybridPowertrainModel,
    MissionDefinition,
    MissionSimulation,
    PhaseCommand,
    evaluate_coupled_mission,
)
from ptdl_full_task_profile import PreliminaryPropulsionAssumptions, preliminary_hybrid_aircraft
from ptdl_ultra_benchmark import UltraCalibration


DEFAULT_ENDURANCE_PROFILE_PATH = Path("ptdl_80h_endurance_profile.json")
ENDURANCE_CALIBRATION_CACHE_PATH = Path(__file__).resolve().parent / "ptdl_endurance_task_calibration_cache.json"
ENDURANCE_CALIBRATION_CACHE_SCHEMA = "full_task_endurance_calibration_dynamic_cd0_v1"
ENDURANCE_PROPULSION_ASSUMPTIONS = PreliminaryPropulsionAssumptions(
    # The endurance profile needs sufficient shaft/generator capability for
    # its takeoff and 2,800 m climb.  This is a declared preliminary sizing
    # assumption, not a published ULTRA engine rating.
    baseline_engine_power_kw=120.0,
    engine_upgrade_kw_per_hybrid_kg=0.0,
    engine_power_ceiling_kw=120.0,
)


@dataclass(frozen=True)
class EnduranceTaskEvaluation:
    design: AirframeDesign
    mission: MissionDefinition
    simulation: MissionSimulation
    result: EvaluationResult
    profile: dict[str, Any]

    @property
    def achieved_duration_h(self) -> float:
        return float(self.simulation.profile["time_s"].iloc[-1] / 3600.0)


@dataclass(frozen=True)
class EnduranceTaskCalibration:
    """Mission-level calibration that reconciles the 72 h ULTRA anchor."""

    efficiency_scale: float
    baseline_duration_h: float
    assumptions: PreliminaryPropulsionAssumptions
    dimension_calibration: UltraCalibration


def load_endurance_profile(path: str | Path = DEFAULT_ENDURANCE_PROFILE_PATH) -> dict[str, Any]:
    """Load the PTDL-specific endurance profile and validate its phase schema."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("requirements", {}).get("target_total_duration_h") != 80.0:
        raise ValueError("The PTDL endurance profile must retain an 80 h target.")
    labels = [segment.get("label") for segment in profile.get("segments", [])]
    expected = [
        "main_climb_to_leveloff",
        "short_leveloff_to_isr_altitude",
        "endurance_loiter",
        "long_descent",
        "approach_descent",
        "flare",
    ]
    if labels != expected:
        raise ValueError("PTDL endurance profile phases do not match the declared task sequence.")
    return profile


def build_endurance_mission(
    profile: dict[str, Any],
    target_total_duration_h: float | None = None,
    speeds_mps: Sequence[float] | None = None,
) -> MissionDefinition:
    """Build the 80 h flight mission, with ISR speed optionally controllable."""
    target_h = float(profile["requirements"]["target_total_duration_h"] if target_total_duration_h is None else target_total_duration_h)
    segments = profile["segments"]
    fixed_duration_s = sum(float(segment.get("duration_s", 0.0)) for segment in segments if "duration_rule" not in segment)
    loiter_duration_s = target_h * 3600.0 - fixed_duration_s
    if loiter_duration_s <= 0.0:
        raise ValueError("Target duration must exceed all fixed-duration phases.")
    requested_speeds = list(speeds_mps) if speeds_mps is not None else [float(segment["speed_target_mps"]) for segment in segments]
    if len(requested_speeds) != len(segments):
        raise ValueError("speeds_mps must provide one value for every task phase.")
    phases: list[PhaseCommand] = []
    for segment, speed in zip(segments, requested_speeds):
        lower_speed, upper_speed = (float(value) for value in segment["speed_bounds_mps"])
        if not lower_speed <= speed <= upper_speed:
            raise ValueError(f"Requested speed for {segment['label']} is outside its declared range.")
        duration_s = loiter_duration_s if segment.get("duration_rule") else float(segment["duration_s"])
        phases.append(
            PhaseCommand(
                name=segment["label"],
                duration_s=duration_s,
                altitude_target_m=float(segment["altitude_target_m"]),
                speed_target_mps=float(speed),
                engine_power_kw=float(segment["engine_power_bounds_kw"][1]),
                motor_power_kw=0.0,
                max_flight_path_angle_deg=float(segment["max_flight_path_angle_deg"]),
                time_step_s=float(segment["time_step_s"]),
            )
        )
    return MissionDefinition(
        phases=tuple(phases),
        initial_altitude_m=0.0,
        initial_speed_mps=50.0,
        required_loiter_s=None,
        terminal_position_error_max_m=None,
    )


def evaluate_endurance_task(
    design: AirframeDesign,
    engine_caps_kw: Sequence[float] | None = None,
    motor_assist_kw: Sequence[float] | None = None,
    speeds_mps: Sequence[float] | None = None,
    profile_path: str | Path = DEFAULT_ENDURANCE_PROFILE_PATH,
    target_total_duration_h: float | None = None,
    calibration: UltraCalibration | None = None,
    assumptions: PreliminaryPropulsionAssumptions = ENDURANCE_PROPULSION_ASSUMPTIONS,
    aero_override: AeroCoefficients | None = None,
    dispatch_policy: HybridDispatchPolicy | None = None,
    engine_model: EngineGeneratorModel | None = None,
    powertrain: HybridPowertrainModel | None = None,
) -> EnduranceTaskEvaluation:
    """Evaluate a design and phase-control schedule with the common evaluator."""
    profile = load_endurance_profile(profile_path)
    mission = build_endurance_mission(profile, target_total_duration_h=target_total_duration_h, speeds_mps=speeds_mps)
    phase_count = len(mission.phases)
    caps = list(engine_caps_kw) if engine_caps_kw is not None else [phase.engine_power_kw for phase in mission.phases]
    motors = list(motor_assist_kw) if motor_assist_kw is not None else [0.0] * phase_count
    if len(caps) != phase_count or len(motors) != phase_count:
        raise ValueError("Engine and motor schedules must contain one value for every task phase.")
    phases = tuple(
        PhaseCommand(
            name=phase.name,
            duration_s=phase.duration_s,
            altitude_target_m=phase.altitude_target_m,
            speed_target_mps=phase.speed_target_mps,
            engine_power_kw=float(cap),
            motor_power_kw=float(motor),
            max_flight_path_angle_deg=phase.max_flight_path_angle_deg,
            time_step_s=phase.time_step_s,
        )
        for phase, cap, motor in zip(mission.phases, caps, motors)
    )
    mission = MissionDefinition(
        phases=phases,
        initial_altitude_m=mission.initial_altitude_m,
        initial_speed_mps=mission.initial_speed_mps,
        required_loiter_s=None,
        terminal_position_error_max_m=None,
    )
    active_calibration = calibration or load_dimension_calibration()
    aircraft = preliminary_hybrid_aircraft(design, active_calibration, assumptions, aero_override=aero_override)
    simulation, result = evaluate_coupled_mission(
        mission,
        aircraft,
        dispatch_policy=dispatch_policy,
        engine_model=engine_model,
        powertrain=powertrain,
    )
    result.constraints.update(geometry_constraints(design, active_calibration.reference))
    result.diagnostics.update(
        {
            "achieved_total_duration_h": float(simulation.profile["time_s"].iloc[-1] / 3600.0),
            "requested_total_duration_h": float(profile["requirements"]["target_total_duration_h"] if target_total_duration_h is None else target_total_duration_h),
            "mission_range_m": simulation.range_m,
            "fuel_mass_kg": aircraft.initial_fuel_kg,
            "engine_power_max_kw": aircraft.engine_electrical_power_max_kw,
            "battery_capacity_mj": aircraft.battery_capacity_mj,
            "aero_source": "vspaero_corrected" if aero_override is not None else "low_order",
            "aero_cl_alpha_per_rad": aircraft.cl_alpha_per_rad,
            "aero_oswald_efficiency": aircraft.oswald_efficiency,
        }
    )
    return EnduranceTaskEvaluation(design, mission, simulation, result, profile)


@lru_cache(maxsize=1)
def calibrate_endurance_task_baseline() -> EnduranceTaskCalibration:
    """Calibrate the full-task powertrain map to the 72 h ULTRA anchor.

    The prior 72 h calibration covered equivalent loiter only.  This second
    calibration uses the same public ULTRA mass, fuel, and payload anchor but
    includes the PTDL endurance task's climb, descent, approach, and flare
    phases.  It estimates the missing system-level engine/propulsion map
    factor and applies that factor uniformly to every later candidate.
    """
    dimension_calibration = load_dimension_calibration()
    if ENDURANCE_CALIBRATION_CACHE_PATH.exists():
        cached = json.loads(ENDURANCE_CALIBRATION_CACHE_PATH.read_text(encoding="utf-8"))
        if cached.get("schema") == ENDURANCE_CALIBRATION_CACHE_SCHEMA:
            scale = float(cached["efficiency_scale"])
            return EnduranceTaskCalibration(
                efficiency_scale=scale,
                baseline_duration_h=float(cached["baseline_duration_h"]),
                assumptions=_scaled_endurance_assumptions(scale),
                dimension_calibration=dimension_calibration,
            )
    lo_scale, hi_scale = 0.80, 1.40
    # The dynamic plant makes each 80 h calibration evaluation more expensive
    # than the previous algebraic model.  Fourteen bisection steps resolve the
    # declared efficiency scale far more tightly than the provisional input
    # data warrant, while keeping routine regression checks practical.
    for _ in range(14):
        scale = 0.5 * (lo_scale + hi_scale)
        assumptions = _scaled_endurance_assumptions(scale)
        evaluation = evaluate_endurance_task(
            BASELINE,
            calibration=dimension_calibration,
            assumptions=assumptions,
        )
        if evaluation.achieved_duration_h >= dimension_calibration.reference.baseline_endurance_h:
            hi_scale = scale
        else:
            lo_scale = scale
    scale = 0.5 * (lo_scale + hi_scale)
    assumptions = _scaled_endurance_assumptions(scale)
    evaluation = evaluate_endurance_task(BASELINE, calibration=dimension_calibration, assumptions=assumptions)
    calibration = EnduranceTaskCalibration(
        efficiency_scale=scale,
        baseline_duration_h=evaluation.achieved_duration_h,
        assumptions=assumptions,
        dimension_calibration=dimension_calibration,
    )
    ENDURANCE_CALIBRATION_CACHE_PATH.write_text(
        json.dumps(
            {
                "schema": ENDURANCE_CALIBRATION_CACHE_SCHEMA,
                "purpose": "Reusable full-task 72-hour inverse calibration for the current low-order reference model.",
                "efficiency_scale": calibration.efficiency_scale,
                "baseline_duration_h": calibration.baseline_duration_h,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return calibration


def export_endurance_baseline(output_dir: str | Path, design: AirframeDesign) -> EnduranceTaskEvaluation:
    """Write the first endurance-task baseline for review and regression."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    task_calibration = calibrate_endurance_task_baseline()
    evaluation = evaluate_endurance_task(
        design,
        calibration=task_calibration.dimension_calibration,
        assumptions=task_calibration.assumptions,
    )
    evaluation.simulation.profile.to_csv(output_path / "ptdl_80h_endurance_baseline_time_history.csv", index=False)
    pd.DataFrame([evaluation.result.to_record()]).to_csv(output_path / "ptdl_80h_endurance_baseline_evaluation.csv", index=False)
    metadata = {
        "model_status": "PTDL 80 h endurance-adapted task baseline with preliminary propulsion assumptions",
        "full_task_72h_calibration": {
            "efficiency_scale": task_calibration.efficiency_scale,
            "baseline_duration_h": task_calibration.baseline_duration_h,
        },
        "design": asdict(design),
        "profile": evaluation.profile,
        "termination_reason": evaluation.simulation.termination_reason,
    }
    (output_path / "ptdl_80h_endurance_baseline_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return evaluation


def export_endurance_extension_verification(
    output_dir: str | Path,
    design: AirframeDesign,
    target_total_duration_h: float = 86.0,
    speeds_mps: Sequence[float] | None = None,
) -> EnduranceTaskEvaluation:
    """Verify endurance margin by extending only the ISR loiter duration."""
    if target_total_duration_h <= 80.0:
        raise ValueError("Extension verification must request more than 80 h.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    task_calibration = calibrate_endurance_task_baseline()
    evaluation = evaluate_endurance_task(
        design,
        speeds_mps=speeds_mps,
        target_total_duration_h=target_total_duration_h,
        calibration=task_calibration.dimension_calibration,
        assumptions=task_calibration.assumptions,
    )
    evaluation.simulation.profile.to_csv(output_path / "ptdl_endurance_extension_time_history.csv", index=False)
    pd.DataFrame([evaluation.result.to_record()]).to_csv(output_path / "ptdl_endurance_extension_evaluation.csv", index=False)
    metadata = {
        "verification_type": "ISR loiter extension under the same full-task phase structure",
        "target_total_duration_h": target_total_duration_h,
        "achieved_total_duration_h": evaluation.achieved_duration_h,
        "termination_reason": evaluation.simulation.termination_reason,
        "design": asdict(design),
        "full_task_72h_calibration": {
            "efficiency_scale": task_calibration.efficiency_scale,
            "baseline_duration_h": task_calibration.baseline_duration_h,
        },
    }
    (output_path / "ptdl_endurance_extension_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return evaluation


def _scaled_endurance_assumptions(scale: float) -> PreliminaryPropulsionAssumptions:
    if not 0.5 <= scale <= 1.8:
        raise ValueError("Full-task efficiency calibration scale is outside its declared range.")
    values = tuple(min(0.60, value * scale) for value in ENDURANCE_PROPULSION_ASSUMPTIONS.engine_efficiency_values)
    return replace(ENDURANCE_PROPULSION_ASSUMPTIONS, engine_efficiency_values=values)
