"""PTDL 80 h endurance-task definition and evaluation adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from 指标体系与评价函数 import EvaluationResult
from 机体参数化设计与气动阻力 import BASELINE, AeroCoefficients, AirframeDesign, geometry_constraints
from 固定起飞质量尺寸标定 import calibrate_dimension_baseline
from ptdl_标定缓存 import load_dimension_calibration, load_task_calibration_values, save_task_calibration_values
from 飞推控闭环物理模型 import (
    EngineGeneratorModel,
    FlightControlParameters,
    HybridDispatchPolicy,
    HybridPowertrainModel,
    MissionDefinition,
    MissionSimulation,
    PhaseCommand,
    LongitudinalDragDisturbance,
    MissionFaultEvent,
    evaluate_coupled_mission,
)
from 全任务剖面定义 import (
    PreliminaryPropulsionAssumptions,
    load_preliminary_propulsion_config,
    preliminary_hybrid_aircraft,
)
from ULTRA基准标定 import UltraCalibration


DEFAULT_ENDURANCE_PROFILE_PATH = Path(__file__).resolve().with_name("八十小时任务剖面_ULTRA基准.json")
ENDURANCE_PROPULSION_ASSUMPTIONS, ULTRA_PROPULSION_CONFIG = load_preliminary_propulsion_config()


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
    initial_speed_mps: float = 50.0,
    disturbances: Sequence[LongitudinalDragDisturbance] = (),
    unpowered_glide_phases: Sequence[str] = (),
    phase_overrides: Mapping[str, Mapping[str, float]] | None = None,
    initial_soc: float | None = None,
    fault_events: Sequence[MissionFaultEvent] = (),
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
        phase = PhaseCommand(
                name=segment["label"],
                duration_s=duration_s,
                altitude_target_m=float(segment["altitude_target_m"]),
                speed_target_mps=float(speed),
                engine_power_kw=float(segment["engine_power_bounds_kw"][1]),
                motor_power_kw=0.0,
                max_flight_path_angle_deg=float(segment["max_flight_path_angle_deg"]),
                time_step_s=float(segment["time_step_s"]),
                unpowered_glide=segment["label"] in unpowered_glide_phases,
                motor_disabled=bool(segment.get("motor_disabled", False)),
            )
        overrides = (phase_overrides or {}).get(phase.name, {})
        invalid = set(overrides).difference(
            {"duration_s", "altitude_target_m", "speed_target_mps", "max_flight_path_angle_deg", "time_step_s"}
        )
        if invalid:
            raise ValueError(f"Unsupported phase overrides for {phase.name}: {sorted(invalid)}")
        phase = replace(phase, **{key: float(value) for key, value in overrides.items()})
        if phase.duration_s <= 0.0 or phase.time_step_s <= 0.0:
            raise ValueError(f"Phase duration and time step must be positive for {phase.name}.")
        phases.append(phase)
    return MissionDefinition(
        phases=tuple(phases),
        initial_altitude_m=0.0,
        initial_speed_mps=float(initial_speed_mps),
        required_loiter_s=None,
        terminal_position_error_max_m=None,
        disturbances=tuple(disturbances),
        initial_soc=initial_soc,
        fault_events=tuple(fault_events),
    )


def evaluate_endurance_task(
    design: AirframeDesign,
    engine_caps_kw: Sequence[float] | None = None,
    motor_assist_kw: Sequence[float] | None = None,
    speeds_mps: Sequence[float] | None = None,
    initial_speed_mps: float = 50.0,
    profile_path: str | Path = DEFAULT_ENDURANCE_PROFILE_PATH,
    target_total_duration_h: float | None = None,
    calibration: UltraCalibration | None = None,
    assumptions: PreliminaryPropulsionAssumptions = ENDURANCE_PROPULSION_ASSUMPTIONS,
    aero_override: AeroCoefficients | None = None,
    dispatch_policy: HybridDispatchPolicy | None = None,
    engine_model: EngineGeneratorModel | None = None,
    powertrain: HybridPowertrainModel | None = None,
    disturbances: Sequence[LongitudinalDragDisturbance] = (),
    unpowered_glide_phases: Sequence[str] = (),
    phase_overrides: Mapping[str, Mapping[str, float]] | None = None,
    flight_controller: FlightControlParameters | None = None,
    enforce_phase_engine_requirements: bool = True,
    initial_soc: float | None = None,
    initial_fuel_kg: float | None = None,
    fault_events: Sequence[MissionFaultEvent] = (),
) -> EnduranceTaskEvaluation:
    """Evaluate a design and phase-control schedule with the common evaluator."""
    profile = load_endurance_profile(profile_path)
    mission = build_endurance_mission(
        profile,
        target_total_duration_h=target_total_duration_h,
        speeds_mps=speeds_mps,
        initial_speed_mps=initial_speed_mps,
        disturbances=disturbances,
        unpowered_glide_phases=unpowered_glide_phases,
        phase_overrides=phase_overrides,
        initial_soc=initial_soc,
        fault_events=fault_events,
    )
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
            unpowered_glide=phase.unpowered_glide,
            motor_disabled=phase.motor_disabled,
        )
        for phase, cap, motor in zip(mission.phases, caps, motors)
    )
    mission = MissionDefinition(
        phases=phases,
        initial_altitude_m=mission.initial_altitude_m,
        initial_speed_mps=mission.initial_speed_mps,
        required_loiter_s=None,
        terminal_position_error_max_m=None,
        disturbances=mission.disturbances,
        initial_soc=mission.initial_soc,
        fault_events=mission.fault_events,
    )
    active_calibration = calibration or calibrate_dimension_baseline()
    aircraft = preliminary_hybrid_aircraft(design, active_calibration, assumptions, aero_override=aero_override)
    if initial_fuel_kg is not None:
        if not np.isfinite(initial_fuel_kg) or not 0.0 < initial_fuel_kg <= aircraft.initial_fuel_kg:
            raise ValueError("initial_fuel_kg must be finite, positive, and no greater than the configured fuel load.")
        aircraft = replace(aircraft, initial_fuel_kg=float(initial_fuel_kg))
    if initial_soc is not None and not 0.0 <= initial_soc <= 1.0:
        raise ValueError("initial_soc must be in [0, 1].")
    simulation, result = evaluate_coupled_mission(
        mission,
        aircraft,
        controller=flight_controller or FlightControlParameters(),
        dispatch_policy=dispatch_policy,
        engine_model=engine_model,
        powertrain=powertrain,
    )
    result.constraints.update(geometry_constraints(design, active_calibration.reference))
    if enforce_phase_engine_requirements:
        for segment in profile["segments"]:
            required_engine_kw = float(segment.get("engine_required_min_kw", 0.0))
            if required_engine_kw <= 0.0:
                continue
            phase_power = simulation.profile.loc[
                simulation.profile["phase"] == segment["label"], "p_engine_kw"
            ]
            minimum_engine_kw = float(phase_power.min()) if not phase_power.empty else 0.0
            result.constraints[f"engine_required_{segment['label']}_kw"] = (
                required_engine_kw - minimum_engine_kw
            )
            result.diagnostics[f"minimum_engine_{segment['label']}_kw"] = minimum_engine_kw
    result.diagnostics.update(
        {
            "achieved_total_duration_h": float(simulation.profile["time_s"].iloc[-1] / 3600.0),
            "requested_total_duration_h": float(profile["requirements"]["target_total_duration_h"] if target_total_duration_h is None else target_total_duration_h),
            "mission_range_m": simulation.range_m,
            "fuel_mass_kg": aircraft.initial_fuel_kg,
            "engine_power_max_kw": aircraft.engine_electrical_power_max_kw,
            "battery_capacity_mj": aircraft.battery_capacity_mj,
            "motor_power_max_kw": aircraft.motor_power_max_kw,
            "engine_rating_multiplier": assumptions.engine_rating_multiplier,
            "motor_rating_multiplier": assumptions.motor_rating_multiplier,
            "battery_energy_multiplier": assumptions.battery_energy_multiplier,
            "aero_source": "vspaero_corrected" if aero_override is not None else "low_order",
            "aero_cl_alpha_per_rad": aircraft.cl_alpha_per_rad,
            "aero_oswald_efficiency": aircraft.oswald_efficiency,
        }
    )
    frame = simulation.profile
    dt_s = frame["time_s"].diff().fillna(0.0).clip(lower=0.0)
    discharge_mj = float((frame["battery_discharge_kw"] * dt_s / 1000.0).sum())
    charge_mj = float((frame["battery_charge_kw"] * dt_s / 1000.0).sum())
    result.diagnostics.update(
        {
            "battery_discharge_energy_mj": discharge_mj,
            "battery_charge_energy_mj": charge_mj,
            "battery_net_energy_mj": discharge_mj - charge_mj,
            "battery_equivalent_full_discharge_cycles": discharge_mj / max(aircraft.battery_capacity_mj, 1e-9),
            "battery_soc_change": float(frame["soc"].iloc[-1] - frame["soc"].iloc[0]),
            "battery_peak_abs_current_a": float(frame["battery_current_a"].abs().max()),
            "battery_diagnostic_basis": "time-integrated bus power; diagnostic only, not a PPT optimization objective",
        }
    )
    interval_dt_s = np.diff(frame["time_s"].to_numpy(dtype=float))
    bus_discharge_kw = frame["battery_discharge_kw"].to_numpy(dtype=float)[1:]
    bus_charge_kw = frame["battery_charge_kw"].to_numpy(dtype=float)[1:]
    terminal_power_kw = frame["battery_terminal_power_kw"].to_numpy(dtype=float)[1:]
    chemical_power_kw = frame["battery_chemical_power_kw"].to_numpy(dtype=float)[1:]
    bus_discharge_energy_mj = float(np.sum(bus_discharge_kw * interval_dt_s / 1000.0))
    bus_charge_energy_mj = float(np.sum(bus_charge_kw * interval_dt_s / 1000.0))
    terminal_discharge_energy_mj = float(
        np.sum(np.maximum(terminal_power_kw, 0.0) * interval_dt_s / 1000.0)
    )
    terminal_charge_energy_mj = float(
        np.sum(np.maximum(-terminal_power_kw, 0.0) * interval_dt_s / 1000.0)
    )
    chemical_discharge_energy_mj = float(
        np.sum(np.maximum(chemical_power_kw, 0.0) * interval_dt_s / 1000.0)
    )
    chemical_charge_energy_mj = float(
        np.sum(np.maximum(-chemical_power_kw, 0.0) * interval_dt_s / 1000.0)
    )
    apparent_discharge_efficiency = (
        bus_discharge_energy_mj / chemical_discharge_energy_mj
        if chemical_discharge_energy_mj > 1e-12
        else float("nan")
    )
    apparent_charge_efficiency = (
        chemical_charge_energy_mj / bus_charge_energy_mj
        if bus_charge_energy_mj > 1e-12
        else float("nan")
    )
    result.diagnostics.update(
        {
            "battery_terminal_discharge_energy_mj": terminal_discharge_energy_mj,
            "battery_terminal_charge_energy_mj": terminal_charge_energy_mj,
            "battery_chemical_discharge_energy_mj": chemical_discharge_energy_mj,
            "battery_chemical_charge_energy_mj": chemical_charge_energy_mj,
            "battery_bus_to_chemical_discharge_efficiency": apparent_discharge_efficiency,
            "battery_chemical_to_bus_charge_efficiency": apparent_charge_efficiency,
            "battery_round_trip_efficiency_estimate": (
                apparent_discharge_efficiency * apparent_charge_efficiency
                if np.isfinite(apparent_discharge_efficiency)
                and np.isfinite(apparent_charge_efficiency)
                else float("nan")
            ),
            "battery_efficiency_diagnostic_basis": (
                "bus energy compared with OCV times current chemical-energy proxy; "
                "includes modeled Rint/converter behavior but is not measured cell efficiency"
            ),
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
    cached_values = load_task_calibration_values(ENDURANCE_PROPULSION_ASSUMPTIONS)
    if cached_values is not None:
        scale, duration_h = cached_values
        assumptions = _scaled_endurance_assumptions(scale)
        return EnduranceTaskCalibration(
            efficiency_scale=scale,
            baseline_duration_h=duration_h,
            assumptions=assumptions,
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
    save_task_calibration_values(
        scale,
        evaluation.achieved_duration_h,
        ENDURANCE_PROPULSION_ASSUMPTIONS,
    )
    return EnduranceTaskCalibration(
        efficiency_scale=scale,
        baseline_duration_h=evaluation.achieved_duration_h,
        assumptions=assumptions,
        dimension_calibration=dimension_calibration,
    )


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
