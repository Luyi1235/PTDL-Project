"""Run an estimated six-phase JSBSim attitude, trajectory, route, and hybrid loop."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import jsbsim

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
MODEL_DIR = ROOT / "高保真参考模型" / "PTDL_EST_6DOF"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PTDL六自由度参考模型 import PTDL6DoFParameters, write_parameter_manifest
from 飞推控闭环物理模型 import DispatchObservation
from 规则混动协同控制 import RuleBasedHybridCoordinator
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 电池能量收支 import BatteryEnergyLedger


FT_PER_M = 3.280839895
MPS_PER_KNOT = 0.514444
LBF_PER_N = 0.224808943
PSF_TO_PA = 47.88025898
KG_TO_LB = 2.20462262185
G = 9.80665
MISSION_HOURS = 80.0
PHASES = (
    ("main_climb_to_leveloff", 3600.0, 2800.0, 40.0, 120.0, 30.0, 8.0),
    ("short_leveloff_to_isr_altitude", 900.0, 3000.0, 36.01, 120.0, 20.0, 3.0),
    ("endurance_loiter", None, 3000.0, 36.01, 100.0, 8.0, 1.0),
    ("long_descent", 6000.0, 800.0, 42.0, 70.0, 0.0, 5.0),
    ("approach_descent", 1200.0, 50.0, 34.0, 80.0, 15.0, 5.0),
    ("flare", 900.0, 0.0, 30.0, 70.0, 15.0, 4.0),
)
DEFAULT_ROUTE_LEG_LENGTH_M = 250000.0
TERMINAL_LANDING_GRACE_S = 7200.0
APPROACH_ALTITUDE_RESPONSE_TIME_MIN_S = 60.0


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_fdm(
    dt_s: float,
    altitude_m: float,
    speed_mps: float,
    model_dir: Path = MODEL_DIR,
) -> jsbsim.FGFDMExec:
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.set_aircraft_path(str(model_dir.parent))
    fdm.set_engine_path(str(MODEL_DIR.parent / "engines"))
    fdm.set_systems_path(str(MODEL_DIR.parent / "systems"))
    if not fdm.load_model(model_dir.name):
        raise RuntimeError("JSBSim cannot load PTDL_EST_6DOF.")
    fdm.set_dt(dt_s)
    fdm["ic/h-sl-ft"] = altitude_m * FT_PER_M
    fdm["ic/vt-kts"] = speed_mps / MPS_PER_KNOT
    rho = 1.225 * max(1.0 - 2.25577e-5 * altitude_m, 1e-3) ** 4.2559
    qbar_pa = 0.5 * rho * speed_mps**2
    cl_required = params_weight_n() / max(qbar_pa * 26.0, 1.0)
    alpha = math.radians(
        np_clip(math.degrees((cl_required - 0.25) / 5.20), -2.0, 10.0)
    )
    fdm["ic/alpha-deg"] = math.degrees(alpha)
    fdm["ic/theta-deg"] = math.degrees(alpha)
    fdm["ic/psi-true-deg"] = 0.0
    if not fdm.run_ic():
        raise RuntimeError("JSBSim initial-condition setup failed.")
    return fdm


def params_weight_n() -> float:
    return PTDL6DoFParameters().mtow_kg * G


def phase_schedule(
    mission_hours: float,
    climb_altitude_m: float,
    loiter_altitude_m: float,
    loiter_speed_mps: float,
    mission_mode: str,
    phase_overrides: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, float | str]]:
    if mission_mode == "endurance_only":
        phase_specs = (
            ("main_climb_to_leveloff", 3600.0, climb_altitude_m, 40.0, 120.0, 30.0, 8.0),
            ("short_leveloff_to_isr_altitude", 900.0, loiter_altitude_m, 36.01, 120.0, 20.0, 3.0),
            ("endurance_loiter", None, loiter_altitude_m, loiter_speed_mps, 100.0, 8.0, 1.0),
        )
    elif mission_mode in {"full", "endurance_then_land"}:
        phase_specs = PHASES
    else:
        raise ValueError(
            "mission_mode must be 'endurance_only', 'full', or 'endurance_then_land'."
        )
    if mission_mode == "endurance_then_land":
        fixed_duration = sum(
            float(item[1])
            for item in phase_specs
            if item[1] is not None
            and item[0] in {"main_climb_to_leveloff", "short_leveloff_to_isr_altitude"}
        )
    else:
        fixed_duration = sum(
            float(item[1]) for item in phase_specs if item[1] is not None
        )
    loiter_duration = mission_hours * 3600.0 - fixed_duration
    if loiter_duration <= 0.0:
        raise ValueError("Mission duration must leave positive time for endurance loiter.")
    schedule: list[dict[str, float | str]] = []
    start = 0.0
    for name, duration, altitude, speed, engine_cap, motor_cap, gamma_limit in phase_specs:
        duration_s = loiter_duration if duration is None else float(duration)
        if mission_mode in {"full", "endurance_then_land"} and name == "main_climb_to_leveloff":
            altitude = climb_altitude_m
        elif mission_mode in {"full", "endurance_then_land"} and name in {
            "short_leveloff_to_isr_altitude",
            "endurance_loiter",
        }:
            altitude = loiter_altitude_m
        if name == "endurance_loiter" and mission_mode in {
            "full",
            "endurance_then_land",
        }:
            speed = loiter_speed_mps
        values: dict[str, float | str] = {
            "name": name,
            "start_s": start,
            "end_s": start + duration_s,
            "duration_s": duration_s,
            "altitude_m": altitude,
            "speed_mps": speed,
            "engine_cap_kw": engine_cap,
            "motor_cap_kw": motor_cap,
            "gamma_limit_deg": gamma_limit,
            "power_off_duration_s": 0.0,
        }
        overrides = (phase_overrides or {}).get(name, {})
        unknown = set(overrides).difference(
            {
                "altitude_m",
                "speed_mps",
                "engine_cap_kw",
                "motor_cap_kw",
                "gamma_limit_deg",
                "power_off_duration_s",
            }
        )
        if unknown:
            raise ValueError(f"Unsupported JSBSim phase overrides for {name}: {sorted(unknown)}")
        values.update({key: float(value) for key, value in overrides.items()})
        if not 0.0 <= float(values["power_off_duration_s"]) < duration_s:
            raise ValueError(
                f"power_off_duration_s for {name} must be non-negative and shorter than the phase."
            )
        values["end_s"] = start + duration_s
        schedule.append(values)
        start += duration_s
    return schedule


def mission_completion_status(
    mission_mode: str,
    flight_profile_completed: bool,
    fuel_feasible: bool,
    touchdown: bool,
    elapsed_s: float,
    requested_duration_s: float,
) -> bool:
    if not flight_profile_completed or not fuel_feasible:
        return False
    if mission_mode == "endurance_only":
        return elapsed_s >= requested_duration_s
    if mission_mode == "full":
        return touchdown
    if mission_mode == "endurance_then_land":
        return touchdown and elapsed_s >= requested_duration_s
    return False


def select_landing_guidance_phase(
    scheduled_phase: dict[str, float | str],
    approach_phase: dict[str, float | str],
    flare_phase: dict[str, float | str],
    altitude_m: float,
    terminal_flare_active: bool,
    flare_gate_altitude_m: float = 150.0,
) -> tuple[dict[str, float | str], bool]:
    scheduled_name = str(scheduled_phase["name"])
    if terminal_flare_active or (
        scheduled_name in {"approach_descent", "flare"}
        and altitude_m <= flare_gate_altitude_m
    ):
        return flare_phase, True
    if scheduled_name == "flare" and altitude_m > flare_gate_altitude_m:
        return approach_phase, False
    return scheduled_phase, False


def flare_target_vertical_speed_up_mps(altitude_m: float) -> float:
    """Keep a bounded, downward flare target; never reverse it into a climb."""
    sink_rate_mps = min(1.0, max(0.25, 0.01 * max(float(altitude_m), 0.0)))
    return -sink_rate_mps


def run(
    output_dir: Path,
    integration_step_s: float = 0.25,
    control_period_s: float = 5.0,
    sample_period_s: float = 60.0,
    mission_hours: float = MISSION_HOURS,
    loiter_speed_mps: float = 32.0,
    loiter_altitude_m: float = 2400.0,
    climb_altitude_m: float = 2200.0,
    route_leg_length_m: float = DEFAULT_ROUTE_LEG_LENGTH_M,
    mission_mode: str = "endurance_only",
    pitch_kp: float = 2.5,
    pitch_kd: float = 2.5,
    *,
    terminal_control_period_s: float = 0.5,
    terminal_pitch_kp: float | None = None,
    terminal_pitch_kd: float | None = None,
    terminal_pitch_ki: float = 0.15,
    model_dir: Path = MODEL_DIR,
    phase_overrides: dict[str, dict[str, float]] | None = None,
    aircraft_overrides: dict[str, float] | None = None,
    initial_soc: float = 0.95,
    battery_capacity_mj: float = 18.0,
    fuel_capacity_kg: float | None = None,
    engine_efficiency_values: tuple[float, ...] | None = None,
    dispatch_policy: object | None = None,
    powertrain_config: dict[str, object] | None = None,
    installed_engine_rating_kw: float | None = None,
    installed_motor_assist_kw: float | None = None,
    initial_altitude_m: float = 1200.0,
    initial_speed_mps: float = 44.0,
    energy_glide_guidance: bool = False,
    plant_health_provider: object | None = None,
    speed_command_event: tuple[str, float, float, float] | None = None,
    recovery_guidance: bool = False,
) -> dict[str, object]:
    if (
        integration_step_s <= 0.0
        or control_period_s < integration_step_s
        or terminal_control_period_s < integration_step_s
    ):
        raise ValueError("Control period must be at least one positive JSBSim integration step.")
    if sample_period_s <= 0.0 or mission_hours <= 0.0:
        raise ValueError("Sample period and mission duration must be positive.")
    if initial_altitude_m < 0.0 or initial_speed_mps <= 0.0:
        raise ValueError("Initial altitude must be nonnegative and speed positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 25.0 <= loiter_speed_mps <= 50.0:
        raise ValueError("Loiter speed must be between 25 and 50 m/s.")
    if route_leg_length_m <= 0.0:
        raise ValueError("Route leg length must be positive.")
    schedule = phase_schedule(
        mission_hours,
        climb_altitude_m,
        loiter_altitude_m,
        loiter_speed_mps,
        mission_mode,
        phase_overrides,
    )
    params = replace(
        PTDL6DoFParameters(),
        **(aircraft_overrides or {}),
    )
    requested_duration_s = mission_hours * 3600.0
    terminal_grace_s = (
        TERMINAL_LANDING_GRACE_S
        if mission_mode in {"full", "endurance_then_land"}
        else 0.0
    )
    scheduled_duration_s = max(float(item["end_s"]) for item in schedule)
    simulation_limit_s = scheduled_duration_s + terminal_grace_s
    fdm = build_fdm(
        integration_step_s,
        initial_altitude_m,
        initial_speed_mps,
        model_dir,
    )
    dt_s = float(fdm.get_delta_t())
    steps_per_control = max(1, round(control_period_s / dt_s))
    control_dt_s = steps_per_control * dt_s

    max_engine_cap_kw = max(float(item["engine_cap_kw"]) for item in schedule)
    max_motor_cap_kw = max(float(item["motor_cap_kw"]) for item in schedule)
    efficiency_values = engine_efficiency_values or tuple(
        min(0.60, value * _endurance_efficiency_scale())
        for value in (0.22, 0.28, 0.315, 0.31, 0.29)
    )
    aircraft = SimpleNamespace(
        engine_electrical_power_max_kw=(
            max_engine_cap_kw if installed_engine_rating_kw is None else float(installed_engine_rating_kw)
        ),
        motor_power_max_kw=(
            max_motor_cap_kw if installed_motor_assist_kw is None else float(installed_motor_assist_kw)
        ),
        battery_capacity_mj=max(0.0, battery_capacity_mj),
        engine_efficiency_power_fractions=(0.0, 0.25, 0.50, 0.75, 1.0),
        engine_efficiency_values=efficiency_values,
    )
    active_powertrain_config = HybridSystemAdapterConfig(
        **(
            powertrain_config
            if powertrain_config is not None
            else {"engine_power_rate_limit_kwps": 10.0}
        )
    )
    adapter = HybridSystemPowertrain(active_powertrain_config)
    power_state = adapter.initial_state(aircraft, initial_soc=initial_soc)
    coordinator = dispatch_policy or RuleBasedHybridCoordinator()
    coordinator.reset()
    battery_energy = BatteryEnergyLedger()
    phase_battery_energy = {}

    route_index = 0
    route_cycles_completed = 0
    route_start_lat = float(fdm["position/lat-geod-deg"])
    route_start_lon = float(fdm["position/long-gc-deg"])
    active_route_waypoints: list[tuple[float, float]] = []
    route_initialized = False
    records: list[dict[str, float | str]] = []
    fuel_used_kg = 0.0
    current_thrust_n = 0.0
    current_engine_kw = 0.0
    current_motor_kw = 0.0
    current_fuel_flow_kgps = 0.0
    max_bank_deg = 0.0
    max_pitch_deg = 0.0
    max_abs_altitude_error_m = 0.0
    max_route_heading_error_deg = 0.0
    minimum_speed_mps = math.inf
    minimum_soc = 1.0
    completed_steps = 0
    touchdown = False
    fuel_exhausted = False
    terminal_flare_active = False
    terminal_pitch_error_integral_rad_s = 0.0
    visited_phases: list[str] = []
    fuel_capacity_kg = (
        params.fuel_mass_kg
        if fuel_capacity_kg is None
        else max(float(fuel_capacity_kg), 0.0)
    )
    fdm["propulsion/tank/contents-lbs[0]"] = fuel_capacity_kg * KG_TO_LB
    total_steps = math.ceil(simulation_limit_s / dt_s)
    next_sample_time_s = 0.0
    route_template_ne_m = (
        (route_leg_length_m, 0.0),
        (route_leg_length_m, route_leg_length_m),
        (0.0, route_leg_length_m),
        (0.0, 0.0),
    )

    control_index = 0
    previous_dispatch_engine_kw = 0.0
    phase_entry_times = {}
    recovery_reference = None
    while completed_steps < total_steps:
        time_s = completed_steps * dt_s
        if time_s >= simulation_limit_s:
            break
        scheduled_phase = next(
            (item for item in schedule if time_s < float(item["end_s"])),
            schedule[-1],
        )
        altitude_m = float(fdm["position/h-sl-ft"]) / FT_PER_M
        phase, terminal_flare_active = select_landing_guidance_phase(
            scheduled_phase,
            schedule[-2],
            schedule[-1],
            altitude_m,
            terminal_flare_active,
        )
        phase_name = str(phase["name"])
        if energy_glide_guidance and phase_name == "long_descent" and altitude_m < 50.0:
            phase = schedule[-1]
            phase_name = str(phase["name"])
            terminal_flare_active = True
        if not visited_phases or visited_phases[-1] != phase_name:
            visited_phases.append(phase_name)
            phase_entry_times[phase_name] = time_s
        command_event_active = False
        if speed_command_event is not None:
            event_phase, event_start, event_duration, event_delta = speed_command_event
            event_elapsed = time_s - phase_entry_times[phase_name]
            command_event_active = (
                phase_name == event_phase and event_start <= event_elapsed < event_start+event_duration
            )
            if command_event_active:
                phase = dict(phase)
                phase["speed_mps"] = float(phase["speed_mps"]) + event_delta
        active_control_period_s = (
            terminal_control_period_s
            if phase_name in {"long_descent", "approach_descent", "flare"}
            else control_period_s
        )
        active_steps_per_control = max(
            1, round(active_control_period_s / dt_s)
        )
        active_control_dt_s = active_steps_per_control * dt_s
        if phase_name == "endurance_loiter" and not route_initialized:
            route_start_lat = float(fdm["position/lat-geod-deg"])
            route_start_lon = float(fdm["position/long-gc-deg"])
            entry_heading = float(fdm["attitude/psi-rad"])
            cos_heading = math.cos(entry_heading)
            sin_heading = math.sin(entry_heading)
            active_route_waypoints = [
                (
                    north * cos_heading - east * sin_heading,
                    north * sin_heading + east * cos_heading,
                )
                for north, east in route_template_ne_m
            ]
            route_initialized = True
        altitude_m = float(fdm["position/h-sl-ft"]) / FT_PER_M
        speed_mps = max(float(fdm["velocities/vt-fps"]) / FT_PER_M, 1.0)
        alpha_rad = float(fdm["aero/alpha-rad"])
        beta_rad = float(fdm["aero/beta-rad"])
        pitch_rad = float(fdm["attitude/theta-rad"])
        bank_rad = float(fdm["attitude/phi-rad"])
        heading_rad = float(fdm["attitude/psi-rad"])
        pitch_rate = float(fdm["velocities/q-rad_sec"])
        roll_rate = float(fdm["velocities/p-rad_sec"])
        yaw_rate = float(fdm["velocities/r-rad_sec"])

        altitude_error = float(phase["altitude_m"]) - altitude_m
        max_gamma = math.radians(float(phase["gamma_limit_deg"]))
        phase_remaining_s = max(float(phase["end_s"]) - time_s, 30.0)
        altitude_response_time_s = (
            240.0
            if phase_name == "endurance_loiter"
            else 900.0
            if phase_name == "main_climb_to_leveloff"
            else 600.0
            if phase_name == "long_descent"
            else max(phase_remaining_s, APPROACH_ALTITUDE_RESPONSE_TIME_MIN_S)
            if phase_name == "approach_descent"
            else max(phase_remaining_s, 120.0)
        )
        vertical_speed_limit = min(speed_mps * math.sin(max_gamma), 2.5)
        vertical_speed_up_mps = -float(fdm["velocities/v-down-fps"]) / FT_PER_M
        desired_vertical_speed = np_clip(
            altitude_error / altitude_response_time_s
            - 0.80 * vertical_speed_up_mps,
            -vertical_speed_limit,
            vertical_speed_limit,
        )
        if phase_name == "flare":
            target_vertical_speed_up_mps = flare_target_vertical_speed_up_mps(
                altitude_m
            )
            desired_vertical_speed = np_clip(
                target_vertical_speed_up_mps,
                -vertical_speed_limit,
                vertical_speed_limit,
            )
        if energy_glide_guidance and phase_name in {"long_descent", "approach_descent", "flare"}:
            from 协同自适应方法 import glide_command
            glide_mass = float(fdm["inertia/weight-lbs"]) / KG_TO_LB
            glide_q = float(fdm["aero/qbar-psf"]) * PSF_TO_PA
            glide_cl = params.cl0 + params.cl_alpha_per_rad * alpha_rad
            glide_ar = params.wingspan_m**2 / params.reference_wing_area_m2
            glide_drag = glide_q * params.reference_wing_area_m2 * (
                params.cd0 + glide_cl**2/(math.pi*glide_ar*params.oswald_efficiency)
            )
            desired_vertical_speed = glide_command(
                speed_mps, float(phase["speed_mps"]), glide_drag,
                glide_mass*G, altitude_m,
            )
            # A descending reachable reference replaces the powered altitude hold.
            altitude_error = desired_vertical_speed * active_control_dt_s
        if recovery_guidance and phase_name == "long_descent":
            from 协同自适应方法 import glide_command
            elapsed = time_s - float(phase["start_s"])
            outage = float(phase.get("power_off_duration_s", 0.0))
            if recovery_reference is None and (elapsed >= outage or altitude_m <= 900.0):
                recovery_reference = (time_s, altitude_m)
            if recovery_reference is None:
                recovery_mass = float(fdm["inertia/weight-lbs"]) / KG_TO_LB
                recovery_q = float(fdm["aero/qbar-psf"]) * PSF_TO_PA
                recovery_cl = params.cl0 + params.cl_alpha_per_rad*alpha_rad
                recovery_ar = params.wingspan_m**2/params.reference_wing_area_m2
                recovery_drag = recovery_q*params.reference_wing_area_m2*(
                    params.cd0+recovery_cl**2/(math.pi*recovery_ar*params.oswald_efficiency))
                desired_vertical_speed = glide_command(
                    speed_mps,float(phase["speed_mps"]),recovery_drag,recovery_mass*G,altitude_m)
                altitude_error = desired_vertical_speed*active_control_dt_s
            else:
                start, height = recovery_reference
                duration = max(float(phase["end_s"])-start,1.)
                slope = (float(phase["altitude_m"])-height)/duration
                reachable_height = height+slope*min(time_s-start,duration)
                altitude_error = reachable_height-altitude_m
                desired_vertical_speed = np_clip(
                    slope+altitude_error/120.-.5*(vertical_speed_up_mps-slope),
                    -vertical_speed_limit,vertical_speed_limit)
        gamma_command = math.atan2(desired_vertical_speed, max(speed_mps, 1.0))
        rho = 1.225 * max(1.0 - 2.25577e-5 * altitude_m, 1e-3) ** 4.2559
        qbar_reference_pa = 0.5 * rho * speed_mps**2
        mass_kg = float(fdm["inertia/weight-lbs"]) / KG_TO_LB
        target_cl = mass_kg * G / max(
            qbar_reference_pa
            * params.reference_wing_area_m2
            * max(math.cos(bank_rad), 0.85),
            1.0,
        )
        alpha_trim_rad = math.radians(
            np_clip(
                math.degrees((target_cl - params.cl0) / params.cl_alpha_per_rad),
                -2.0,
                12.0,
            )
        )
        elevator_trim_rad = (
            params.cm0 + params.cm_alpha_per_rad * alpha_trim_rad
        ) / -params.cm_elevator_per_rad
        theta_command = float(
            np_clip(
                alpha_trim_rad + gamma_command,
                math.radians(-5.0),
                math.radians(14.0),
            )
        )

        heading_command = heading_rad
        if phase_name == "endurance_loiter":
            target_north, target_east = active_route_waypoints[route_index]
            lat = math.radians(float(fdm["position/lat-geod-deg"]))
            north_m = math.radians(float(fdm["position/lat-geod-deg"]) - route_start_lat) * 6_371_000.0
            east_m = (
                math.radians(float(fdm["position/long-gc-deg"]) - route_start_lon)
                * 6_371_000.0
                * max(math.cos(lat), 1e-4)
            )
            delta_north = target_north - north_m
            delta_east = target_east - east_m
            distance = math.hypot(delta_north, delta_east)
            if distance < max(10000.0, 0.04 * route_leg_length_m):
                route_index = (route_index + 1) % len(active_route_waypoints)
                if route_index == 0:
                    route_cycles_completed += 1
                target_north, target_east = active_route_waypoints[route_index]
                delta_north = target_north - north_m
                delta_east = target_east - east_m
            heading_command = math.atan2(delta_east, delta_north)
        heading_error = wrap_pi(heading_command - heading_rad)
        max_route_heading_error_deg = max(
            max_route_heading_error_deg, abs(math.degrees(heading_error))
        )
        bank_command = float(
            np_clip(0.30 * heading_error, math.radians(-8.0), math.radians(8.0))
        )
        phase_elapsed = time_s - float(phase["start_s"])
        speed_target = float(phase["speed_mps"])
        speed_error = speed_target - speed_mps
        qbar_pa = float(fdm["aero/qbar-psf"]) * PSF_TO_PA
        aspect_ratio = params.wingspan_m**2 / params.reference_wing_area_m2
        lift_coefficient_for_drag = (
            params.cl0 + params.cl_alpha_per_rad * alpha_rad
        )
        induced_drag_factor = 1.0 / (
            math.pi * aspect_ratio * params.oswald_efficiency
        )
        cd = (
            params.cd0
            + induced_drag_factor * lift_coefficient_for_drag**2
            + 0.35 * beta_rad**2
        )
        drag_n = qbar_pa * params.reference_wing_area_m2 * cd
        speed_acceleration_command = np_clip(0.08 * speed_error, -0.8, 0.8)
        required_shaft_kw = max(
            0.0,
            (
                drag_n
                + mass_kg * G * math.sin(gamma_command)
                + mass_kg * speed_acceleration_command
            )
            * speed_mps
            / params.propulsive_efficiency
            / 1000.0,
        )
        speed_tracking_power_kw = (
            max(drag_n + mass_kg * speed_acceleration_command, 0.0)
            * speed_mps
            / params.propulsive_efficiency
            / 1000.0
        )
        required_shaft_kw = max(required_shaft_kw, speed_tracking_power_kw)
        engine_cap_kw = min(float(phase["engine_cap_kw"]), aircraft.engine_electrical_power_max_kw)
        motor_cap_kw = min(float(phase["motor_cap_kw"]), aircraft.motor_power_max_kw)
        power_off_duration_s = float(phase.get("power_off_duration_s", 0.0))
        power_loss_active = 0.0 <= phase_elapsed < power_off_duration_s
        if recovery_guidance and phase_name == "long_descent" and recovery_reference is not None:
            power_loss_active = False
        if energy_glide_guidance and phase_name in {"long_descent", "approach_descent", "flare"}:
            power_loss_active = True
        if power_loss_active:
            engine_cap_kw = 0.0
            motor_cap_kw = 0.0
        efficiency_curve = adapter.engine_efficiency_curve(power_state, altitude_m, engine_cap_kw)
        observation = DispatchObservation(
            phase_name=phase_name,
            time_step_s=active_control_dt_s,
            soc=float(power_state.battery.soc),
            altitude_error_m=altitude_error,
            speed_error_mps=speed_error,
            required_propulsion_kw=required_shaft_kw,
            engine_power_cap_kw=engine_cap_kw,
            motor_power_min_kw=-min(18.0, motor_cap_kw),
            motor_power_max_kw=motor_cap_kw,
            battery_capacity_mj=aircraft.battery_capacity_mj,
            battery_soc_min=active_powertrain_config.battery_soc_min,
            battery_soc_max=active_powertrain_config.battery_soc_max,
            battery_soc_reserve=active_powertrain_config.battery_soc_min,
            engine_power_rate_limit_kwps=active_powertrain_config.engine_power_rate_limit_kwps,
            engine_efficiency_power_fractions=efficiency_curve[0],
            engine_efficiency_values=efficiency_curve[1],
            phase_elapsed_s=phase_elapsed,
            mission_elapsed_s=time_s,
            required_motor_dc_kw=adapter.motor_dc_demand(power_state, required_shaft_kw),
        )
        dispatch = coordinator.dispatch(observation)
        plant_health = (
            plant_health_provider(phase_name, phase_elapsed, time_s)
            if plant_health_provider is not None else None
        )
        effective_engine_cap_kw = (
            engine_cap_kw
            if dispatch.engine_power_cap_kw is None
            else min(engine_cap_kw, max(0.0, dispatch.engine_power_cap_kw))
        )
        scheduled_motor_min_kw = -min(18.0, motor_cap_kw)
        effective_motor_min_kw = (
            scheduled_motor_min_kw
            if dispatch.motor_power_min_kw is None
            else max(scheduled_motor_min_kw, dispatch.motor_power_min_kw)
        )
        effective_motor_max_kw = (
            motor_cap_kw
            if dispatch.motor_power_max_kw is None
            else min(motor_cap_kw, max(0.0, dispatch.motor_power_max_kw))
        )
        if plant_health is not None:
            effective_engine_cap_kw = min(
                effective_engine_cap_kw, engine_cap_kw*plant_health.engine_available_fraction)
            effective_motor_max_kw = min(
                effective_motor_max_kw, motor_cap_kw*plant_health.motor_available_fraction)
            effective_motor_min_kw = max(
                effective_motor_min_kw, scheduled_motor_min_kw*plant_health.motor_available_fraction)
        previous_soc = float(power_state.battery.soc)
        output = adapter.step(
            power_state,
            SimpleNamespace(
                phase_name=phase_name,
                altitude_m=altitude_m,
                engine_target_is_dc=dispatch.engine_target_is_dc,
                engine_target_kw=dispatch.engine_target_kw,
                engine_power_cap_kw=effective_engine_cap_kw,
                propulsion_power_request_kw=required_shaft_kw,
                motor_power_min_kw=effective_motor_min_kw,
                motor_power_max_kw=effective_motor_max_kw,
                battery_available_energy_fraction=(
                    plant_health.battery_available_energy_fraction
                    if plant_health is not None else dispatch.battery_available_energy_fraction
                ),
            ),
            active_control_dt_s,
            aircraft,
        )
        power_state = output.state
        if hasattr(coordinator, "update_powertrain_feedback"):
            coordinator.update_powertrain_feedback(output)
        if hasattr(coordinator, "update_feedback"):
            coordinator.update_feedback(
                dt_s=active_control_dt_s, soc_before=previous_soc, soc_after=output.soc,
                current_a=output.battery_current_a,
                nominal_capacity_ah=power_state.nominal_battery_parameters["BAT_Capacity_Ah"],
                command_kw=dispatch.engine_target_kw, engine_kw=output.engine_power_kw,
                nominal_cap_kw=engine_cap_kw, speed_error_mps=speed_error,
                altitude_error_m=altitude_error,
                soc_min=active_powertrain_config.battery_soc_min,
                soc_max=active_powertrain_config.battery_soc_max,
                coulombic_efficiency=power_state.nominal_battery_parameters["BAT_CoulombicEfficiency"],
            )
        battery_energy.update(
            output.battery_terminal_power_kw, output.battery_chemical_power_kw,
            active_control_dt_s,
        )
        phase_battery_energy.setdefault(phase_name, BatteryEnergyLedger()).update(
            output.battery_terminal_power_kw, output.battery_chemical_power_kw,
            active_control_dt_s,
        )
        current_engine_kw = output.engine_power_kw
        current_motor_kw = output.motor_power_kw
        current_fuel_flow_kgps = output.fuel_flow_kgps
        current_thrust_n = (
            output.propulsor_input_power_kw * 1000.0 * params.propulsive_efficiency
            / max(speed_mps, 1.0)
        )
        fuel_used_kg += current_fuel_flow_kgps * active_control_dt_s
        if fuel_used_kg >= fuel_capacity_kg:
            fuel_used_kg = fuel_capacity_kg
            fuel_exhausted = True
        remaining_fuel_kg = max(fuel_capacity_kg - fuel_used_kg, 0.0)
        fdm["propulsion/tank/contents-lbs[0]"] = remaining_fuel_kg * KG_TO_LB
        minimum_soc = min(minimum_soc, output.soc)
        fdm["propulsion/ptdl-thrust-lbf"] = current_thrust_n * LBF_PER_N

        if time_s + 1e-9 >= next_sample_time_s or control_index == 0 or fuel_exhausted:
            row = {
                "time_s": time_s,
                "phase": phase_name,
                "phase_elapsed_s": phase_elapsed,
                "latitude_deg": float(fdm["position/lat-geod-deg"]),
                "longitude_deg": float(fdm["position/long-gc-deg"]),
                "altitude_m": altitude_m,
                "altitude_target_m": altitude_m + altitude_error,
                "altitude_error_m": altitude_error,
                "vertical_speed_up_mps": vertical_speed_up_mps,
                "desired_vertical_speed_mps": desired_vertical_speed,
                "gamma_command_deg": math.degrees(gamma_command),
                "pitch_error_deg": math.degrees(pitch_rad - theta_command),
                "speed_mps": speed_mps,
                "speed_target_mps": speed_target,
                "mass_kg": mass_kg,
                "estimated_drag_n": drag_n,
                "required_propulsion_power_kw": required_shaft_kw,
                "engine_target_kw": dispatch.engine_target_kw,
                "previous_engine_target_kw": previous_dispatch_engine_kw,
                "control_period_s": active_control_dt_s,
                "soc_before": previous_soc,
                "soc_target": dispatch.soc_target,
                "dispatch_mode": dispatch.mode,
                "engine_power_cap_kw": effective_engine_cap_kw,
                "power_loss_active": float(power_loss_active),
                "speed_command_event_active": float(command_event_active),
                "motor_power_min_kw": effective_motor_min_kw,
                "motor_power_max_kw": effective_motor_max_kw,
                "bank_deg": math.degrees(bank_rad),
                "bank_command_deg": math.degrees(bank_command),
                "pitch_deg": math.degrees(pitch_rad),
                "pitch_command_deg": math.degrees(theta_command),
                "pitch_rate_degps": math.degrees(pitch_rate),
                "elevator_command_norm": float(fdm["fcs/elevator-cmd-norm"]),
                "elevator_actuator_norm": float(fdm["fcs/elevator-actuator-norm"]),
                "elevator_position_deg": math.degrees(
                    float(fdm["fcs/elevator-pos-rad"])
                ),
                "heading_deg": math.degrees(heading_rad) % 360.0,
                "heading_command_deg": math.degrees(heading_command) % 360.0,
                "heading_error_deg": math.degrees(heading_error),
                "alpha_deg": math.degrees(alpha_rad),
                "engine_power_kw": current_engine_kw,
                "motor_bus_power_kw": current_motor_kw,
                "shaft_power_kw": output.propulsor_input_power_kw,
                "fuel_flow_kgps": current_fuel_flow_kgps,
                "battery_current_a": output.battery_current_a,
                "battery_terminal_power_kw": output.battery_terminal_power_kw,
                "battery_chemical_power_kw": output.battery_chemical_power_kw,
                "battery_available_fraction": power_state.battery_available_fraction,
                "battery_effective_capacity_ah": (
                    power_state.nominal_battery_parameters["BAT_Capacity_Ah"]
                    * power_state.battery_available_fraction
                ),
                "battery_isolated_charge_ah": power_state.battery_isolated_charge_ah,
                "battery_reconnection_count": power_state.battery_reconnection_count,
                "estimated_battery_available_fraction": (
                    coordinator.estimator.battery_fraction if hasattr(coordinator, "estimator") else 1.0
                ),
                "estimated_engine_available_fraction": (
                    coordinator.estimator.engine_fraction if hasattr(coordinator, "estimator") else 1.0
                ),
                "fuel_used_kg": fuel_used_kg,
                "soc": output.soc,
                "route_waypoint_index": route_index,
            }
            records.append(row)
            max_bank_deg = max(max_bank_deg, abs(row["bank_deg"]))
            max_pitch_deg = max(max_pitch_deg, abs(row["pitch_deg"]))
            max_abs_altitude_error_m = max(max_abs_altitude_error_m, abs(altitude_error))
            minimum_speed_mps = min(minimum_speed_mps, speed_mps)
            next_sample_time_s = time_s + sample_period_s
        previous_dispatch_engine_kw = dispatch.engine_target_kw

        if fuel_exhausted:
            break

        for _ in range(active_steps_per_control):
            if completed_steps >= total_steps:
                break
            pitch_now = float(fdm["attitude/theta-rad"])
            bank_now = float(fdm["attitude/phi-rad"])
            pitch_rate_now = float(fdm["velocities/q-rad_sec"])
            roll_rate_now = float(fdm["velocities/p-rad_sec"])
            beta_now = float(fdm["aero/beta-rad"])
            yaw_rate_now = float(fdm["velocities/r-rad_sec"])
            terminal_phase = phase_name in {
                "long_descent",
                "approach_descent",
                "flare",
            }
            active_pitch_kp = (
                terminal_pitch_kp
                if terminal_phase and terminal_pitch_kp is not None
                else pitch_kp
            )
            active_pitch_kd = (
                terminal_pitch_kd
                if terminal_phase and terminal_pitch_kd is not None
                else pitch_kd
            )
            pitch_error_now = pitch_now - theta_command
            integral_limit = (
                math.radians(8.0) / terminal_pitch_ki
                if terminal_pitch_ki > 0.0
                else 0.0
            )
            proposed_integral = terminal_pitch_error_integral_rad_s
            if terminal_phase and terminal_pitch_ki > 0.0:
                proposed_integral = np_clip(
                    terminal_pitch_error_integral_rad_s
                    + pitch_error_now * dt_s,
                    -integral_limit,
                    integral_limit,
                )
            elevator_base_rad = (
                elevator_trim_rad
                + active_pitch_kp * pitch_error_now
                + active_pitch_kd * pitch_rate_now
            )
            elevator_rad = elevator_base_rad
            if terminal_phase and terminal_pitch_ki > 0.0:
                elevator_with_integral_rad = (
                    elevator_base_rad
                    + terminal_pitch_ki * proposed_integral
                )
                if (
                    abs(elevator_with_integral_rad)
                    <= math.radians(params.elevator_limit_deg)
                    or elevator_with_integral_rad * pitch_error_now < 0.0
                ):
                    terminal_pitch_error_integral_rad_s = proposed_integral
                    elevator_rad = elevator_with_integral_rad
                else:
                    elevator_rad += (
                        terminal_pitch_ki * terminal_pitch_error_integral_rad_s
                    )
            aileron_rad = (
                1.8 * (bank_command - bank_now) - 1.2 * roll_rate_now
            )
            rudder_rad = 0.45 * beta_now + 0.18 * yaw_rate_now
            fdm["fcs/elevator-cmd-norm"] = np_clip(
                elevator_rad / math.radians(params.elevator_limit_deg), -1.0, 1.0
            )
            fdm["fcs/aileron-cmd-norm"] = np_clip(
                aileron_rad / math.radians(params.aileron_limit_deg), -1.0, 1.0
            )
            fdm["fcs/rudder-cmd-norm"] = np_clip(
                rudder_rad / math.radians(params.rudder_limit_deg), -1.0, 1.0
            )
            if not fdm.run():
                raise RuntimeError(f"JSBSim stopped at integration step {completed_steps}.")
            completed_steps += 1
            altitude_now = float(fdm["position/h-sl-ft"]) / FT_PER_M
            if phase_name == "flare" and altitude_now <= 1.0:
                touchdown = True
                break
        if touchdown:
            break
        current_speed = float(fdm["velocities/vt-fps"]) / FT_PER_M
        if not math.isfinite(current_speed) or current_speed < 15.0:
            current_altitude = float(fdm["position/h-sl-ft"]) / FT_PER_M
            current_alpha_deg = math.degrees(float(fdm["aero/alpha-rad"]))
            current_pitch_deg = math.degrees(float(fdm["attitude/theta-rad"]))
            partial_path = output_dir / "PTDL_JSBSim闭环失败前时序.csv"
            if records:
                with partial_path.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
            raise RuntimeError(
                "Estimated six-DoF envelope exited: "
                f"phase={phase_name}, speed={current_speed:.2f} m/s, "
                f"altitude={current_altitude:.1f} m, alpha={current_alpha_deg:.2f} deg, "
                f"pitch={current_pitch_deg:.2f} deg, thrust={current_thrust_n:.1f} N, "
                f"engine={current_engine_kw:.2f} kW, motor={current_motor_kw:.2f} kW "
                f"at {time_s:.1f} s; partial history={partial_path}."
            )
        control_index += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest_path = write_parameter_manifest(output_dir / "PTDL_六自由度参数清单.json")
    flight_profile_completed = touchdown or completed_steps >= total_steps
    fuel_margin_kg = fuel_capacity_kg - fuel_used_kg
    fuel_feasible = not fuel_exhausted and fuel_margin_kg >= 0.0
    phase_metrics: dict[str, dict[str, float | int]] = {}
    for name in visited_phases:
        phase_rows = [row for row in records if row["phase"] == name]
        if not phase_rows:
            continue
        phase_metrics[name] = {
            "sample_count": len(phase_rows),
            "duration_sampled_s": float(
                phase_rows[-1]["time_s"] - phase_rows[0]["time_s"]
            ),
            "altitude_error_abs_max_m": max(
                abs(float(row["altitude_error_m"])) for row in phase_rows
            ),
            "altitude_error_abs_mean_m": sum(
                abs(float(row["altitude_error_m"])) for row in phase_rows
            ) / len(phase_rows),
            "speed_error_abs_max_mps": max(
                abs(float(row["speed_mps"]) - float(row["speed_target_mps"]))
                for row in phase_rows
            ),
            "speed_min_mps": min(float(row["speed_mps"]) for row in phase_rows),
            "pitch_abs_max_deg": max(abs(float(row["pitch_deg"])) for row in phase_rows),
            "bank_abs_max_deg": max(abs(float(row["bank_deg"])) for row in phase_rows),
            "soc_start": float(phase_rows[0]["soc"]),
            "soc_end": float(phase_rows[-1]["soc"]),
            "fuel_used_end_kg": float(phase_rows[-1]["fuel_used_kg"]),
            "route_heading_error_abs_max_deg": max(
                abs(float(row["heading_error_deg"])) for row in phase_rows
            ),
        }
    summary = {
        "battery_energy_accounting": battery_energy.summary(),
        "phase_battery_energy_accounting": {
            name: ledger.summary() for name, ledger in phase_battery_energy.items()
        },
        "battery_availability_model": (
            "equivalent parallel-string isolation with charge-conserving quasistatic reconnection"
        ),
        "model_id": "PTDL-EST-6DOF-ULTRA-SCALE",
        "status": "estimated_model_software_closed_loop_not_calibrated_or_flight_validated",
        "mission_mode": mission_mode,
        "candidate_specific_model": model_dir != MODEL_DIR,
        "mission_profile": (
            "climb, leveloff, and endurance loiter; post-endurance descent and landing are excluded"
            if mission_mode == "endurance_only"
            else "80 h minimum elapsed flight before powered descent, approach, flare, and touchdown"
            if mission_mode == "endurance_then_land"
            else "six-phase schedule; loiter duration is residual to requested mission duration"
        ),
        "mission_hours_requested": mission_hours,
        "minimum_flight_hours_before_landing": (
            mission_hours if mission_mode == "endurance_then_land" else None
        ),
        "scheduled_profile_duration_s": scheduled_duration_s,
        "terminal_landing_grace_s": terminal_grace_s,
        "simulation_duration_s": completed_steps * dt_s,
        "integration_step_s": dt_s,
        "guidance_and_power_update_period_s": control_dt_s,
        "terminal_guidance_and_power_update_period_s": (
            max(1, round(terminal_control_period_s / dt_s)) * dt_s
        ),
        "pitch_gains": {
            "general": {"kp": pitch_kp, "kd": pitch_kd},
            "terminal": {
                "kp": pitch_kp if terminal_pitch_kp is None else terminal_pitch_kp,
                "kd": pitch_kd if terminal_pitch_kd is None else terminal_pitch_kd,
                "ki": terminal_pitch_ki,
            },
        },
        "approach_altitude_response_time_min_s": (
            APPROACH_ALTITUDE_RESPONSE_TIME_MIN_S
        ),
        "integration_steps": completed_steps,
        "sample_count": len(records),
        "phase_sequence": visited_phases,
        "phase_duration_s": {
            str(item["name"]): float(item["duration_s"]) for item in schedule
        },
        "waypoint_count": len(route_template_ne_m),
        "task_scenario": {
            "climb_altitude_m": climb_altitude_m,
            "loiter_altitude_m": loiter_altitude_m,
            "loiter_speed_mps": loiter_speed_mps,
            "route_leg_length_m": route_leg_length_m,
            "engine_electrical_power_max_kw": aircraft.engine_electrical_power_max_kw,
            "efficiency_scale_from_full_task_ULTRA_anchor": _endurance_efficiency_scale(),
            "candidate_phase_overrides": phase_overrides or {},
        },
        "route_cycles_completed_sampled": route_cycles_completed,
        "attitude_trajectory_route_feedback_enabled": True,
        "hybrid_component_adapter_enabled": True,
        "energy_management_policy": type(coordinator).__name__,
        "energy_glide_guidance": energy_glide_guidance,
        "recovery_guidance": recovery_guidance,
        "recovery_reference_time_height": recovery_reference,
        "powertrain_adapter_configuration": asdict(active_powertrain_config),
        "powertrain_source": adapter.metadata,
        "powertrain_aggregate_sizing": {
            "engine_electrical_power_max_kw": aircraft.engine_electrical_power_max_kw,
            "motor_power_max_kw": aircraft.motor_power_max_kw,
            "battery_capacity_mj": aircraft.battery_capacity_mj,
            "engine_rating_scale": power_state.engine_rating_scale,
            "motor_branches": power_state.motor_branches,
            "battery_parallel_equivalent": power_state.battery_parallel_equivalent,
        },
        "engine_map_altitude_clamped_steps": power_state.engine_map_altitude_clamped_steps,
        "maximum_abs_bank_deg_sampled": max_bank_deg,
        "maximum_abs_pitch_deg_sampled": max_pitch_deg,
        "maximum_abs_elevator_command_norm_sampled": max(
            (abs(float(row["elevator_command_norm"])) for row in records),
            default=0.0,
        ),
        "maximum_abs_elevator_position_deg_sampled": max(
            (abs(float(row["elevator_position_deg"])) for row in records),
            default=0.0,
        ),
        "maximum_abs_altitude_error_m_at_control_updates": max_abs_altitude_error_m,
        "maximum_route_heading_error_deg": max_route_heading_error_deg,
        "minimum_speed_mps_sampled": minimum_speed_mps,
        "terminal_speed_mps": float(fdm["velocities/vt-fps"]) / FT_PER_M,
        "terminal_vertical_speed_up_mps": -float(fdm["velocities/v-down-fps"]) / FT_PER_M,
        "terminal_alpha_deg": math.degrees(float(fdm["aero/alpha-rad"])),
        "terminal_altitude_m": float(fdm["position/h-sl-ft"]) / FT_PER_M,
        "minimum_soc": minimum_soc,
        "terminal_soc": float(power_state.battery.soc),
        "fuel_used_kg_estimated": fuel_used_kg,
        "fuel_capacity_kg_assumed": fuel_capacity_kg,
        "fuel_margin_kg_estimated": fuel_margin_kg,
        "fuel_feasible_under_assumed_capacity": fuel_feasible,
        "phase_metrics_sampled": phase_metrics,
        "touchdown_detected": touchdown,
        "touchdown_required_for_mission_completion": mission_mode
        in {"full", "endurance_then_land"},
        "fuel_exhausted_before_profile_end": fuel_exhausted,
        "flight_profile_completed": flight_profile_completed,
        "requested_duration_completed": completed_steps * dt_s >= requested_duration_s,
        "terminal_landing_window_completed": (
            mission_mode not in {"full", "endurance_then_land"}
            or touchdown
            or completed_steps * dt_s >= simulation_limit_s
        ),
        "mission_completed": mission_completion_status(
            mission_mode=mission_mode,
            flight_profile_completed=flight_profile_completed,
            fuel_feasible=fuel_feasible,
            touchdown=touchdown,
            elapsed_s=completed_steps * dt_s,
            requested_duration_s=requested_duration_s,
        ),
        "outputs": [str(csv_path), str(manifest_path)],
        "limitations": [
            "Aircraft mass, inertia, aerodynamic derivatives, and powertrain scaling are estimated assumptions.",
            "The route is a repeated four-waypoint local pattern; terrain, wind, navigation error, and airspace constraints are omitted.",
            "Guidance and energy management update more slowly than the JSBSim integration step.",
            "The 80 h simulation is software evidence only, not a calibrated performance or safety result.",
        ],
    }
    (output_dir / "PTDL_JSBSim全任务闭环摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def np_clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _endurance_efficiency_scale() -> float:
    cache_path = CORE / "ptdl_endurance_task_calibration_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    return float(cache["efficiency_scale"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration-step-s", type=float, default=1.0 / 60.0)
    parser.add_argument("--control-period-s", type=float, default=2.0)
    parser.add_argument("--sample-period-s", type=float, default=60.0)
    parser.add_argument("--mission-hours", type=float, default=80.0)
    parser.add_argument("--loiter-speed-mps", type=float, default=32.0)
    parser.add_argument("--loiter-altitude-m", type=float, default=2400.0)
    parser.add_argument("--climb-altitude-m", type=float, default=2200.0)
    parser.add_argument(
        "--mission-mode",
        choices=("endurance_only", "full", "endurance_then_land"),
        default="endurance_only",
    )
    parser.add_argument("--pitch-kp", type=float, default=2.5)
    parser.add_argument("--pitch-kd", type=float, default=2.5)
    parser.add_argument("--terminal-pitch-kp", type=float)
    parser.add_argument("--terminal-pitch-kd", type=float)
    parser.add_argument("--terminal-pitch-ki", type=float, default=0.15)
    parser.add_argument(
        "--route-leg-length-m",
        type=float,
        default=DEFAULT_ROUTE_LEG_LENGTH_M,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "PTDL_JSBSim全任务姿态轨迹航迹闭环_20261011",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output_dir,
                args.integration_step_s,
                args.control_period_s,
                args.sample_period_s,
                args.mission_hours,
                args.loiter_speed_mps,
                args.loiter_altitude_m,
                args.climb_altitude_m,
                args.route_leg_length_m,
                args.mission_mode,
                args.pitch_kp,
                args.pitch_kd,
                terminal_pitch_kp=args.terminal_pitch_kp,
                terminal_pitch_kd=args.terminal_pitch_kd,
                terminal_pitch_ki=args.terminal_pitch_ki,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
