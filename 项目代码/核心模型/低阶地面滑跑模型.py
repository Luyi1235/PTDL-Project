"""Preliminary longitudinal takeoff and landing ground-roll screening model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np

from 飞推控闭环物理模型 import (
    AircraftParameters,
    FUEL_LHV_J_PER_KG,
    G,
    isa_density,
    parasite_drag_coefficient,
    thrust_from_power,
    HybridPowertrainRequest,
)


@dataclass(frozen=True)
class GroundRollAssumptions:
    rolling_friction_coefficient: float = 0.035
    braking_deceleration_mps2: float = 2.0
    rotation_stall_speed_factor: float = 1.20
    touchdown_speed_factor: float = 1.15
    lift_coefficient_fraction: float = 0.80
    time_step_s: float = 0.1
    maximum_roll_time_s: float = 300.0
    headwind_mps: float = 0.0
    runway_slope_deg: float = 0.0
    shaft_power_response_time_constant_s: float = 1.5
    shaft_power_rate_limit_kwps: float = 60.0
    initial_shaft_power_kw: float = 0.0


@dataclass(frozen=True)
class GroundRollResult:
    phase: str
    feasible: bool
    distance_m: float
    duration_s: float
    terminal_speed_mps: float
    peak_speed_mps: float
    shaft_energy_mj: float
    mean_shaft_power_kw: float
    engine_shaft_energy_mj: float
    motor_shaft_energy_mj: float
    estimated_fuel_used_kg: float
    estimated_battery_energy_used_mj: float
    terminal_soc_estimated: float
    termination_reason: str
    assumptions: dict[str, Any]


def _forces(
    aircraft: AircraftParameters,
    speed_mps: float,
    altitude_m: float,
    cl: float,
    headwind_mps: float,
) -> tuple[float, float, float]:
    rho = isa_density(altitude_m)
    airspeed = max(0.0, speed_mps + headwind_mps)
    q = 0.5 * rho * airspeed**2
    lift = q * aircraft.wing_area_m2 * cl
    cd = parasite_drag_coefficient(aircraft, rho, max(airspeed, 1.0))
    cd += cl**2 / (np.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    drag = q * aircraft.wing_area_m2 * cd
    return float(rho), float(lift), float(drag)


def simulate_ground_roll(
    aircraft: AircraftParameters,
    phase: str,
    available_shaft_power_kw: float,
    altitude_m: float = 0.0,
    assumptions: GroundRollAssumptions = GroundRollAssumptions(),
    initial_speed_mps: float | None = None,
    mass_kg: float | None = None,
    engine_power_fraction: float = 1.0,
    initial_soc: float | None = None,
    battery_capacity_mj: float | None = None,
    powertrain: Any | None = None,
) -> GroundRollResult:
    """Integrate ground distance until liftoff or stop using a point-mass balance."""
    if phase not in {"takeoff", "landing"}:
        raise ValueError("phase must be 'takeoff' or 'landing'.")
    if available_shaft_power_kw < 0.0:
        raise ValueError("available_shaft_power_kw must be nonnegative.")
    if not 0.0 <= engine_power_fraction <= 1.0:
        raise ValueError("engine_power_fraction must be between 0 and 1.")
    active_soc = aircraft.initial_soc if initial_soc is None else float(initial_soc)
    active_battery_capacity_mj = (
        aircraft.battery_capacity_mj
        if battery_capacity_mj is None
        else float(battery_capacity_mj)
    )
    if not 0.0 <= active_soc <= 1.0 or active_battery_capacity_mj <= 0.0:
        raise ValueError("SOC must be in [0, 1] and battery capacity must be positive.")
    numeric_assumptions = (
        assumptions.rolling_friction_coefficient,
        assumptions.braking_deceleration_mps2,
        assumptions.rotation_stall_speed_factor,
        assumptions.touchdown_speed_factor,
        assumptions.lift_coefficient_fraction,
        assumptions.time_step_s,
        assumptions.maximum_roll_time_s,
        assumptions.headwind_mps,
        assumptions.runway_slope_deg,
        assumptions.shaft_power_response_time_constant_s,
        assumptions.shaft_power_rate_limit_kwps,
        assumptions.initial_shaft_power_kw,
    )
    if not all(isfinite(value) for value in numeric_assumptions):
        raise ValueError("Ground-roll assumptions must be finite.")
    if (
        assumptions.time_step_s <= 0.0
        or assumptions.maximum_roll_time_s <= 0.0
        or assumptions.rolling_friction_coefficient < 0.0
        or assumptions.braking_deceleration_mps2 < 0.0
        or assumptions.rotation_stall_speed_factor <= 0.0
        or assumptions.touchdown_speed_factor <= 0.0
        or assumptions.shaft_power_response_time_constant_s <= 0.0
        or assumptions.shaft_power_rate_limit_kwps < 0.0
        or assumptions.initial_shaft_power_kw < 0.0
        or not 0.0 < assumptions.lift_coefficient_fraction <= 1.0
        or abs(assumptions.runway_slope_deg) >= 15.0
    ):
        raise ValueError("Ground-roll assumptions are outside their physical screening bounds.")
    rho = isa_density(altitude_m)
    mass = aircraft.initial_mass_kg if mass_kg is None else float(mass_kg)
    if mass <= 0.0:
        raise ValueError("mass_kg must be positive.")
    weight_n = mass * G
    runway_grade = np.sin(np.deg2rad(assumptions.runway_slope_deg))
    cl_ground = assumptions.lift_coefficient_fraction * aircraft.cl_max
    stall_speed = np.sqrt(2.0 * weight_n / max(rho * aircraft.wing_area_m2 * aircraft.cl_max, 1e-9))
    target_speed = (
        assumptions.rotation_stall_speed_factor * stall_speed
        if phase == "takeoff"
        else assumptions.touchdown_speed_factor * stall_speed
    )
    speed = (
        (0.0 if phase == "takeoff" else target_speed)
        if initial_speed_mps is None
        else float(initial_speed_mps)
    )
    if speed < 0.0:
        raise ValueError("initial_speed_mps must be nonnegative.")
    distance = 0.0
    elapsed = 0.0
    peak_speed = speed
    shaft_energy_mj = 0.0
    engine_shaft_energy_mj = 0.0
    motor_shaft_energy_mj = 0.0
    shaft_power_kw = min(available_shaft_power_kw, assumptions.initial_shaft_power_kw)
    component_state = powertrain.initial_state(aircraft, active_soc) if powertrain else None
    component_fuel_kg = 0.0
    component_battery_energy_mj = 0.0
    reason = "maximum_roll_time"
    while elapsed < assumptions.maximum_roll_time_s:
        dt = min(assumptions.time_step_s, assumptions.maximum_roll_time_s - elapsed)
        rho, lift, drag = _forces(
            aircraft, speed, altitude_m, cl_ground, assumptions.headwind_mps
        )
        if phase == "takeoff" and speed >= target_speed and lift >= weight_n:
            reason = "liftoff"
            break
        if phase == "landing" and speed <= 1e-6:
            reason = "stopped"
            break
        if phase == "takeoff":
            response_fraction = 1.0 - np.exp(
                -dt / assumptions.shaft_power_response_time_constant_s
            )
            requested_power_change = (
                available_shaft_power_kw - shaft_power_kw
            ) * response_fraction
            maximum_power_change = assumptions.shaft_power_rate_limit_kwps * dt
            power_change = float(
                np.clip(
                    requested_power_change,
                    -maximum_power_change,
                    maximum_power_change,
                )
            )
            next_shaft_power_kw = float(
                np.clip(
                    shaft_power_kw + power_change,
                    0.0,
                    available_shaft_power_kw,
                )
            )
            step_shaft_power_kw = 0.5 * (shaft_power_kw + next_shaft_power_kw)
            step_engine_fraction = engine_power_fraction
            if powertrain is not None:
                propeller_eta = max(aircraft.propulsive_efficiency, 1e-9)
                component_output = powertrain.step(
                    component_state,
                    HybridPowertrainRequest(
                        "takeoff_roll",
                        available_shaft_power_kw / propeller_eta * engine_power_fraction,
                        aircraft.engine_electrical_power_max_kw,
                        step_shaft_power_kw / propeller_eta,
                        0.0, aircraft.motor_power_max_kw, altitude_m=altitude_m,
                        engine_target_is_dc=True,
                    ),
                    dt, aircraft,
                )
                component_state = component_output.state
                step_shaft_power_kw = component_output.propulsor_input_power_kw * propeller_eta
                component_fuel_kg += component_output.fuel_flow_kgps * dt
                component_battery_energy_mj += component_output.battery_chemical_power_kw * dt / 1000.0
                step_engine_fraction = min(
                    component_output.engine_power_kw / max(component_output.motor_dc_power_kw, 1e-9), 1.0
                )
            thrust, _ = thrust_from_power(
                step_shaft_power_kw * 1000.0,
                max(speed, 0.1),
                rho,
                aircraft.prop_disk_area_m2,
                aircraft.induced_loss_factor,
            )
            normal_force = max(0.0, weight_n - lift)
            acceleration = (
                thrust
                - drag
                - assumptions.rolling_friction_coefficient * normal_force
                - weight_n * runway_grade
            ) / mass
            shaft_energy_mj += step_shaft_power_kw * dt / 1000.0
            engine_shaft_energy_mj += (
                step_shaft_power_kw * step_engine_fraction * dt / 1000.0
            )
            motor_shaft_energy_mj += (
                step_shaft_power_kw * (1.0 - step_engine_fraction) * dt / 1000.0
            )
            shaft_power_kw = next_shaft_power_kw
        else:
            braking = assumptions.braking_deceleration_mps2
            rolling = assumptions.rolling_friction_coefficient * max(0.0, weight_n - lift) / mass
            acceleration = -(braking + rolling + drag / mass) - G * runway_grade
        if phase == "landing" and acceleration < 0.0:
            dt = min(dt, speed / -acceleration)
        speed_next = max(0.0, speed + acceleration * dt)
        distance += 0.5 * (speed + speed_next) * dt
        speed = speed_next
        peak_speed = max(peak_speed, speed)
        elapsed += dt
        if (
            phase == "takeoff"
            and acceleration <= 0.0
            and speed < target_speed
            and shaft_power_kw >= available_shaft_power_kw - 1e-9
        ):
            reason = "insufficient_acceleration"
            break
    success = reason in {"liftoff", "stopped"}
    propulsive_efficiency = max(aircraft.propulsive_efficiency, 1e-9)
    generator_efficiency = max(aircraft.engine_generator_efficiency, 1e-9)
    discharge_efficiency = max(aircraft.battery_discharge_efficiency, 1e-9)
    estimated_fuel_used_kg = (
        engine_shaft_energy_mj
        / (
            propulsive_efficiency
            * generator_efficiency
            * (FUEL_LHV_J_PER_KG / 1.0e6)
        )
    )
    estimated_battery_energy_used_mj = (
        motor_shaft_energy_mj / (propulsive_efficiency * discharge_efficiency)
    )
    terminal_soc = float(
        np.clip(
            active_soc - estimated_battery_energy_used_mj / active_battery_capacity_mj,
            0.0,
            1.0,
        )
    )
    if powertrain is not None:
        estimated_fuel_used_kg = component_fuel_kg
        estimated_battery_energy_used_mj = component_battery_energy_mj
        terminal_soc = float(component_state.battery.soc)
    return GroundRollResult(
        phase=phase,
        feasible=success,
        distance_m=float(distance),
        duration_s=float(elapsed),
        terminal_speed_mps=float(speed),
        peak_speed_mps=float(peak_speed),
        shaft_energy_mj=float(shaft_energy_mj),
        mean_shaft_power_kw=float(
            shaft_energy_mj * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        ),
        engine_shaft_energy_mj=float(engine_shaft_energy_mj),
        motor_shaft_energy_mj=float(motor_shaft_energy_mj),
        estimated_fuel_used_kg=float(estimated_fuel_used_kg),
        estimated_battery_energy_used_mj=float(estimated_battery_energy_used_mj),
        terminal_soc_estimated=terminal_soc,
        termination_reason=reason,
        assumptions={
            **asdict(assumptions),
            "powertrain_source": powertrain.metadata if powertrain is not None else "reference_energy_balance",
        },
    )
