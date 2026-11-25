"""Reconstructed flight-propulsion-control model for the PTDL algorithm case.

This is a clean, low-order fixed-wing longitudinal model rebuilt from the
methodological parts of the supplied references.  It keeps the reusable parts
of the earlier work: segmented mission states, two-dimensional point-mass
dynamics, a momentum-theory propulsion relation, slipstream-augmented wing
aerodynamics, and state continuity between mission phases.

It intentionally does not import or reuse the senior project's OpenMDAO,
Dymos, OpenVSP, or outer-loop optimization code.  The default numerical values
are *reference-model placeholders*, not PTDL design data.  They must be
replaced with validated airframe, engine, motor, and battery data before a
PTDL performance result is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from 线性动态系统 import (
    first_order_state_space_step,
    first_order_transfer_step,
)
from 指标体系与评价函数 import (
    BatteryLimits,
    EvaluationConfig,
    EvaluationResult,
    FlightEnvelope,
    MissionRequirements,
    PowerLimits,
    evaluate_mission,
)


G = 9.80665
FUEL_LHV_J_PER_KG = 43.0e6
AIR_DYNAMIC_VISCOSITY_KG_PER_M_S = 1.789e-5


@dataclass(frozen=True)
class AircraftParameters:
    """Reference values derived from the prior fixed-wing study where available."""

    empty_mass_kg: float = 560.78
    initial_fuel_kg: float = 800.0
    wing_area_m2: float = 26.0
    wing_span_m: float = 24.99
    cd0: float = 0.022
    # ``cd0`` is the geometry-specific value at the reference Reynolds
    # number. Dimension-driven aircraft populate these fields so the plant
    # can update parasite drag with actual altitude and speed during a mission.
    reference_chord_m: float = 0.0
    parasite_drag_reference_reynolds: float = 0.0
    oswald_efficiency: float = 0.84
    cl_alpha_per_rad: float = 5.2
    cl_max: float = 1.30
    cl_min: float = -0.60
    prop_disk_area_m2: float = 8.0
    induced_loss_factor: float = 1.18
    propulsive_efficiency: float = 0.86
    slipstream_factor: float = 0.55
    engine_electrical_power_max_kw: float = 160.0
    engine_generator_efficiency: float = 0.30
    engine_efficiency_power_fractions: tuple[float, ...] = (0.0, 1.0)
    engine_efficiency_values: tuple[float, ...] = (0.30, 0.30)
    motor_power_min_kw: float = -30.0
    motor_power_max_kw: float = 80.0
    battery_capacity_mj: float = 360.0
    battery_discharge_efficiency: float = 0.95
    battery_charge_efficiency: float = 0.95
    initial_soc: float = 0.85
    soc_safe: float = 0.15
    soc_reserve: float = 0.20
    # Reference dynamic values.  They are deliberately centralized so a
    # component model supplied by the propulsion team can replace them.
    engine_response_time_constant_s: float = 4.0
    engine_power_rate_limit_kwps: float = 10.0
    motor_response_time_constant_s: float = 0.4
    motor_power_rate_limit_kwps: float = 80.0
    battery_bus_voltage_v: float = 400.0

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span_m**2 / self.wing_area_m2

    @property
    def initial_mass_kg(self) -> float:
        return self.empty_mass_kg + self.initial_fuel_kg


@dataclass(frozen=True)
class PhaseCommand:
    """Stage command used by the low-order closed-loop flight model."""

    name: str
    duration_s: float
    altitude_target_m: float
    speed_target_mps: float
    engine_power_kw: float
    motor_power_kw: float
    max_flight_path_angle_deg: float = 10.0
    time_step_s: float = 20.0
    unpowered_glide: bool = False
    engine_shutdown: bool = False
    motor_disabled: bool = False


@dataclass(frozen=True)
class MissionDefinition:
    phases: tuple[PhaseCommand, ...]
    required_range_m: float | None = None
    required_loiter_s: float | None = None
    terminal_position_error_max_m: float | None = None
    initial_altitude_m: float = 0.0
    initial_speed_mps: float = 42.0
    initial_soc: float | None = None
    disturbances: tuple["LongitudinalDragDisturbance", ...] = ()
    fault_events: tuple["MissionFaultEvent", ...] = ()


@dataclass(frozen=True)
class MissionFaultEvent:
    """Replaceable health-state injection over a named mission-phase interval."""

    phase_name: str
    start_phase_elapsed_s: float
    duration_s: float
    engine_available_fraction: float = 1.0
    motor_available_fraction: float = 1.0
    battery_available_energy_fraction: float = 1.0
    sensor_data_valid: bool = True

    def __post_init__(self) -> None:
        values = (
            self.start_phase_elapsed_s,
            self.duration_s,
            self.engine_available_fraction,
            self.motor_available_fraction,
            self.battery_available_energy_fraction,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Fault-event values must be finite.")
        if self.start_phase_elapsed_s < 0.0 or self.duration_s <= 0.0:
            raise ValueError("Fault event start must be nonnegative and duration positive.")
        if any(not 0.0 <= value <= 1.0 for value in values[2:]):
            raise ValueError("Fault availability fractions must be in [0, 1].")


@dataclass(frozen=True)
class LongitudinalDragDisturbance:
    """Equivalent longitudinal drag event used for response screening."""

    phase_name: str
    start_time_s: float
    duration_s: float
    additional_drag_n: float
    response_step_s: float = 5.0
    recovery_window_s: float = 900.0
    recovery_dwell_s: float = 30.0
    headwind_step_mps: float = 0.0
    headwind_gust_amplitude_mps: float = 0.0
    headwind_gust_period_s: float = 10.0


@dataclass(frozen=True)
class FlightControlParameters:
    """Longitudinal guidance and attitude-loop parameters for the reference plant."""

    altitude_to_gamma_gain: float = 3.0e-4
    gamma_tracking_gain: float = 0.045
    max_gamma_rate_radps: float = np.deg2rad(0.20)
    alpha_min_rad: float = np.deg2rad(-5.0)
    # Keep the flight controller consistent with the 2.2.1 evaluation envelope.
    alpha_max_rad: float = np.deg2rad(12.0)
    alpha_time_constant_s: float = 1.5
    alpha_rate_limit_radps: float = np.deg2rad(4.0)
    speed_floor_mps: float = 18.0
    max_longitudinal_accel_mps2: float = 1.2
    speed_tracking_time_constant_s: float = 30.0
    glide_speed_path_angle_gain_rad_per_mps: float = 0.02
    glide_entry_acceleration_duration_s: float = 0.0
    # Long endurance trim is numerically stable to 120 s with the discrete
    # gamma loop below.  Transient phases retain their smaller mission steps.
    max_integration_step_s: float = 120.0


@dataclass(frozen=True)
class DispatchObservation:
    """State supplied to an external hybrid energy-management policy."""

    phase_name: str
    time_step_s: float
    soc: float
    altitude_error_m: float
    speed_error_mps: float
    required_propulsion_kw: float
    engine_power_cap_kw: float
    motor_power_min_kw: float
    motor_power_max_kw: float
    battery_capacity_mj: float = 0.0
    battery_discharge_efficiency: float = 0.95
    battery_charge_efficiency: float = 0.95
    battery_soc_min: float = 0.15
    battery_soc_max: float = 1.0
    battery_soc_reserve: float = 0.20
    engine_power_rate_limit_kwps: float = 10.0
    engine_efficiency_power_fractions: tuple[float, ...] = (0.0, 1.0)
    engine_efficiency_values: tuple[float, ...] = (0.30, 0.30)
    phase_elapsed_s: float = 0.0
    mission_elapsed_s: float = 0.0
    required_motor_dc_kw: float | None = None


@dataclass(frozen=True)
class PowerDispatch:
    """Coordinated generator target; motor power closes the instantaneous balance."""

    engine_target_kw: float
    soc_target: float
    mode: str
    engine_power_cap_kw: float | None = None
    motor_power_min_kw: float | None = None
    motor_power_max_kw: float | None = None
    battery_available_energy_fraction: float = 1.0
    engine_target_is_dc: bool = False


class HybridDispatchPolicy(Protocol):
    """Interface for hybrid power-split controllers used by ``simulate_mission``."""

    def reset(self) -> None: ...

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch: ...


@dataclass(frozen=True)
class EngineGeneratorRequest:
    """Electrical-power request accepted by a replaceable engine-generator model."""

    target_electrical_power_kw: float
    power_cap_kw: float
    phase_name: str


@dataclass(frozen=True)
class EngineGeneratorOutput:
    """Output contract for a propulsion-team engine-generator implementation."""

    state: Any
    electrical_power_kw: float
    fuel_flow_kgps: float


class EngineGeneratorModel(Protocol):
    """Replaceable aviation-engine and generator model interface.

    A dynamics-school model only needs to implement these two methods.  Its
    internal state may contain speed, torque, manifold pressure, thermal state,
    start-stop state, BSFC-map interpolation, or any other validated detail.
    The common flight-control solver only relies on the declared output power
    and fuel flow.
    """

    def initial_state(self, aircraft: AircraftParameters) -> Any: ...

    def step(
        self,
        state: Any,
        request: EngineGeneratorRequest,
        dt_s: float,
        aircraft: AircraftParameters,
    ) -> EngineGeneratorOutput: ...


@dataclass(frozen=True)
class ReferenceEngineGeneratorState:
    electrical_power_kw: float = 0.0


class ReferenceEngineGenerator:
    """First-order engine-generator placeholder with a configurable efficiency map.

    This model is intentionally modest.  It keeps response lag and power-rate
    limits visible in the coupled dynamics, but it is not a substitute for an
    engine map.  Replace this class with a validated component model when one
    becomes available.
    """

    def initial_state(self, aircraft: AircraftParameters) -> ReferenceEngineGeneratorState:
        return ReferenceEngineGeneratorState()

    def step(
        self,
        state: ReferenceEngineGeneratorState,
        request: EngineGeneratorRequest,
        dt_s: float,
        aircraft: AircraftParameters,
    ) -> EngineGeneratorOutput:
        cap_kw = min(max(request.power_cap_kw, 0.0), aircraft.engine_electrical_power_max_kw)
        target_kw = float(np.clip(request.target_electrical_power_kw, 0.0, cap_kw))
        lagged_kw = first_order_state_space_step(
            state.electrical_power_kw,
            target_kw,
            dt_s,
            max(aircraft.engine_response_time_constant_s, 1e-6),
        )
        maximum_change_kw = max(aircraft.engine_power_rate_limit_kwps, 0.0) * dt_s
        power_kw = float(np.clip(lagged_kw, state.electrical_power_kw - maximum_change_kw, state.electrical_power_kw + maximum_change_kw))
        power_kw = float(np.clip(power_kw, 0.0, cap_kw))
        efficiency = engine_efficiency_at_power(power_kw, aircraft)
        fuel_flow = power_kw * 1000.0 / max(efficiency * FUEL_LHV_J_PER_KG, 1e-9)
        return EngineGeneratorOutput(
            state=ReferenceEngineGeneratorState(power_kw),
            electrical_power_kw=power_kw,
            fuel_flow_kgps=float(fuel_flow),
        )


@dataclass(frozen=True)
class HybridPowertrainRequest:
    """Power-flow request formed by the flight controller and dispatch policy."""

    phase_name: str
    engine_target_kw: float
    engine_power_cap_kw: float
    propulsion_power_request_kw: float
    motor_power_min_kw: float | None = None
    motor_power_max_kw: float | None = None
    battery_available_energy_fraction: float = 1.0
    altitude_m: float = 0.0
    engine_target_is_dc: bool = False


@dataclass(frozen=True)
class HybridPowertrainOutput:
    """Common output contract for a complete replaceable hybrid powertrain."""

    state: Any
    engine_power_kw: float
    motor_power_kw: float
    propulsor_input_power_kw: float
    fuel_flow_kgps: float
    soc: float
    battery_current_a: float
    battery_terminal_power_kw: float = 0.0
    battery_chemical_power_kw: float = 0.0


class HybridPowertrainModel(Protocol):
    """Replaceable full powertrain interface for later topology-specific models."""

    def initial_state(self, aircraft: AircraftParameters, initial_soc: float) -> Any: ...

    def step(
        self,
        state: Any,
        request: HybridPowertrainRequest,
        dt_s: float,
        aircraft: AircraftParameters,
    ) -> HybridPowertrainOutput: ...


@dataclass(frozen=True)
class ReferenceHybridPowertrainState:
    engine_state: Any
    motor_power_kw: float
    soc: float


class ReferenceHybridPowertrain:
    """Equivalent DC-bus hybrid powertrain with replaceable engine model.

    ``engine_power_kw`` denotes engine-generator electrical-bus output and
    ``motor_power_kw`` denotes the battery contribution to that bus: positive
    values discharge the battery for propulsion and negative values absorb
    generator surplus for charging.  The default topology is therefore an
    energy-flow equivalent model, not a claim about the final PTDL mechanical
    layout.  A series, parallel, or power-split implementation can replace the
    entire class through ``HybridPowertrainModel`` without changing the flight
    dynamics or the 2.2 evaluation interface.
    """

    def __init__(self, engine_model: EngineGeneratorModel | None = None) -> None:
        self.engine_model = engine_model or ReferenceEngineGenerator()

    def initial_state(self, aircraft: AircraftParameters, initial_soc: float) -> ReferenceHybridPowertrainState:
        return ReferenceHybridPowertrainState(
            engine_state=self.engine_model.initial_state(aircraft),
            motor_power_kw=0.0,
            soc=float(np.clip(initial_soc, 0.0, 1.0)),
        )

    def step(
        self,
        state: ReferenceHybridPowertrainState,
        request: HybridPowertrainRequest,
        dt_s: float,
        aircraft: AircraftParameters,
    ) -> HybridPowertrainOutput:
        engine = self.engine_model.step(
            state.engine_state,
            EngineGeneratorRequest(request.engine_target_kw, request.engine_power_cap_kw, request.phase_name),
            dt_s,
            aircraft,
        )
        desired_motor_kw = request.propulsion_power_request_kw - engine.electrical_power_kw
        charge_room_mj = max(0.0, 1.0 - state.soc) * aircraft.battery_capacity_mj
        charge_room_kw = charge_room_mj * 1000.0 / max(aircraft.battery_charge_efficiency * dt_s, 1e-9)
        motor_min_kw = aircraft.motor_power_min_kw if request.motor_power_min_kw is None else request.motor_power_min_kw
        motor_max_kw = aircraft.motor_power_max_kw if request.motor_power_max_kw is None else request.motor_power_max_kw
        lower_kw = max(motor_min_kw, -charge_room_kw)
        desired_motor_kw = float(np.clip(desired_motor_kw, lower_kw, motor_max_kw))
        lagged_motor_kw = first_order_transfer_step(
            state.motor_power_kw,
            desired_motor_kw,
            dt_s,
            max(aircraft.motor_response_time_constant_s, 1e-6),
        )
        maximum_change_kw = max(aircraft.motor_power_rate_limit_kwps, 0.0) * dt_s
        motor_kw = float(np.clip(lagged_motor_kw, state.motor_power_kw - maximum_change_kw, state.motor_power_kw + maximum_change_kw))
        motor_kw = float(np.clip(motor_kw, lower_kw, motor_max_kw))
        battery_energy_mj = (
            max(motor_kw, 0.0) / max(aircraft.battery_discharge_efficiency, 1e-9)
            - aircraft.battery_charge_efficiency * max(-motor_kw, 0.0)
        ) * dt_s / 1000.0
        soc = float(np.clip(state.soc - battery_energy_mj / max(aircraft.battery_capacity_mj, 1e-9), 0.0, 1.0))
        return HybridPowertrainOutput(
            state=ReferenceHybridPowertrainState(engine.state, motor_kw, soc),
            engine_power_kw=engine.electrical_power_kw,
            motor_power_kw=motor_kw,
            propulsor_input_power_kw=max(0.0, engine.electrical_power_kw + motor_kw),
            fuel_flow_kgps=engine.fuel_flow_kgps,
            soc=soc,
            battery_current_a=float(motor_kw * 1000.0 / max(aircraft.battery_bus_voltage_v, 1e-9)),
            battery_terminal_power_kw=float(
                motor_kw / max(aircraft.battery_discharge_efficiency, 1e-9)
                if motor_kw >= 0.0
                else motor_kw * aircraft.battery_charge_efficiency
            ),
            battery_chemical_power_kw=float(
                motor_kw / max(aircraft.battery_discharge_efficiency, 1e-9)
                if motor_kw >= 0.0
                else motor_kw * aircraft.battery_charge_efficiency
            ),
        )


@dataclass(frozen=True)
class MissionSimulation:
    profile: pd.DataFrame
    response_metrics: dict[str, float]
    terminal_fuel_kg: float
    range_m: float
    loiter_s: float
    terminal_position_error_m: float
    termination_reason: str


def reference_evaluation_config(
    aircraft: AircraftParameters = AircraftParameters(),
    mission: MissionDefinition | None = None,
) -> EvaluationConfig:
    """Build the matching 2.2.1 evaluator config for the reference model.

    The limits inherit only values already present in the reference model.  The
    speed/altitude/load boundaries are provisional envelope assumptions.
    """
    return EvaluationConfig(
        mission=MissionRequirements(
            endurance_min_h=80.0,
            terminal_soc_min=aircraft.soc_reserve,
            soc_min=aircraft.soc_safe,
            fuel_reserve_min_kg=0.0,
            range_min_m=mission.required_range_m if mission else None,
            loiter_min_s=mission.required_loiter_s if mission else None,
            terminal_position_error_max_m=mission.terminal_position_error_max_m if mission else None,
        ),
        flight=FlightEnvelope(
            speed_min_mps=18.0,
            speed_max_mps=90.0,
            altitude_min_m=0.0,
            altitude_max_m=9000.0,
            angle_of_attack_min_rad=np.deg2rad(-5.0),
            angle_of_attack_max_rad=np.deg2rad(12.0),
            load_factor_max=2.5,
        ),
        power=PowerLimits(
            engine_power_max_kw=aircraft.engine_electrical_power_max_kw,
            motor_power_min_kw=aircraft.motor_power_min_kw,
            motor_power_max_kw=aircraft.motor_power_max_kw,
            engine_power_rate_max_kwps=10.0,
            motor_power_rate_max_kwps=10.0,
        ),
        battery=BatteryLimits(
            discharge_power_max_kw=aircraft.motor_power_max_kw,
            charge_power_max_kw=abs(aircraft.motor_power_min_kw),
            current_abs_max_a=max(abs(aircraft.motor_power_min_kw), aircraft.motor_power_max_kw) * 1000.0 / 400.0,
            discharge_efficiency=aircraft.battery_discharge_efficiency,
            charge_efficiency=aircraft.battery_charge_efficiency,
        ),
    )


def evaluate_coupled_mission(
    mission: MissionDefinition,
    aircraft: AircraftParameters = AircraftParameters(),
    controller: FlightControlParameters = FlightControlParameters(),
    dispatch_policy: HybridDispatchPolicy | None = None,
    engine_model: EngineGeneratorModel | None = None,
    powertrain: HybridPowertrainModel | None = None,
) -> tuple[MissionSimulation, EvaluationResult]:
    """Run the reconstructed model and evaluate it with the 2.2.1 definition."""
    simulation = simulate_mission(mission, aircraft, controller, dispatch_policy, engine_model, powertrain)
    result = evaluate_mission(
        simulation.profile,
        reference_evaluation_config(aircraft, mission),
        response_metrics=simulation.response_metrics,
        terminal_fuel_kg=simulation.terminal_fuel_kg,
        range_m=simulation.range_m,
        loiter_s=simulation.loiter_s,
        terminal_position_error_m=simulation.terminal_position_error_m,
    )
    result.diagnostics.update(
        {f"response_{name}": float(value) for name, value in simulation.response_metrics.items()}
    )
    return simulation, result


def reference_endurance_mission(loiter_duration_h: float = 80.0) -> MissionDefinition:
    """Return a replaceable five-stage mission definition for architecture tests.

    The 80 h loiter duration is a requested mission duration, not proof that
    the reference aircraft can achieve it.  The simulation ends early if fuel
    or battery safety limits are reached.
    """
    return MissionDefinition(
        phases=(
            PhaseCommand("takeoff_initial_climb", 900.0, 1200.0, 52.0, 125.0, 32.0, 12.0, 5.0),
            PhaseCommand("main_climb", 3600.0, 7600.0, 67.0, 115.0, 18.0, 9.0, 10.0),
            PhaseCommand("endurance_loiter", loiter_duration_h * 3600.0, 7600.0, 70.0, 58.0, 0.0, 3.0, 120.0),
            PhaseCommand("return_descent", 3600.0, 300.0, 55.0, 30.0, -8.0, 6.0, 20.0),
            PhaseCommand("approach", 900.0, 30.0, 35.0, 42.0, 8.0, 5.0, 5.0),
        ),
        required_loiter_s=loiter_duration_h * 3600.0,
    )


def simulate_mission(
    mission: MissionDefinition,
    aircraft: AircraftParameters = AircraftParameters(),
    controller: FlightControlParameters = FlightControlParameters(),
    dispatch_policy: HybridDispatchPolicy | None = None,
    engine_model: EngineGeneratorModel | None = None,
    powertrain: HybridPowertrainModel | None = None,
) -> MissionSimulation:
    """Simulate the complete segmented mission and produce the 2.2.1 profile.

    The plant closes the physical loop at every integration step: guidance
    generates a requested flight path and propulsion power, the hybrid system
    delivers limited actual power, the propulsor converts that power to thrust,
    and two-dimensional point-mass equations update speed, flight-path angle,
    altitude, range, fuel and SOC.  Thus a power shortfall can no longer be
    hidden by directly advancing the commanded flight state.
    """
    if powertrain is not None and engine_model is not None:
        raise ValueError("Provide either a complete powertrain or an engine model replacement, not both.")
    active_powertrain = powertrain or ReferenceHybridPowertrain(engine_model)
    if dispatch_policy is not None:
        dispatch_policy.reset()
    initial_soc = aircraft.initial_soc if mission.initial_soc is None else mission.initial_soc
    state = _State(
        range_m=0.0,
        altitude_m=mission.initial_altitude_m,
        speed_mps=mission.initial_speed_mps,
        gamma_rad=0.0,
        fuel_used_kg=0.0,
        soc=initial_soc,
        alpha_rad=0.0,
        powertrain_state=active_powertrain.initial_state(aircraft, initial_soc),
    )
    rows: list[dict[str, float | str]] = []
    time_s = 0.0
    loiter_s = 0.0
    previous_theta = 0.0
    termination_reason = "mission_complete"

    simulation_phases: list[PhaseCommand] = []
    for phase in mission.phases:
        if phase.unpowered_glide:
            if controller.glide_entry_acceleration_duration_s > 0.0:
                simulation_phases.append(
                    PhaseCommand(
                        name=f"{phase.name}_entry_acceleration",
                        duration_s=controller.glide_entry_acceleration_duration_s,
                        altitude_target_m=phase.altitude_target_m,
                        speed_target_mps=phase.speed_target_mps,
                        engine_power_kw=phase.engine_power_kw,
                        motor_power_kw=0.0,
                        max_flight_path_angle_deg=phase.max_flight_path_angle_deg,
                        time_step_s=min(1.0, phase.time_step_s),
                    )
                )
            shutdown_rate_kwps = min(aircraft.engine_power_rate_limit_kwps, 10.0)
            shutdown_duration_s = (
                aircraft.engine_electrical_power_max_kw / max(shutdown_rate_kwps, 1e-9)
                + 1.0
            )
            simulation_phases.append(
                PhaseCommand(
                    name=f"{phase.name}_engine_shutdown",
                    duration_s=shutdown_duration_s,
                    altitude_target_m=phase.altitude_target_m,
                    speed_target_mps=phase.speed_target_mps,
                    engine_power_kw=aircraft.engine_electrical_power_max_kw,
                    motor_power_kw=0.0,
                    max_flight_path_angle_deg=phase.max_flight_path_angle_deg,
                    time_step_s=min(1.0, phase.time_step_s),
                    engine_shutdown=True,
                )
            )
        simulation_phases.append(phase)

    for phase in simulation_phases:
        elapsed = 0.0
        while elapsed < phase.duration_s - 1e-9:
            # Only the long loiter segment is near trim for hours.  Climb,
            # descent, approach and flare retain a shorter step because their
            # speed-gamma coupling is genuinely transient.
            phase_step_limit_s = (
                controller.max_integration_step_s
                if phase.name == "endurance_loiter"
                else min(controller.max_integration_step_s, 20.0)
            )
            if phase.unpowered_glide:
                phase_step_limit_s = min(phase_step_limit_s, 5.0)
            dt = min(phase.time_step_s, phase_step_limit_s, phase.duration_s - elapsed)
            active_events = [
                event for event in mission.disturbances
                if event.phase_name == phase.name
                and elapsed < event.start_time_s + event.duration_s + event.recovery_window_s
                and elapsed + dt > event.start_time_s
            ]
            if active_events:
                phase_step_limit_s = min(
                    phase_step_limit_s,
                    min(event.response_step_s for event in active_events),
                )
            dt = min(phase.time_step_s, phase_step_limit_s, phase.duration_s - elapsed)
            # Land exactly on external-input edges so a coarse cruise step
            # cannot shift or shorten a prescribed perturbation or fault.
            boundaries = [
                boundary
                for event in mission.disturbances if event.phase_name == phase.name
                for boundary in (event.start_time_s, event.start_time_s + event.duration_s,
                                 event.start_time_s + event.duration_s + event.recovery_window_s)
                if elapsed + 1e-8 < boundary < elapsed + dt - 1e-8
            ]
            boundaries += [
                boundary
                for event in mission.fault_events if event.phase_name == phase.name
                for boundary in (event.start_phase_elapsed_s,
                                 event.start_phase_elapsed_s + event.duration_s)
                if elapsed + 1e-8 < boundary < elapsed + dt - 1e-8
            ]
            if boundaries:
                dt = min(boundaries) - elapsed
            disturbance_drag_n = sum(
                event.additional_drag_n
                for event in mission.disturbances
                if event.phase_name == phase.name
                and event.start_time_s <= elapsed < event.start_time_s + event.duration_s
            )
            active_headwind_mps = sum(
                event.headwind_step_mps
                + event.headwind_gust_amplitude_mps
                * np.sin(
                    2.0
                    * np.pi
                    * max(0.0, elapsed - event.start_time_s)
                    / max(event.headwind_gust_period_s, 1e-9)
                )
                for event in mission.disturbances
                if event.phase_name == phase.name
                and event.start_time_s <= elapsed < event.start_time_s + event.duration_s
            )
            active_phase = phase
            if phase.unpowered_glide:
                active_phase = PhaseCommand(
                    name=phase.name,
                    duration_s=phase.duration_s,
                    altitude_target_m=phase.altitude_target_m,
                    speed_target_mps=phase.speed_target_mps,
                    engine_power_kw=0.0,
                    motor_power_kw=0.0,
                    max_flight_path_angle_deg=phase.max_flight_path_angle_deg,
                    time_step_s=phase.time_step_s,
                    unpowered_glide=True,
                )
            fault_active = any(
                event.phase_name == phase.name
                and event.start_phase_elapsed_s <= elapsed
                < event.start_phase_elapsed_s + event.duration_s
                for event in mission.fault_events
            )
            derivative = _step(
                state,
                active_phase,
                dt,
                phase.duration_s - elapsed,
                aircraft,
                controller,
                dispatch_policy,
                active_powertrain,
                disturbance_drag_n,
                active_headwind_mps,
                phase_elapsed_s=elapsed,
                mission_elapsed_s=time_s,
            )
            state = derivative.next_state
            time_s += dt
            elapsed += dt
            if phase.name == "endurance_loiter":
                loiter_s += dt
            q_radps = (derivative.theta_rad - previous_theta) / dt
            previous_theta = derivative.theta_rad
            rows.append(
                {
                    "time_s": time_s,
                    "phase_elapsed_s": elapsed,
                    "phase": phase.name,
                    "fault_active": fault_active,
                    "fuel_flow_kgps": derivative.fuel_flow_kgps,
                    "battery_discharge_kw": max(derivative.motor_power_kw, 0.0),
                    "battery_charge_kw": max(-derivative.motor_power_kw, 0.0),
                    "soc": state.soc,
                    "h_m": state.altitude_m,
                    "v_mps": state.speed_mps,
                    "p_engine_kw": derivative.engine_power_kw,
                    "p_motor_kw": derivative.motor_power_kw,
                    "p_motor_target_kw": derivative.motor_target_kw,
                    "p_engine_target_kw": derivative.engine_target_kw,
                    "p_propulsor_input_kw": derivative.propulsor_input_power_kw,
                    "p_propulsion_request_kw": derivative.propulsion_power_request_kw,
                    "power_mode": derivative.power_mode,
                    "propulsion_mode": derivative.power_mode,
                    "soc_target": derivative.soc_target,
                    "e_h_m": phase.altitude_target_m - state.altitude_m,
                    "e_v_mps": phase.speed_target_mps - state.speed_mps,
                    "e_theta_rad": derivative.gamma_command_rad - state.gamma_rad,
                    "q_radps": q_radps,
                    "alpha_rad": derivative.alpha_rad,
                    "load_factor": derivative.load_factor,
                    "battery_current_a": derivative.battery_current_a,
                    "battery_terminal_power_kw": derivative.battery_terminal_power_kw,
                    "battery_chemical_power_kw": derivative.battery_chemical_power_kw,
                    "battery_temperature_c": np.nan,
                    "range_m": state.range_m,
                    "mass_kg": derivative.mass_kg,
                    "thrust_n": derivative.thrust_n,
                    "drag_n": derivative.drag_n,
                    "disturbance_drag_n": disturbance_drag_n,
                    "equivalent_headwind_mps": active_headwind_mps,
                    "cd0": derivative.cd0,
                    "lift_n": derivative.lift_n,
                    "lift_shortfall_n": derivative.lift_shortfall_n,
                    "power_shortfall_kw": derivative.power_shortfall_kw,
                    "gamma_command_rad": derivative.gamma_command_rad,
                    "gamma_rad": state.gamma_rad,
                    "alpha_command_rad": derivative.alpha_command_rad,
                }
            )
            if aircraft.initial_fuel_kg - state.fuel_used_kg <= 0.0:
                termination_reason = "fuel_depleted"
                break
            if state.soc <= aircraft.soc_safe:
                termination_reason = "battery_soc_safe_limit"
                break
            # A short transient shortage is now part of the physical closed
            # loop.  Terminate only when it has driven the vehicle to the
            # minimum numerical/safety speed, not merely because a command was
            # temporarily infeasible.
            if derivative.speed_floor_limited and derivative.power_shortfall_kw > 0.5:
                termination_reason = (
                    "unpowered_glide_envelope_violation"
                    if phase.unpowered_glide
                    else "propulsion_power_shortfall"
                )
                break
            if derivative.speed_floor_limited and derivative.lift_shortfall_n > 0.01 * derivative.mass_kg * G:
                termination_reason = "lift_shortfall"
                break
        if termination_reason != "mission_complete":
            break

    if not rows:
        raise RuntimeError("Mission simulation produced no time-history rows.")
    # The integration loop records states at the end of each step.  Add a
    # compatible t=0 row so duration and trapezoidal objectives use the same
    # mission clock as the dynamics, rather than losing the first time step.
    initial_row = dict(rows[0])
    initial_phase = mission.phases[0]
    initial_row.update(
        {
            "time_s": 0.0,
            "phase_elapsed_s": 0.0,
            "h_m": mission.initial_altitude_m,
            "v_mps": mission.initial_speed_mps,
            "soc": aircraft.initial_soc if mission.initial_soc is None else mission.initial_soc,
            "range_m": 0.0,
            "mass_kg": aircraft.initial_mass_kg,
            "e_h_m": initial_phase.altitude_target_m - mission.initial_altitude_m,
            "e_v_mps": initial_phase.speed_target_mps - mission.initial_speed_mps,
            "q_radps": 0.0,
            "fault_active": False,
        }
    )
    profile = pd.DataFrame([initial_row, *rows])
    phase_offsets = _phase_entry_times(profile)
    profile["disturbance_delta_h_m"] = np.nan
    profile["disturbance_delta_v_mps"] = np.nan
    for event in mission.disturbances:
        if event.phase_name not in phase_offsets:
            continue
        phase_mask = profile["phase"] == event.phase_name
        phase_times = profile.loc[phase_mask, "time_s"]
        event_start = phase_offsets[event.phase_name] + event.start_time_s
        baseline = profile.loc[
            phase_mask
            & (phase_times >= event_start - 30.0)
            & (phase_times < event_start)
        ]
        if baseline.empty:
            baseline = profile.loc[phase_mask & (phase_times < event_start)].tail(1)
        if baseline.empty:
            continue
        profile.loc[phase_mask, "disturbance_delta_h_m"] = (
            profile.loc[phase_mask, "h_m"] - float(baseline["h_m"].mean())
        )
        profile.loc[phase_mask, "disturbance_delta_v_mps"] = (
            profile.loc[phase_mask, "v_mps"] - float(baseline["v_mps"].mean())
        )
    # The evaluator accepts optional current/temperature only when present and finite.
    profile = profile.drop(columns="battery_temperature_c")
    terminal_error = abs(mission.phases[-1].altitude_target_m - state.altitude_m)
    response_metrics = _response_metrics_from_tracking(profile, mission)
    return MissionSimulation(
        profile=profile,
        response_metrics=response_metrics,
        terminal_fuel_kg=max(0.0, aircraft.initial_fuel_kg - state.fuel_used_kg),
        range_m=state.range_m,
        loiter_s=loiter_s,
        terminal_position_error_m=terminal_error,
        termination_reason=termination_reason,
    )


def _first_continuous_band_time(
    time_s: np.ndarray,
    in_band: np.ndarray,
    dwell_s: float,
) -> float | None:
    if len(time_s) < 2:
        return None
    start: int | None = None
    for index, value in enumerate(in_band):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(in_band) - 1):
            end = index if value and index == len(in_band) - 1 else index - 1
            if end > start and time_s[end] - time_s[start] >= dwell_s:
                return float(time_s[start])
            start = None
    return None


def _response_metrics_from_tracking(profile: pd.DataFrame, mission: MissionDefinition) -> dict[str, float]:
    """Measure settling and disturbance recovery from actual state histories."""
    settling_s = 0.0
    switching_s = 0.0
    phase_offsets = _phase_entry_times(profile)
    for phase in mission.phases:
        phase_data = profile.loc[profile["phase"] == phase.name]
        if phase_data.empty:
            continue
        if phase.name != "endurance_loiter":
            altitude_tolerance_m = max(10.0, 0.01 * max(abs(phase.altitude_target_m), 100.0))
            speed_tolerance_mps = 0.8
            within_band = (
                phase_data["e_h_m"].abs().to_numpy() <= altitude_tolerance_m
            ) & (phase_data["e_v_mps"].abs().to_numpy() <= speed_tolerance_mps)
            times = phase_data["time_s"].to_numpy(float)
            settled_at = _first_continuous_band_time(times, within_band, 30.0)
            settling_s += (
                settled_at - float(times[0])
                if settled_at is not None
                else float(times[-1] - times[0])
            )
    recovery_times: list[float] = []
    recovery_successes: list[bool] = []
    peak_speed_deviations: list[float] = []
    peak_altitude_deviations: list[float] = []
    for event in mission.disturbances:
        if event.phase_name not in phase_offsets:
            recovery_successes.append(False)
            recovery_times.append(event.recovery_window_s)
            continue
        phase_data = profile.loc[profile["phase"] == event.phase_name]
        event_start = phase_offsets[event.phase_name] + event.start_time_s
        event_end = event_start + event.duration_s
        baseline_window = phase_data.loc[
            (phase_data["time_s"] >= event_start - 30.0)
            & (phase_data["time_s"] < event_start)
        ]
        if baseline_window.empty:
            baseline_window = phase_data.loc[phase_data["time_s"] < event_start].tail(1)
        if baseline_window.empty:
            recovery_successes.append(False)
            recovery_times.append(event.recovery_window_s)
            continue
        baseline_h = float(baseline_window["h_m"].mean())
        baseline_v = float(baseline_window["v_mps"].mean())
        window = phase_data.loc[
            (phase_data["time_s"] >= event_end)
            & (phase_data["time_s"] <= event_end + event.recovery_window_s)
        ]
        full_response = phase_data.loc[
            (phase_data["time_s"] >= event_start)
            & (phase_data["time_s"] <= event_end + event.recovery_window_s)
        ]
        peak_speed_deviations.append(
            float((full_response["v_mps"] - baseline_v).abs().max()) if not full_response.empty else 0.0
        )
        peak_altitude_deviations.append(
            float((full_response["h_m"] - baseline_h).abs().max()) if not full_response.empty else 0.0
        )
        tolerance_h = max(10.0, 0.01 * max(abs(
            next(phase.altitude_target_m for phase in mission.phases if phase.name == event.phase_name)
        ), 100.0))
        tolerance_v = 0.8
        in_band = (
            (np.abs(window["h_m"].to_numpy(float) - baseline_h) <= tolerance_h)
            & (np.abs(window["v_mps"].to_numpy(float) - baseline_v) <= tolerance_v)
        )
        times = window["time_s"].to_numpy(float)
        recovered_at = _first_continuous_band_time(times, in_band, event.recovery_dwell_s)
        recovery_successes.append(recovered_at is not None)
        recovery_times.append(
            recovered_at - event_end
            if recovered_at is not None
            else event.recovery_window_s
        )
        event_modes = window["power_mode"].astype(str).to_numpy()
        mode_changes = np.flatnonzero(event_modes[1:] != event_modes[:-1]) + 1
        event_switch_responses: list[float] = []
        event_times = window["time_s"].to_numpy(float)
        engine_actual = window["p_engine_kw"].to_numpy(float)
        engine_target = window["p_engine_target_kw"].to_numpy(float)
        motor_actual = window["p_motor_kw"].to_numpy(float)
        motor_target = window["p_motor_target_kw"].to_numpy(float)
        for change_index in mode_changes:
            next_change = next(
                (int(index) for index in mode_changes if index > change_index),
                len(event_times) - 1,
            )
            if event_times[next_change] - event_times[change_index] < 10.0:
                continue
            engine_slice = slice(change_index, next_change + 1)
            settled = (
                (
                    np.abs(engine_actual[engine_slice] - engine_target[engine_slice])
                    <= np.maximum(1.0, 0.02 * np.abs(engine_target[engine_slice]))
                )
                & (
                    np.abs(motor_actual[engine_slice] - motor_target[engine_slice])
                    <= np.maximum(1.0, 0.02 * np.abs(motor_target[engine_slice]))
                )
            )
            completion = _first_continuous_band_time(
                event_times[engine_slice] - event_times[change_index], settled, 10.0
            )
            event_switch_responses.append(
                completion
                if completion is not None
                else min(event.recovery_window_s, float(event_times[next_change] - event_times[change_index]))
            )
        switching_s = max(switching_s, max(event_switch_responses, default=0.0))
    settling_horizon_s = sum(
        phase.duration_s for phase in mission.phases if phase.name != "endurance_loiter"
    )
    recovery_horizon_s = max(
        (event.recovery_window_s for event in mission.disturbances),
        default=0.0,
    )
    recovery_success = all(recovery_successes) if recovery_successes else True
    failure_penalty_s = (
        1.1 * settling_horizon_s + 2.0 * recovery_horizon_s
        if not recovery_success
        else 0.0
    )
    return {
        "settling_time_s": float(settling_s),
        "recovery_time_s": float(max(recovery_times, default=0.0)),
        "recovery_success": float(recovery_success),
        "failure_penalty_s": float(failure_penalty_s),
        "peak_speed_deviation_mps": float(max(peak_speed_deviations, default=0.0)),
        "peak_altitude_deviation_m": float(max(peak_altitude_deviations, default=0.0)),
        "mode_switch_time_s": float(switching_s),
    }


def _phase_entry_times(profile: pd.DataFrame) -> dict[str, float]:
    """Recover actual phase boundaries from the emitted timeline."""
    entries: dict[str, float] = {}
    previous_phase: str | None = None
    previous_time_s = 0.0
    for row in profile.loc[:, ["time_s", "phase"]].itertuples(index=False):
        current_phase = str(row.phase)
        if current_phase != previous_phase and current_phase not in entries:
            entries[current_phase] = previous_time_s
        previous_phase = current_phase
        previous_time_s = float(row.time_s)
    return entries


@dataclass(frozen=True)
class _State:
    range_m: float
    altitude_m: float
    speed_mps: float
    gamma_rad: float
    fuel_used_kg: float
    soc: float
    alpha_rad: float
    powertrain_state: Any


@dataclass(frozen=True)
class _Step:
    next_state: _State
    engine_power_kw: float
    motor_power_kw: float
    propulsor_input_power_kw: float
    propulsion_power_request_kw: float
    engine_target_kw: float
    power_mode: str
    soc_target: float
    fuel_flow_kgps: float
    alpha_rad: float
    alpha_command_rad: float
    theta_rad: float
    gamma_command_rad: float
    load_factor: float
    battery_current_a: float
    mass_kg: float
    thrust_n: float
    drag_n: float
    cd0: float
    lift_n: float
    lift_shortfall_n: float
    power_shortfall_kw: float
    speed_floor_limited: bool
    motor_target_kw: float = 0.0
    battery_terminal_power_kw: float = 0.0
    battery_chemical_power_kw: float = 0.0


def _step(
    state: _State,
    command: PhaseCommand,
    dt: float,
    remaining_phase_s: float,
    aircraft: AircraftParameters,
    controller: FlightControlParameters,
    dispatch_policy: HybridDispatchPolicy | None,
    powertrain: HybridPowertrainModel,
    disturbance_drag_n: float = 0.0,
    headwind_mps: float = 0.0,
    phase_elapsed_s: float = 0.0,
    mission_elapsed_s: float = 0.0,
) -> _Step:
    rho = isa_density(state.altitude_m)
    mass = max(aircraft.empty_mass_kg, aircraft.initial_mass_kg - state.fuel_used_kg)
    gamma_max = np.deg2rad(command.max_flight_path_angle_deg)
    # Guidance produces a bounded terminal-state request.  Actual gamma and
    # speed are integrated below from available thrust and aerodynamic force.
    hdot_command = np.clip(
        1.2 * (command.altitude_target_m - state.altitude_m) / max(remaining_phase_s, dt),
        -state.speed_mps * np.sin(gamma_max),
        state.speed_mps * np.sin(gamma_max),
    )
    gamma_command = float(np.arcsin(np.clip(hdot_command / max(state.speed_mps, controller.speed_floor_mps), -0.98, 0.98)))
    if command.unpowered_glide:
        speed_error = command.speed_target_mps - state.speed_mps
        gamma_command = float(
            np.clip(
                gamma_command
                - controller.glide_speed_path_angle_gain_rad_per_mps * speed_error,
                -gamma_max,
                gamma_max,
            )
    )
    speed_tracking_time_constant_s = max(
        controller.speed_tracking_time_constant_s, dt
    )
    speed_dot_command = np.clip(
        1.2
        * (command.speed_target_mps - state.speed_mps)
        / speed_tracking_time_constant_s,
        -controller.max_longitudinal_accel_mps2,
        controller.max_longitudinal_accel_mps2,
    )
    gamma_error = gamma_command - state.gamma_rad
    # Exact zero-order-hold form of a first-order gamma loop.  Unlike a raw
    # Euler gain, it remains stable when the long loiter phase is advanced with
    # a larger multi-rate integration step during outer-loop optimization.
    gamma_rate_command = float(np.clip(
        (1.0 - np.exp(-controller.gamma_tracking_gain * dt)) * gamma_error / max(dt, 1e-9),
        -controller.max_gamma_rate_radps,
        controller.max_gamma_rate_radps,
    ))
    speed_forces = max(state.speed_mps + headwind_mps, controller.speed_floor_mps)
    q_nominal = 0.5 * rho * speed_forces**2
    lift_command = mass * (G * np.cos(state.gamma_rad) + speed_forces * gamma_rate_command)
    cl_command = lift_command / max(q_nominal * aircraft.wing_area_m2, 1.0)
    alpha_command = float(np.clip(
        cl_command / max(aircraft.cl_alpha_per_rad, 1e-6),
        controller.alpha_min_rad,
        controller.alpha_max_rad,
    ))
    alpha_lagged = first_order_transfer_step(
        state.alpha_rad,
        alpha_command,
        dt,
        max(controller.alpha_time_constant_s, 1e-6),
    )
    maximum_alpha_change = controller.alpha_rate_limit_radps * dt
    alpha = float(np.clip(alpha_lagged, state.alpha_rad - maximum_alpha_change, state.alpha_rad + maximum_alpha_change))
    cl_command_limited = float(np.clip(aircraft.cl_alpha_per_rad * alpha_command, aircraft.cl_min, aircraft.cl_max))
    cd0 = parasite_drag_coefficient(aircraft, rho, speed_forces)
    cd_command = cd0 + cl_command_limited**2 / (np.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    drag_command = q_nominal * aircraft.wing_area_m2 * cd_command
    thrust_request = max(0.0, drag_command + disturbance_drag_n + mass * speed_dot_command + mass * G * np.sin(gamma_command))
    induced_request = induced_velocity_from_thrust(thrust_request, speed_forces, rho, aircraft.prop_disk_area_m2)
    required_propulsion_kw = thrust_request * (speed_forces + aircraft.induced_loss_factor * induced_request) / max(aircraft.propulsive_efficiency, 1e-6) / 1000.0
    engine_cap_kw = (
        0.0
        if command.unpowered_glide
        else min(max(command.engine_power_kw, 0.0), aircraft.engine_electrical_power_max_kw)
    )
    phase_motor_max_kw = (
        0.0
        if command.unpowered_glide or command.engine_shutdown or command.motor_disabled
        else min(max(command.motor_power_kw, 0.0), max(aircraft.motor_power_max_kw, 0.0))
    )
    if dispatch_policy is None:
        motor_command_kw = 0.0 if command.motor_disabled else command.motor_power_kw
        engine_target_kw = float(np.clip(required_propulsion_kw - motor_command_kw, 0.0, engine_cap_kw))
        power_mode = "scheduled"
        soc_target = float("nan")
        active_engine_cap_kw = engine_cap_kw
        active_motor_min_kw = aircraft.motor_power_min_kw
        active_motor_max_kw = aircraft.motor_power_max_kw
        if command.motor_disabled:
            active_motor_min_kw = 0.0
            active_motor_max_kw = 0.0
    else:
        efficiency_curve = (
            powertrain.engine_efficiency_curve(state.powertrain_state, state.altitude_m, engine_cap_kw)
            if hasattr(powertrain, "engine_efficiency_curve")
            else (aircraft.engine_efficiency_power_fractions, aircraft.engine_efficiency_values)
        )
        dispatch = dispatch_policy.dispatch(
            DispatchObservation(
                phase_name=command.name,
                time_step_s=dt,
                soc=state.soc,
                altitude_error_m=command.altitude_target_m - state.altitude_m,
                speed_error_mps=command.speed_target_mps - state.speed_mps,
                required_propulsion_kw=required_propulsion_kw,
                engine_power_cap_kw=engine_cap_kw,
                motor_power_min_kw=(
                    0.0 if command.motor_disabled else aircraft.motor_power_min_kw
                ),
                motor_power_max_kw=phase_motor_max_kw,
                battery_capacity_mj=aircraft.battery_capacity_mj,
                battery_discharge_efficiency=aircraft.battery_discharge_efficiency,
                battery_charge_efficiency=aircraft.battery_charge_efficiency,
                battery_soc_min=max(0.20, aircraft.soc_safe),
                battery_soc_max=1.0,
                battery_soc_reserve=aircraft.soc_reserve,
                engine_power_rate_limit_kwps=aircraft.engine_power_rate_limit_kwps,
                engine_efficiency_power_fractions=efficiency_curve[0],
                engine_efficiency_values=efficiency_curve[1],
                phase_elapsed_s=phase_elapsed_s,
                mission_elapsed_s=mission_elapsed_s,
                required_motor_dc_kw=(
                    powertrain.motor_dc_demand(state.powertrain_state, required_propulsion_kw)
                    if hasattr(powertrain, "motor_dc_demand") else None
                ),
            )
        )
        active_engine_cap_kw = min(
            engine_cap_kw,
            engine_cap_kw if dispatch.engine_power_cap_kw is None else max(dispatch.engine_power_cap_kw, 0.0),
        )
        active_motor_min_kw = aircraft.motor_power_min_kw if dispatch.motor_power_min_kw is None else max(aircraft.motor_power_min_kw, dispatch.motor_power_min_kw)
        active_motor_max_kw = phase_motor_max_kw if dispatch.motor_power_max_kw is None else min(phase_motor_max_kw, dispatch.motor_power_max_kw)
        if command.unpowered_glide:
            active_engine_cap_kw = 0.0
            active_motor_min_kw = 0.0
            active_motor_max_kw = 0.0
            engine_target_kw = 0.0
        elif command.motor_disabled:
            active_motor_min_kw = 0.0
            active_motor_max_kw = 0.0
        if active_motor_min_kw > active_motor_max_kw:
            active_motor_min_kw = active_motor_max_kw = 0.0
        engine_target_kw = float(np.clip(dispatch.engine_target_kw, 0.0, active_engine_cap_kw))
        power_mode = dispatch.mode
        soc_target = dispatch.soc_target
    if command.unpowered_glide:
        active_engine_cap_kw = 0.0
        active_motor_min_kw = 0.0
        active_motor_max_kw = 0.0
        engine_target_kw = 0.0
        power_mode = "unpowered_glide"
    elif command.motor_disabled:
        active_motor_min_kw = 0.0
        active_motor_max_kw = 0.0
    elif command.engine_shutdown:
        active_engine_cap_kw = aircraft.engine_electrical_power_max_kw
        active_motor_min_kw = 0.0
        active_motor_max_kw = 0.0
        engine_target_kw = 0.0
        power_mode = "engine_shutdown_transition"
    powertrain_output = powertrain.step(
        state.powertrain_state,
        HybridPowertrainRequest(
            command.name,
            engine_target_kw,
            active_engine_cap_kw,
            0.0 if command.engine_shutdown else required_propulsion_kw,
            active_motor_min_kw,
            active_motor_max_kw,
            dispatch.battery_available_energy_fraction if dispatch_policy is not None else 1.0,
            altitude_m=state.altitude_m,
            engine_target_is_dc=dispatch.engine_target_is_dc if dispatch_policy is not None else False,
        ),
        dt,
        aircraft,
    )
    if dispatch_policy is not None and hasattr(dispatch_policy, "update_powertrain_feedback"):
        dispatch_policy.update_powertrain_feedback(powertrain_output)
    usable_propulsor_power_w = powertrain_output.propulsor_input_power_kw * 1000.0 * aircraft.propulsive_efficiency
    thrust, induced_velocity = thrust_from_power(
        usable_propulsor_power_w,
        speed_forces,
        rho,
        aircraft.prop_disk_area_m2,
        aircraft.induced_loss_factor,
    )
    # The present point-mass plant uses a single consistent free-stream dynamic
    # pressure in the normal-force controller and aerodynamic plant.  A future
    # propeller-wing interaction model may replace this with a validated local
    # slipstream field; mixing a slipstream-corrected plant with an uncorrected
    # controller creates an artificial normal-force overshoot.
    q_effective = q_nominal
    cl = float(np.clip(aircraft.cl_alpha_per_rad * alpha, aircraft.cl_min, aircraft.cl_max))
    cd = cd0 + cl**2 / (np.pi * aircraft.aspect_ratio * aircraft.oswald_efficiency)
    lift = q_effective * aircraft.wing_area_m2 * cl
    drag = q_effective * aircraft.wing_area_m2 * cd + disturbance_drag_n
    lift_shortfall_n = max(0.0, lift_command - lift)
    speed_dot_actual = float(np.clip(
        (thrust - drag - mass * G * np.sin(state.gamma_rad)) / mass,
        -controller.max_longitudinal_accel_mps2,
        controller.max_longitudinal_accel_mps2,
    ))
    gamma_dot_actual = float(np.clip(
        (lift - mass * G * np.cos(state.gamma_rad)) / max(mass * speed_forces, 1e-6),
        -controller.max_gamma_rate_radps,
        controller.max_gamma_rate_radps,
    ))
    raw_speed_next = state.speed_mps + speed_dot_actual * dt
    speed_floor_limited = raw_speed_next <= controller.speed_floor_mps
    speed_next = max(controller.speed_floor_mps, raw_speed_next)
    gamma_next = float(np.clip(state.gamma_rad + gamma_dot_actual * dt, -gamma_max, gamma_max))
    altitude_next = max(0.0, state.altitude_m + 0.5 * (state.speed_mps * np.sin(state.gamma_rad) + speed_next * np.sin(gamma_next)) * dt)
    range_next = state.range_m + 0.5 * (state.speed_mps * np.cos(state.gamma_rad) + speed_next * np.cos(gamma_next)) * dt
    fuel_next = state.fuel_used_kg + powertrain_output.fuel_flow_kgps * dt
    power_shortfall_kw = max(0.0, required_propulsion_kw - powertrain_output.propulsor_input_power_kw)
    theta = gamma_next + alpha
    return _Step(
        next_state=_State(
            range_next,
            altitude_next,
            speed_next,
            gamma_next,
            fuel_next,
            powertrain_output.soc,
            alpha,
            powertrain_output.state,
        ),
        engine_power_kw=powertrain_output.engine_power_kw,
        motor_power_kw=powertrain_output.motor_power_kw,
        propulsor_input_power_kw=powertrain_output.propulsor_input_power_kw,
        propulsion_power_request_kw=required_propulsion_kw,
        engine_target_kw=engine_target_kw,
        power_mode=power_mode,
        soc_target=soc_target,
        fuel_flow_kgps=powertrain_output.fuel_flow_kgps,
        alpha_rad=float(alpha),
        alpha_command_rad=alpha_command,
        theta_rad=float(theta),
        gamma_command_rad=gamma_command,
        load_factor=float(lift / max(mass * G, 1.0)),
        battery_current_a=powertrain_output.battery_current_a,
        mass_kg=float(mass),
        thrust_n=float(thrust),
        drag_n=float(drag),
        cd0=float(cd0),
        lift_n=float(lift),
        lift_shortfall_n=float(lift_shortfall_n),
        power_shortfall_kw=float(power_shortfall_kw),
        speed_floor_limited=speed_floor_limited,
        motor_target_kw=float(required_propulsion_kw - engine_target_kw),
        battery_terminal_power_kw=float(
            getattr(
                powertrain_output,
                "battery_terminal_power_kw",
                powertrain_output.motor_power_kw,
            )
        ),
        battery_chemical_power_kw=float(
            getattr(
                powertrain_output,
                "battery_chemical_power_kw",
                getattr(
                    powertrain_output,
                    "battery_terminal_power_kw",
                    powertrain_output.motor_power_kw,
                ),
            )
        ),
    )


def thrust_from_power(power_w: float, airspeed_mps: float, rho_kgpm3: float, disk_area_m2: float, kappa: float) -> tuple[float, float]:
    """Solve the momentum-theory power-thrust relation by bounded bisection."""
    if power_w <= 1.0:
        return 0.0, 0.0
    low, high = 0.0, max(10.0, 3.0 * power_w ** (2.0 / 3.0) * (rho_kgpm3 * disk_area_m2) ** (1.0 / 3.0))
    for _ in range(60):
        thrust = 0.5 * (low + high)
        vi = -0.5 * airspeed_mps + np.sqrt(0.25 * airspeed_mps**2 + thrust / (2.0 * rho_kgpm3 * disk_area_m2))
        required = thrust * (airspeed_mps + kappa * vi)
        if required > power_w:
            high = thrust
        else:
            low = thrust
    thrust = 0.5 * (low + high)
    vi = -0.5 * airspeed_mps + np.sqrt(0.25 * airspeed_mps**2 + thrust / (2.0 * rho_kgpm3 * disk_area_m2))
    return float(thrust), float(vi)


def engine_efficiency_at_power(engine_power_kw: float, aircraft: AircraftParameters) -> float:
    """Interpolate a declared engine-generator efficiency map.

    The default two-point map reproduces the former constant-efficiency model.
    More detailed maps can be supplied as fractions of maximum electrical power
    and are clipped to a conservative numerical range.
    """
    fractions = np.asarray(aircraft.engine_efficiency_power_fractions, dtype=float)
    efficiencies = np.asarray(aircraft.engine_efficiency_values, dtype=float)
    if len(fractions) != len(efficiencies) or len(fractions) < 2:
        raise ValueError("Engine efficiency map must contain matching arrays with at least two points.")
    if not np.all(np.diff(fractions) > 0.0):
        raise ValueError("Engine efficiency map fractions must be strictly increasing.")
    # Preserve the original constant-efficiency contract.  The default map is
    # only a placeholder; callers that change engine_generator_efficiency
    # without supplying a map must still obtain that new scalar efficiency.
    if np.array_equal(fractions, np.array([0.0, 1.0])) and np.array_equal(efficiencies, np.array([0.30, 0.30])):
        return float(np.clip(aircraft.engine_generator_efficiency, 0.05, 0.60))
    load_fraction = engine_power_kw / max(aircraft.engine_electrical_power_max_kw, 1e-9)
    efficiency = float(np.interp(load_fraction, fractions, efficiencies))
    return float(np.clip(efficiency, 0.05, 0.60))


def induced_velocity_from_thrust(thrust_n: float, airspeed_mps: float, rho_kgpm3: float, disk_area_m2: float) -> float:
    return float(-0.5 * airspeed_mps + np.sqrt(0.25 * airspeed_mps**2 + thrust_n / (2.0 * rho_kgpm3 * disk_area_m2)))


def turbulent_skin_friction_coefficient(reynolds: float) -> float:
    """Return a low-speed turbulent flat-plate skin-friction coefficient."""
    re = max(float(reynolds), 1.0e5)
    # The commonly quoted ``-1700/Re`` correction reverses the physical trend
    # near the one-million Reynolds-number range used by this mission model.
    # Keep the monotonic turbulent correlation until a transition-resolving
    # component model is available.
    cf = 0.455 / max(np.log10(re) ** 2.58, 1e-6)
    return float(np.clip(cf, 0.0010, 0.0080))


def parasite_drag_coefficient(aircraft: AircraftParameters, rho_kgpm3: float, airspeed_mps: float) -> float:
    """Update calibrated component-build-up ``CD0`` for current Reynolds number.

    Aircraft without a dimensional drag reference retain constant ``CD0``.
    This keeps generic reference models backward compatible while dimensional
    PTDL candidates receive an altitude/speed correction during the mission.
    """
    if aircraft.parasite_drag_reference_reynolds <= 0.0 or aircraft.reference_chord_m <= 0.0:
        return float(aircraft.cd0)
    reynolds = rho_kgpm3 * max(airspeed_mps, 1.0) * aircraft.reference_chord_m / AIR_DYNAMIC_VISCOSITY_KG_PER_M_S
    ratio = turbulent_skin_friction_coefficient(reynolds) / turbulent_skin_friction_coefficient(aircraft.parasite_drag_reference_reynolds)
    return float(np.clip(aircraft.cd0 * ratio, 0.55 * aircraft.cd0, 1.80 * aircraft.cd0))


def isa_density(altitude_m: float) -> float:
    """Troposphere/lower-stratosphere ISA density suitable for the reference model."""
    h = max(0.0, altitude_m)
    if h <= 11000.0:
        temperature = 288.15 - 0.0065 * h
        pressure = 101325.0 * (temperature / 288.15) ** 5.25588
    else:
        temperature = 216.65
        pressure = 22632.06 * np.exp(-G * (h - 11000.0) / (287.05287 * temperature))
    return float(pressure / (287.05287 * temperature))
