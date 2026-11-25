"""Mission-scale adapters for the Hybrid_System component models.

The source components use SI electrical/mechanical units internally. The PTDL
plant contract uses kW, kg/s, and fractional SOC; conversions happen here.
The PMSM is evaluated at a steady operating point because its 1 ms dynamics
cannot be integrated over an 80 h optimization mission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, exp, isfinite, pi
from pathlib import Path
from typing import Any

from 动力模型.battery_rint.battery_rint import BatteryRint
from 动力模型.engine_map_genset.engine_map import EngineMap
from 动力模型.generator_gearbox.generator_gearbox import GeneratorGearbox
from 动力模型.motor_pmsm.pmsm_bus_limited import PMSMBusLimited
from 动力模型.full_system.init_hybrid_system import init_hybrid_system


POWERTRAIN_VERSION = "1005-20261023"


@dataclass(frozen=True)
class HybridSystemAdapterConfig:
    """Declared component operating point and numerical settings.

    Power ratings are read from the component initializers, not scaled to the
    PTDL aircraft. ``engine_speed_rpm`` and ``motor_speed_rpm`` are component
    shaft operating points, not flight-condition variables.
    """

    engine_speed_rpm: float = 5000.0
    motor_speed_rpm: float = 2100.0
    motor_branches: int = 2
    engine_map_path: str | Path | None = None
    engine_fuel_flow_scale: float = 1.0
    engine_power_rate_limit_kwps: float = 10.0
    battery_resistance_multiplier: float = 1.0
    inverter_efficiency: float = 0.95
    battery_soc_min: float = 0.20
    battery_soc_max: float = 1.00
    engine_time_constant_s: float = 0.0
    engine_command_delay_s: float = 0.0
    engine_efficiency_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if any(not isfinite(v) or v < 0 for v in (
            self.engine_time_constant_s, self.engine_command_delay_s
        )):
            raise ValueError("Engine lag and delay must be finite and nonnegative")
        if not isfinite(self.engine_speed_rpm) or self.engine_speed_rpm <= 0:
            raise ValueError("engine_speed_rpm must be positive and finite")
        if not isfinite(self.motor_speed_rpm) or self.motor_speed_rpm <= 0:
            raise ValueError("motor_speed_rpm must be positive and finite")
        if self.motor_branches < 1:
            raise ValueError("motor_branches must be positive")
        if not isfinite(self.engine_fuel_flow_scale) or self.engine_fuel_flow_scale <= 0:
            raise ValueError("engine_fuel_flow_scale must be positive and finite")
        if not isfinite(self.engine_efficiency_multiplier) or self.engine_efficiency_multiplier <= 0:
            raise ValueError("engine_efficiency_multiplier must be positive and finite")
        if not isfinite(self.engine_power_rate_limit_kwps) or self.engine_power_rate_limit_kwps <= 0:
            raise ValueError("engine_power_rate_limit_kwps must be positive and finite")
        if not isfinite(self.battery_resistance_multiplier) or self.battery_resistance_multiplier <= 0:
            raise ValueError("battery_resistance_multiplier must be positive and finite")
        if not 0 < self.inverter_efficiency <= 1:
            raise ValueError("inverter_efficiency must be in (0, 1]")
        if not 0 <= self.battery_soc_min < self.battery_soc_max <= 1:
            raise ValueError("battery SOC bounds must satisfy 0 <= min < max <= 1")


@dataclass
class HybridSystemPowertrainState:
    engine_map: EngineMap
    generator: GeneratorGearbox
    battery: BatteryRint
    motor: PMSMBusLimited
    engine_rating_scale: float
    battery_parallel_equivalent: float
    motor_shaft_rating_kw: float
    motor_branches: int
    engine_power_max_kw: float
    engine_efficiency_power_fractions: tuple[float, ...]
    engine_efficiency_values: tuple[float, ...]
    engine_power_kw: float = 0.0
    nominal_battery_parameters: dict[str, Any] = field(default_factory=dict)
    battery_available_fraction: float = 1.0
    battery_isolated_charge_ah: float = 0.0
    battery_reconnection_count: int = 0
    engine_elapsed_s: float = 0.0
    engine_command_history: list[tuple[float, float]] = field(default_factory=list)
    engine_map_altitude_clamped_steps: int = 0


@dataclass(frozen=True)
class HybridSystemPowertrainOutput:
    state: HybridSystemPowertrainState
    engine_power_kw: float
    motor_power_kw: float
    propulsor_input_power_kw: float
    fuel_flow_kgps: float
    soc: float
    battery_current_a: float
    battery_terminal_power_kw: float
    battery_chemical_power_kw: float
    motor_dc_power_kw: float = 0.0
    unabsorbed_bus_power_kw: float = 0.0
    bus_power_deficit_kw: float = 0.0


class HybridSystemPowertrain:
    """Adapt the supplied engine, generator, Rint battery, and PMSM models.

    Source ratings anchor each module's efficiency and operating equations.
    Aggregate engine, motor, and battery ratings are scaled from the PTDL
    aircraft sizing estimates passed to ``initial_state``. This is an
    equivalent sizing model, not a claim that the prototype component hardware
    has these aggregate ratings. Battery positive power means discharge.
    """

    def __init__(self, config: HybridSystemAdapterConfig = HybridSystemAdapterConfig()) -> None:
        self.config = config
        source = init_hybrid_system()
        self.generator_parameters = dict(source["generator"])
        self.generator_parameters["GEN_N_engine_design_rpm"] = config.engine_speed_rpm
        self.generator_parameters["GEN_GearRatio"] = (
            config.engine_speed_rpm / self.generator_parameters["GEN_N_design_rpm"]
        )
        self.battery_parameters = dict(source["battery"])
        self.battery_parameters["BAT_SOC_min"] = config.battery_soc_min
        self.battery_parameters["BAT_SOC_max"] = config.battery_soc_max
        for key in ("BAT_Rdis_pack_Ohm", "BAT_Rchg_pack_Ohm"):
            self.battery_parameters[key] = tuple(
                value * config.battery_resistance_multiplier
                for value in self.battery_parameters[key]
            )
        self.motor_parameters = dict(source["motor"])
        self.ems_parameters = dict(source["ems"])
        self.ems_parameters["PE_Eta_inverter"] = config.inverter_efficiency
        self.ems_parameters["PE_Eta_rectifier"] = (
            0.96 / self.generator_parameters["GEN_Eta"]
        )
        self.rectifier_efficiency = float(self.ems_parameters["PE_Eta_rectifier"])
        self.dcdc_efficiency = float(self.ems_parameters["PE_Eta_dcdc"])
        self.generator_rated_ac_kw = self.generator_parameters["GEN_P_elec_design_W"] / 1000.0
        self.motor_rated_mechanical_kw = self.motor_parameters["MOT_P_rated_W"] / 1000.0
        self.engine_map = EngineMap(config.engine_map_path) if config.engine_map_path else EngineMap()
        if not self.engine_map.rpm[0] <= config.engine_speed_rpm <= self.engine_map.rpm[-1]:
            raise ValueError("engine_speed_rpm must be inside the engine map range [2800, 6100]")
        if config.motor_speed_rpm > self.motor_parameters["MOT_N_limit_rpm"]:
            raise ValueError("motor_speed_rpm exceeds the supplied motor speed limit")
        self._efficiency_cache: dict[tuple[float, float, float], tuple[tuple, tuple]] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        p = self.battery_parameters
        motor = self.motor_parameters
        return {
            "model": "1005 engine-generator / 81s3p Rint / EMRAX 228 MV LC",
            "source_version": POWERTRAIN_VERSION,
            "fuel_model": "altitude-speed-throttle map; direct supplied fuel flow",
            "engine_map_path": str(self.engine_map.csv_path),
            "engine_map_convergence_status": self.engine_map.convergence_status,
            "engine_map_altitude_range_m": list(self.engine_map.altitude_m[::2]),
            "engine_map_speed_range_rpm": [self.engine_map.rpm[0], self.engine_map.rpm[-1]],
            "altitude_outside_map_policy": "clamp to nearest altitude boundary and count steps",
            "low_load_policy": "below 30 degree throttle, retain the boundary fuel flow; zero fuel only when stopped",
            "engine_map_converged_points": self.engine_map.converged_count,
            "engine_map_total_points": len(self.engine_map._rows),
            "engine_speed_rpm": self.config.engine_speed_rpm,
            "engine_fuel_flow_scale_override": self.config.engine_fuel_flow_scale,
            "engine_efficiency_multiplier": self.config.engine_efficiency_multiplier,
            "engine_power_rate_limit_kwps": self.config.engine_power_rate_limit_kwps,
            "engine_time_constant_s": self.config.engine_time_constant_s,
            "engine_command_delay_s": self.config.engine_command_delay_s,
            "generator_rated_ac_kw": self.generator_rated_ac_kw,
            "engine_rated_mechanical_kw": self.generator_parameters["GEN_P_engine_design_W"] / 1000.0,
            "generator_design_speed_rpm": self.generator_parameters["GEN_N_design_rpm"],
            "generator_dc_rating_kw": self.generator_rated_ac_kw * self.rectifier_efficiency,
            "motor_branches": self.config.motor_branches,
            "motor_speed_rpm": self.config.motor_speed_rpm,
            "motor_rated_mechanical_kw_per_branch": self.motor_rated_mechanical_kw,
            "motor_product": motor["MOT_ProductName"],
            "motor_dc_input_max_kw_per_branch": motor["MOT_P_input_max_W"] / 1000.0,
            "battery_nominal_energy_wh": p["BAT_Energy_nominal_Wh"],
            "battery_capacity_ah": p["BAT_Capacity_Ah"],
            "battery_series_cells": p["BAT_Ns"],
            "battery_parallel_cells": p["BAT_Np"],
            "battery_nominal_voltage_v": p["BAT_Vnom_pack_V"],
            "battery_resistance_multiplier": self.config.battery_resistance_multiplier,
            "battery_soc_min": self.config.battery_soc_min,
            "battery_soc_max": self.config.battery_soc_max,
            "motor_solver": "steady_operating_point",
            "warning": (
                "Aggregate ratings follow PTDL preliminary sizing assumptions; "
                "this is an equivalent model, not validated hardware."
            ),
        }

    def initial_state(self, aircraft: Any, initial_soc: float) -> HybridSystemPowertrainState:
        if not self.config.battery_soc_min <= initial_soc <= self.config.battery_soc_max:
            raise ValueError("initial_soc is outside the component battery SOC limits")
        battery_parameters = dict(self.battery_parameters)
        battery_parameters["BAT_SOC_init"] = float(initial_soc)
        nominal_battery_wh = float(battery_parameters["BAT_Energy_nominal_Wh"])
        target_battery_wh = max(float(aircraft.battery_capacity_mj), 0.0) * 1e6 / 3600.0
        battery_parallel_equivalent = max(target_battery_wh / nominal_battery_wh, 1e-6)
        battery_parameters["BAT_Capacity_Ah"] *= battery_parallel_equivalent
        battery_parameters["BAT_Energy_nominal_Wh"] *= battery_parallel_equivalent
        for key in ("BAT_Rdis_pack_Ohm", "BAT_Rchg_pack_Ohm"):
            battery_parameters[key] = tuple(
                resistance / battery_parallel_equivalent
                for resistance in battery_parameters[key]
            )

        battery = BatteryRint(battery_parameters)
        generator = GeneratorGearbox(self.generator_parameters)
        motor_parameters = dict(self.motor_parameters)
        # The supplied Hybrid_System path is series-hybrid: the traction motor
        # must carry generator power as well as the battery-assist allowance.
        # `motor_power_max_kw` is the battery-assist capability in the PTDL
        # mission contract, not the installed traction-motor nameplate rating.
        engine_bus_rating_kw = max(
            float(aircraft.engine_electrical_power_max_kw), 0.0
        )
        battery_assist_rating_kw = max(float(aircraft.motor_power_max_kw), 0.0)
        motor_shaft_rating_kw = engine_bus_rating_kw + battery_assist_rating_kw
        motor_branches = max(
            self.config.motor_branches,
            int(ceil(motor_shaft_rating_kw / self.motor_rated_mechanical_kw)),
        )
        motor = PMSMBusLimited(motor_parameters, inverter_efficiency=self.config.inverter_efficiency)
        motor.initialize_operating_point(self.config.motor_speed_rpm, 0.0)
        native_generator_dc_kw = self.generator_rated_ac_kw * self.rectifier_efficiency
        engine_rating_scale = max(
            float(aircraft.engine_electrical_power_max_kw) / max(native_generator_dc_kw, 1e-12),
            1e-12,
        )
        return HybridSystemPowertrainState(
            self.engine_map,
            generator,
            battery,
            motor,
            engine_rating_scale,
            battery_parallel_equivalent,
            motor_shaft_rating_kw,
            motor_branches,
            max(float(aircraft.engine_electrical_power_max_kw), 0.0),
            tuple(float(value) for value in aircraft.engine_efficiency_power_fractions),
            tuple(float(value) for value in aircraft.engine_efficiency_values),
            nominal_battery_parameters=dict(battery_parameters),
        )

    def set_battery_availability(
        self, state: HybridSystemPowertrainState, fraction: float
    ) -> None:
        """Equivalent parallel-string isolation with charge-conserving reconnection.

        Offline strings retain their charge. Reconnection assumes quasistatic
        equalization; switching currents and thermal transients are not modeled.
        This represents availability, not irreversible cell aging.
        """
        fraction = float(fraction)
        if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("battery availability must be finite and in [0, 1]")
        old_fraction = state.battery_available_fraction
        if fraction == old_fraction:
            return
        nominal = state.nominal_battery_parameters
        capacity_ah = float(nominal["BAT_Capacity_Ah"])
        active_charge = capacity_ah * old_fraction * state.battery.soc
        if fraction < old_fraction:
            transferred = capacity_ah * (old_fraction - fraction) * state.battery.soc
            state.battery_isolated_charge_ah += transferred
            active_charge -= transferred
        else:
            transferred = state.battery_isolated_charge_ah * (
                (fraction - old_fraction) / (1.0 - old_fraction)
            )
            state.battery_isolated_charge_ah -= transferred
            active_charge += transferred
            state.battery_reconnection_count += 1
        if fraction > 0.0:
            state.battery.soc = active_charge / (capacity_ah * fraction)
        scale = max(fraction, 1e-12)
        for key in ("BAT_Capacity_Ah", "BAT_Energy_nominal_Wh"):
            state.battery.parameters[key] = float(nominal[key]) * scale
        for key in ("BAT_Rdis_pack_Ohm", "BAT_Rchg_pack_Ohm"):
            state.battery.parameters[key] = tuple(float(r) / scale for r in nominal[key])
        state.battery_available_fraction = fraction

    def _engine_output(
        self,
        state: HybridSystemPowertrainState,
        target_dc_kw: float,
        power_cap_kw: float,
        altitude_m: float = 0.0,
    ) -> tuple[float, float]:
        max_dc_kw = self.generator_rated_ac_kw * self.rectifier_efficiency * state.engine_rating_scale
        target_dc_w = min(
            max(float(target_dc_kw), 0.0),
            max(float(power_cap_kw), 0.0),
            max_dc_kw,
        ) * 1000.0
        target_ac_w = target_dc_w / max(self.rectifier_efficiency, 1e-12)
        if target_ac_w <= 1e-9:
            return 0.0, 0.0
        gp = state.generator.parameters
        active_scale = state.engine_rating_scale
        engine_mechanical_request_w = (
            target_ac_w / active_scale / (gp["GEN_Eta"] * gp["GEN_GearEta"])
        )
        engine = state.engine_map.operating_point_for_power(
            engine_mechanical_request_w,
            self.config.engine_speed_rpm,
            altitude_m=altitude_m,
        )
        generator = state.generator.outputs(
            target_ac_w,
            self.config.engine_speed_rpm,
            engine["BrakeTorque_Nm"] * active_scale,
        )
        dc_kw = generator["P_generator_ac_actual_W"] * self.rectifier_efficiency / 1000.0
        fuel_scale = (
            active_scale * self.config.engine_fuel_flow_scale
            / self.config.engine_efficiency_multiplier
        )
        return max(dc_kw, 0.0), max(engine["FuelFlow_kg_s"] * fuel_scale, 0.0)

    def engine_efficiency_curve(self, state, altitude_m, power_cap_kw):
        """Prediction curve at this altitude and scheduled generator cap."""
        key = (float(altitude_m), float(power_cap_kw), state.engine_rating_scale)
        if key not in self._efficiency_cache:
            fractions = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
            efficiencies = []
            for fraction in fractions:
                power, fuel = self._engine_output(
                    state, max(fraction, 0.05) * power_cap_kw, power_cap_kw, altitude_m
                )
                efficiencies.append(max(power * 1000.0 / max(fuel * 43e6, 1e-9), 1e-6))
            if len(self._efficiency_cache) >= 512:
                self._efficiency_cache.clear()
            self._efficiency_cache[key] = fractions, tuple(efficiencies)
        return self._efficiency_cache[key]

    def _motor_output(
        self,
        state: HybridSystemPowertrainState,
        requested_shaft_kw: float,
        bus_voltage_v: float,
        available_dc_kw: float,
    ) -> tuple[float, float]:
        """Return total mechanical shaft kW and total motor DC demand kW."""
        if requested_shaft_kw <= 0.0 or available_dc_kw <= 0.0:
            return 0.0, 0.0
        motor = state.motor
        omega = self.config.motor_speed_rpm * pi / 30.0
        branch_rating_w = motor.parameters["MOT_P_rated_W"]
        branch_request_w = min(
            max(requested_shaft_kw, 0.0) * 1000.0 / state.motor_branches,
            branch_rating_w,
        )
        max_torque = motor.parameters["MOT_Iq_limit_A"] * motor.parameters["MOT_Kt_NmPerA"]
        requested_torque = min(branch_request_w / omega, max_torque)
        branch_dc_w = min(
            max(available_dc_kw, 0.0) * 1000.0 / state.motor_branches,
            motor.parameters["MOT_P_input_max_W"],
        )

        def evaluate(torque: float) -> dict[str, float | bool]:
            return motor.steady_outputs(
                self.config.motor_speed_rpm,
                torque,
                max(bus_voltage_v, 0.0),
                dc_power_available_w=branch_dc_w,
            )

        operating = evaluate(requested_torque)
        if not operating["equilibrium_feasible"] and requested_torque > 1e-9:
            low, high = 0.0, requested_torque
            operating = evaluate(low)
            if not operating["equilibrium_feasible"]:
                return 0.0, 0.0
            for _ in range(14):
                midpoint = 0.5 * (low + high)
                candidate = evaluate(midpoint)
                if candidate["equilibrium_feasible"]:
                    low, operating = midpoint, candidate
                else:
                    high = midpoint
        shaft_kw = max(float(operating["P_shaft_W"]), 0.0) * state.motor_branches / 1000.0
        motor_ac_w = max(float(operating["P_elec_W"]), 0.0) * state.motor_branches
        motor_dc_kw = motor_ac_w / max(self.config.inverter_efficiency, 1e-12) / 1000.0
        return shaft_kw, motor_dc_kw

    def motor_dc_demand(self, state, shaft_kw):
        voltage = float(state.battery.outputs(0.0)["Voc_pack_V"])
        return self._motor_output(state, shaft_kw, voltage, 1e9)[1]

    def step(
        self,
        state: HybridSystemPowertrainState,
        request: Any,
        dt_s: float,
        aircraft: Any,
    ) -> Any:
        del aircraft
        dt_s = float(dt_s)
        if not isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        altitude_m = float(getattr(request, "altitude_m", 0.0))
        if not isfinite(altitude_m):
            raise ValueError("altitude_m must be finite")
        if not self.engine_map.altitude_m[0] <= altitude_m <= self.engine_map.altitude_m[-1]:
            state.engine_map_altitude_clamped_steps += 1

        self.set_battery_availability(
            state, getattr(request, "battery_available_energy_fraction", 1.0)
        )
        battery = state.battery
        battery_capacity_wh = float(battery.parameters["BAT_Energy_nominal_Wh"])
        battery_discharge_limit = max(float(battery.outputs(1e12)["P_bat_W"]), 0.0)
        battery_charge_limit = max(-float(battery.outputs(-1e12)["P_bat_W"]), 0.0)
        battery_discharge_limit = min(
            battery_discharge_limit,
            max(battery.soc - self.config.battery_soc_min, 0.0) * battery_capacity_wh * 3600.0 / dt_s,
        )
        battery_charge_limit = min(
            battery_charge_limit,
            max(self.config.battery_soc_max - battery.soc, 0.0) * battery_capacity_wh * 3600.0 / dt_s,
        )
        if state.battery_available_fraction == 0.0:
            battery_discharge_limit = battery_charge_limit = 0.0
        motor_max_kw = request.motor_power_max_kw
        motor_min_kw = request.motor_power_min_kw
        battery_bus_discharge_kw = battery_discharge_limit * self.dcdc_efficiency / 1000.0
        battery_bus_charge_kw = battery_charge_limit / max(self.dcdc_efficiency, 1e-12) / 1000.0
        if motor_max_kw is not None:
            battery_bus_discharge_kw = min(battery_bus_discharge_kw, max(float(motor_max_kw), 0.0))
        if motor_min_kw is not None:
            battery_bus_charge_kw = min(battery_bus_charge_kw, max(-float(motor_min_kw), 0.0))

        voltage = float(battery.outputs(0.0)["Voc_pack_V"])
        requested_shaft_kw = max(float(request.propulsion_power_request_kw), 0.0)
        estimated_shaft_kw, estimated_motor_dc_kw = self._motor_output(
            state,
            requested_shaft_kw,
            voltage,
            1e9,
        )
        motor_efficiency = (
            estimated_shaft_kw / estimated_motor_dc_kw
            if estimated_motor_dc_kw > 1e-9
            else 1.0
        )
        requested_battery_bus_kw = request.propulsion_power_request_kw - request.engine_target_kw
        if request.motor_power_max_kw is not None:
            requested_battery_bus_kw = min(requested_battery_bus_kw, request.motor_power_max_kw)
        if request.motor_power_min_kw is not None:
            requested_battery_bus_kw = max(requested_battery_bus_kw, request.motor_power_min_kw)
        engine_bus_target_kw = max(
            requested_shaft_kw / max(motor_efficiency, 1e-9) - requested_battery_bus_kw,
            0.0,
        )
        if getattr(request, "engine_target_is_dc", False):
            engine_bus_target_kw = max(float(request.engine_target_kw), 0.0)
        if self.config.engine_command_delay_s > 0:
            history = state.engine_command_history
            history.append((state.engine_elapsed_s, engine_bus_target_kw))
            cutoff = state.engine_elapsed_s - self.config.engine_command_delay_s
            while len(history) > 1 and history[1][0] <= cutoff + 1e-9:
                history.pop(0)
            # Average the delayed ZOH command over this interval. A one-second
            # delay must not become a 300-second outage on the mission grid.
            end = cutoff + dt_s
            boundaries = [cutoff] + [t for t, _ in history if cutoff < t < end] + [end]
            command_integral = 0.0
            for left, right in zip(boundaries, boundaries[1:]):
                active = [value for t, value in history if t <= left + 1e-9]
                command_integral += (active[-1] if active else 0.0) * (right-left)
            engine_bus_target_kw = command_integral/dt_s
        if self.config.engine_time_constant_s > 0:
            engine_bus_target_kw = state.engine_power_kw + (
                engine_bus_target_kw-state.engine_power_kw
            ) * (1.0-exp(-dt_s/self.config.engine_time_constant_s))
        state.engine_elapsed_s += dt_s
        maximum_engine_change_kw = self.config.engine_power_rate_limit_kwps * dt_s
        rate_limited_engine_target_kw = min(
            max(
                engine_bus_target_kw,
                state.engine_power_kw - maximum_engine_change_kw,
            ),
            state.engine_power_kw + maximum_engine_change_kw,
        )
        engine_dc_kw, fuel_flow = self._engine_output(
            state,
            rate_limited_engine_target_kw,
            request.engine_power_cap_kw,
            altitude_m,
        )
        state.engine_power_kw = engine_dc_kw
        actual_shaft_kw = 0.0
        actual_motor_dc_kw = 0.0
        battery_result: dict[str, float] = battery.outputs(0.0)
        battery_bus_kw = battery_bus_discharge_kw
        for _ in range(3):
            available_bus_kw = max(engine_dc_kw + max(battery_bus_kw, 0.0), 0.0)
            actual_shaft_kw, actual_motor_dc_kw = self._motor_output(
                state,
                requested_shaft_kw,
                voltage,
                available_bus_kw,
            )
            desired_battery_bus_kw = actual_motor_dc_kw - engine_dc_kw
            desired_battery_bus_kw = min(
                max(desired_battery_bus_kw, -battery_bus_charge_kw),
                battery_bus_discharge_kw,
            )
            if desired_battery_bus_kw >= 0:
                terminal_request_w = desired_battery_bus_kw * 1000.0 / max(self.dcdc_efficiency, 1e-12)
            else:
                terminal_request_w = desired_battery_bus_kw * 1000.0 * self.dcdc_efficiency
            terminal_request_w = min(
                max(terminal_request_w, -battery_charge_limit),
                battery_discharge_limit,
            )
            battery_result = battery.outputs(terminal_request_w)
            terminal_actual_w = float(battery_result["P_bat_W"])
            battery_bus_kw = (
                terminal_actual_w * self.dcdc_efficiency / 1000.0
                if terminal_actual_w >= 0
                else terminal_actual_w / max(self.dcdc_efficiency, 1e-12) / 1000.0
            )
            voltage = float(battery_result["V_bat_V"])

        battery.step(terminal_request_w, dt_s, max_step_s=dt_s)
        return HybridSystemPowertrainOutput(
            state=state,
            engine_power_kw=engine_dc_kw,
            motor_power_kw=battery_bus_kw,
            propulsor_input_power_kw=actual_shaft_kw,
            fuel_flow_kgps=fuel_flow,
            soc=float(battery.soc),
            battery_current_a=float(battery_result["I_bat_A"]),
            battery_terminal_power_kw=float(battery_result["P_bat_W"]) / 1000.0,
            battery_chemical_power_kw=float(
                battery_result["Voc_pack_V"] * battery_result["I_bat_A"] / 1000.0
            ),
            motor_dc_power_kw=actual_motor_dc_kw,
            unabsorbed_bus_power_kw=max(engine_dc_kw + battery_bus_kw - actual_motor_dc_kw, 0.0),
            bus_power_deficit_kw=max(actual_motor_dc_kw - engine_dc_kw - battery_bus_kw, 0.0),
        )


def _linear_interpolate(value: float, x_values: tuple[float, ...], y_values: tuple[float, ...]) -> float:
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("Engine-efficiency calibration arrays must have matching nonzero lengths")
    if value <= x_values[0]:
        return y_values[0]
    for left in range(len(x_values) - 1):
        if value <= x_values[left + 1]:
            span = x_values[left + 1] - x_values[left]
            fraction = (value - x_values[left]) / span if span else 0.0
            return y_values[left] + fraction * (y_values[left + 1] - y_values[left])
    return y_values[-1]
