"""Intelligent, safety-supervised energy management prototypes for PTDL 1.2.2.

The online controller is deliberately hierarchical.  Fuzzy inference produces
an interpretable power-split request, while the flight-propulsion simulator
still enforces physical power/SOC boundaries.  A future neural predictor may
adjust the SOC reference, but never writes unrestricted actuator commands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Protocol

import numpy as np

from 飞推控闭环物理模型 import (
    DispatchObservation,
    HybridDispatchPolicy,
    MissionFaultEvent,
    PowerDispatch,
)
from 规则混动协同控制 import DEFAULT_SOC_TARGETS


# The intelligent controller maintains a terminal power reserve rather than
# monotonically consuming the battery after loiter.  These references will be
# replaced by an NN-predicted trajectory after high-fidelity training data are
# available.
FUZZY_SOC_TARGETS = {
    "main_climb_to_leveloff": 0.76,
    "short_leveloff_to_isr_altitude": 0.70,
    "endurance_loiter": 0.62,
    "long_descent": 0.55,
    "approach_descent": 0.48,
    "flare": 0.38,
}

PROACTIVE_ASSIST_PHASES = {"main_climb_to_leveloff", "short_leveloff_to_isr_altitude"}
NEURAL_PHASES = (
    "main_climb_to_leveloff",
    "short_leveloff_to_isr_altitude",
    "endurance_loiter",
    "long_descent",
    "approach_descent",
    "flare",
)


def neural_imitation_feature_vector(
    phase_name: str,
    time_step_s: float,
    soc: float,
    altitude_error_m: float,
    speed_error_mps: float,
    required_propulsion_kw: float,
    engine_power_cap_kw: float,
    previous_engine_target_kw: float,
    phase_elapsed_s: float,
    phase_duration_s: float,
) -> np.ndarray:
    cap = max(float(engine_power_cap_kw), 1e-6)
    return np.asarray(
        [
            *[float(phase_name == phase) for phase in NEURAL_PHASES],
            float(time_step_s) / 300.0,
            float(soc),
            float(altitude_error_m) / 3000.0,
            float(speed_error_mps) / 20.0,
            float(required_propulsion_kw) / cap,
            float(previous_engine_target_kw) / cap,
            float(phase_elapsed_s) / max(float(phase_duration_s), 1.0),
        ],
        dtype=float,
    )


class NeuralImitationCoordinator:
    """Offline-MPC imitation policy with online power/rate safety projection."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "ptdl_mpc_imitation_mlp_v1":
            raise ValueError("Unsupported neural MPC-imitation model schema.")
        if tuple(payload.get("phase_labels", ())) != NEURAL_PHASES:
            raise ValueError("Neural imitation phase schema does not match this model.")
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=float)
        self.feature_scale = np.maximum(
            np.asarray(payload["feature_scale"], dtype=float), 1e-9
        )
        self.target_mean = float(payload["target_mean"])
        self.target_scale = max(float(payload["target_scale"]), 1e-9)
        self.coefs = [np.asarray(item, dtype=float) for item in payload["coefs"]]
        self.intercepts = [np.asarray(item, dtype=float) for item in payload["intercepts"]]
        self.phase_durations = {
            str(name): float(value)
            for name, value in payload["phase_durations_s"].items()
        }
        self.soc_targets = {
            str(name): float(value)
            for name, value in payload["phase_soc_targets"].items()
        }
        self._previous_engine_target_kw: float | None = None

    def reset(self) -> None:
        self._previous_engine_target_kw = None

    def predict_engine_fraction(self, features: np.ndarray) -> float:
        activation = (features - self.feature_mean) / self.feature_scale
        for index, (weights, bias) in enumerate(zip(self.coefs, self.intercepts)):
            activation = activation @ weights + bias
            if index < len(self.coefs) - 1:
                activation = np.tanh(activation)
        return float(self.target_mean + self.target_scale * activation[0])

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        previous = (
            0.0
            if self._previous_engine_target_kw is None
            else self._previous_engine_target_kw
        )
        features = neural_imitation_feature_vector(
            observation.phase_name,
            observation.time_step_s,
            observation.soc,
            observation.altitude_error_m,
            observation.speed_error_mps,
            observation.required_propulsion_kw,
            observation.engine_power_cap_kw,
            previous,
            observation.phase_elapsed_s,
            self.phase_durations.get(observation.phase_name, 1.0),
        )
        predicted_fraction = float(np.clip(self.predict_engine_fraction(features), 0.0, 1.0))
        requested_engine_kw = predicted_fraction * observation.engine_power_cap_kw
        feasible_max = max(0.0, observation.engine_power_cap_kw)
        soc_target = self.soc_targets.get(
            observation.phase_name, observation.battery_soc_reserve
        )
        charge_allowed = observation.soc < min(
            observation.battery_soc_max, soc_target
        )
        motor_min = (
            min(0.0, observation.motor_power_min_kw)
            if charge_allowed
            else 0.0
        )
        motor_max = max(0.0, observation.motor_power_max_kw)
        feasible_min = min(
            feasible_max,
            max(
                0.0,
                observation.required_propulsion_kw - motor_max,
            ),
        )
        feasible_max = min(
            feasible_max,
            max(0.0, observation.required_propulsion_kw - motor_min),
        )
        requested_engine_kw = float(np.clip(requested_engine_kw, feasible_min, feasible_max))
        if self._previous_engine_target_kw is not None:
            slew = max(0.0, observation.engine_power_rate_limit_kwps) * max(
                observation.time_step_s, 0.0
            )
            requested_engine_kw = float(
                np.clip(
                    requested_engine_kw,
                    self._previous_engine_target_kw - slew,
                    self._previous_engine_target_kw + slew,
                )
            )
        requested_engine_kw = float(
            np.clip(requested_engine_kw, feasible_min, feasible_max)
        )
        self._previous_engine_target_kw = requested_engine_kw
        return PowerDispatch(
            engine_target_kw=requested_engine_kw,
            soc_target=soc_target,
            mode="neural_mpc_imitation_safety_projected",
            engine_power_cap_kw=observation.engine_power_cap_kw,
            motor_power_min_kw=observation.motor_power_min_kw,
            motor_power_max_kw=observation.motor_power_max_kw,
        )


class SOCReferencePredictor(Protocol):
    """Offline-trained predictor interface; not an actuator interface."""

    def predict_soc_target(self, observation: DispatchObservation, nominal_target: float) -> float: ...


@dataclass(frozen=True)
class FuzzyEnergyManagementParameters:
    soc_targets: dict[str, float]
    default_soc_target: float = 0.50
    target_correction_limit: float = 0.06
    soc_deadband: float = 0.015
    maximum_charge_bias_kw: float = 18.0
    maximum_assist_bias_kw: float = 12.0
    tracking_reserve_limit_kw: float = 18.0
    engine_target_rate_limit_kwps: float = 3.0

    @classmethod
    def default(cls) -> "FuzzyEnergyManagementParameters":
        return cls(soc_targets=dict(FUZZY_SOC_TARGETS))


@dataclass(frozen=True)
class HealthState:
    """Health information accepted by the fault-supervisory layer.

    Component derating becomes physically effective once the corresponding
    dynamic component model is supplied.  The present wrapper already applies
    the engine availability cap to the decision layer.
    """

    engine_available_fraction: float = 1.0
    motor_available_fraction: float = 1.0
    battery_available_energy_fraction: float = 1.0
    sensor_data_valid: bool = True


class FuzzyAdaptiveCoordinator:
    """Interpretable fuzzy supervisor for real-time hybrid power allocation."""

    def __init__(
        self,
        parameters: FuzzyEnergyManagementParameters | None = None,
        predictor: SOCReferencePredictor | None = None,
    ) -> None:
        self.parameters = parameters or FuzzyEnergyManagementParameters.default()
        self.predictor = predictor
        self._previous_engine_target_kw: float | None = None

    def reset(self) -> None:
        self._previous_engine_target_kw = None

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        parameters = self.parameters
        nominal_target = float(parameters.soc_targets.get(observation.phase_name, parameters.default_soc_target))
        predicted_target = nominal_target if self.predictor is None else self.predictor.predict_soc_target(observation, nominal_target)
        soc_target = float(np.clip(predicted_target, nominal_target - parameters.target_correction_limit, nominal_target + parameters.target_correction_limit))
        target_error = soc_target - observation.soc

        # Fuzzy memberships use normalized, bounded quantities and therefore
        # remain meaningful when the actual component power ratings are updated.
        soc_low = _ramp(target_error, parameters.soc_deadband, 0.18)
        soc_high = _ramp(-target_error, parameters.soc_deadband, 0.18)
        demand_ratio = observation.required_propulsion_kw / max(observation.engine_power_cap_kw, 1e-6)
        demand_high = _ramp(demand_ratio, 0.60, 1.0)
        tracking_ratio = min(
            1.0,
            max(observation.speed_error_mps, 0.0) / 5.0 + max(observation.altitude_error_m, 0.0) / 3000.0,
        )

        charge_bias_kw = parameters.maximum_charge_bias_kw * soc_low * (1.0 - 0.35 * demand_high)
        assist_membership = max(soc_high, tracking_ratio)
        # At high demand the generator must retain priority; the battery only
        # fills a genuine generator-capability gap.  Proactive discharge is
        # reserved for lower-demand conditions where SOC recovery is feasible.
        allow_proactive_assist = observation.phase_name in PROACTIVE_ASSIST_PHASES
        assist_bias_kw = parameters.maximum_assist_bias_kw * assist_membership * (1.0 - demand_high) if allow_proactive_assist else 0.0
        tracking_reserve_kw = parameters.tracking_reserve_limit_kw * tracking_ratio * (1.0 - demand_high) if allow_proactive_assist else 0.0
        engine_target_kw = observation.required_propulsion_kw + charge_bias_kw - assist_bias_kw - tracking_reserve_kw
        # Project the fuzzy command onto the instantaneous propulsion-feasible
        # set before rate limiting.  The learning/fuzzy layer may shape a power
        # split, but it may not create a generator-plus-motor shortfall.
        feasible_lower_kw = max(0.0, observation.required_propulsion_kw - observation.motor_power_max_kw)
        engine_target_kw = float(np.clip(engine_target_kw, feasible_lower_kw, observation.engine_power_cap_kw))
        if self._previous_engine_target_kw is not None:
            maximum_change = parameters.engine_target_rate_limit_kwps * observation.time_step_s
            engine_target_kw = float(np.clip(engine_target_kw, self._previous_engine_target_kw - maximum_change, self._previous_engine_target_kw + maximum_change))
        engine_target_kw = max(engine_target_kw, feasible_lower_kw)
        self._previous_engine_target_kw = engine_target_kw

        motor_balance_kw = observation.required_propulsion_kw - engine_target_kw
        if not observation.motor_power_min_kw <= motor_balance_kw <= observation.motor_power_max_kw:
            mode = "safety_limited"
        elif motor_balance_kw > 0.5:
            mode = "fuzzy_motor_assist"
        elif motor_balance_kw < -0.5:
            mode = "fuzzy_battery_charge"
        else:
            mode = "fuzzy_engine_follow"
        return PowerDispatch(engine_target_kw, soc_target, mode)


@dataclass(frozen=True)
class MPCParameters:
    horizon_steps: int = 3
    engine_action_levels: int = 3
    demand_slope_memory: float = 0.5
    demand_slope_limit_kwps: float = 0.02
    soc_terminal_weight_mj: float = 4.0
    power_shortfall_penalty_mj_per_kw: float = 1.0e5
    soc_violation_penalty_mj: float = 1.0e5
    engine_slew_penalty_mj_per_kw: float = 0.002
    default_soc_target: float = 0.50
    phase_soc_targets: tuple[tuple[str, float], ...] = tuple(DEFAULT_SOC_TARGETS.items())


def fuel_energy_mj(power_kw, duration_s, efficiency):
    """Convert electrical kW and seconds to upstream fuel energy in MJ."""
    return np.asarray(power_kw) * duration_s / 1000.0 / np.asarray(efficiency)


class MPCEnergyManagementCoordinator:
    """Small receding-horizon power-split MPC using the declared component maps."""

    def __init__(self, parameters: MPCParameters | None = None) -> None:
        self.parameters = parameters or MPCParameters()
        if self.parameters.horizon_steps < 1 or self.parameters.engine_action_levels < 2:
            raise ValueError("MPC horizon must be positive and action levels must be at least two.")
        self._previous_demand_kw: float | None = None
        self._demand_slope_kwps = 0.0
        self._previous_engine_target_kw: float | None = None
        self._sequences = np.asarray(
            list(product(range(self.parameters.engine_action_levels), repeat=self.parameters.horizon_steps)),
            dtype=float,
        )

    def reset(self) -> None:
        self._previous_demand_kw = None
        self._demand_slope_kwps = 0.0
        self._previous_engine_target_kw = None

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        parameters = self.parameters
        dt_s = max(float(observation.time_step_s), 1e-6)
        demand_kw = max(0.0, float(observation.required_propulsion_kw))
        if self._previous_demand_kw is not None:
            measured_slope = (demand_kw - self._previous_demand_kw) / dt_s
            memory = float(np.clip(parameters.demand_slope_memory, 0.0, 1.0))
            slope_limit = max(0.0, parameters.demand_slope_limit_kwps)
            self._demand_slope_kwps = (
                memory * self._demand_slope_kwps + (1.0 - memory) * measured_slope
            )
            self._demand_slope_kwps = float(
                np.clip(self._demand_slope_kwps, -slope_limit, slope_limit)
            )
        self._previous_demand_kw = demand_kw
        slope_limit = max(0.0, parameters.demand_slope_limit_kwps)
        demands = np.clip(
            demand_kw + self._demand_slope_kwps * dt_s * np.arange(1, parameters.horizon_steps + 1),
            0.0,
            observation.engine_power_cap_kw + observation.motor_power_max_kw,
        )

        capacity_mj = max(float(observation.battery_capacity_mj), 1e-6)
        discharge_efficiency = float(np.clip(observation.battery_discharge_efficiency, 0.05, 1.0))
        charge_efficiency = float(np.clip(observation.battery_charge_efficiency, 0.05, 1.0))
        soc_min = float(np.clip(observation.battery_soc_min, 0.0, 0.99))
        soc_max = float(np.clip(observation.battery_soc_max, soc_min + 1e-6, 1.0))
        soc_target = max(
            float(observation.battery_soc_reserve),
            dict(parameters.phase_soc_targets).get(observation.phase_name, parameters.default_soc_target),
        )
        soc_target = float(np.clip(soc_target, soc_min, soc_max))
        fractions = np.asarray(observation.engine_efficiency_power_fractions, dtype=float)
        efficiencies = np.asarray(observation.engine_efficiency_values, dtype=float)
        if len(fractions) != len(efficiencies) or len(fractions) < 2:
            fractions = np.array([0.0, 1.0])
            efficiencies = np.array([0.30, 0.30])
        rate_limit_kw = max(0.0, float(observation.engine_power_rate_limit_kwps)) * dt_s
        previous_engine_kw = (
            demand_kw
            if self._previous_engine_target_kw is None
            else self._previous_engine_target_kw
        )

        scores = np.zeros(len(self._sequences), dtype=float)
        predicted_soc = np.full(len(self._sequences), float(observation.soc), dtype=float)
        prior_engine = np.full(len(self._sequences), previous_engine_kw, dtype=float)
        first_engine_targets = np.zeros(len(self._sequences), dtype=float)
        for horizon_index, forecast_demand_kw in enumerate(demands):
            motor_min = float(observation.motor_power_min_kw)
            motor_max = float(observation.motor_power_max_kw)
            feasible_engine_min = max(0.0, forecast_demand_kw - motor_max)
            feasible_engine_max = min(observation.engine_power_cap_kw, forecast_demand_kw - motor_min)
            if feasible_engine_max < feasible_engine_min:
                scores += parameters.power_shortfall_penalty_mj_per_kw * (
                    feasible_engine_min - feasible_engine_max
                )
                feasible_engine_max = feasible_engine_min
            lower = np.maximum(feasible_engine_min, prior_engine - rate_limit_kw)
            upper = np.minimum(feasible_engine_max, prior_engine + rate_limit_kw)
            infeasible_rate = np.maximum(0.0, lower - upper)
            scores += parameters.power_shortfall_penalty_mj_per_kw * infeasible_rate
            upper = np.maximum(upper, lower)
            fractions_grid = self._sequences[:, horizon_index] / max(
                self.parameters.engine_action_levels - 1, 1
            )
            engine_targets = lower + fractions_grid * (upper - lower)
            engine_targets = np.clip(engine_targets, 0.0, observation.engine_power_cap_kw)
            motor_kw = forecast_demand_kw - engine_targets
            motor_kw = np.clip(motor_kw, motor_min, motor_max)

            load_fractions = engine_targets / max(observation.engine_power_cap_kw, 1e-9)
            engine_efficiency = np.clip(
                np.interp(load_fractions, fractions, efficiencies), 0.05, 0.60
            )
            engine_energy_mj = fuel_energy_mj(engine_targets, dt_s, engine_efficiency)
            battery_energy_mj = (
                np.maximum(motor_kw, 0.0) / discharge_efficiency
                - charge_efficiency * np.maximum(-motor_kw, 0.0)
            ) * dt_s / 1000.0
            predicted_soc -= battery_energy_mj / capacity_mj
            scores += engine_energy_mj + battery_energy_mj
            scores += parameters.engine_slew_penalty_mj_per_kw * np.abs(engine_targets - prior_engine)
            scores += parameters.soc_violation_penalty_mj * (
                np.maximum(0.0, soc_min - predicted_soc) ** 2
                + np.maximum(0.0, predicted_soc - soc_max) ** 2
            )
            if horizon_index == 0:
                first_engine_targets[:] = engine_targets
            prior_engine = engine_targets

        scores += (
            parameters.soc_terminal_weight_mj
            * capacity_mj
            * (predicted_soc - soc_target) ** 2
        )
        selected = int(np.argmin(scores))
        engine_target_kw = float(first_engine_targets[selected])
        self._previous_engine_target_kw = engine_target_kw
        motor_balance_kw = demand_kw - engine_target_kw
        if motor_balance_kw > 0.5:
            mode = "mpc_motor_assist"
        elif motor_balance_kw < -0.5:
            mode = "mpc_battery_charge"
        else:
            mode = "mpc_engine_follow"
        return PowerDispatch(engine_target_kw, soc_target, mode)


class FaultTolerantCoordinator:
    """Health-aware wrapper for any energy-management policy."""

    def __init__(self, coordinator: HybridDispatchPolicy, health: HealthState = HealthState()) -> None:
        self.coordinator = coordinator
        self.health = health

    def reset(self) -> None:
        self.coordinator.reset()

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        health = self.health
        engine_cap = observation.engine_power_cap_kw * float(np.clip(health.engine_available_fraction, 0.0, 1.0))
        derated = replace(
            observation,
            battery_capacity_mj=observation.battery_capacity_mj * health.battery_available_energy_fraction,
            engine_power_cap_kw=engine_cap,
            motor_power_min_kw=observation.motor_power_min_kw * float(np.clip(health.motor_available_fraction, 0.0, 1.0)),
            motor_power_max_kw=observation.motor_power_max_kw * float(np.clip(health.motor_available_fraction, 0.0, 1.0)),
        )
        if not health.sensor_data_valid:
            # Conservative fallback: the generator tracks current demand and
            # does not intentionally discharge the battery without valid SOC.
            fallback_soc_target = 0.50
            if health.battery_available_energy_fraction < 0.75:
                fallback_soc_target = min(
                    0.90,
                    fallback_soc_target
                    + 0.08
                    * (0.75 - health.battery_available_energy_fraction)
                    / 0.75,
                )
            return PowerDispatch(
                min(derated.required_propulsion_kw, derated.engine_power_cap_kw),
                fallback_soc_target,
                "sensor_fallback",
                derated.engine_power_cap_kw,
                derated.motor_power_min_kw,
                derated.motor_power_max_kw,
                health.battery_available_energy_fraction,
            )
        dispatch = self.coordinator.dispatch(derated)
        engine_cap = max(0.0, derated.engine_power_cap_kw)
        motor_min = min(0.0, derated.motor_power_min_kw)
        motor_max = max(0.0, derated.motor_power_max_kw)
        demand = max(0.0, derated.required_propulsion_kw)
        feasible_engine_min = max(0.0, demand - motor_max)
        feasible_engine_max = min(engine_cap, demand - motor_min)
        if feasible_engine_min <= feasible_engine_max:
            projected_engine_kw = float(
                np.clip(
                    dispatch.engine_target_kw,
                    feasible_engine_min,
                    feasible_engine_max,
                )
            )
        else:
            # Preserve actuator bounds when available power cannot meet demand.
            projected_engine_kw = float(
                np.clip(dispatch.engine_target_kw, 0.0, engine_cap)
            )
        mode = dispatch.mode
        if abs(projected_engine_kw - dispatch.engine_target_kw) > 1e-8:
            mode = f"{mode}_safety_projected"
        if health.battery_available_energy_fraction < 0.75:
            # Preserve additional reserve when capacity/SOH is degraded.
            protected_target = min(0.90, dispatch.soc_target + 0.08 * (0.75 - health.battery_available_energy_fraction) / 0.75)
            return PowerDispatch(
                projected_engine_kw,
                protected_target,
                f"derated_{mode}",
                engine_cap,
                motor_min,
                motor_max,
                health.battery_available_energy_fraction,
            )
        return PowerDispatch(
            projected_engine_kw,
            dispatch.soc_target,
            mode,
            engine_cap,
            motor_min,
            motor_max,
            health.battery_available_energy_fraction,
        )


class FaultTolerantFuzzyCoordinator(FaultTolerantCoordinator):
    """Backward-compatible name for the health-aware fuzzy controller."""


class ScheduledFaultCoordinator:
    """Apply health-aware dispatch only during configured mission intervals."""

    def __init__(
        self,
        coordinator: HybridDispatchPolicy,
        events: tuple[MissionFaultEvent, ...],
    ) -> None:
        self.coordinator = coordinator
        self.events = tuple(events)

    def reset(self) -> None:
        self.coordinator.reset()

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        active = next(
            (
                event
                for event in self.events
                if event.phase_name == observation.phase_name
                and event.start_phase_elapsed_s <= observation.phase_elapsed_s
                < event.start_phase_elapsed_s + event.duration_s
            ),
            None,
        )
        health = (
            HealthState()
            if active is None
            else HealthState(
                engine_available_fraction=active.engine_available_fraction,
                motor_available_fraction=active.motor_available_fraction,
                battery_available_energy_fraction=active.battery_available_energy_fraction,
                sensor_data_valid=active.sensor_data_valid,
            )
        )
        return FaultTolerantCoordinator(self.coordinator, health).dispatch(observation)


def _ramp(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("Fuzzy membership upper limit must exceed lower limit.")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
