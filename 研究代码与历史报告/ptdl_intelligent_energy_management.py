"""Intelligent, safety-supervised energy management prototypes for PTDL 1.2.2.

The online controller is deliberately hierarchical.  Fuzzy inference produces
an interpretable power-split request, while the flight-propulsion simulator
still enforces physical power/SOC boundaries.  A future neural predictor may
adjust the SOC reference, but never writes unrestricted actuator commands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from ptdl_flight_propulsion_control import DispatchObservation, PowerDispatch
from ptdl_hybrid_coordination import DEFAULT_SOC_TARGETS


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
        # In cruise and descent, intentional battery assist remains disabled at
        # trim; it becomes available only when the tracking residual is large.
        # This lets the same controller use the stored motor envelope for a
        # transient wind-recovery manoeuvre without consuming it during normal
        # long-endurance operation.
        allow_transient_assist = allow_proactive_assist or tracking_ratio >= 0.35
        assist_bias_kw = parameters.maximum_assist_bias_kw * assist_membership * (1.0 - demand_high) if allow_transient_assist else 0.0
        # Hold part of the motor envelope as a recovery reserve while the
        # tracking error is small.  As the error grows, the reserve is released
        # back to propulsion.  This creates a physical energy-versus-response
        # trade-off only when the engine transient and motor boundary are
        # active; it does not artificially change a power-abundant trim point.
        held_motor_reserve_kw = (
            parameters.tracking_reserve_limit_kw * (1.0 - tracking_ratio) if allow_proactive_assist else 0.0
        )
        available_motor_max_kw = max(0.0, observation.motor_power_max_kw - held_motor_reserve_kw)
        engine_target_kw = observation.required_propulsion_kw + charge_bias_kw - assist_bias_kw
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
        if not observation.motor_power_min_kw <= motor_balance_kw <= available_motor_max_kw:
            mode = "safety_limited"
        elif motor_balance_kw > 0.5:
            mode = "fuzzy_motor_assist"
        elif motor_balance_kw < -0.5:
            mode = "fuzzy_battery_charge"
        else:
            mode = "fuzzy_engine_follow"
        return PowerDispatch(
            engine_target_kw,
            soc_target,
            mode,
            motor_power_max_kw=available_motor_max_kw,
            propulsion_power_limit_kw=observation.engine_power_cap_kw + available_motor_max_kw,
        )


class FaultTolerantFuzzyCoordinator:
    """Health-aware wrapper with sensor fallback and power-availability derating."""

    def __init__(self, coordinator: FuzzyAdaptiveCoordinator, health: HealthState = HealthState()) -> None:
        self.coordinator = coordinator
        self.health = health
        self._estimated_engine_available_fraction = 1.0

    @property
    def estimated_engine_available_fraction(self) -> float:
        """Filtered availability estimate for logging and SOC-reserve adaptation.

        The current low-order plant exposes the effective generator capability
        directly.  When the propulsion institute's dynamic engine model is
        connected, ``measured_available_cap_kw`` can replace this input while
        retaining the same estimator and dispatch interface.
        """
        return self._estimated_engine_available_fraction

    def reset(self) -> None:
        self.coordinator.reset()
        self._estimated_engine_available_fraction = 1.0

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        health = self.health
        engine_cap = observation.engine_power_cap_kw * float(np.clip(health.engine_available_fraction, 0.0, 1.0))
        observed_engine_fraction = engine_cap / max(observation.engine_power_cap_kw, 1e-6)
        # A first-order monitor avoids a discontinuous SOC-reference jump when
        # an engine availability change is detected.  The 20 s time constant
        # represents an intentionally replaceable health-monitoring filter.
        monitor_gain = 1.0 - np.exp(-observation.time_step_s / 20.0)
        self._estimated_engine_available_fraction += monitor_gain * (observed_engine_fraction - self._estimated_engine_available_fraction)
        self._estimated_engine_available_fraction = float(np.clip(self._estimated_engine_available_fraction, 0.0, 1.0))
        derated = replace(
            observation,
            engine_power_cap_kw=engine_cap,
            motor_power_min_kw=observation.motor_power_min_kw * float(np.clip(health.motor_available_fraction, 0.0, 1.0)),
            motor_power_max_kw=observation.motor_power_max_kw * float(np.clip(health.motor_available_fraction, 0.0, 1.0)),
        )
        if not health.sensor_data_valid:
            # Conservative fallback: the generator tracks current demand and
            # does not intentionally discharge the battery without valid SOC.
            return PowerDispatch(
                min(derated.required_propulsion_kw, derated.engine_power_cap_kw),
                0.50,
                "sensor_fallback",
                derated.engine_power_cap_kw,
                derated.motor_power_min_kw,
                derated.motor_power_max_kw,
                derated.engine_power_cap_kw + derated.motor_power_max_kw,
            )
        dispatch = self.coordinator.dispatch(derated)
        engine_reserve = 0.12 * (1.0 - self._estimated_engine_available_fraction)
        battery_reserve = 0.10 * (1.0 - float(np.clip(health.battery_available_energy_fraction, 0.0, 1.0)))
        protected_target = min(0.90, dispatch.soc_target + engine_reserve + battery_reserve)
        if engine_reserve > 1e-4 or battery_reserve > 1e-4:
            # Preserve energy before a derated generator reaches its limit;
            # high tracking errors still release the motor reserve inside the
            # fuzzy coordinator to avoid sacrificing controllability.
            return PowerDispatch(
                dispatch.engine_target_kw,
                protected_target,
                f"health_adaptive_{dispatch.mode}",
                derated.engine_power_cap_kw,
                derated.motor_power_min_kw,
                derated.motor_power_max_kw,
                min(
                    derated.engine_power_cap_kw + derated.motor_power_max_kw,
                    dispatch.propulsion_power_limit_kw
                    if dispatch.propulsion_power_limit_kw is not None
                    else derated.engine_power_cap_kw + derated.motor_power_max_kw,
                ),
            )
        return PowerDispatch(
            dispatch.engine_target_kw,
            dispatch.soc_target,
            dispatch.mode,
            derated.engine_power_cap_kw,
            derated.motor_power_min_kw,
            derated.motor_power_max_kw,
            min(
                derated.engine_power_cap_kw + derated.motor_power_max_kw,
                dispatch.propulsion_power_limit_kw
                if dispatch.propulsion_power_limit_kw is not None
                else derated.engine_power_cap_kw + derated.motor_power_max_kw,
            ),
        )


def _ramp(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("Fuzzy membership upper limit must exceed lower limit.")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
