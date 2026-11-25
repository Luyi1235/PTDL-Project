"""Rule-based hybrid flight-control coordination model for PTDL 1.2.1.

The controller is intentionally transparent: flight tracking determines power
demand in ``ptdl_flight_propulsion_control``; this module allocates that demand
between generator and battery using phase-dependent SOC references, tracking
reserve, and generator slew limits.  Component values remain replaceable
preliminary assumptions until propulsion test data are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ptdl_flight_propulsion_control import DispatchObservation, PowerDispatch


DEFAULT_SOC_TARGETS = {
    "main_climb_to_leveloff": 0.72,
    "short_leveloff_to_isr_altitude": 0.66,
    "endurance_loiter": 0.55,
    "long_descent": 0.42,
    "approach_descent": 0.34,
    "flare": 0.30,
}


@dataclass(frozen=True)
class HybridCoordinationParameters:
    """Preliminary, centralized parameters for the 1.2.1 coordination model."""

    soc_targets: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SOC_TARGETS))
    default_soc_target: float = 0.50
    soc_deadband: float = 0.02
    soc_feedback_gain_kw: float = 105.0
    proactive_discharge_limit_kw: float = 10.0
    proactive_charge_limit_kw: float = 18.0
    speed_error_assist_gain_kw_per_mps: float = 2.5
    altitude_error_assist_gain_kw_per_m: float = 0.0010
    tracking_assist_limit_kw: float = 18.0
    engine_target_rate_limit_kwps: float = 3.0


class RuleBasedHybridCoordinator:
    """SOC-feedback power-split controller with phase-continuous dispatch."""

    def __init__(self, parameters: HybridCoordinationParameters = HybridCoordinationParameters()) -> None:
        self.parameters = parameters
        self._previous_engine_target_kw: float | None = None

    def reset(self) -> None:
        self._previous_engine_target_kw = None

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        parameters = self.parameters
        soc_target = float(parameters.soc_targets.get(observation.phase_name, parameters.default_soc_target))
        soc_error = soc_target - observation.soc
        if abs(soc_error) <= parameters.soc_deadband:
            soc_correction_kw = 0.0
        else:
            soc_correction_kw = float(
                np.clip(
                    parameters.soc_feedback_gain_kw * soc_error,
                    -parameters.proactive_discharge_limit_kw,
                    parameters.proactive_charge_limit_kw,
                )
            )
        tracking_assist_kw = float(
            np.clip(
                parameters.speed_error_assist_gain_kw_per_mps * max(observation.speed_error_mps, 0.0)
                + parameters.altitude_error_assist_gain_kw_per_m * max(observation.altitude_error_m, 0.0),
                0.0,
                parameters.tracking_assist_limit_kw,
            )
        )
        desired_engine_kw = observation.required_propulsion_kw + soc_correction_kw - tracking_assist_kw
        desired_engine_kw = float(np.clip(desired_engine_kw, 0.0, observation.engine_power_cap_kw))
        if self._previous_engine_target_kw is not None:
            maximum_change = parameters.engine_target_rate_limit_kwps * observation.time_step_s
            desired_engine_kw = float(
                np.clip(
                    desired_engine_kw,
                    self._previous_engine_target_kw - maximum_change,
                    self._previous_engine_target_kw + maximum_change,
                )
            )
        self._previous_engine_target_kw = desired_engine_kw
        balance_motor_kw = observation.required_propulsion_kw - desired_engine_kw
        if balance_motor_kw > 0.5:
            mode = "motor_assist"
        elif balance_motor_kw < -0.5:
            mode = "battery_charge"
        else:
            mode = "engine_follow"
        return PowerDispatch(desired_engine_kw, soc_target, mode)
