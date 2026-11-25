"""Identifiable online health estimates and phase-scheduled MPC."""

from dataclasses import dataclass
import math

import numpy as np

from 智能能量管理与容错调度 import (
    FaultTolerantCoordinator, HealthState, MPCParameters,
    MPCEnergyManagementCoordinator,
)


def identify_delayed_response(command, response, dt_s, maximum_delay_steps=8):
    """Fit y[k+1] = a*y[k] + b*u[k-d] + c on excited data only."""
    u, y = np.asarray(command, float), np.asarray(response, float)
    if len(u) != len(y) or len(y) < 30 or dt_s <= 0:
        raise ValueError("Identification needs equal traces, >=30 samples and positive dt.")
    if not np.isfinite(u).all() or not np.isfinite(y).all() or np.std(u) < 1e-6:
        raise ValueError("Finite, excited commands are required.")
    start = maximum_delay_steps
    split = max(start + 10, int(len(y) * 0.7))
    if split >= len(y) - 3:
        raise ValueError("Insufficient independent validation samples.")
    trials = []
    for delay in range(maximum_delay_steps + 1):
        k = np.arange(start, len(y) - 1)
        x = np.column_stack((y[k], u[k-delay], np.ones(len(k))))
        train = k < split
        coeff = np.linalg.lstsq(x[train], y[k[train]+1], rcond=None)[0]
        a, b, c = coeff
        if not 0 <= a < 1 or b <= 0:
            continue
        # Delay selection uses training only; validation remains untouched.
        error = x @ coeff - y[k+1]
        trials.append({
            "delay_steps": delay, "delay_s": delay * dt_s,
            "time_constant_s": -dt_s / math.log(max(a, 1e-12)),
            "steady_gain": b / (1-a), "offset": c / (1-a),
            "training_rmse": float(np.sqrt(np.mean(error[train]**2))),
            "validation_rmse": float(np.sqrt(np.mean(error[~train]**2))),
            "a": float(a), "b": float(b), "c": float(c),
        })
    if not trials:
        raise ValueError("No stable identifiable first-order response.")
    return min(trials, key=lambda r: r["training_rmse"])


@dataclass
class ResidualHealthEstimator:
    """Estimate availability from measured response, never from injected truth.

    Engine ratio is observable only after a near-rating command has settled.
    Capacity uses coulomb/SOC increments, excluding switch jumps and SOC clamps.
    Flight residual is an anomaly indicator, not an independent component label.
    """

    battery_fraction: float = 1.0
    engine_fraction: float = 1.0
    updates: int = 0
    capacity_observations: int = 0
    engine_observations: int = 0
    tracking_risk: float = 0.0
    last_engine_command: float = 0.0
    settled_s: float = 0.0

    def update(self, *, dt_s, soc_before, soc_after, current_a,
               nominal_capacity_ah, command_kw, engine_kw, nominal_cap_kw,
               speed_error_mps, altitude_error_m, soc_min=0.2, soc_max=1.0,
               coulombic_efficiency=1.0):
        values = (dt_s, soc_before, soc_after, current_a, nominal_capacity_ah,
                  command_kw, engine_kw, nominal_cap_kw, speed_error_mps, altitude_error_m)
        if not all(math.isfinite(v) for v in values) or dt_s <= 0 or nominal_capacity_ah <= 0:
            raise ValueError("Health observations must be finite with positive dt/capacity.")
        self.updates += 1
        self.tracking_risk = 0.9*self.tracking_risk + 0.1*min(
            1.0, abs(speed_error_mps)/10 + abs(altitude_error_m)/500)
        delta = soc_before - soc_after
        if (abs(current_a) > 0.5 and abs(delta) > 1e-7
                and soc_min+1e-5 < soc_after < soc_max-1e-5):
            fraction = coulombic_efficiency*current_a*dt_s/(3600*nominal_capacity_ah*delta)
            if 0.1 <= fraction <= 1.10:
                self.battery_fraction += 0.35*(min(1.0, fraction)-self.battery_fraction)
                self.capacity_observations += 1
        self.settled_s = (
            self.settled_s + dt_s if abs(command_kw-self.last_engine_command) < 0.2 else 0.0
        )
        if nominal_cap_kw > 0 and command_kw >= 0.9*nominal_cap_kw and self.settled_s >= 10:
            ratio = np.clip(engine_kw/nominal_cap_kw, 0.1, 1.0)
            self.engine_fraction += 0.2*(float(ratio)-self.engine_fraction)
            self.engine_observations += 1
        self.last_engine_command = command_kw

    def health(self):
        return HealthState(
            engine_available_fraction=self.engine_fraction,
            battery_available_energy_fraction=self.battery_fraction,
        )


class EstimatedHealthCoordinator:
    """Health estimates constrain decisions without modifying physical health."""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.estimator = ResidualHealthEstimator()

    def reset(self):
        self.coordinator.reset()
        self.estimator = ResidualHealthEstimator()

    def update_feedback(self, **measurements):
        self.estimator.update(**measurements)

    def dispatch(self, observation):
        from dataclasses import replace
        protected = replace(
            observation,
            battery_soc_reserve=min(
                observation.battery_soc_max,
                observation.battery_soc_reserve+0.05*self.estimator.tracking_risk,
            ),
        )
        result = FaultTolerantCoordinator(self.coordinator, self.estimator.health()).dispatch(protected)
        # Physical isolation belongs exclusively to the plant fault source.
        return replace(result, battery_available_energy_fraction=1.0)


class PhaseScheduledMPC:
    """Switch parameter sets while retaining demand and actuator history."""

    def __init__(self, phase_parameters=None):
        self.phase_parameters = phase_parameters or {}
        self.controller = MPCEnergyManagementCoordinator()
        self.active_phase = None

    def reset(self):
        self.controller.reset()
        self.active_phase = None

    def dispatch(self, observation):
        if observation.phase_name != self.active_phase:
            parameters = MPCParameters(**self.phase_parameters.get(observation.phase_name, {}))
            fresh = MPCEnergyManagementCoordinator(parameters)
            for name in ("_previous_demand_kw", "_demand_slope_kwps", "_previous_engine_target_kw"):
                setattr(fresh, name, getattr(self.controller, name))
            self.controller = fresh
            self.active_phase = observation.phase_name
        return self.controller.dispatch(observation)


def glide_command(speed_mps, target_speed_mps, drag_n, weight_n, altitude_m):
    """Energy-consistent descent with speed priority and a terminal flare."""
    if min(speed_mps, target_speed_mps, weight_n) <= 0:
        raise ValueError("Positive speed and weight required.")
    sink_feedforward = -drag_n/weight_n * speed_mps
    speed_feedback = 0.35*(speed_mps-target_speed_mps)
    descent = float(np.clip(sink_feedforward+speed_feedback, -5.0, 0.5))
    if altitude_m < 12:
        descent = max(descent, -max(0.35, 0.08*altitude_m))
    return descent
