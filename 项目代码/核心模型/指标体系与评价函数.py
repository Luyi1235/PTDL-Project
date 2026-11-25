"""PTDL 2.2.1 multi-objective indicator and constraint evaluator.

This module evaluates a simulated full-mission time history. It does not create
the flight dynamics itself: upstream models may be a simplified Python model,
RCAIDE, or a coupled propulsion/flight-control model.

All values in the demo configuration are illustrative only. Replace them with
the validated PTDL aircraft, propulsion, battery, and mission requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


FUEL_LHV_MJ_PER_KG = 43.0


@dataclass(frozen=True)
class MissionRequirements:
    """Mission-level requirements. All constraint functions use g(z) <= 0."""

    endurance_min_h: float = 80.0
    terminal_soc_min: float = 0.20
    soc_min: float = 0.15
    fuel_reserve_min_kg: Optional[float] = None
    range_min_m: Optional[float] = None
    loiter_min_s: Optional[float] = None
    terminal_position_error_max_m: Optional[float] = None


@dataclass(frozen=True)
class FlightEnvelope:
    speed_min_mps: Optional[float] = None
    speed_max_mps: Optional[float] = None
    altitude_min_m: Optional[float] = None
    altitude_max_m: Optional[float] = None
    angle_of_attack_min_rad: Optional[float] = None
    angle_of_attack_max_rad: Optional[float] = None
    load_factor_max: Optional[float] = None


@dataclass(frozen=True)
class PowerLimits:
    engine_power_min_kw: float = 0.0
    engine_power_max_kw: Optional[float] = None
    motor_power_min_kw: Optional[float] = None
    motor_power_max_kw: Optional[float] = None
    engine_power_rate_max_kwps: Optional[float] = None
    motor_power_rate_max_kwps: Optional[float] = None


@dataclass(frozen=True)
class BatteryLimits:
    discharge_power_max_kw: Optional[float] = None
    charge_power_max_kw: Optional[float] = None
    current_abs_max_a: Optional[float] = None
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    discharge_efficiency: float = 0.95
    charge_efficiency: float = 0.95


@dataclass(frozen=True)
class ControlQualitySettings:
    height_error_allow_m: float = 30.0
    speed_error_allow_mps: float = 3.0
    pitch_error_allow_rad: float = np.deg2rad(3.0)
    pitch_rate_allow_radps: float = np.deg2rad(5.0)
    height_weight: float = 0.30
    speed_weight: float = 0.30
    pitch_weight: float = 0.20
    pitch_rate_weight: float = 0.20
    peak_weight: float = 0.50
    recovery_time_weight: float = 1.0
    switch_time_weight: float = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    mission: MissionRequirements
    flight: FlightEnvelope
    power: PowerLimits
    battery: BatteryLimits
    control: ControlQualitySettings = field(default_factory=ControlQualitySettings)


@dataclass
class EvaluationResult:
    objectives: dict[str, float]
    constraints: dict[str, float]
    diagnostics: dict[str, float]

    @property
    def feasible(self) -> bool:
        return all(value <= 0.0 for value in self.constraints.values())

    @property
    def total_violation(self) -> float:
        return float(sum(max(0.0, value) for value in self.constraints.values()))

    def to_record(self) -> dict[str, float | bool]:
        record: dict[str, float | bool] = {}
        record.update(self.objectives)
        record.update({f"g_{name}": value for name, value in self.constraints.items()})
        record.update(self.diagnostics)
        record["feasible"] = self.feasible
        record["total_violation"] = self.total_violation
        return record


@dataclass(frozen=True)
class ObjectiveNormalizer:
    """Fixed reference normalization for objective comparisons and MOEA/D."""

    labels: tuple[str, ...]
    ideal: np.ndarray
    reference_worst: np.ndarray

    @classmethod
    def from_samples(
        cls,
        samples: pd.DataFrame,
        labels: Sequence[str] = ("J_energy_mj", "J_stability", "J_response_s"),
        worst_quantile: float = 0.95,
    ) -> "ObjectiveNormalizer":
        if not 0.0 < worst_quantile <= 1.0:
            raise ValueError("worst_quantile must be in (0, 1].")
        values = samples.loc[:, labels].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Samples for normalization must be finite.")
        ideal = np.min(values, axis=0)
        reference_worst = np.quantile(values, worst_quantile, axis=0)
        span = reference_worst - ideal
        scale = np.maximum(np.maximum(np.abs(ideal), np.abs(reference_worst)), 1.0)
        minimum_span = np.maximum(16.0 * np.spacing(scale), 1e-12)
        reference_worst = np.where(
            span > minimum_span,
            reference_worst,
            ideal + minimum_span,
        )
        return cls(tuple(labels), ideal, reference_worst)

    def transform(self, objectives: Mapping[str, float]) -> dict[str, float]:
        raw = np.asarray([objectives[label] for label in self.labels], dtype=float)
        span = self.reference_worst - self.ideal
        scale = np.maximum(
            np.maximum(np.abs(self.ideal), np.abs(self.reference_worst)), 1.0
        )
        minimum_span = np.maximum(16.0 * np.spacing(scale), 1e-12)
        normalized = np.zeros_like(raw)
        np.divide(
            raw - self.ideal,
            span,
            out=normalized,
            where=span > minimum_span,
        )
        return {label: float(value) for label, value in zip(self.labels, normalized)}


def generate_simplex_weights(resolution: int = 4) -> pd.DataFrame:
    """Return a complete three-objective simplex weight scan for MOEA/D."""
    if resolution < 1:
        raise ValueError("resolution must be at least 1.")
    weights = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            weights.append((i / resolution, j / resolution, k / resolution))
    return pd.DataFrame(weights, columns=["w_energy", "w_stability", "w_response"])


def weighted_preference_score(
    normalized_objectives: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Post-process a feasible Pareto solution; never replace Pareto sorting."""
    if set(normalized_objectives) != set(weights):
        raise ValueError("Objective and weight labels must match exactly.")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0):
        raise ValueError("Preference weights must sum to 1.")
    return float(sum(normalized_objectives[label] * weights[label] for label in weights))


def evaluate_mission(
    profile: pd.DataFrame,
    config: EvaluationConfig,
    response_metrics: Optional[Mapping[str, float]] = None,
    terminal_fuel_kg: Optional[float] = None,
    range_m: Optional[float] = None,
    loiter_s: Optional[float] = None,
    terminal_position_error_m: Optional[float] = None,
) -> EvaluationResult:
    """Evaluate one candidate control strategy.

    Required profile columns are: time_s, fuel_flow_kgps, battery_discharge_kw,
    battery_charge_kw, soc, h_m, v_mps, p_engine_kw, p_motor_kw, e_h_m,
    e_v_mps, e_theta_rad, q_radps. Optional columns are alpha_rad,
    load_factor, battery_current_a and battery_temperature_c.
    """
    required = (
        "time_s", "fuel_flow_kgps", "battery_discharge_kw", "battery_charge_kw",
        "soc", "h_m", "v_mps", "p_engine_kw", "p_motor_kw", "e_h_m",
        "e_v_mps", "e_theta_rad", "q_radps",
    )
    missing = sorted(set(required).difference(profile.columns))
    if missing:
        raise ValueError(f"Profile is missing required columns: {', '.join(missing)}")

    data = profile.sort_values("time_s").reset_index(drop=True)
    time_s = data["time_s"].to_numpy(dtype=float)
    if len(time_s) < 2 or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("time_s must contain at least two strictly increasing values.")
    if not np.isfinite(data.loc[:, required].to_numpy(dtype=float)).all():
        raise ValueError("Required profile values must be finite.")

    duration_s = float(time_s[-1] - time_s[0])
    duration_h = duration_s / 3600.0
    fuel_used_kg = float(np.trapezoid(data["fuel_flow_kgps"].to_numpy(dtype=float), time_s))
    battery_net_mj = float(
        np.trapezoid(
            data["battery_discharge_kw"].to_numpy(dtype=float) / config.battery.discharge_efficiency
            - config.battery.charge_efficiency * data["battery_charge_kw"].to_numpy(dtype=float),
            time_s,
        ) / 1000.0
    )
    battery_discharge_bus_mj = float(
        np.trapezoid(data["battery_discharge_kw"].to_numpy(dtype=float), time_s) / 1000.0
    )
    battery_charge_bus_mj = float(
        np.trapezoid(data["battery_charge_kw"].to_numpy(dtype=float), time_s) / 1000.0
    )
    battery_discharge_cell_mj = (
        battery_discharge_bus_mj / config.battery.discharge_efficiency
    )
    battery_charge_cell_mj = (
        battery_charge_bus_mj * config.battery.charge_efficiency
    )
    battery_conversion_loss_mj = max(
        0.0,
        battery_discharge_cell_mj
        - battery_discharge_bus_mj
        + battery_charge_bus_mj
        - battery_charge_cell_mj,
    )
    battery_throughput_input_mj = battery_discharge_cell_mj + battery_charge_bus_mj
    battery_energy_management_efficiency = (
        (battery_discharge_bus_mj + battery_charge_cell_mj)
        / battery_throughput_input_mj
        if battery_throughput_input_mj > 1e-12
        else 1.0
    )
    energy_mj = fuel_used_kg * FUEL_LHV_MJ_PER_KG + battery_net_mj

    control = config.control
    e_h = data["e_h_m"].to_numpy(dtype=float) / control.height_error_allow_m
    e_v = data["e_v_mps"].to_numpy(dtype=float) / control.speed_error_allow_mps
    e_theta = data["e_theta_rad"].to_numpy(dtype=float) / control.pitch_error_allow_rad
    q = data["q_radps"].to_numpy(dtype=float) / control.pitch_rate_allow_radps
    integrated_tracking = np.trapezoid(
        control.height_weight * e_h**2
        + control.speed_weight * e_v**2
        + control.pitch_weight * e_theta**2
        + control.pitch_rate_weight * q**2,
        time_s,
    ) / duration_s
    peak_error = float(max(np.max(np.abs(e_h)), np.max(np.abs(e_v)), np.max(np.abs(e_theta))))
    stability = float(integrated_tracking + control.peak_weight * peak_error)

    response_metrics = response_metrics or {}
    settling_s = float(response_metrics.get("settling_time_s", 0.0))
    recovery_s = float(response_metrics.get("recovery_time_s", 0.0))
    switching_s = float(response_metrics.get("mode_switch_time_s", 0.0))
    recovery_success = response_metrics.get("recovery_success")
    failure_penalty_s = (
        float(response_metrics.get("failure_penalty_s", 0.0))
        if recovery_success is not None and recovery_success < 1.0
        else 0.0
    )
    response_s = (
        control.recovery_time_weight * recovery_s
        + control.switch_time_weight * switching_s
        + failure_penalty_s
    )

    constraints: dict[str, float] = {
        "endurance_80h": config.mission.endurance_min_h - duration_h,
        "terminal_soc": config.mission.terminal_soc_min - float(data["soc"].iloc[-1]),
        "minimum_soc": config.mission.soc_min - float(data["soc"].min()),
    }
    _add_interval_constraint(constraints, "speed", data["v_mps"], config.flight.speed_min_mps, config.flight.speed_max_mps)
    _add_interval_constraint(constraints, "altitude", data["h_m"], config.flight.altitude_min_m, config.flight.altitude_max_m)
    _add_interval_constraint(constraints, "engine_power", data["p_engine_kw"], config.power.engine_power_min_kw, config.power.engine_power_max_kw)
    _add_interval_constraint(constraints, "motor_power", data["p_motor_kw"], config.power.motor_power_min_kw, config.power.motor_power_max_kw)
    _add_upper_constraint(constraints, "battery_discharge_power", data["battery_discharge_kw"], config.battery.discharge_power_max_kw)
    _add_upper_constraint(constraints, "battery_charge_power", data["battery_charge_kw"], config.battery.charge_power_max_kw)
    _add_rate_constraint(constraints, "engine_power_rate", data["p_engine_kw"], time_s, config.power.engine_power_rate_max_kwps)
    _add_rate_constraint(constraints, "motor_power_rate", data["p_motor_kw"], time_s, config.power.motor_power_rate_max_kwps)
    if "alpha_rad" in data:
        _add_interval_constraint(constraints, "angle_of_attack", data["alpha_rad"], config.flight.angle_of_attack_min_rad, config.flight.angle_of_attack_max_rad)
    if "load_factor" in data:
        _add_upper_constraint(constraints, "load_factor", data["load_factor"], config.flight.load_factor_max)
    if "battery_current_a" in data:
        _add_upper_constraint(constraints, "battery_current", np.abs(data["battery_current_a"]), config.battery.current_abs_max_a)
    if "battery_temperature_c" in data:
        _add_interval_constraint(constraints, "battery_temperature", data["battery_temperature_c"], config.battery.temperature_min_c, config.battery.temperature_max_c)
    if config.mission.fuel_reserve_min_kg is not None and terminal_fuel_kg is not None:
        constraints["terminal_fuel"] = config.mission.fuel_reserve_min_kg - terminal_fuel_kg
    if config.mission.range_min_m is not None and range_m is not None:
        constraints["range"] = config.mission.range_min_m - range_m
    if config.mission.loiter_min_s is not None and loiter_s is not None:
        constraints["loiter"] = config.mission.loiter_min_s - loiter_s
    if config.mission.terminal_position_error_max_m is not None and terminal_position_error_m is not None:
        constraints["terminal_position_error"] = terminal_position_error_m - config.mission.terminal_position_error_max_m

    diagnostics = {
        "endurance_h": duration_h,
        "endurance_margin_h": duration_h - config.mission.endurance_min_h,
        "fuel_used_kg": fuel_used_kg,
        "battery_net_energy_mj": battery_net_mj,
        "battery_discharge_bus_energy_mj": battery_discharge_bus_mj,
        "battery_charge_bus_energy_mj": battery_charge_bus_mj,
        "battery_conversion_loss_mj": battery_conversion_loss_mj,
        "battery_energy_management_efficiency": float(
            np.clip(battery_energy_management_efficiency, 0.0, 1.0)
        ),
        "battery_efficiency_assessment_valid": float(
            battery_throughput_input_mj > 1e-12
        ),
        "terminal_soc": float(data["soc"].iloc[-1]),
        "minimum_soc": float(data["soc"].min()),
        "max_engine_power_kw": float(data["p_engine_kw"].max()),
        "max_motor_power_kw": float(data["p_motor_kw"].max()),
        "max_battery_discharge_kw": float(data["battery_discharge_kw"].max()),
        "max_battery_charge_kw": float(data["battery_charge_kw"].max()),
        "max_height_error_m": float(np.abs(data["e_h_m"]).max()),
        "max_speed_error_mps": float(np.abs(data["e_v_mps"]).max()),
        "max_pitch_error_rad": float(np.abs(data["e_theta_rad"]).max()),
        "max_pitch_rate_radps": float(np.abs(data["q_radps"]).max()),
        "response_phase_settling_time_s": settling_s,
        "response_disturbance_recovery_time_s": recovery_s,
        "response_mode_switch_time_s": switching_s,
        "response_failure_penalty_s": failure_penalty_s,
    }
    return EvaluationResult(
        objectives={"J_energy_mj": energy_mj, "J_stability": stability, "J_response_s": response_s},
        constraints=constraints,
        diagnostics=diagnostics,
    )


def _add_interval_constraint(
    constraints: dict[str, float], name: str, values: pd.Series, lower: Optional[float], upper: Optional[float]
) -> None:
    if lower is not None:
        constraints[f"{name}_min"] = lower - float(values.min())
    if upper is not None:
        constraints[f"{name}_max"] = float(values.max()) - upper


def _add_upper_constraint(constraints: dict[str, float], name: str, values: pd.Series, upper: Optional[float]) -> None:
    if upper is not None:
        constraints[f"{name}_max"] = float(np.max(values)) - upper


def _add_rate_constraint(
    constraints: dict[str, float], name: str, values: pd.Series, time_s: np.ndarray, upper: Optional[float]
) -> None:
    if upper is not None:
        rate = np.abs(np.diff(values.to_numpy(dtype=float)) / np.diff(time_s))
        excess = float(rate.max()) - upper
        tolerance = 1e-10 * max(1., abs(upper))
        constraints[f"{name}_max"] = 0. if abs(excess) <= tolerance else excess


def demo_configuration() -> EvaluationConfig:
    """Illustrative configuration for unit testing only; not PTDL design data."""
    return EvaluationConfig(
        mission=MissionRequirements(endurance_min_h=80.0, terminal_soc_min=0.20, soc_min=0.15),
        flight=FlightEnvelope(speed_min_mps=20.0, speed_max_mps=55.0, altitude_min_m=0.0, altitude_max_m=6000.0),
        power=PowerLimits(engine_power_max_kw=120.0, motor_power_min_kw=-40.0, motor_power_max_kw=80.0, engine_power_rate_max_kwps=5.0, motor_power_rate_max_kwps=8.0),
        battery=BatteryLimits(discharge_power_max_kw=80.0, charge_power_max_kw=40.0, current_abs_max_a=250.0, temperature_min_c=0.0, temperature_max_c=50.0),
    )


if __name__ == "__main__":
    time_s = np.linspace(0.0, 80.5 * 3600.0, 161)
    demo_profile = pd.DataFrame(
        {
            "time_s": time_s,
            "fuel_flow_kgps": np.full_like(time_s, 0.00017),
            "battery_discharge_kw": np.full_like(time_s, 2.0),
            "battery_charge_kw": np.zeros_like(time_s),
            "soc": np.linspace(0.90, 0.25, len(time_s)),
            "h_m": np.full_like(time_s, 2500.0),
            "v_mps": np.full_like(time_s, 35.0),
            "p_engine_kw": np.full_like(time_s, 65.0),
            "p_motor_kw": np.full_like(time_s, 2.0),
            "e_h_m": np.zeros_like(time_s),
            "e_v_mps": np.zeros_like(time_s),
            "e_theta_rad": np.zeros_like(time_s),
            "q_radps": np.zeros_like(time_s),
        }
    )
    demo_result = evaluate_mission(demo_profile, demo_configuration(), {"settling_time_s": 20.0})
    print(pd.Series(demo_result.to_record()).to_string())
