"""PTDL 2.2.2 constrained multi-objective control optimization prototype.

This module implements a reproducible algorithm-level delivery for the PTDL
project.  NSGA-II is the baseline optimizer and MOEA/D is the comparison
optimizer.  Both call the 2.2.1 evaluator through one common problem class.

The built-in mission simulator is intentionally a low-order surrogate used to
verify the optimization chain before the real coupled model is available.  It
is NOT an aircraft-performance prediction model and its constants are NOT PTDL
design data.  Replace ``simulate_surrogate_mission`` with an RCAIDE or coupled
flight-propulsion-control adapter when validated models become available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from 指标体系与评价函数 import (
    EvaluationResult,
    ObjectiveNormalizer,
    demo_configuration,
    evaluate_mission,
    generate_simplex_weights,
    weighted_preference_score,
)
from 飞推控闭环物理模型 import (
    AircraftParameters,
    FlightControlParameters,
    HybridDispatchPolicy,
    HybridPowertrainModel,
    LongitudinalDragDisturbance,
    MissionDefinition,
    evaluate_coupled_mission,
    reference_endurance_mission,
)
from 机体参数化设计与气动阻力 import (
    AeroCoefficients,
    AirframeDesign,
    aero_from_design,
    geometry_constraints,
)
from 全任务剖面定义 import (
    DEFAULT_PROFILE_PATH,
    PreliminaryPropulsionAssumptions,
    build_full_task_mission,
    load_source_profile,
    preliminary_hybrid_aircraft,
    load_preliminary_propulsion_config,
    source_profile_duration_h,
)
from 固定起飞质量尺寸标定 import calibrate_dimension_baseline
from ULTRA基准标定 import UltraCalibration
from 八十小时续航任务模型 import (
    DEFAULT_ENDURANCE_PROFILE_PATH,
    ULTRA_PROPULSION_CONFIG,
    build_endurance_mission,
    calibrate_endurance_task_baseline,
    evaluate_endurance_task,
    load_endurance_profile,
)
from 响应指标 import apply_paired_response_metrics, zero_amplitude_disturbances


OBJECTIVE_LABELS = ("J_energy_mj", "J_stability", "J_response_s")
DECISION_LABELS = (
    "duration_h",
    "cruise_speed_mps",
    "cruise_altitude_m",
    "engine_power_climb_kw",
    "engine_power_cruise_kw",
    "engine_power_return_kw",
    "motor_power_climb_kw",
    "motor_power_cruise_kw",
    "motor_power_return_kw",
    "control_gain",
)
# The proxy optimizes motor assist only.  Charging needs an explicit engine-
# generator fuel map, so it is intentionally excluded until the real model is
# connected; allowing negative motor power here would create free energy.
LOWER_BOUNDS = np.array([79.0, 28.0, 1000.0, 45.0, 45.0, 45.0, 0.0, 0.0, 0.0, 0.60])
UPPER_BOUNDS = np.array([82.0, 45.0, 4500.0, 105.0, 105.0, 105.0, 35.0, 35.0, 35.0, 1.80])


@dataclass(frozen=True)
class CandidateEvaluation:
    decision: np.ndarray
    result: EvaluationResult

    @property
    def objectives(self) -> np.ndarray:
        return np.asarray([self.result.objectives[label] for label in OBJECTIVE_LABELS], dtype=float)

    @property
    def violation(self) -> float:
        return self.result.total_violation

    @property
    def feasible(self) -> bool:
        return self.result.feasible


@dataclass
class OptimizerRun:
    algorithm: str
    decisions: np.ndarray
    evaluations: list[CandidateEvaluation]
    history: pd.DataFrame
    evaluated_archive: list[CandidateEvaluation] | None = None
    generation_pareto_archive: dict[int, list[CandidateEvaluation]] | None = None

    @property
    def objectives(self) -> np.ndarray:
        return np.vstack([item.objectives for item in self.evaluations])

    @property
    def violations(self) -> np.ndarray:
        return np.asarray([item.violation for item in self.evaluations], dtype=float)


class PTDLControlProblem:
    """Decision-to-mission adapter shared by all optimizers."""

    lower_bounds = LOWER_BOUNDS
    upper_bounds = UPPER_BOUNDS
    decision_labels = DECISION_LABELS

    def __init__(self) -> None:
        self.config = demo_configuration()

    def evaluate(self, decision: Sequence[float]) -> CandidateEvaluation:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        profile, response_metrics = simulate_surrogate_mission(x)
        result = evaluate_mission(profile, self.config, response_metrics=response_metrics)
        return CandidateEvaluation(x, result)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds, size=(count, len(self.lower_bounds)))


class CoupledPTDLControlProblem:
    """MOEA/D adapter using the reconstructed flight-propulsion-control model.

    Unlike ``PTDLControlProblem``, this class does not use the synthetic
    optimization landscape.  Its candidate evaluation is the real low-order
    coupled-model calculation.  Aircraft parameters remain externally
    configurable because the reference values are not PTDL design data.
    """

    decision_labels = (
        "cruise_speed_mps",
        "cruise_altitude_m",
        "engine_limit_climb_kw",
        "engine_limit_loiter_kw",
        "engine_limit_return_kw",
        "motor_assist_climb_kw",
        "motor_assist_loiter_kw",
        "motor_assist_return_kw",
    )

    def __init__(self, aircraft: AircraftParameters = AircraftParameters(), loiter_duration_h: float = 80.0) -> None:
        self.aircraft = aircraft
        self.loiter_duration_h = loiter_duration_h
        self.lower_bounds = np.array([
            58.0, 6500.0, 80.0, 35.0, 20.0, 0.0,
            aircraft.motor_power_min_kw, aircraft.motor_power_min_kw,
        ])
        self.upper_bounds = np.array([
            78.0, 8200.0, aircraft.engine_electrical_power_max_kw,
            min(120.0, aircraft.engine_electrical_power_max_kw),
            min(100.0, aircraft.engine_electrical_power_max_kw),
            aircraft.motor_power_max_kw, aircraft.motor_power_max_kw,
            aircraft.motor_power_max_kw,
        ])

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds, size=(count, len(self.lower_bounds)))

    def evaluate(self, decision: Sequence[float]) -> CandidateEvaluation:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        mission = _coupled_mission_from_decision(x, self.loiter_duration_h)
        _, result = evaluate_coupled_mission(mission, self.aircraft)
        return CandidateEvaluation(x, result)


class SourceProfileControlProblem:
    """MOEA/D control adapter for the supplied ULTRA checkout profile.

    The source profile's 70 m/s ISR requirement is preserved.  With an 80 h
    target this is expected to be a feasibility audit under the current MTOW,
    not a claim that control optimization alone can overcome the energy budget.
    """

    phase_labels = (
        "main_climb_to_leveloff",
        "short_leveloff_to_isr_altitude",
        "isr_cruise_alpha0_5000km",
        "long_descent",
        "approach_descent",
        "flare",
    )
    decision_labels = tuple(f"engine_cap_{label}_kw" for label in phase_labels) + tuple(
        f"motor_assist_{label}_kw" for label in phase_labels
    )

    def __init__(
        self,
        design: AirframeDesign = AirframeDesign(payload_mass_kg=100.0, hybrid_system_mass_kg=120.0),
        profile_path: str | Path = DEFAULT_PROFILE_PATH,
        target_total_duration_h: float = 80.0,
        assumptions: PreliminaryPropulsionAssumptions | None = None,
    ) -> None:
        self.design = design
        self.profile_path = Path(profile_path)
        self.target_total_duration_h = target_total_duration_h
        self.assumptions = assumptions or load_preliminary_propulsion_config()[0]
        self.sizing_config = load_preliminary_propulsion_config()[1]
        self.calibration = calibrate_dimension_baseline()
        self.source_profile = load_source_profile(profile_path)
        self.base_mission = build_full_task_mission(self.source_profile, target_total_duration_h)
        self.aircraft = preliminary_hybrid_aircraft(design, self.calibration, self.assumptions)
        engine_lower = np.array([70.0, 65.0, 20.0, 10.0, 25.0, 20.0])
        engine_upper = np.full(len(self.phase_labels), self.aircraft.engine_electrical_power_max_kw)
        motor_upper = np.full(len(self.phase_labels), min(30.0, self.aircraft.motor_power_max_kw))
        self.lower_bounds = np.concatenate((np.minimum(engine_lower, engine_upper), np.zeros(len(self.phase_labels))))
        self.upper_bounds = np.concatenate((engine_upper, motor_upper))

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        samples = rng.uniform(self.lower_bounds, self.upper_bounds, size=(count, len(self.lower_bounds)))
        if count:
            # Feasibility anchor: maximum available engine schedule with no
            # battery use.  It gives MOEA/D a fuel-limited reference before
            # it explores battery-assist schedules.
            phase_count = len(self.phase_labels)
            samples[0, :phase_count] = self.upper_bounds[:phase_count]
            samples[0, phase_count:] = 0.0
        return samples

    def mission_from_decision(self, decision: Sequence[float]) -> MissionDefinition:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        engine_caps = x[: len(self.phase_labels)]
        motor_assist = x[len(self.phase_labels) :]
        phases = tuple(
            replace(phase, engine_power_kw=float(engine), motor_power_kw=float(motor))
            for phase, engine, motor in zip(self.base_mission.phases, engine_caps, motor_assist)
        )
        return replace(self.base_mission, phases=phases)

    def evaluate(self, decision: Sequence[float]) -> CandidateEvaluation:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        simulation, result = evaluate_coupled_mission(self.mission_from_decision(x), self.aircraft)
        result.constraints.update(geometry_constraints(self.design, self.calibration.reference))
        result.diagnostics.update(
            {
                "achieved_total_duration_h": float(simulation.profile["time_s"].iloc[-1] / 3600.0),
                "source_profile_duration_h": source_profile_duration_h(self.source_profile),
                "requested_total_duration_h": self.target_total_duration_h,
            }
        )
        return CandidateEvaluation(x, result)


class PTDLEnduranceJointProblem:
    """Joint dimension, mass-allocation, and control problem for PTDL 80 h."""

    design_labels = (
        "wing_area_m2",
        "aspect_ratio",
        "taper_ratio",
        "sweep_deg",
        "tip_twist_deg",
        "fuselage_length_m",
        "fuselage_diameter_m",
        "propeller_diameter_m",
        "payload_mass_kg",
        "hybrid_system_mass_kg",
    )

    def __init__(
        self,
        profile_path: str | Path = DEFAULT_ENDURANCE_PROFILE_PATH,
        aero_provider: Callable[[AirframeDesign, UltraCalibration], AeroCoefficients] | None = None,
        powertrain_factory: Callable[[], HybridPowertrainModel] | None = None,
        dispatch_policy_factory: Callable[[], HybridDispatchPolicy] | None = None,
        seed_decisions: Sequence[Sequence[float]] = (),
        disturbances: Sequence[LongitudinalDragDisturbance] = (),
        assumptions: PreliminaryPropulsionAssumptions | None = None,
        design_feature_bounds: Mapping[str, tuple[float, float]] | None = None,
        uncertainty_scenarios: Sequence[Mapping[str, object]] = (),
        sixdof_validator: Callable[..., Mapping[str, Mapping[str, float]]] | None = None,
        stepwise_aero_validator: Callable[..., Mapping[str, object]] | None = None,
        control_parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        response_definition: str = "legacy",
    ) -> None:
        self.profile_path = Path(profile_path)
        self.profile = load_endurance_profile(profile_path)
        self.task_calibration = calibrate_endurance_task_baseline()
        self.calibration = self.task_calibration.dimension_calibration
        self.assumptions = assumptions or self.task_calibration.assumptions
        self.sizing_config = ULTRA_PROPULSION_CONFIG
        self.aero_provider = aero_provider
        self.sixdof_validator = sixdof_validator
        self.stepwise_aero_validator = stepwise_aero_validator
        self.control_parameter_bounds = dict(control_parameter_bounds or {})
        if response_definition not in {"legacy", "recovery-integral"}:
            raise ValueError("Unknown response objective definition")
        self.response_definition = response_definition
        allowed_controls = {"speed_tracking_time_constant_s", "gamma_tracking_gain",
                            "alpha_time_constant_s"}
        for name, limits in self.control_parameter_bounds.items():
            if name not in allowed_controls or len(limits) != 2:
                raise ValueError(f"Unsupported flight-control parameter: {name}")
            if not np.isfinite(limits).all() or not 0 < limits[0] < limits[1]:
                raise ValueError("Flight-control bounds must be finite positive intervals")
        self.powertrain_factory = powertrain_factory
        self.dispatch_policy_factory = dispatch_policy_factory
        self.disturbances = tuple(disturbances)
        self.uncertainty_scenarios = tuple(dict(item) for item in uncertainty_scenarios)
        allowed_scenario_keys = {
            "label",
            "engine_rating_factor",
            "motor_rating_factor",
            "battery_energy_factor",
            "engine_efficiency_factor",
        }
        for scenario in self.uncertainty_scenarios:
            unknown = set(scenario).difference(allowed_scenario_keys)
            if unknown:
                raise ValueError(f"Unknown robust-scenario fields: {sorted(unknown)}")
            for key in allowed_scenario_keys.difference({"label"}):
                value = float(scenario.get(key, 1.0))
                if not np.isfinite(value) or not 0.5 <= value <= 1.5:
                    raise ValueError(f"{key} must be a finite multiplier in [0.5, 1.5].")
        self.seed_decisions = tuple(np.asarray(item, dtype=float) for item in seed_decisions)
        self.design_feature_bounds = dict(design_feature_bounds or {})
        self.base_mission = build_endurance_mission(self.profile)
        self.phase_labels = tuple(phase.name for phase in self.base_mission.phases)
        self.decision_labels = (
            self.design_labels
            + ("isr_cruise_speed_mps",)
            + tuple(f"engine_cap_{label}_kw" for label in self.phase_labels)
            + tuple(f"motor_assist_{label}_kw" for label in self.phase_labels)
        )
        design_lower = np.array([22.0, 18.0, 0.30, 0.0, -3.0, 7.5, 0.65, 2.1, 100.0, 0.0])
        design_upper = np.array([40.0, 31.0, 0.70, 8.0, 1.5, 10.0, 0.95, 3.0, 205.0, 180.0])
        for label, (feature_min, feature_max) in self.design_feature_bounds.items():
            if label not in self.design_labels[:7] or not np.isfinite([feature_min, feature_max]).all():
                raise ValueError(f"Invalid aerodynamic feature bound for {label!r}.")
            if feature_min >= feature_max:
                raise ValueError(f"Aerodynamic feature bounds for {label!r} must be increasing.")
            feature_index = self.design_labels.index(label)
            design_lower[feature_index] = max(design_lower[feature_index], feature_min)
            design_upper[feature_index] = min(design_upper[feature_index], feature_max)
            if design_lower[feature_index] >= design_upper[feature_index]:
                raise ValueError(f"Aerodynamic feature bounds do not overlap the design range for {label!r}.")
        engine_lower = np.asarray([segment["engine_power_bounds_kw"][0] for segment in self.profile["segments"]], dtype=float)
        engine_upper = np.asarray(
            [segment["engine_power_bounds_kw"][1] for segment in self.profile["segments"]],
            dtype=float,
        ) * self.assumptions.engine_rating_multiplier
        motor_upper = np.asarray(
            [segment["motor_assist_bounds_kw"][1] for segment in self.profile["segments"]],
            dtype=float,
        ) * self.assumptions.motor_rating_multiplier
        motor_upper = np.asarray(
            [
                0.0 if segment.get("motor_disabled", False) else upper
                for segment, upper in zip(self.profile["segments"], motor_upper)
            ],
            dtype=float,
        )
        self.lower_bounds = np.concatenate((design_lower, [30.0], engine_lower, np.zeros(len(self.phase_labels))))
        self.upper_bounds = np.concatenate((design_upper, [42.0], engine_upper, motor_upper))
        if self.control_parameter_bounds:
            self.decision_labels += tuple(self.control_parameter_bounds)
            self.lower_bounds = np.concatenate((self.lower_bounds,
                                                [b[0] for b in self.control_parameter_bounds.values()]))
            self.upper_bounds = np.concatenate((self.upper_bounds,
                                                [b[1] for b in self.control_parameter_bounds.values()]))

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        samples = rng.uniform(self.lower_bounds, self.upper_bounds, size=(count, len(self.lower_bounds)))
        if count:
            # The screened high-aspect-ratio design with a moderate hybrid
            # package is a transparent feasibility anchor, not a claimed optimum.
            samples[0, :10] = np.array([31.988386, 28.921683, 0.322985, 3.026379, 0.216925, 8.241671, 0.707050, 2.152803, 100.0, 0.0])
            samples[0, 10] = 30.0
            phase_count = len(self.phase_labels)
            samples[0, 11 : 11 + phase_count] = self.upper_bounds[11 : 11 + phase_count]
            samples[0, 11 + phase_count : 11 + 2 * phase_count] = 0.0
            defaults = FlightControlParameters()
            samples[0, 11 + 2 * phase_count:] = [
                np.clip(getattr(defaults, k), *bounds)
                for k, bounds in self.control_parameter_bounds.items()
            ]
            for index, decision in enumerate(self.seed_decisions[:count]):
                if self.control_parameter_bounds and decision.shape == (11 + 2 * phase_count,):
                    decision = np.concatenate((decision, samples[0, 11 + 2 * phase_count:]))
                if decision.shape != self.lower_bounds.shape:
                    raise ValueError(f"Seed decision must have {len(self.lower_bounds)} values.")
                samples[index] = np.clip(decision, self.lower_bounds, self.upper_bounds)
        return samples

    def design_from_decision(self, decision: Sequence[float]) -> AirframeDesign:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        return AirframeDesign(**{label: float(value) for label, value in zip(self.design_labels, x[: len(self.design_labels)])})

    def evaluate(self, decision: Sequence[float]) -> CandidateEvaluation:
        x = np.clip(np.asarray(decision, dtype=float), self.lower_bounds, self.upper_bounds)
        design = self.design_from_decision(x)
        phase_count = len(self.phase_labels)
        loiter_index = self.phase_labels.index("endurance_loiter")
        loiter_phase = self.base_mission.phases[loiter_index]
        loiter_speed = float(x[10])
        engine_caps = x[11 : 11 + phase_count]
        motor_assist = x[11 + phase_count : 11 + 2 * phase_count]
        flight_controller = FlightControlParameters(**dict(zip(
            self.control_parameter_bounds, x[11 + 2 * phase_count:])))
        speeds = [phase.speed_target_mps for phase in self.base_mission.phases]
        speeds[self.phase_labels.index("endurance_loiter")] = loiter_speed
        stepwise_aero_result = None
        vspaero_failure = None
        if self.stepwise_aero_validator is not None:
            stepwise_aero_result = self.stepwise_aero_validator.evaluate_candidate(
                design,
                self.calibration,
                speed_mps=loiter_speed,
                altitude_m=float(loiter_phase.altitude_target_m),
            )
            aero_override = stepwise_aero_result["aero"]
        elif self.aero_provider is None:
            aero_override = None
        elif getattr(self.aero_provider, "accepts_operating_point", False):
            aero_override = self.aero_provider(
                design,
                self.calibration,
                speed_mps=loiter_speed,
                altitude_m=float(loiter_phase.altitude_target_m),
            )
            vspaero_failure = getattr(self.aero_provider, "last_failure", None)
        else:
            aero_override = self.aero_provider(design, self.calibration)
            vspaero_failure = getattr(self.aero_provider, "last_failure", None)
        scenarios = self.uncertainty_scenarios or ({"label": "nominal"},)
        scenario_evaluations = []
        scenario_labels: list[str] = []
        for scenario_index, scenario in enumerate(scenarios):
            label = str(scenario.get("label", f"scenario_{scenario_index + 1}"))
            scenario_labels.append(label)
            scenario_assumptions = replace(
                self.assumptions,
                engine_rating_multiplier=(
                    self.assumptions.engine_rating_multiplier
                    * float(scenario.get("engine_rating_factor", 1.0))
                ),
                motor_rating_multiplier=(
                    self.assumptions.motor_rating_multiplier
                    * float(scenario.get("motor_rating_factor", 1.0))
                ),
                battery_energy_multiplier=(
                    self.assumptions.battery_energy_multiplier
                    * float(scenario.get("battery_energy_factor", 1.0))
                ),
                engine_efficiency_values=tuple(
                    float(np.clip(value * float(scenario.get("engine_efficiency_factor", 1.0)), 0.05, 0.60))
                    for value in self.assumptions.engine_efficiency_values
                ),
            )
            evaluation = evaluate_endurance_task(
                design,
                engine_caps_kw=engine_caps,
                motor_assist_kw=motor_assist,
                speeds_mps=speeds,
                profile_path=self.profile_path,
                calibration=self.calibration,
                assumptions=scenario_assumptions,
                aero_override=aero_override,
                dispatch_policy=self.dispatch_policy_factory() if self.dispatch_policy_factory is not None else None,
                powertrain=self.powertrain_factory() if self.powertrain_factory is not None else None,
                disturbances=self.disturbances,
                flight_controller=flight_controller,
            )
            if self.disturbances:
                nominal_evaluation = evaluate_endurance_task(
                    design,
                    engine_caps_kw=engine_caps,
                    motor_assist_kw=motor_assist,
                    speeds_mps=speeds,
                    profile_path=self.profile_path,
                    calibration=self.calibration,
                    assumptions=scenario_assumptions,
                    aero_override=aero_override,
                    dispatch_policy=self.dispatch_policy_factory() if self.dispatch_policy_factory is not None else None,
                    powertrain=self.powertrain_factory() if self.powertrain_factory is not None else None,
                    disturbances=zero_amplitude_disturbances(self.disturbances),
                    flight_controller=flight_controller,
                )
                apply_paired_response_metrics(evaluation, nominal_evaluation, self.disturbances,
                                              definition=self.response_definition)
            scenario_evaluations.append(evaluation)

        if self.uncertainty_scenarios:
            results = [item.result for item in scenario_evaluations]
            objective_names = set().union(*(item.objectives for item in results))
            constraint_names = set().union(*(item.constraints for item in results))
            result = EvaluationResult(
                objectives={
                    name: max(float(item.objectives.get(name, float("-inf"))) for item in results)
                    for name in objective_names
                },
                constraints={
                    name: max(float(item.constraints.get(name, 0.0)) for item in results)
                    for name in constraint_names
                },
                diagnostics={
                    "robust_scenario_count": float(len(results)),
                    "robust_feasible_scenario_count": float(sum(item.feasible for item in results)),
                    **{
                        f"scenario_{label}_{objective}": float(result.objectives[objective])
                        for label, result in zip(scenario_labels, results)
                        for objective in OBJECTIVE_LABELS
                    },
                },
            )
        else:
            result = scenario_evaluations[0].result

        if vspaero_failure:
            result.constraints["vspaero_candidate_failed"] = 1.0
            result.diagnostics["vspaero_candidate_failed"] = 1.0

        if stepwise_aero_result is not None:
            result.constraints.update(stepwise_aero_result["constraints"])
            result.diagnostics.update(stepwise_aero_result["diagnostics"])

        if self.sixdof_validator is not None:
            loiter_label = self.phase_labels[loiter_index]
            scenario_power_limits: list[float] = []
            for scenario_evaluation in scenario_evaluations:
                mission_profile = scenario_evaluation.simulation.profile
                if {"phase", "p_propulsor_input_kw"}.issubset(mission_profile.columns):
                    loiter_power = mission_profile.loc[
                        mission_profile["phase"] == loiter_label,
                        "p_propulsor_input_kw",
                    ].to_numpy(dtype=float)
                    loiter_power = loiter_power[np.isfinite(loiter_power)]
                    if loiter_power.size:
                        scenario_power_limits.append(
                            max(0.0, float(np.median(loiter_power)))
                        )
            sixdof_available_power_kw = (
                min(scenario_power_limits)
                if scenario_power_limits
                else float(engine_caps[loiter_index] + motor_assist[loiter_index])
            )
            sixdof_kwargs = {
                "design": design,
                "aero": aero_override or aero_from_design(design, self.calibration),
                "calibration": self.calibration,
                "cruise_speed_mps": loiter_speed,
                "cruise_altitude_m": float(
                    self.base_mission.phases[loiter_index].altitude_target_m
                ),
                "available_power_kw": sixdof_available_power_kw,
            }
            if getattr(self.sixdof_validator, "full_mission", False):
                sixdof_kwargs.update(
                    {
                        "engine_caps_kw": dict(zip(self.phase_labels, engine_caps)),
                        "motor_assist_kw": dict(zip(self.phase_labels, motor_assist)),
                        "phase_speeds_mps": dict(zip(self.phase_labels, speeds)),
                        "phase_altitudes_m": {
                            phase.name: float(phase.altitude_target_m)
                            for phase in self.base_mission.phases
                        },
                        "initial_soc": float(
                            scenario_evaluations[0].simulation.profile["soc"].iloc[0]
                        ),
                    }
                )
            if getattr(self.sixdof_validator, "accepts_powertrain_sizing", False):
                sixdof_kwargs["powertrain_sizing"] = {
                    "engine_electrical_power_max_kw": result.diagnostics["engine_power_max_kw"],
                    "motor_power_max_kw": result.diagnostics["motor_power_max_kw"],
                    "battery_capacity_mj": result.diagnostics["battery_capacity_mj"],
                }
            sixdof_result = self.sixdof_validator(**sixdof_kwargs)
            result.constraints.update(sixdof_result["constraints"])
            result.diagnostics.update(sixdof_result["diagnostics"])
        result.diagnostics.update({
            f"flight_controller_{name}": float(value)
            for name, value in zip(self.control_parameter_bounds, x[11 + 2 * phase_count:])
        })
        return CandidateEvaluation(x, result)


def _coupled_mission_from_decision(decision: np.ndarray, loiter_duration_h: float) -> MissionDefinition:
    (
        cruise_speed,
        cruise_altitude,
        engine_climb,
        engine_loiter,
        engine_return,
        motor_climb,
        motor_loiter,
        motor_return,
    ) = decision
    reference = reference_endurance_mission(loiter_duration_h)
    phases = []
    for phase in reference.phases:
        if phase.name == "main_climb":
            phases.append(
                replace(
                    phase,
                    altitude_target_m=float(cruise_altitude),
                    engine_power_kw=float(engine_climb),
                    motor_power_kw=float(motor_climb),
                )
            )
        elif phase.name == "endurance_loiter":
            phases.append(
                replace(
                    phase,
                    altitude_target_m=float(cruise_altitude),
                    speed_target_mps=float(cruise_speed),
                    engine_power_kw=float(engine_loiter),
                    motor_power_kw=float(motor_loiter),
                )
            )
        elif phase.name == "return_descent":
            phases.append(
                replace(
                    phase,
                    engine_power_kw=float(engine_return),
                    motor_power_kw=float(motor_return),
                )
            )
        else:
            phases.append(phase)
    return MissionDefinition(
        phases=tuple(phases),
        required_range_m=reference.required_range_m,
        required_loiter_s=reference.required_loiter_s,
        terminal_position_error_max_m=reference.terminal_position_error_max_m,
    )


def simulate_surrogate_mission(decision: np.ndarray, points: int = 81) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate a low-order full-mission history for algorithm verification.

    The mapping intentionally includes energy-management, tracking, and
    response trade-offs, while retaining the same profile contract used by the
    2.2.1 evaluator.  It must be replaced by a validated flight/propulsion
    model before making any PTDL performance claim.
    """
    (
        duration_h,
        cruise_speed,
        cruise_altitude,
        p_eng_climb,
        p_eng_cruise,
        p_eng_return,
        p_motor_climb,
        p_motor_cruise,
        p_motor_return,
        control_gain,
    ) = decision
    time_s = np.linspace(0.0, duration_h * 3600.0, points)
    tau = time_s / time_s[-1]
    knots = np.array([0.0, 0.16, 0.78, 1.0])
    p_engine = np.interp(tau, knots, [p_eng_climb, p_eng_cruise, p_eng_cruise, p_eng_return])
    p_motor = np.interp(tau, knots, [p_motor_climb, p_motor_cruise, p_motor_cruise, p_motor_return])

    # The constants below only create a smooth, constrained surrogate landscape.
    fuel_flow = (
        8.0e-5
        + 1.10e-6 * p_engine
        + 1.7e-7 * (cruise_speed - 30.0) ** 2
        + 4.0e-5 * (control_gain - 0.60) ** 2
    )
    battery_discharge = np.maximum(p_motor, 0.0)
    battery_charge = np.maximum(-p_motor, 0.0)
    net_battery_kw = battery_discharge / 0.95 - 0.95 * battery_charge
    energy_mj = _cumulative_trapezoid(net_battery_kw, time_s) / 1000.0
    soc = 0.86 - energy_mj / 10000.0

    phase_wave = np.sin(2.0 * np.pi * tau)
    transition = np.cos(4.0 * np.pi * tau)
    height_error = (16.0 / control_gain) * phase_wave + 0.18 * (p_motor - np.mean(p_motor))
    speed_error = (2.2 / control_gain) * transition + 0.06 * (p_motor - np.mean(p_motor))
    pitch_error = np.deg2rad(1.7 / control_gain) * phase_wave
    pitch_rate = np.deg2rad(2.8 / control_gain) * transition
    altitude = cruise_altitude + height_error
    speed = cruise_speed + speed_error

    response_metrics = {
        "settling_time_s": float(max(45.0, 260.0 / control_gain - 2.2 * p_motor_climb)),
        "recovery_time_s": float(max(20.0, 80.0 / control_gain - 0.8 * p_motor_return + 0.25 * np.ptp(p_motor))),
        "mode_switch_time_s": float(25.0 + 0.08 * (np.ptp(p_engine) + np.ptp(p_motor))),
    }
    profile = pd.DataFrame(
        {
            "time_s": time_s,
            "fuel_flow_kgps": np.full(points, fuel_flow),
            "battery_discharge_kw": battery_discharge,
            "battery_charge_kw": battery_charge,
            "soc": soc,
            "h_m": altitude,
            "v_mps": speed,
            "p_engine_kw": p_engine,
            "p_motor_kw": p_motor,
            "e_h_m": height_error,
            "e_v_mps": speed_error,
            "e_theta_rad": pitch_error,
            "q_radps": pitch_rate,
        }
    )
    return profile, response_metrics


def _cumulative_trapezoid(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    integrated = np.zeros_like(values)
    integrated[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(time_s))
    return integrated


def dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    """Deb feasibility rules followed by Pareto dominance for feasible points."""
    if left.feasible and not right.feasible:
        return True
    if right.feasible and not left.feasible:
        return False
    if not left.feasible and not right.feasible:
        return left.violation < right.violation
    return bool(np.all(left.objectives <= right.objectives) and np.any(left.objectives < right.objectives))


def non_dominated_sort(evaluations: Sequence[CandidateEvaluation]) -> list[list[int]]:
    remaining = list(range(len(evaluations)))
    fronts: list[list[int]] = []
    while remaining:
        front = [
            index
            for index in remaining
            if not any(
                other != index and dominates(evaluations[other], evaluations[index])
                for other in remaining
            )
        ]
        fronts.append(front)
        remaining = [index for index in remaining if index not in front]
    return fronts


def crowding_distances(front: Sequence[int], evaluations: Sequence[CandidateEvaluation]) -> dict[int, float]:
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    values = np.vstack([evaluations[index].objectives for index in front])
    for objective_index in range(values.shape[1]):
        ordering = np.argsort(values[:, objective_index])
        distance[front[ordering[0]]] = float("inf")
        distance[front[ordering[-1]]] = float("inf")
        span = values[ordering[-1], objective_index] - values[ordering[0], objective_index]
        if span <= 1e-12:
            continue
        for position in range(1, len(ordering) - 1):
            index = front[ordering[position]]
            distance[index] += (values[ordering[position + 1], objective_index] - values[ordering[position - 1], objective_index]) / span
    return distance


def _rank_and_crowding(evaluations: Sequence[CandidateEvaluation]) -> tuple[np.ndarray, dict[int, float], list[list[int]]]:
    fronts = non_dominated_sort(evaluations)
    rank = np.empty(len(evaluations), dtype=int)
    crowding: dict[int, float] = {}
    for level, front in enumerate(fronts):
        for index in front:
            rank[index] = level
        crowding.update(crowding_distances(front, evaluations))
    return rank, crowding, fronts


def _sbx_and_mutate(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    eta: float = 15.0,
    mutation_probability: float | None = None,
) -> np.ndarray:
    u = rng.random(parent_a.size)
    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
    child = 0.5 * ((1.0 + beta) * parent_a + (1.0 - beta) * parent_b)
    probability = mutation_probability if mutation_probability is not None else 1.0 / parent_a.size
    mask = rng.random(parent_a.size) < probability
    child[mask] += rng.normal(0.0, 0.08 * (upper[mask] - lower[mask]))
    return np.clip(child, lower, upper)


class NSGAII:
    def __init__(
        self,
        population_size: int = 36,
        generations: int = 35,
        seed: int = 20260831,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.rng = np.random.default_rng(seed)
        self.progress_callback = progress_callback

    def run(self, problem: PTDLControlProblem) -> OptimizerRun:
        population = problem.sample(self.population_size, self.rng)
        evaluations = [problem.evaluate(x) for x in population]
        if self.progress_callback:
            self.progress_callback(0, self.generations)
        evaluated_archive = list(evaluations)
        generation_pareto_archive = {
            0: _feasible_pareto_snapshot(evaluated_archive)
        }
        history_rows = [_history_row(0, evaluations, evaluated_archive)]
        history_rows[-1]["historical_pareto_count"] = float(len(generation_pareto_archive[0]))
        for generation in range(1, self.generations + 1):
            rank, crowding, _ = _rank_and_crowding(evaluations)
            offspring: list[np.ndarray] = []
            while len(offspring) < self.population_size:
                parent_indices = [self._tournament(rank, crowding), self._tournament(rank, crowding)]
                offspring.append(
                    _sbx_and_mutate(
                        population[parent_indices[0]], population[parent_indices[1]], problem.lower_bounds, problem.upper_bounds, self.rng
                    )
                )
            offspring_array = np.vstack(offspring)
            offspring_evaluations = [problem.evaluate(x) for x in offspring_array]
            evaluated_archive.extend(offspring_evaluations)
            population, evaluations = _nsga_environmental_selection(
                np.vstack([population, offspring_array]), evaluations + offspring_evaluations, self.population_size
            )
            history_rows.append(_history_row(generation, evaluations, evaluated_archive))
            generation_pareto_archive[generation] = _feasible_pareto_snapshot(evaluated_archive)
            history_rows[-1]["historical_pareto_count"] = float(
                len(generation_pareto_archive[generation])
            )
            if self.progress_callback:
                self.progress_callback(generation, self.generations)
        return OptimizerRun(
            "NSGA-II",
            population,
            evaluations,
            pd.DataFrame(history_rows),
            evaluated_archive,
            generation_pareto_archive,
        )

    def _tournament(self, rank: np.ndarray, crowding: Mapping[int, float]) -> int:
        left, right = self.rng.integers(0, self.population_size, size=2)
        if rank[left] != rank[right]:
            return int(left if rank[left] < rank[right] else right)
        return int(left if crowding[left] >= crowding[right] else right)


def _nsga_environmental_selection(
    decisions: np.ndarray, evaluations: list[CandidateEvaluation], population_size: int
) -> tuple[np.ndarray, list[CandidateEvaluation]]:
    selected: list[int] = []
    for front in non_dominated_sort(evaluations):
        remaining = population_size - len(selected)
        if len(front) <= remaining:
            selected.extend(front)
            continue
        crowding = crowding_distances(front, evaluations)
        selected.extend(sorted(front, key=lambda index: crowding[index], reverse=True)[:remaining])
        break
    return decisions[selected], [evaluations[index] for index in selected]


class MOEAD:
    def __init__(
        self,
        resolution: int = 5,
        generations: int = 60,
        neighborhood_size: int = 5,
        seed: int = 20260832,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.resolution = resolution
        self.generations = generations
        self.neighborhood_size = neighborhood_size
        self.rng = np.random.default_rng(seed)
        self.progress_callback = progress_callback

    def run(self, problem: PTDLControlProblem) -> OptimizerRun:
        weights = generate_simplex_weights(self.resolution).to_numpy(dtype=float)
        population = problem.sample(len(weights), self.rng)
        evaluations = [problem.evaluate(x) for x in population]
        evaluated_archive = list(evaluations)
        distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        neighborhoods = np.argsort(distances, axis=1)[:, : min(self.neighborhood_size, len(weights))]
        scale = np.array([3000.0, 1.0, 400.0])
        ideal = _feasible_ideal(evaluations)
        history_rows = [_history_row(0, evaluations, evaluated_archive)]
        generation_pareto_archive = {
            0: _feasible_pareto_snapshot(evaluated_archive)
        }
        history_rows[-1]["historical_pareto_count"] = float(len(generation_pareto_archive[0]))
        if self.progress_callback:
            self.progress_callback(0, self.generations)
        for generation in range(1, self.generations + 1):
            for subproblem, neighborhood in enumerate(neighborhoods):
                parent_a, parent_b = self.rng.choice(neighborhood, size=2, replace=True)
                child_x = _sbx_and_mutate(
                    population[parent_a], population[parent_b], problem.lower_bounds, problem.upper_bounds, self.rng
                )
                child = problem.evaluate(child_x)
                evaluated_archive.append(child)
                if child.feasible:
                    ideal = np.minimum(ideal, child.objectives)
                for neighbor in neighborhood:
                    if _moead_replaces(child, evaluations[neighbor], weights[neighbor], ideal, scale):
                        population[neighbor] = child_x
                        evaluations[neighbor] = child
            history_rows.append(_history_row(generation, evaluations, evaluated_archive))
            generation_pareto_archive[generation] = _feasible_pareto_snapshot(evaluated_archive)
            history_rows[-1]["historical_pareto_count"] = float(
                len(generation_pareto_archive[generation])
            )
            if self.progress_callback:
                self.progress_callback(generation, self.generations)
        return OptimizerRun(
            "MOEA/D",
            population,
            evaluations,
            pd.DataFrame(history_rows),
            evaluated_archive,
            generation_pareto_archive,
        )


class MOPSO:
    """Exploratory constrained multi-objective PSO with an external Pareto archive."""

    def __init__(
        self,
        resolution: int = 3,
        generations: int = 3,
        seed: int = 20261002,
        inertia: float = 0.55,
        cognitive: float = 1.4,
        social: float = 1.4,
        archive_limit: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if resolution < 1 or generations < 1 or archive_limit < 1:
            raise ValueError("resolution, generations, and archive_limit must be positive.")
        self.resolution = resolution
        self.generations = generations
        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)
        self.archive_limit = archive_limit
        self.rng = np.random.default_rng(seed)
        self.progress_callback = progress_callback

    def run(self, problem: PTDLControlProblem) -> OptimizerRun:
        population_size = len(generate_simplex_weights(self.resolution))
        positions = problem.sample(population_size, self.rng)
        span = problem.upper_bounds - problem.lower_bounds
        velocities = self.rng.uniform(-0.08, 0.08, size=positions.shape) * span
        evaluations = [problem.evaluate(position) for position in positions]
        personal_best_positions = positions.copy()
        personal_best_evaluations = list(evaluations)
        evaluated_archive = list(evaluations)
        pareto_archive = _mopso_archive(evaluations, self.archive_limit)
        generation_pareto_archive = {
            0: list(pareto_archive)
        }
        history_rows = [_history_row(0, evaluations, evaluated_archive)]
        history_rows[-1]["historical_pareto_count"] = float(len(pareto_archive))
        if self.progress_callback:
            self.progress_callback(0, self.generations)

        for generation in range(1, self.generations + 1):
            leaders = _mopso_archive(evaluations + pareto_archive, self.archive_limit)
            for index in range(population_size):
                leader = leaders[int(self.rng.integers(0, len(leaders)))]
                r1 = self.rng.random(positions.shape[1])
                r2 = self.rng.random(positions.shape[1])
                velocities[index] = (
                    self.inertia * velocities[index]
                    + self.cognitive
                    * r1
                    * (personal_best_positions[index] - positions[index])
                    + self.social * r2 * (leader.decision - positions[index])
                )
                velocities[index] = np.clip(velocities[index], -0.20 * span, 0.20 * span)
                positions[index] = np.clip(
                    positions[index] + velocities[index],
                    problem.lower_bounds,
                    problem.upper_bounds,
                )
                evaluations[index] = problem.evaluate(positions[index])
                evaluated_archive.append(evaluations[index])
                if _mopso_prefer(
                    evaluations[index],
                    personal_best_evaluations[index],
                    self.rng,
                ):
                    personal_best_positions[index] = positions[index].copy()
                    personal_best_evaluations[index] = evaluations[index]
            pareto_archive = _mopso_archive(
                pareto_archive + evaluations,
                self.archive_limit,
            )
            history_rows.append(_history_row(generation, evaluations, evaluated_archive))
            history_rows[-1]["historical_pareto_count"] = float(len(pareto_archive))
            generation_pareto_archive[generation] = list(pareto_archive)
            if self.progress_callback:
                self.progress_callback(generation, self.generations)

        return OptimizerRun(
            "MOPSO",
            positions,
            evaluations,
            pd.DataFrame(history_rows),
            evaluated_archive,
            generation_pareto_archive,
        )


class HybridNSGAIIMOPSO:
    """Hybrid optimizer mixing NSGA-II elitism with MOPSO position updates."""

    def __init__(
        self,
        resolution: int = 3,
        generations: int = 3,
        seed: int = 20261004,
        inertia: float = 0.55,
        cognitive: float = 1.4,
        social: float = 1.4,
        archive_limit: int = 100,
        initial_ga_probability: float = 0.5,
        adaptation_rate: float = 0.25,
        operator_probability_bounds: tuple[float, float] = (0.2, 0.8),
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if resolution < 1 or generations < 1 or archive_limit < 1:
            raise ValueError("resolution, generations, and archive_limit must be positive.")
        lower_probability, upper_probability = operator_probability_bounds
        if not 0.0 < lower_probability < upper_probability < 1.0:
            raise ValueError("operator_probability_bounds must lie strictly inside (0, 1).")
        if not lower_probability <= initial_ga_probability <= upper_probability:
            raise ValueError("initial_ga_probability must be within operator_probability_bounds.")
        if not 0.0 < adaptation_rate <= 1.0:
            raise ValueError("adaptation_rate must be in (0, 1].")
        self.resolution = resolution
        self.generations = generations
        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)
        self.archive_limit = archive_limit
        self.ga_probability = float(initial_ga_probability)
        self.adaptation_rate = float(adaptation_rate)
        self.operator_probability_bounds = operator_probability_bounds
        self.rng = np.random.default_rng(seed)
        self.progress_callback = progress_callback

    def run(self, problem: PTDLControlProblem) -> OptimizerRun:
        population_size = len(generate_simplex_weights(self.resolution))
        positions = problem.sample(population_size, self.rng)
        span = problem.upper_bounds - problem.lower_bounds
        velocities = self.rng.uniform(-0.08, 0.08, size=positions.shape) * span
        evaluations = [problem.evaluate(position) for position in positions]
        personal_best_positions = positions.copy()
        personal_best_evaluations = list(evaluations)
        evaluated_archive = list(evaluations)
        pareto_archive = _mopso_archive(evaluations, self.archive_limit)
        generation_pareto_archive = {0: list(pareto_archive)}
        history_rows = [_history_row(0, evaluations, evaluated_archive)]
        history_rows[0]["historical_pareto_count"] = float(len(pareto_archive))
        history_rows[0]["hybrid_ga_probability"] = self.ga_probability
        history_rows[0]["hybrid_ga_improvement_rate"] = float("nan")
        history_rows[0]["hybrid_pso_improvement_rate"] = float("nan")
        if self.progress_callback:
            self.progress_callback(0, self.generations)

        for generation in range(1, self.generations + 1):
            rank, crowding, _ = _rank_and_crowding(evaluations)
            leaders = _mopso_archive(evaluations + pareto_archive, self.archive_limit)
            offspring_positions: list[np.ndarray] = []
            offspring_velocities: list[np.ndarray] = []
            offspring_evaluations: list[CandidateEvaluation] = []
            offspring_best_positions: list[np.ndarray] = []
            offspring_best_evaluations: list[CandidateEvaluation] = []
            ga_count = int(round(self.ga_probability * population_size))
            ga_count = min(population_size - 1, max(1, ga_count))
            operator_is_ga = np.zeros(population_size, dtype=bool)
            operator_is_ga[self.rng.permutation(population_size)[:ga_count]] = True
            ga_rewards: list[float] = []
            pso_rewards: list[float] = []

            for index in range(population_size):
                if not operator_is_ga[index]:
                    leader = leaders[int(self.rng.integers(0, len(leaders)))]
                    velocity = (
                        self.inertia * velocities[index]
                        + self.cognitive
                        * self.rng.random(span.size)
                        * (personal_best_positions[index] - positions[index])
                        + self.social
                        * self.rng.random(span.size)
                        * (leader.decision - positions[index])
                    )
                    velocity = np.clip(velocity, -0.20 * span, 0.20 * span)
                    child_position = np.clip(
                        positions[index] + velocity,
                        problem.lower_bounds,
                        problem.upper_bounds,
                    )
                    child_evaluation = problem.evaluate(child_position)
                    pso_rewards.append(
                        _effective_improvement(
                            child_evaluation, evaluations[index]
                        )
                    )
                    if _mopso_prefer(
                        child_evaluation,
                        personal_best_evaluations[index],
                        self.rng,
                    ):
                        best_position = child_position.copy()
                        best_evaluation = child_evaluation
                    else:
                        best_position = personal_best_positions[index].copy()
                        best_evaluation = personal_best_evaluations[index]
                else:
                    parent_a = self._tournament(rank, crowding)
                    parent_b = self._tournament(rank, crowding)
                    child_position = _sbx_and_mutate(
                        positions[parent_a],
                        positions[parent_b],
                        problem.lower_bounds,
                        problem.upper_bounds,
                        self.rng,
                    )
                    velocity = np.zeros_like(child_position)
                    child_evaluation = problem.evaluate(child_position)
                    ga_rewards.append(
                        max(
                            _effective_improvement(
                                child_evaluation, evaluations[parent_a]
                            ),
                            _effective_improvement(
                                child_evaluation, evaluations[parent_b]
                            ),
                        )
                    )
                    best_position = child_position.copy()
                    best_evaluation = child_evaluation
                offspring_positions.append(child_position)
                offspring_velocities.append(velocity)
                offspring_evaluations.append(child_evaluation)
                offspring_best_positions.append(best_position)
                offspring_best_evaluations.append(best_evaluation)

            evaluated_archive.extend(offspring_evaluations)
            combined_positions = np.vstack([positions, np.vstack(offspring_positions)])
            combined_velocities = np.vstack([velocities, np.vstack(offspring_velocities)])
            combined_best_positions = np.vstack(
                [personal_best_positions, np.vstack(offspring_best_positions)]
            )
            combined_evaluations = evaluations + offspring_evaluations
            combined_best_evaluations = (
                personal_best_evaluations + offspring_best_evaluations
            )
            selected: list[int] = []
            for front in non_dominated_sort(combined_evaluations):
                remaining = population_size - len(selected)
                if len(front) <= remaining:
                    selected.extend(front)
                else:
                    front_crowding = crowding_distances(front, combined_evaluations)
                    selected.extend(
                        sorted(front, key=lambda item: front_crowding[item], reverse=True)[
                            :remaining
                        ]
                    )
                    break
            positions = combined_positions[selected]
            velocities = combined_velocities[selected]
            personal_best_positions = combined_best_positions[selected]
            evaluations = [combined_evaluations[index] for index in selected]
            personal_best_evaluations = [
                combined_best_evaluations[index] for index in selected
            ]
            pareto_archive = _mopso_archive(
                pareto_archive + offspring_evaluations,
                self.archive_limit,
            )
            history_rows.append(_history_row(generation, evaluations, evaluated_archive))
            history_rows[-1]["historical_pareto_count"] = float(len(pareto_archive))
            ga_reward = float(np.mean(ga_rewards)) if ga_rewards else 0.0
            pso_reward = float(np.mean(pso_rewards)) if pso_rewards else 0.0
            lower_probability, upper_probability = self.operator_probability_bounds
            self.ga_probability = float(
                np.clip(
                    self.ga_probability
                    + self.adaptation_rate * (ga_reward - pso_reward),
                    lower_probability,
                    upper_probability,
                )
            )
            history_rows[-1]["hybrid_ga_probability"] = self.ga_probability
            history_rows[-1]["hybrid_ga_improvement_rate"] = ga_reward
            history_rows[-1]["hybrid_pso_improvement_rate"] = pso_reward
            generation_pareto_archive[generation] = list(pareto_archive)
            if self.progress_callback:
                self.progress_callback(generation, self.generations)

        return OptimizerRun(
            "Hybrid NSGA-II/MOPSO",
            positions,
            evaluations,
            pd.DataFrame(history_rows),
            evaluated_archive,
            generation_pareto_archive,
        )

    def _tournament(
        self, rank: np.ndarray, crowding: Mapping[int, float]
    ) -> int:
        left, right = self.rng.integers(0, len(rank), size=2)
        if rank[left] != rank[right]:
            return int(left if rank[left] < rank[right] else right)
        return int(left if crowding[left] >= crowding[right] else right)


def _effective_improvement(
    candidate: CandidateEvaluation, incumbent: CandidateEvaluation
) -> float:
    if candidate.feasible != incumbent.feasible:
        return float(candidate.feasible)
    if not candidate.feasible:
        return float(candidate.violation < incumbent.violation - 1e-12)
    return float(
        np.all(candidate.objectives <= incumbent.objectives)
        and np.any(candidate.objectives < incumbent.objectives)
    )


def _mopso_prefer(
    candidate: CandidateEvaluation,
    incumbent: CandidateEvaluation,
    rng: np.random.Generator,
) -> bool:
    if candidate.feasible != incumbent.feasible:
        return candidate.feasible
    if not candidate.feasible:
        if not np.isclose(candidate.violation, incumbent.violation):
            return candidate.violation < incumbent.violation
        return bool(rng.integers(0, 2))
    candidate_dominates = np.all(candidate.objectives <= incumbent.objectives) and np.any(
        candidate.objectives < incumbent.objectives
    )
    incumbent_dominates = np.all(incumbent.objectives <= candidate.objectives) and np.any(
        incumbent.objectives < candidate.objectives
    )
    if candidate_dominates:
        return True
    if incumbent_dominates:
        return False
    return bool(rng.integers(0, 2))


def _mopso_archive(
    evaluations: Sequence[CandidateEvaluation],
    archive_limit: int,
) -> list[CandidateEvaluation]:
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        return [min(evaluations, key=lambda item: item.violation)]
    unique: dict[tuple[float, ...], CandidateEvaluation] = {}
    for item in feasible:
        unique.setdefault(tuple(np.round(item.decision, 12)), item)
    candidates = list(unique.values())
    front_indices = non_dominated_sort(candidates)[0]
    archive = [candidates[index] for index in front_indices]
    if len(archive) <= archive_limit:
        return archive
    crowding = crowding_distances(list(range(len(archive))), archive)
    keep = sorted(range(len(archive)), key=lambda index: crowding[index], reverse=True)[
        :archive_limit
    ]
    return [archive[index] for index in keep]


def _feasible_ideal(evaluations: Sequence[CandidateEvaluation]) -> np.ndarray:
    feasible = [item.objectives for item in evaluations if item.feasible]
    return np.min(feasible, axis=0) if feasible else np.min([item.objectives for item in evaluations], axis=0)


def _moead_replaces(
    candidate: CandidateEvaluation,
    incumbent: CandidateEvaluation,
    weight: np.ndarray,
    ideal: np.ndarray,
    scale: np.ndarray,
) -> bool:
    if candidate.feasible and not incumbent.feasible:
        return True
    if incumbent.feasible and not candidate.feasible:
        return False
    if not candidate.feasible and not incumbent.feasible:
        return candidate.violation < incumbent.violation
    candidate_value = np.max(weight * np.abs((candidate.objectives - ideal) / scale))
    incumbent_value = np.max(weight * np.abs((incumbent.objectives - ideal) / scale))
    # Equal scalarization values should retain the incumbent.  Replacing on a
    # tie lets the last evaluated child overwrite an entire neighbourhood,
    # which collapses otherwise distinct MOEA/D subproblems prematurely.
    return bool(candidate_value < incumbent_value - 1e-12)


def _history_row(
    generation: int,
    evaluations: Sequence[CandidateEvaluation],
    evaluated_archive: Sequence[CandidateEvaluation] | None = None,
) -> dict[str, float]:
    feasible = [item for item in evaluations if item.feasible]
    archive = list(evaluated_archive or evaluations)
    archive_feasible = [item for item in archive if item.feasible]
    row: dict[str, float] = {
        "generation": generation,
        "feasible_count": float(len(feasible)),
        "feasible_ratio": float(len(feasible) / len(evaluations)),
        "minimum_total_violation": float(min(item.violation for item in evaluations)),
        "evaluated_feasible_count": float(len(archive_feasible)),
    }
    for position, label in enumerate(OBJECTIVE_LABELS):
        row[f"population_best_{label}"] = (
            float(min(item.objectives[position] for item in feasible)) if feasible else float("nan")
        )
        row[f"best_{label}"] = (
            float(min(item.objectives[position] for item in archive_feasible))
            if archive_feasible else float("nan")
        )
    return row


def pareto_indices(evaluations: Sequence[CandidateEvaluation]) -> list[int]:
    feasible_indices = [index for index, item in enumerate(evaluations) if item.feasible]
    if not feasible_indices:
        return []
    feasible = [evaluations[index] for index in feasible_indices]
    return [feasible_indices[index] for index in non_dominated_sort(feasible)[0]]


def _feasible_pareto_snapshot(
    evaluations: Sequence[CandidateEvaluation],
) -> list[CandidateEvaluation]:
    """Return a de-duplicated feasible front for a cumulative evaluation archive."""
    unique: list[CandidateEvaluation] = []
    seen: set[tuple[float, ...]] = set()
    for item in evaluations:
        if not item.feasible:
            continue
        key = tuple(np.round(item.decision, 12))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return [unique[index] for index in pareto_indices(unique)]


def run_coupled_moead_baseline(
    output_dir: str | Path,
    aircraft: AircraftParameters = AircraftParameters(),
    loiter_duration_h: float = 80.0,
    resolution: int = 3,
    generations: int = 12,
    seed: int = 20260904,
) -> dict[str, object]:
    """Run MOEA/D on the reconstructed coupled model and export an audit set.

    The function deliberately exports the least-violating candidate even when
    no design reaches the 80 h hard constraint.  This makes the first run a
    feasibility audit of the provisional model rather than an invalid Pareto
    result.  Once validated PTDL parameters are supplied, the same interface
    will automatically export the feasible Pareto set.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problem = CoupledPTDLControlProblem(aircraft=aircraft, loiter_duration_h=loiter_duration_h)
    run = MOEAD(resolution=resolution, generations=generations, seed=seed).run(problem)
    run.history.to_csv(output_path / "moea_d_convergence.csv", index=False)

    final_records = _candidate_records(run, range(len(run.evaluations)), labels=problem.decision_labels)
    for record, item in zip(final_records, run.evaluations):
        record["feasible"] = item.feasible
    pd.DataFrame(final_records).to_csv(output_path / "moea_d_final_population.csv", index=False)

    best_index = int(np.argmin(run.violations))
    best_candidate = run.evaluations[best_index]
    best_mission = _coupled_mission_from_decision(best_candidate.decision, loiter_duration_h)
    simulation, _ = evaluate_coupled_mission(best_mission, aircraft)
    closest_record = _candidate_records(run, [best_index], labels=problem.decision_labels)[0]
    closest_record.update(
        {
            "feasible": best_candidate.feasible,
            "termination_reason": simulation.termination_reason,
            "achieved_duration_h": float(simulation.profile["time_s"].iloc[-1] / 3600.0),
            "achieved_loiter_h": float(simulation.loiter_s / 3600.0),
        }
    )
    pd.DataFrame([closest_record]).to_csv(output_path / "least_violating_candidate.csv", index=False)
    simulation.profile.to_csv(output_path / "least_violating_time_history.csv", index=False)

    pareto = pareto_indices(run.evaluations)
    pd.DataFrame(_candidate_records(run, pareto, labels=problem.decision_labels)).to_csv(
        output_path / "moea_d_feasible_pareto.csv", index=False
    )
    metadata = {
        "model_status": "reconstructed low-order fixed-wing flight-propulsion-control baseline; reference values are not PTDL design data",
        "optimization_method": "MOEA/D",
        "objectives": list(OBJECTIVE_LABELS),
        "hard_constraint": "T_endurance >= 80 h",
        "loiter_requirement_h": loiter_duration_h,
        "decision_labels": list(problem.decision_labels),
        "moead": {
            "resolution": resolution,
            "weights": int((resolution + 1) * (resolution + 2) / 2),
            "generations": generations,
            "seed": seed,
        },
        "final_feasible_count": len(pareto),
        "least_violating_candidate_index": best_index,
    }
    (output_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"run": run, "least_violating": best_candidate, "output_dir": output_path}


def run_source_profile_moead_audit(
    output_dir: str | Path,
    design: AirframeDesign = AirframeDesign(payload_mass_kg=100.0, hybrid_system_mass_kg=120.0),
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    target_total_duration_h: float = 80.0,
    resolution: int = 2,
    generations: int = 4,
    seed: int = 20260905,
) -> dict[str, object]:
    """Optimize source-profile control schedules and export a feasibility audit.

    It intentionally keeps the JSON ISR speed and altitude intact.  A zero
    feasible count is therefore a meaningful energy-budget result, not a
    failed optimizer run: the exported least-violating schedule identifies the
    control-limit and endurance shortfall to be resolved by mission or design
    changes.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problem = SourceProfileControlProblem(design, profile_path, target_total_duration_h)
    run = MOEAD(resolution=resolution, generations=generations, seed=seed).run(problem)
    run.history.to_csv(output_path / "source_profile_moead_convergence.csv", index=False)
    final_records = _candidate_records(run, range(len(run.evaluations)), labels=problem.decision_labels)
    for record, item in zip(final_records, run.evaluations):
        record["feasible"] = item.feasible
    pd.DataFrame(final_records).to_csv(output_path / "source_profile_moead_final_population.csv", index=False)

    best_index = int(np.argmin(run.violations))
    best_candidate = run.evaluations[best_index]
    simulation, _ = evaluate_coupled_mission(problem.mission_from_decision(best_candidate.decision), problem.aircraft)
    least_record = _candidate_records(run, [best_index], labels=problem.decision_labels)[0]
    least_record.update(
        {
            "feasible": best_candidate.feasible,
            "achieved_total_duration_h": float(simulation.profile["time_s"].iloc[-1] / 3600.0),
            "termination_reason": simulation.termination_reason,
            "terminal_fuel_kg": simulation.terminal_fuel_kg,
            "terminal_soc": float(simulation.profile["soc"].iloc[-1]),
        }
    )
    pd.DataFrame([least_record]).to_csv(output_path / "source_profile_moead_least_violating.csv", index=False)
    simulation.profile.to_csv(output_path / "source_profile_moead_least_violating_time_history.csv", index=False)
    pareto = pareto_indices(run.evaluations)
    pd.DataFrame(_candidate_records(run, pareto, labels=problem.decision_labels)).to_csv(
        output_path / "source_profile_moead_feasible_pareto.csv", index=False
    )
    metadata = {
        "model_status": "source-JSON-preserving MOEA/D control feasibility audit with preliminary hybrid assumptions",
        "source_profile_path": str(Path(profile_path)),
        "source_profile_initial_duration_h": source_profile_duration_h(problem.source_profile),
        "source_cruise_speed_mps": problem.source_profile["reference"]["cruise_speed_mps"],
        "source_cruise_altitude_m": problem.source_profile["reference"]["cruise_altitude_m"],
        "hard_constraint": "T_endurance >= 80 h",
        "target_total_duration_h": target_total_duration_h,
        "design": {
            "wing_area_m2": design.wing_area_m2,
            "aspect_ratio": design.aspect_ratio,
            "payload_mass_kg": design.payload_mass_kg,
            "hybrid_system_mass_kg": design.hybrid_system_mass_kg,
        },
        "engine_power_max_kw": problem.aircraft.engine_electrical_power_max_kw,
        "propulsion_assumptions": {
            "engine_rating_multiplier": problem.assumptions.engine_rating_multiplier,
            "motor_rating_multiplier": problem.assumptions.motor_rating_multiplier,
            "battery_energy_multiplier": problem.assumptions.battery_energy_multiplier,
        },
        "decision_labels": list(problem.decision_labels),
        "moead": {
            "resolution": resolution,
            "weights": int((resolution + 1) * (resolution + 2) / 2),
            "generations": generations,
            "seed": seed,
        },
        "final_feasible_count": len(pareto),
        "least_violating_candidate_index": best_index,
    }
    (output_path / "source_profile_moead_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"run": run, "least_violating": best_candidate, "problem": problem, "output_dir": output_path}


def run_ptdl_80h_joint_moead(
    output_dir: str | Path,
    profile_path: str | Path = DEFAULT_ENDURANCE_PROFILE_PATH,
    resolution: int = 2,
    generations: int = 4,
    seed: int = 20260906,
    aero_provider: Callable[[AirframeDesign, UltraCalibration], AeroCoefficients] | None = None,
    aero_model_label: str = "low_order",
    powertrain_factory: Callable[[], HybridPowertrainModel] | None = None,
    dispatch_policy_factory: Callable[[], HybridDispatchPolicy] | None = None,
    seed_decisions: Sequence[Sequence[float]] = (),
    algorithm: str = "MOEA/D",
        disturbances: Sequence[LongitudinalDragDisturbance] = (),
        design_feature_bounds: Mapping[str, tuple[float, float]] | None = None,
        assumptions: PreliminaryPropulsionAssumptions | None = None,
    uncertainty_scenarios: Sequence[Mapping[str, object]] = (),
    sixdof_validator: Callable[..., Mapping[str, Mapping[str, float]]] | None = None,
    stepwise_aero_validator: Callable[..., Mapping[str, object]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    response_definition: str = "legacy",
) -> dict[str, object]:
    """Run a seeded PTDL 80 h joint-design experiment with either optimizer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problem = PTDLEnduranceJointProblem(
        profile_path,
        aero_provider=aero_provider,
        powertrain_factory=powertrain_factory,
        dispatch_policy_factory=dispatch_policy_factory,
        seed_decisions=seed_decisions,
        disturbances=disturbances,
        design_feature_bounds=design_feature_bounds,
        assumptions=assumptions,
        uncertainty_scenarios=uncertainty_scenarios,
        sixdof_validator=sixdof_validator,
        stepwise_aero_validator=stepwise_aero_validator,
        response_definition=response_definition,
    )
    normalized_algorithm = (
        algorithm.strip().upper().replace("-", "").replace("/", "").replace(" ", "")
    )
    population_size = int((resolution + 1) * (resolution + 2) / 2)
    if normalized_algorithm == "MOEAD":
        algorithm_name = "MOEA/D"
        prefix = "ptdl_80h_joint_moead"
        optimizer = MOEAD(
            resolution=resolution,
            generations=generations,
            seed=seed,
            progress_callback=progress_callback,
        )
    elif normalized_algorithm == "NSGAII":
        algorithm_name = "NSGA-II"
        prefix = "ptdl_80h_joint_nsga_ii"
        optimizer = NSGAII(
            population_size=population_size,
            generations=generations,
            seed=seed,
            progress_callback=progress_callback,
        )
    elif normalized_algorithm == "MOPSO":
        algorithm_name = "MOPSO"
        prefix = "ptdl_80h_joint_mopso"
        optimizer = MOPSO(
            resolution=resolution,
            generations=generations,
            seed=seed,
            progress_callback=progress_callback,
        )
    elif normalized_algorithm in {"HYBRIDNSGAIIMOPSO", "NSGAIIMOPS"}:
        algorithm_name = "Hybrid NSGA-II/MOPSO"
        prefix = "ptdl_80h_joint_hybrid_nsga_ii_mopso"
        optimizer = HybridNSGAIIMOPSO(
            resolution=resolution,
            generations=generations,
            seed=seed,
            progress_callback=progress_callback,
        )
    elif normalized_algorithm in {"PTDL自适应分块", "PTDLBLOCKADAPTIVE"}:
        from PTDL分块自适应优化 import PTDLBlockAdaptive
        algorithm_name = "PTDL-自适应分块"
        prefix = "ptdl_80h_joint_block_adaptive"
        optimizer = PTDLBlockAdaptive(
            resolution=resolution, generations=generations, seed=seed,
            progress_callback=progress_callback,
        )
    elif normalized_algorithm in {"PTDL耦合代理辅助", "PTDLCOUPLEDSURROGATE", "PTDL耦合引导",
                                  "PTDL耦合贡献学习", "PTDL耦合贡献学习保先验", "PTDL去重贡献学习"}:
        from PTDL耦合代理辅助优化 import PTDLCoupledSurrogateSearch
        assisted = normalized_algorithm in {"PTDL耦合代理辅助", "PTDLCOUPLEDSURROGATE"}
        policy = {"PTDL耦合贡献学习": "credit",
                  "PTDL耦合贡献学习保先验": "credit-prior",
                  "PTDL去重贡献学习": "credit-prior"}.get(normalized_algorithm, "uniform")
        names = {
            "PTDL耦合贡献学习": ("PTDL-耦合贡献学习", "ptdl_80h_joint_coupled_credit"),
            "PTDL耦合贡献学习保先验": ("PTDL-耦合贡献学习保先验", "ptdl_80h_joint_coupled_credit_prior"),
            "PTDL去重贡献学习": ("PTDL-去重贡献学习", "ptdl_80h_joint_coupled_unique"),
        }
        algorithm_name, prefix = names.get(normalized_algorithm, (
            ("PTDL-耦合代理辅助", "ptdl_80h_joint_coupled_surrogate") if assisted
            else ("PTDL-耦合引导", "ptdl_80h_joint_coupled_guided")))
        optimizer = PTDLCoupledSurrogateSearch(
            resolution=resolution, generations=generations, seed=seed,
            progress_callback=progress_callback, use_surrogate=assisted, proposal_policy=policy,
            deduplicate=normalized_algorithm == "PTDL去重贡献学习",
        )
    else:
        raise ValueError(
            "Unsupported optimizer; choose a registered PTDL optimizer."
        )
    started_at = perf_counter()
    run = optimizer.run(problem)
    runtime_s = perf_counter() - started_at
    archive_evaluations = list(run.evaluated_archive or run.evaluations)
    for seed_decision in problem.seed_decisions:
        if not any(np.array_equal(seed_decision, item.decision) for item in archive_evaluations):
            archive_evaluations.append(problem.evaluate(seed_decision))
    unique_archive: list[CandidateEvaluation] = []
    seen_decisions: set[tuple[float, ...]] = set()
    for item in archive_evaluations:
        key = tuple(np.round(item.decision, 12))
        if key not in seen_decisions:
            seen_decisions.add(key)
            unique_archive.append(item)
    archive_evaluations = unique_archive
    archive_run = OptimizerRun(
        run.algorithm,
        np.vstack([item.decision for item in archive_evaluations]),
        archive_evaluations,
        run.history,
        archive_evaluations,
    )
    run.history.to_csv(output_path / f"{prefix}_convergence.csv", index=False)
    records = _candidate_records(run, range(len(run.evaluations)), labels=problem.decision_labels)
    for record, item in zip(records, run.evaluations):
        record["feasible"] = item.feasible
    pd.DataFrame(records).to_csv(output_path / f"{prefix}_final_population.csv", index=False)
    final_population_pareto = pareto_indices(run.evaluations)
    final_pareto_records = _candidate_records(
        run, final_population_pareto, labels=problem.decision_labels
    )
    for record, index in zip(final_pareto_records, final_population_pareto):
        record["feasible"] = run.evaluations[index].feasible
    pd.DataFrame(final_pareto_records).to_csv(
        output_path / f"{prefix}_final_population_feasible_pareto.csv",
        index=False,
    )
    evaluated_records = _candidate_records(
        archive_run,
        range(len(archive_run.evaluations)),
        labels=problem.decision_labels,
    )
    for record, item in zip(evaluated_records, archive_run.evaluations):
        record["feasible"] = item.feasible
    pd.DataFrame(evaluated_records).to_csv(
        output_path / f"{prefix}_evaluated_candidates.csv",
        index=False,
    )
    best_index = int(np.argmin(run.violations))
    best_candidate = run.evaluations[best_index]
    x = best_candidate.decision
    phase_count = len(problem.phase_labels)
    speeds = [phase.speed_target_mps for phase in problem.base_mission.phases]
    loiter_index = problem.phase_labels.index("endurance_loiter")
    speeds[loiter_index] = float(x[10])
    design = problem.design_from_decision(x)
    if stepwise_aero_validator is not None:
        stepwise_replay = stepwise_aero_validator.evaluate_candidate(
            design,
            problem.calibration,
            speed_mps=float(x[10]),
            altitude_m=float(problem.base_mission.phases[loiter_index].altitude_target_m),
        )
        aero_override = stepwise_replay["aero"]
    elif aero_provider is None:
        aero_override = None
    elif getattr(aero_provider, "accepts_operating_point", False):
        aero_override = aero_provider(
            design,
            problem.calibration,
            speed_mps=float(x[10]),
            altitude_m=float(problem.base_mission.phases[loiter_index].altitude_target_m),
        )
    else:
        aero_override = aero_provider(design, problem.calibration)
    best_evaluation = evaluate_endurance_task(
        design,
        engine_caps_kw=x[11 : 11 + phase_count],
        motor_assist_kw=x[11 + phase_count :],
        speeds_mps=speeds,
        profile_path=profile_path,
        calibration=problem.calibration,
        assumptions=problem.assumptions,
        aero_override=aero_override,
        dispatch_policy=problem.dispatch_policy_factory() if problem.dispatch_policy_factory is not None else None,
        powertrain=problem.powertrain_factory() if problem.powertrain_factory is not None else None,
        disturbances=problem.disturbances,
    )
    best_record = _candidate_records(run, [best_index], labels=problem.decision_labels)[0]
    best_record.update(
        {
            "feasible": best_candidate.feasible,
            "achieved_total_duration_h": best_evaluation.achieved_duration_h,
            "termination_reason": best_evaluation.simulation.termination_reason,
            "terminal_fuel_kg": best_evaluation.simulation.terminal_fuel_kg,
            "terminal_soc": float(best_evaluation.simulation.profile["soc"].iloc[-1]),
        }
    )
    pd.DataFrame([best_record]).to_csv(output_path / f"{prefix}_least_violating.csv", index=False)
    best_evaluation.simulation.profile.to_csv(output_path / f"{prefix}_least_violating_time_history.csv", index=False)
    pareto = pareto_indices(archive_run.evaluations)
    pd.DataFrame(_candidate_records(archive_run, pareto, labels=problem.decision_labels)).to_csv(
        output_path / f"{prefix}_feasible_pareto.csv", index=False
    )
    generation_records: list[dict[str, object]] = []
    generation_counts: dict[str, int] = {}
    for generation, snapshot in sorted((run.generation_pareto_archive or {}).items()):
        generation_counts[str(generation)] = len(snapshot)
        snapshot_run = OptimizerRun(
            run.algorithm,
            np.vstack([item.decision for item in snapshot]) if snapshot else np.empty((0, len(problem.decision_labels))),
            snapshot,
            run.history,
        )
        for candidate_index, record in enumerate(
            _candidate_records(snapshot_run, range(len(snapshot)), labels=problem.decision_labels)
        ):
            record["generation"] = generation
            record["archive_candidate_index"] = candidate_index
            record["feasible"] = True
            generation_records.append(record)
    pd.DataFrame(generation_records).to_csv(
        output_path / f"{prefix}_generation_feasible_pareto.csv",
        index=False,
    )
    metadata = {
        "model_status": f"preliminary joint dimension-mass-control {algorithm_name} on the PTDL 80 h endurance-adapted profile",
        "hard_constraint": "T_endurance >= 80 h",
        "decision_count": len(problem.decision_labels),
        "decision_labels": list(problem.decision_labels),
        "design_feature_bounds": problem.design_feature_bounds,
        "robust_uncertainty_scenarios": list(problem.uncertainty_scenarios),
        "robust_objective_aggregation": (
            "worst case (maximum) objective and constraint value across declared scenarios"
            if problem.uncertainty_scenarios
            else "single nominal scenario"
        ),
        "optimizer": {
            "algorithm": algorithm_name,
            "resolution": resolution,
            "population_size": population_size,
            "generations": generations,
            "seed": seed,
            "runtime_s": runtime_s,
        },
        "final_feasible_count": sum(item.feasible for item in run.evaluations),
        "final_population_pareto_count": len(final_population_pareto),
        "feasible_archive_count": len(pareto),
        "evaluated_candidate_count": len(archive_evaluations),
        "archive_basis": "历代累计评估候选中的可行非支配外部档案；逐代快照包含第0代",
        "comparison_metric_basis": "历代累计可行 Pareto 外部档案",
        "generation_pareto_counts": generation_counts,
        "generation_pareto_file": f"{prefix}_generation_feasible_pareto.csv",
        "final_population_pareto_file": f"{prefix}_final_population_feasible_pareto.csv",
        "least_violating_candidate_index": best_index,
        "aero_model": aero_model_label,
        "ultra_reference_basis": problem.sizing_config["reference_basis"],
        "preliminary_propulsion_assumptions": {
            "engine_rating_multiplier": problem.assumptions.engine_rating_multiplier,
            "motor_rating_multiplier": problem.assumptions.motor_rating_multiplier,
            "battery_energy_multiplier": problem.assumptions.battery_energy_multiplier,
            "engine_power_limit_kw_at_selected_design": best_evaluation.result.diagnostics.get(
                "engine_power_max_kw"
            ),
            "motor_power_limit_kw_at_selected_design": best_evaluation.result.diagnostics.get(
                "motor_power_max_kw"
            ),
            "battery_capacity_mj_at_selected_design": best_evaluation.result.diagnostics.get(
                "battery_capacity_mj"
            ),
        },
        "powertrain_model": (
            "reference_dc_bus"
            if powertrain_factory is None
            else getattr(
                getattr(powertrain_factory, "func", powertrain_factory),
                "__name__",
                type(powertrain_factory).__name__,
            )
        ),
        "dispatch_policy_model": (
            "scheduled_rule_dispatch"
            if dispatch_policy_factory is None
            else getattr(
                getattr(dispatch_policy_factory, "func", dispatch_policy_factory),
                "__name__",
                type(dispatch_policy_factory).__name__,
            )
        ),
        "assumption_status": "ULTRA aircraft and mission anchors with project-selected preliminary component sizing multipliers; no supplier or dynamometer calibration",
    }
    (output_path / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "run": run,
        "feasible_archive": archive_run,
        "least_violating": best_candidate,
        "problem": problem,
        "output_dir": output_path,
    }


def run_delivery_experiment(
    output_dir: str | Path,
    seed: int = 20260831,
    nsga_population: int = 36,
    nsga_generations: int = 35,
    moead_resolution: int = 5,
    moead_generations: int = 60,
) -> dict[str, object]:
    """Run the two algorithms and write all 2.2.2 prototype outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problem = PTDLControlProblem()
    runs = [
        NSGAII(nsga_population, nsga_generations, seed).run(problem),
        MOEAD(moead_resolution, moead_generations, seed + 1).run(problem),
    ]
    all_pareto_records: list[dict[str, float | str]] = []
    all_objective_records: list[dict[str, float]] = []
    for run in runs:
        run.history.to_csv(output_path / f"{_slug(run.algorithm)}_convergence.csv", index=False)
        archive = run.evaluated_archive or run.evaluations
        archive_run = OptimizerRun(run.algorithm, run.decisions, archive, run.history)
        records = _candidate_records(archive_run, pareto_indices(archive))
        pd.DataFrame(records).to_csv(output_path / f"{_slug(run.algorithm)}_pareto_candidates.csv", index=False)
        all_pareto_records.extend(records)
        all_objective_records.extend(
            {label: item.result.objectives[label] for label in OBJECTIVE_LABELS}
            for item in run.evaluations
            if item.feasible
        )

    representatives = _export_representatives(runs, problem, output_path, all_objective_records)
    comparison = pd.DataFrame([_comparison_row(run) for run in runs])
    comparison.to_csv(output_path / "algorithm_comparison.csv", index=False)
    _write_convergence_svg(runs, output_path / "convergence_feasibility.svg")
    _write_convergence_png(runs, output_path / "convergence_feasibility.png")
    metadata = {
        "model_status": "low-order surrogate for algorithm-chain verification only; not PTDL design data",
        "endurance_constraint": "T_endurance >= 80 h",
        "objective_labels": list(OBJECTIVE_LABELS),
        "decision_labels": list(DECISION_LABELS),
        "nsga_ii": {"population": nsga_population, "generations": nsga_generations, "seed": seed},
        "moead": {"resolution": moead_resolution, "weights": int((moead_resolution + 1) * (moead_resolution + 2) / 2), "generations": moead_generations, "seed": seed + 1},
    }
    (output_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"runs": runs, "comparison": comparison, "representatives": representatives, "output_dir": output_path}


def _candidate_records(
    run: OptimizerRun,
    indices: Iterable[int],
    labels: Sequence[str] = DECISION_LABELS,
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for index in indices:
        item = run.evaluations[index]
        record: dict[str, float | str] = {"algorithm": run.algorithm, "candidate_index": float(index)}
        record.update({label: float(value) for label, value in zip(labels, item.decision)})
        record.update(item.result.objectives)
        record.update({f"g_{name}": value for name, value in item.result.constraints.items()})
        record.update({f"diagnostic_{name}": value for name, value in item.result.diagnostics.items()})
        record["total_violation"] = item.violation
        records.append(record)
    return records


def _export_representatives(
    runs: Sequence[OptimizerRun],
    problem: PTDLControlProblem,
    output_path: Path,
    objective_records: Sequence[Mapping[str, float]],
) -> pd.DataFrame:
    all_candidates: list[tuple[str, int, CandidateEvaluation]] = []
    for run in runs:
        archive = run.evaluated_archive or run.evaluations
        all_candidates.extend((run.algorithm, index, archive[index]) for index in pareto_indices(archive))
    if not all_candidates:
        raise RuntimeError("No feasible Pareto candidates were produced by the prototype run.")
    normalizer = ObjectiveNormalizer.from_samples(pd.DataFrame(objective_records), labels=OBJECTIVE_LABELS, worst_quantile=0.95)
    preferences = {
        "energy_priority": {"J_energy_mj": 0.60, "J_stability": 0.20, "J_response_s": 0.20},
        "stability_priority": {"J_energy_mj": 0.20, "J_stability": 0.60, "J_response_s": 0.20},
        "response_priority": {"J_energy_mj": 0.20, "J_stability": 0.20, "J_response_s": 0.60},
        "balanced": {"J_energy_mj": 1.0 / 3.0, "J_stability": 1.0 / 3.0, "J_response_s": 1.0 / 3.0},
    }
    representative_rows: list[dict[str, float | str]] = []
    control_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []
    for preference_name, weights in preferences.items():
        scores = [weighted_preference_score(normalizer.transform(item.result.objectives), weights) for _, _, item in all_candidates]
        algorithm, index, candidate = all_candidates[int(np.argmin(scores))]
        profile, _ = simulate_surrogate_mission(candidate.decision)
        representative_rows.append(
            {
                "preference": preference_name,
                "algorithm": algorithm,
                "candidate_index": index,
                "preference_score": float(min(scores)),
                **{label: float(value) for label, value in zip(DECISION_LABELS, candidate.decision)},
                **candidate.result.objectives,
            }
        )
        control_frames.append(
            pd.DataFrame(
                {
                    "preference": preference_name,
                    "algorithm": algorithm,
                    "time_s": profile["time_s"],
                    "p_engine_kw": profile["p_engine_kw"],
                    "p_motor_kw": profile["p_motor_kw"],
                    "battery_discharge_kw": profile["battery_discharge_kw"],
                    "battery_charge_kw": profile["battery_charge_kw"],
                }
            )
        )
        state_frames.append(
            pd.DataFrame(
                {
                    "preference": preference_name,
                    "algorithm": algorithm,
                    "time_s": profile["time_s"],
                    "soc": profile["soc"],
                    "h_m": profile["h_m"],
                    "v_mps": profile["v_mps"],
                    "e_h_m": profile["e_h_m"],
                    "e_v_mps": profile["e_v_mps"],
                    "e_theta_rad": profile["e_theta_rad"],
                    "q_radps": profile["q_radps"],
                }
            )
        )
    representatives = pd.DataFrame(representative_rows)
    representatives.to_csv(output_path / "representative_solutions.csv", index=False)
    pd.concat(control_frames, ignore_index=True).to_csv(output_path / "candidate_control_sequences.csv", index=False)
    pd.concat(state_frames, ignore_index=True).to_csv(output_path / "candidate_state_trajectories.csv", index=False)
    return representatives


def _comparison_row(run: OptimizerRun) -> dict[str, float | str]:
    indices = pareto_indices(run.evaluations)
    feasible = [item for item in run.evaluations if item.feasible]
    row: dict[str, float | str] = {
        "algorithm": run.algorithm,
        "population_size": len(run.evaluations),
        "feasible_count": len(feasible),
        "feasible_ratio": len(feasible) / len(run.evaluations),
        "pareto_count": len(indices),
        "minimum_total_violation": min(item.violation for item in run.evaluations),
    }
    for position, label in enumerate(OBJECTIVE_LABELS):
        row[f"best_{label}"] = min(item.objectives[position] for item in feasible) if feasible else float("nan")
    return row


def _write_convergence_svg(runs: Sequence[OptimizerRun], path: Path) -> None:
    """Write a dependency-free feasibility convergence plot as SVG."""
    width, height, pad = 760, 380, 55
    colors = {"NSGA-II": "#2563eb", "MOEA/D": "#dc2626"}
    max_generation = max(int(run.history["generation"].max()) for run in runs)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#475569"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#475569"/>',
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="Arial" font-size="14">Generation</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="14">Feasible ratio</text>',
    ]
    for tick in range(6):
        y = height - pad - tick * (height - 2 * pad) / 5
        lines.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        lines.append(f'<text x="{pad-8}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick/5:.1f}</text>')
    for run in runs:
        points = []
        for _, row in run.history.iterrows():
            x = pad + row["generation"] / max_generation * (width - 2 * pad)
            y = height - pad - row["feasible_ratio"] * (height - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[run.algorithm]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        lines.append(f'<text x="{width-180}" y="{45 + 20 * list(colors).index(run.algorithm)}" font-family="Arial" font-size="13" fill="{color}">{run.algorithm}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_convergence_png(runs: Sequence[OptimizerRun], path: Path) -> None:
    """Write a portable raster version of the feasibility convergence curve."""
    width, height, pad = 1000, 520, 75
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {"NSGA-II": "#2563eb", "MOEA/D": "#dc2626"}
    max_generation = max(int(run.history["generation"].max()) for run in runs)
    draw.line((pad, height - pad, width - pad, height - pad), fill="#475569", width=2)
    draw.line((pad, pad, pad, height - pad), fill="#475569", width=2)
    for tick in range(6):
        y = height - pad - tick * (height - 2 * pad) / 5
        draw.line((pad, y, width - pad, y), fill="#e2e8f0", width=1)
        draw.text((pad - 40, y - 7), f"{tick / 5:.1f}", fill="#334155")
    draw.text((width // 2 - 45, height - 42), "Generation", fill="#0f172a")
    draw.text((18, 20), "Feasible ratio", fill="#0f172a")
    for position, run in enumerate(runs):
        points = []
        for _, row in run.history.iterrows():
            x = pad + row["generation"] / max_generation * (width - 2 * pad)
            y = height - pad - row["feasible_ratio"] * (height - 2 * pad)
            points.append((float(x), float(y)))
        draw.line(points, fill=colors[run.algorithm], width=4)
        draw.text((width - 220, 30 + 26 * position), run.algorithm, fill=colors[run.algorithm])
    image.save(path)


def _slug(name: str) -> str:
    return name.lower().replace("/", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PTDL 2.2.2 prototype delivery experiment.")
    parser.add_argument("--output-dir", default="ptdl_2_2_2_results", help="Directory for CSV, SVG and metadata outputs.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller run suitable for automated tests.")
    parser.add_argument(
        "--coupled-baseline",
        action="store_true",
        help="Run MOEA/D on the reconstructed flight-propulsion-control model instead of the legacy proxy comparison.",
    )
    args = parser.parse_args()
    if args.coupled_baseline:
        result = run_coupled_moead_baseline(
            args.output_dir,
            resolution=2 if args.quick else 3,
            generations=2 if args.quick else 12,
        )
        least_violating = result["least_violating"]
        print(f"MOEA/D feasible candidates: {sum(item.feasible for item in result['run'].evaluations)}")
        print(f"Least total violation: {least_violating.violation:.3f}")
        print(f"Results written to: {Path(args.output_dir).resolve()}")
        return
    kwargs = {"output_dir": args.output_dir}
    if args.quick:
        kwargs.update({"nsga_population": 18, "nsga_generations": 8, "moead_resolution": 3, "moead_generations": 12})
    result = run_delivery_experiment(**kwargs)
    print(result["comparison"].to_string(index=False))
    print(f"Results written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
