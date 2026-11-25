"""Fixed-MTOW dimensional screening for the PTDL ULTRA reference aircraft.

This module is the outer sizing layer before the full flight-propulsion-control
MOEA/D problem is assembled.  It uses physical dimensions and a fixed takeoff
mass allocation; it does not treat CD0 or Oswald efficiency as decision
variables.  The initial study condition is the project-approved equivalent
loiter definition, so it is deliberately not presented as a full-mission
hybrid-control result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ptdl_2_2_1_evaluator import EvaluationResult
from ptdl_airframe_design import AirframeDesign, AeroCoefficients, AirframeMetrics, aero_from_design, aircraft_from_design, design_metrics, geometry_constraints
from ptdl_flight_propulsion_control import MissionDefinition, PhaseCommand, MissionSimulation, evaluate_coupled_mission, simulate_mission
from ptdl_ultra_benchmark import UltraCalibration, calibrate_ultra_baseline


@dataclass(frozen=True)
class SizingEvaluation:
    design: AirframeDesign
    cruise_speed_mps: float
    metrics: AirframeMetrics
    aerodynamics: AeroCoefficients
    simulation: MissionSimulation
    result: EvaluationResult

    @property
    def achieved_endurance_h(self) -> float:
        return float(self.simulation.profile["time_s"].iloc[-1] / 3600.0)


def equivalent_loiter_mission(
    cruise_speed_mps: float,
    calibration: UltraCalibration,
    duration_cap_h: float = 100.0,
) -> MissionDefinition:
    """Create the common initial 80 h feasibility case.

    The mission begins already at the loiter state.  Climb, transit, payload
    duty cycle, reserve policy, and return are intentionally excluded here and
    will enter the subsequent full-task-profile control optimization.
    """
    reference = calibration.reference
    return MissionDefinition(
        phases=(
            PhaseCommand(
                "endurance_loiter",
                duration_cap_h * 3600.0,
                reference.calibration_altitude_m,
                cruise_speed_mps,
                reference.assumed_engine_electrical_power_max_kw,
                0.0,
                max_flight_path_angle_deg=1.0,
                time_step_s=120.0,
            ),
        ),
        # In this pure-loiter screening case, endurance and loiter duration
        # are the same requirement.  Keep only the evaluator's endurance
        # constraint to avoid double-counting the identical 80 h shortfall.
        required_loiter_s=None,
        initial_altitude_m=reference.calibration_altitude_m,
        initial_speed_mps=cruise_speed_mps,
    )


def evaluate_fixed_mtow_design(
    design: AirframeDesign,
    cruise_speed_mps: float,
    calibration: UltraCalibration | None = None,
    duration_cap_h: float = 100.0,
) -> SizingEvaluation:
    """Evaluate one dimensional design against the 80 h hard constraint."""
    if not 28.0 <= cruise_speed_mps <= 42.0:
        raise ValueError("cruise_speed_mps must stay within the declared study range.")
    active_calibration = calibration or calibrate_dimension_baseline()
    metrics = design_metrics(design, active_calibration.reference)
    aerodynamics = aero_from_design(design, active_calibration, cruise_speed_mps)
    aircraft = aircraft_from_design(design, active_calibration)
    mission = equivalent_loiter_mission(cruise_speed_mps, active_calibration, duration_cap_h)
    simulation, result = evaluate_coupled_mission(mission, aircraft)

    # A candidate is feasible only when its 2.2.1 mission constraints and its
    # dimensional packaging/mass constraints are both satisfied.
    result.constraints.update(geometry_constraints(design, active_calibration.reference))
    result.diagnostics.update(
        {
            "achieved_endurance_h": float(simulation.profile["time_s"].iloc[-1] / 3600.0),
            "wing_span_m": metrics.wing_span_m,
            "fuel_mass_kg": metrics.fuel_mass_kg,
            "payload_mass_kg": metrics.payload_mass_kg,
            "hybrid_system_mass_kg": metrics.hybrid_system_mass_kg,
            "cd0_derived": aerodynamics.cd0,
            "oswald_efficiency_derived": aerodynamics.oswald_efficiency,
        }
    )
    return SizingEvaluation(design, cruise_speed_mps, metrics, aerodynamics, simulation, result)


def calibrate_dimension_baseline() -> UltraCalibration:
    """Calibrate the dimension-derived baseline back to the 72 h anchor.

    The original open-data benchmark used provisional values for propeller disk
    area and induced efficiency.  Once those quantities are derived from the
    actual dimensional model, a second, one-time baseline calibration is
    necessary to avoid treating a model-interface change as an aircraft design
    improvement or degradation.  This remains an inverse calibration, not an
    independent validation of proprietary ULTRA aerodynamic data.
    """
    open_data_calibration = calibrate_ultra_baseline()
    reference = open_data_calibration.reference
    mission = equivalent_loiter_mission(
        reference.cruise_speed_mps,
        open_data_calibration,
        duration_cap_h=reference.target_endurance_h + 20.0,
    )
    lo_cd0, hi_cd0 = 0.002, 0.080
    for _ in range(26):
        cd0 = 0.5 * (lo_cd0 + hi_cd0)
        candidate_calibration = replace(open_data_calibration, cd0=cd0)
        simulation = simulate_mission(mission, aircraft_from_design(AirframeDesign(), candidate_calibration))
        duration_h = float(simulation.profile["time_s"].iloc[-1] / 3600.0)
        if duration_h >= reference.baseline_endurance_h:
            lo_cd0 = cd0
        else:
            hi_cd0 = cd0
    cd0 = 0.5 * (lo_cd0 + hi_cd0)
    calibrated = replace(open_data_calibration, cd0=cd0)
    simulation = simulate_mission(mission, aircraft_from_design(AirframeDesign(), calibrated))
    profile = simulation.profile
    return replace(
        calibrated,
        baseline_endurance_h=float(profile["time_s"].iloc[-1] / 3600.0),
        baseline_fuel_used_kg=reference.fuel_mass_kg - simulation.terminal_fuel_kg,
        cruise_power_kw=float(profile["p_engine_kw"].tail(min(100, len(profile))).mean()),
    )


def sample_designs(count: int, seed: int = 20260904) -> list[tuple[AirframeDesign, float]]:
    """Generate a reproducible exploratory design of experiments."""
    if count < 1:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(seed)
    samples = [(AirframeDesign(), 36.01)]
    for _ in range(count):
        samples.append(
            (
                AirframeDesign(
                    wing_area_m2=float(rng.uniform(22.0, 40.0)),
                    aspect_ratio=float(rng.uniform(18.0, 31.0)),
                    taper_ratio=float(rng.uniform(0.30, 0.70)),
                    sweep_deg=float(rng.uniform(0.0, 8.0)),
                    tip_twist_deg=float(rng.uniform(-3.0, 1.5)),
                    fuselage_length_m=float(rng.uniform(7.5, 10.0)),
                    fuselage_diameter_m=float(rng.uniform(0.65, 0.95)),
                    propeller_diameter_m=float(rng.uniform(2.1, 3.0)),
                    payload_mass_kg=float(rng.uniform(100.0, 205.0)),
                    hybrid_system_mass_kg=float(rng.uniform(0.0, 180.0)),
                ),
                float(rng.uniform(30.0, 40.0)),
            )
        )
    return samples


def run_fixed_mtow_screen(
    output_dir: str | Path,
    sample_count: int = 48,
    seed: int = 20260904,
) -> pd.DataFrame:
    """Run the preliminary sizing screen and export a traceable result set."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calibration = calibrate_dimension_baseline()
    records: list[dict[str, float | bool | str]] = []
    for index, (design, speed) in enumerate(sample_designs(sample_count, seed)):
        evaluation = evaluate_fixed_mtow_design(design, speed, calibration)
        record: dict[str, float | bool | str] = {"candidate_index": index, "cruise_speed_mps": speed}
        record.update(asdict(design))
        record.update(asdict(evaluation.metrics))
        record.update(
            {
                "cd0_derived": evaluation.aerodynamics.cd0,
                "oswald_efficiency_derived": evaluation.aerodynamics.oswald_efficiency,
                "achieved_endurance_h": evaluation.achieved_endurance_h,
                "termination_reason": evaluation.simulation.termination_reason,
                "feasible": evaluation.result.feasible,
            }
        )
        record.update(evaluation.result.objectives)
        record.update({f"g_{name}": value for name, value in evaluation.result.constraints.items()})
        record["total_violation"] = evaluation.result.total_violation
        records.append(record)

    frame = pd.DataFrame(records).sort_values(["feasible", "achieved_endurance_h"], ascending=[False, False])
    frame.to_csv(output_path / "fixed_mtow_sizing_screen.csv", index=False)
    metadata = {
        "model_status": "preliminary fixed-MTOW equivalent-loiter sizing screen; not a full-task-profile or certified aircraft result",
        "baseline": "dimension-layer inverse-calibrated ULTRA 72 h equivalent loiter",
        "hard_constraint": "T_endurance >= 80 h",
        "fuel_rule": "fuel mass is residual MTOW after dimension-derived conventional mass, payload, and hybrid-system mass",
        "span_limit_m": 32.0,
        "payload_min_kg": 100.0,
        "hybrid_system_max_kg": 220.0,
        "sample_count_excluding_baseline": sample_count,
        "seed": seed,
    }
    (output_path / "fixed_mtow_sizing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    output = run_fixed_mtow_screen("ptdl_fixed_mtow_sizing_results")
    print(output.head(10).to_string(index=False))
