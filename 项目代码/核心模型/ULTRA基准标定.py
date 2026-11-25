"""Open-data-calibrated ULTRA endurance benchmark for PTDL.

This module is a transparent benchmark model, not a full digital twin of the
DZYNE/AFRL ULTRA aircraft. Public data establish selected geometry, mass,
fuel, and payload scale anchors. The 70 kt speed below is a project reference
operating point, not an official ULTRA cruise-speed specification. Publicly unavailable aerodynamic and
propulsion maps are represented by declared assumptions, with CD0 inversely
calibrated so that the reference model reproduces 72 h equivalent loiter.

The purpose is to establish a defensible 72 h baseline before assessing the
PTDL 80 h requirement under identical fuel, payload, and takeoff-mass limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from 飞推控闭环物理模型 import (
    AircraftParameters,
    MissionDefinition,
    PhaseCommand,
    evaluate_coupled_mission,
    simulate_mission,
)


KNOT_TO_MPS = 0.514444
FOOT_TO_M = 0.3048
POUND_TO_KG = 0.45359237
US_GALLON_TO_L = 3.785411784


@dataclass(frozen=True)
class UltraOpenData:
    """Public ULTRA scale figures and transparent operating-point assumptions."""

    baseline_endurance_h: float = 72.0
    target_endurance_h: float = 80.0
    # Project reference mission point; not an official ULTRA cruise rating.
    cruise_speed_knots: float = 70.0
    wingspan_ft: float = 82.0
    mtow_lb: float = 3000.0
    fuel_capacity_us_gal: float = 162.0
    payload_lb: float = 400.0
    jet_a_density_kg_per_l: float = 0.80
    assumed_wing_area_m2: float = 26.0
    assumed_oswald_efficiency: float = 0.84
    assumed_cl_alpha_per_rad: float = 5.2
    assumed_propulsive_efficiency: float = 0.86
    assumed_engine_generator_efficiency: float = 0.30
    assumed_engine_electrical_power_max_kw: float = 80.0
    calibration_altitude_m: float = 3000.0

    @property
    def cruise_speed_mps(self) -> float:
        return self.cruise_speed_knots * KNOT_TO_MPS

    @property
    def wingspan_m(self) -> float:
        return self.wingspan_ft * FOOT_TO_M

    @property
    def mtow_kg(self) -> float:
        return self.mtow_lb * POUND_TO_KG

    @property
    def payload_kg(self) -> float:
        return self.payload_lb * POUND_TO_KG

    @property
    def fuel_mass_kg(self) -> float:
        return self.fuel_capacity_us_gal * US_GALLON_TO_L * self.jet_a_density_kg_per_l

    @property
    def nonfuel_mass_kg(self) -> float:
        """Airframe, propulsion system, avionics, and declared payload combined."""
        return self.mtow_kg - self.fuel_mass_kg


@dataclass(frozen=True)
class UltraCalibration:
    cd0: float
    baseline_endurance_h: float
    baseline_fuel_used_kg: float
    cruise_power_kw: float
    reference: UltraOpenData


def ultra_equivalent_loiter_mission(
    duration_h: float,
    reference: UltraOpenData,
    speed_mps: float | None = None,
) -> MissionDefinition:
    """Return a steady-state loiter mission with no unsubstantiated transit radius.

    It represents the level, constant-speed part of the public endurance claim.
    Transit, climb, descent, winds, payload duty cycles, and reserves are added
    only after their real mission inputs are available.
    """
    cruise_speed = reference.cruise_speed_mps if speed_mps is None else speed_mps
    return MissionDefinition(
        phases=(
            PhaseCommand(
                "equivalent_loiter",
                duration_h * 3600.0,
                reference.calibration_altitude_m,
                cruise_speed,
                reference.assumed_engine_electrical_power_max_kw,
                0.0,
                max_flight_path_angle_deg=1.0,
                time_step_s=120.0,
            ),
        ),
        required_loiter_s=duration_h * 3600.0,
        initial_altitude_m=reference.calibration_altitude_m,
        initial_speed_mps=cruise_speed,
        initial_soc=0.50,
    )


def ultra_aircraft(reference: UltraOpenData, cd0: float) -> AircraftParameters:
    """Build a fixed-MTOW conventional baseline; no hidden hybrid energy credit."""
    return AircraftParameters(
        empty_mass_kg=reference.nonfuel_mass_kg,
        initial_fuel_kg=reference.fuel_mass_kg,
        wing_area_m2=reference.assumed_wing_area_m2,
        wing_span_m=reference.wingspan_m,
        cd0=cd0,
        oswald_efficiency=reference.assumed_oswald_efficiency,
        cl_alpha_per_rad=reference.assumed_cl_alpha_per_rad,
        prop_disk_area_m2=8.0,
        induced_loss_factor=1.18,
        propulsive_efficiency=reference.assumed_propulsive_efficiency,
        engine_electrical_power_max_kw=reference.assumed_engine_electrical_power_max_kw,
        engine_generator_efficiency=reference.assumed_engine_generator_efficiency,
        motor_power_min_kw=0.0,
        motor_power_max_kw=0.0,
        battery_capacity_mj=1.0,
        initial_soc=0.50,
        soc_safe=0.0,
        soc_reserve=0.0,
    )


def calibrate_ultra_baseline(reference: UltraOpenData = UltraOpenData()) -> UltraCalibration:
    """Inverse-calibrate CD0 to the project 72 h equivalent-loiter baseline."""
    lo_cd0, hi_cd0 = 0.002, 0.080
    mission = ultra_equivalent_loiter_mission(reference.target_endurance_h + 20.0, reference)
    for _ in range(26):
        cd0 = 0.5 * (lo_cd0 + hi_cd0)
        simulation = simulate_mission(mission, ultra_aircraft(reference, cd0))
        if simulation.profile["time_s"].iloc[-1] / 3600.0 >= reference.baseline_endurance_h:
            lo_cd0 = cd0
        else:
            hi_cd0 = cd0
    cd0 = 0.5 * (lo_cd0 + hi_cd0)
    aircraft = ultra_aircraft(reference, cd0)
    simulation = simulate_mission(mission, aircraft)
    profile = simulation.profile
    duration_h = float(profile["time_s"].iloc[-1] / 3600.0)
    return UltraCalibration(
        cd0=cd0,
        baseline_endurance_h=duration_h,
        baseline_fuel_used_kg=reference.fuel_mass_kg - simulation.terminal_fuel_kg,
        cruise_power_kw=float(profile["p_engine_kw"].tail(min(100, len(profile))).mean()),
        reference=reference,
    )


def endurance_at_speed(calibration: UltraCalibration, speed_mps: float, duration_cap_h: float = 120.0) -> float:
    mission = ultra_equivalent_loiter_mission(duration_cap_h, calibration.reference, speed_mps=speed_mps)
    simulation = simulate_mission(mission, ultra_aircraft(calibration.reference, calibration.cd0))
    return float(simulation.profile["time_s"].iloc[-1] / 3600.0)


def required_cd0_factor_for_target(calibration: UltraCalibration) -> float:
    """Return the largest CD0 multiplier that still reaches the 80 h target."""
    reference = calibration.reference
    target = reference.target_endurance_h
    lo_factor, hi_factor = 0.45, 1.0
    for _ in range(24):
        factor = 0.5 * (lo_factor + hi_factor)
        mission = ultra_equivalent_loiter_mission(target + 20.0, reference)
        simulation = simulate_mission(mission, ultra_aircraft(reference, calibration.cd0 * factor))
        duration_h = simulation.profile["time_s"].iloc[-1] / 3600.0
        if duration_h >= target:
            lo_factor = factor
        else:
            hi_factor = factor
    return float(0.5 * (lo_factor + hi_factor))


def required_engine_efficiency_for_target(calibration: UltraCalibration) -> float:
    """Return the lowest constant generator efficiency that reaches 80 h.

    This is an energy-budget sensitivity, not a claim about an available
    production engine.  A real hybrid assessment must replace the constant
    efficiency with a measured engine BSFC map and battery/motor efficiency map.
    """
    reference = calibration.reference
    mission = ultra_equivalent_loiter_mission(reference.target_endurance_h + 20.0, reference)
    lo_efficiency, hi_efficiency = 0.25, 0.50
    for _ in range(24):
        efficiency = 0.5 * (lo_efficiency + hi_efficiency)
        aircraft = replace(
            ultra_aircraft(reference, calibration.cd0),
            engine_generator_efficiency=efficiency,
        )
        duration_h = simulate_mission(mission, aircraft).profile["time_s"].iloc[-1] / 3600.0
        if duration_h >= reference.target_endurance_h:
            hi_efficiency = efficiency
        else:
            lo_efficiency = efficiency
    return float(0.5 * (lo_efficiency + hi_efficiency))


def run_ultra_benchmark(output_dir: str | Path, reference: UltraOpenData = UltraOpenData()) -> dict[str, object]:
    """Calibrate the 72 h ULTRA benchmark and run transparent 80 h checks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calibration = calibrate_ultra_baseline(reference)
    baseline_mission = ultra_equivalent_loiter_mission(reference.target_endurance_h + 20.0, reference)
    baseline_simulation, baseline_evaluation = evaluate_coupled_mission(
        baseline_mission,
        ultra_aircraft(reference, calibration.cd0),
    )

    speeds = np.linspace(24.0, 44.0, 81)
    endurance = np.array([endurance_at_speed(calibration, speed) for speed in speeds])
    speed_scan = pd.DataFrame({"cruise_speed_mps": speeds, "cruise_speed_knots": speeds / KNOT_TO_MPS, "endurance_h": endurance})
    speed_scan.to_csv(output_path / "ultra_speed_endurance_scan.csv", index=False)
    best_speed_row = speed_scan.loc[speed_scan["endurance_h"].idxmax()]

    cd0_factor = required_cd0_factor_for_target(calibration)
    required_engine_efficiency = required_engine_efficiency_for_target(calibration)
    fixed_speed_target_h = endurance_at_speed(calibration, reference.cruise_speed_mps, reference.target_endurance_h + 20.0)
    summary = pd.DataFrame(
        [
            {
                "case": "72h_calibrated_baseline",
                "cruise_speed_knots": reference.cruise_speed_knots,
                "endurance_h": calibration.baseline_endurance_h,
                "cd0_factor": 1.0,
                "engine_generator_efficiency": reference.assumed_engine_generator_efficiency,
                "description": "Open-data inverse-calibrated equivalent loiter baseline",
            },
            {
                "case": "speed_only_best_case",
                "cruise_speed_knots": float(best_speed_row["cruise_speed_knots"]),
                "endurance_h": float(best_speed_row["endurance_h"]),
                "cd0_factor": 1.0,
                "engine_generator_efficiency": reference.assumed_engine_generator_efficiency,
                "description": "Same fuel, MTOW, payload, and aerodynamics; speed scan only",
            },
            {
                "case": "80h_fixed_speed_requirement",
                "cruise_speed_knots": reference.cruise_speed_knots,
                "endurance_h": fixed_speed_target_h,
                "cd0_factor": cd0_factor,
                "engine_generator_efficiency": reference.assumed_engine_generator_efficiency,
                "description": "Largest drag-polar CD0 multiplier that reaches 80 h at 70 kt",
            },
            {
                "case": "80h_fixed_speed_efficiency_requirement",
                "cruise_speed_knots": reference.cruise_speed_knots,
                "endurance_h": reference.target_endurance_h,
                "cd0_factor": 1.0,
                "engine_generator_efficiency": required_engine_efficiency,
                "description": "Constant engine-generator efficiency required for 80 h at 70 kt",
            },
        ]
    )
    summary["cd0_reduction_percent"] = 100.0 * (1.0 - summary["cd0_factor"])
    summary.to_csv(output_path / "ultra_80h_feasibility_summary.csv", index=False)
    baseline_simulation.profile.to_csv(output_path / "ultra_72h_baseline_time_history.csv", index=False)
    metadata = {
        "model_status": "open-data-calibrated ULTRA benchmark, not a certified or proprietary digital twin",
        "calibration_definition": "72 h equivalent pure loiter at 70 kt and assumed 3000 m altitude",
        "public_inputs": {
            "wingspan_ft": reference.wingspan_ft,
            "mtow_lb": reference.mtow_lb,
            "fuel_capacity_us_gal": reference.fuel_capacity_us_gal,
            "payload_lb": reference.payload_lb,
        },
        "reference_operating_point_assumptions": {
            "cruise_speed_knots": reference.cruise_speed_knots,
            "calibration_altitude_m": reference.calibration_altitude_m,
        },
        "assumptions": {
            "fuel_density_kg_per_l": reference.jet_a_density_kg_per_l,
            "wing_area_m2": reference.assumed_wing_area_m2,
            "propulsive_efficiency": reference.assumed_propulsive_efficiency,
            "engine_generator_efficiency": reference.assumed_engine_generator_efficiency,
            "calibration_altitude_m": reference.calibration_altitude_m,
        },
        "calibrated_cd0": calibration.cd0,
        "required_engine_generator_efficiency_for_80h": required_engine_efficiency,
        "baseline_evaluation_constraints": baseline_evaluation.constraints,
    }
    (output_path / "ultra_model_metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    return {
        "calibration": calibration,
        "baseline_simulation": baseline_simulation,
        "baseline_evaluation": baseline_evaluation,
        "speed_scan": speed_scan,
        "summary": summary,
        "output_dir": output_path,
    }


if __name__ == "__main__":
    result = run_ultra_benchmark("ptdl_ultra_benchmark_results")
    print(result["summary"].to_string(index=False))
