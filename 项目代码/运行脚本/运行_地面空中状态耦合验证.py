"""Estimated state handoff from takeoff roll through air task to landing roll."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import (
    calibrate_endurance_task_baseline,
    evaluate_endurance_task,
    load_endurance_profile,
)
from 低阶地面滑跑模型 import GroundRollAssumptions, simulate_ground_roll
from 全任务剖面定义 import preliminary_hybrid_aircraft
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from VSPAERO气动代理模型 import VSPAeroSurrogate


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "动力下降模式正式重优化_20261003"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE23点.json"
DESIGN_LABELS = (
    "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
    "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m",
    "payload_mass_kg", "hybrid_system_mass_kg",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "地面空中状态耦合估计验证_20261004",
    )
    parser.add_argument("--shaft-efficiency", type=float, default=0.90)
    parser.add_argument("--ground-accessory-load-kw", type=float, default=1.0)
    parser.add_argument("--power-response-time-constant-s", type=float, default=1.5)
    parser.add_argument("--power-rate-limit-kwps", type=float, default=60.0)
    args = parser.parse_args()
    if not 0.5 <= args.shaft_efficiency <= 1.0:
        raise SystemExit("--shaft-efficiency must be in [0.5, 1.0].")
    if args.ground_accessory_load_kw < 0.0:
        raise SystemExit("--ground-accessory-load-kw must be nonnegative.")

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = [float(value) for value in candidate["decision"]]
    design = AirframeDesign(**dict(zip(DESIGN_LABELS, decision[:10])))
    calibration = calibrate_endurance_task_baseline()
    surrogate = VSPAeroSurrogate.from_json(SURROGATE_PATH)
    aero = surrogate.corrected_aero(design, calibration.dimension_calibration)
    aircraft = preliminary_hybrid_aircraft(
        design, calibration.dimension_calibration, calibration.assumptions,
        aero_override=aero,
    )
    profile = load_endurance_profile()
    speeds = [float(segment["speed_target_mps"]) for segment in profile["segments"]]
    speeds[2] = decision[10]
    engine_caps = decision[11:17]
    motor_assist = decision[17:23]

    # Candidate schedule powers are bus-side estimates; map them to shaft power
    # with a replaceable efficiency rather than equating electrical and shaft power.
    takeoff_shaft_kw = max(0.0, engine_caps[0] + motor_assist[0]) * args.shaft_efficiency
    ground_assumptions = GroundRollAssumptions(
        shaft_power_response_time_constant_s=args.power_response_time_constant_s,
        shaft_power_rate_limit_kwps=args.power_rate_limit_kwps,
    )
    takeoff = simulate_ground_roll(
        aircraft, "takeoff", takeoff_shaft_kw, assumptions=ground_assumptions,
    )
    if not takeoff.feasible:
        raise SystemExit(f"Estimated takeoff roll did not reach liftoff: {takeoff.termination_reason}")
    total_takeoff_bus_power_kw = max(0.0, engine_caps[0]) + max(0.0, motor_assist[0])
    engine_bus_fraction = (
        max(0.0, engine_caps[0]) / total_takeoff_bus_power_kw
        if total_takeoff_bus_power_kw > 0.0
        else 0.0
    )
    motor_bus_fraction = 1.0 - engine_bus_fraction
    takeoff_bus_energy_mj = takeoff.shaft_energy_mj / args.shaft_efficiency
    engine_bus_energy_mj = takeoff_bus_energy_mj * engine_bus_fraction
    motor_bus_energy_mj = takeoff_bus_energy_mj * motor_bus_fraction
    ground_fuel_used_kg = engine_bus_energy_mj / max(
        aircraft.engine_generator_efficiency * 43.0, 1e-9
    )
    takeoff_battery_energy_mj = (
        motor_bus_energy_mj
        + args.ground_accessory_load_kw * takeoff.duration_s / 1000.0
    ) / max(aircraft.battery_discharge_efficiency, 1e-9)
    air_initial_soc = aircraft.initial_soc - takeoff_battery_energy_mj / aircraft.battery_capacity_mj
    air_initial_fuel_kg = aircraft.initial_fuel_kg - ground_fuel_used_kg
    if air_initial_soc <= aircraft.soc_safe:
        raise SystemExit("Estimated takeoff energy use would cross the configured SOC safety floor.")

    air = evaluate_endurance_task(
        design,
        engine_caps_kw=engine_caps,
        motor_assist_kw=motor_assist,
        speeds_mps=speeds,
        initial_speed_mps=takeoff.terminal_speed_mps,
        initial_soc=air_initial_soc,
        initial_fuel_kg=air_initial_fuel_kg,
        calibration=calibration.dimension_calibration,
        assumptions=calibration.assumptions,
        aero_override=aero,
        dispatch_policy=MPCEnergyManagementCoordinator(),
        powertrain=HybridSystemPowertrain(config=HybridSystemAdapterConfig()),
    )
    flare = air.simulation.profile.loc[
        air.simulation.profile["phase"] == "flare"
    ].iloc[-1]
    landing = simulate_ground_roll(
        aircraft,
        "landing",
        0.0,
        assumptions=ground_assumptions,
        initial_speed_mps=float(flare["v_mps"]),
        mass_kg=float(flare["mass_kg"]),
    )
    landing_battery_energy_mj = (
        args.ground_accessory_load_kw
        * landing.duration_s
        / 1000.0
        / max(aircraft.battery_discharge_efficiency, 1e-9)
    )
    final_soc_after_landing = max(
        0.0,
        float(air.simulation.profile["soc"].iloc[-1])
        - landing_battery_energy_mj / aircraft.battery_capacity_mj,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    air.simulation.profile.to_csv(output_dir / "空中任务时序.csv", index=False)
    rows = [
        {
            "segment": "takeoff_roll",
            "feasible": takeoff.feasible,
            "duration_s": takeoff.duration_s,
            "distance_m": takeoff.distance_m,
            "initial_speed_mps": 0.0,
            "terminal_speed_mps": takeoff.terminal_speed_mps,
            "terminal_altitude_m": 0.0,
            "terminal_mass_kg": aircraft.empty_mass_kg + air_initial_fuel_kg,
            "fuel_used_estimate_kg": ground_fuel_used_kg,
            "soc_after_segment_estimate": air_initial_soc,
        },
        {
            "segment": "air_task",
            "feasible": air.result.feasible,
            "duration_s": air.achieved_duration_h * 3600.0,
            "distance_m": air.simulation.range_m,
            "initial_speed_mps": air.mission.initial_speed_mps,
            "terminal_speed_mps": float(air.simulation.profile["v_mps"].iloc[-1]),
            "terminal_altitude_m": float(air.simulation.profile["h_m"].iloc[-1]),
            "terminal_mass_kg": float(air.simulation.profile["mass_kg"].iloc[-1]),
            "fuel_used_estimate_kg": float(air.simulation.profile["mass_kg"].iloc[0])
            - float(air.simulation.profile["mass_kg"].iloc[-1]),
            "soc_after_segment_estimate": float(air.simulation.profile["soc"].iloc[-1]),
        },
        {
            "segment": "landing_roll",
            "feasible": landing.feasible,
            "duration_s": landing.duration_s,
            "distance_m": landing.distance_m,
            "initial_speed_mps": float(flare["v_mps"]),
            "terminal_speed_mps": landing.terminal_speed_mps,
            "terminal_altitude_m": 0.0,
            "terminal_mass_kg": float(flare["mass_kg"]),
            "fuel_used_estimate_kg": 0.0,
            "soc_after_segment_estimate": final_soc_after_landing,
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "地面空中状态交接.csv", index=False)
    metadata = {
        "candidate": str(args.candidate.resolve()),
        "status": "estimated kinematic state handoff; not calibrated or validated",
        "assumptions": {
            "shaft_efficiency": args.shaft_efficiency,
            "ground_roll": asdict(ground_assumptions),
            "takeoff_shaft_power_kw": takeoff_shaft_kw,
            "integrated_takeoff_shaft_energy_mj": takeoff.shaft_energy_mj,
            "integrated_takeoff_bus_energy_mj": takeoff_bus_energy_mj,
            "engine_bus_energy_mj": engine_bus_energy_mj,
            "motor_bus_energy_mj": motor_bus_energy_mj,
            "engine_bus_fraction_estimate": engine_bus_fraction,
            "motor_bus_fraction_estimate": motor_bus_fraction,
            "ground_accessory_load_kw": args.ground_accessory_load_kw,
            "estimated_takeoff_fuel_used_kg": ground_fuel_used_kg,
            "estimated_takeoff_battery_energy_mj": takeoff_battery_energy_mj,
            "air_initial_fuel_kg": air_initial_fuel_kg,
            "air_initial_soc": air_initial_soc,
            "estimated_landing_battery_energy_mj": landing_battery_energy_mj,
            "mission_end_soc_after_landing_estimate": final_soc_after_landing,
        },
        "handoffs": {
            "takeoff_terminal_speed_to_air_initial_speed_mps": takeoff.terminal_speed_mps,
            "flare_terminal_speed_to_landing_initial_speed_mps": float(flare["v_mps"]),
            "flare_terminal_mass_to_landing_mass_kg": float(flare["mass_kg"]),
        },
        "air_task_feasible": air.result.feasible,
        "air_task_duration_h": air.achieved_duration_h,
        "air_task_terminal_soc": float(air.simulation.profile["soc"].iloc[-1]),
        "air_task_terminal_fuel_kg": max(
            0.0,
            float(air.simulation.profile["mass_kg"].iloc[-1])
            - float(aircraft.empty_mass_kg),
        ),
        "limitations": [
            "Takeoff shaft energy is integrated from the transient low-order power response; source split remains an estimated constant bus-power share.",
            "Takeoff electrical power is converted to shaft power with a user-replaceable efficiency estimate.",
            "Shaft-power spin-up uses a first-order response and rate limit; braking regeneration is omitted.",
            "Engine fuel use is estimated from generator efficiency and 43 MJ/kg fuel lower heating value.",
            "Runway wind, slope, surface, braking and ground effect remain low-order assumptions.",
            "This software handoff test is not a takeoff/landing performance validation.",
        ],
    }
    (output_dir / "地面空中状态耦合说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
