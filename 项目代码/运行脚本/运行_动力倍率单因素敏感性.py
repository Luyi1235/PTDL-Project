"""Vary engine, motor, and battery sizing multipliers one at a time."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import calibrate_endurance_task_baseline, evaluate_endurance_task, load_endurance_profile
from 飞推控闭环物理模型 import LongitudinalDragDisturbance
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from VSPAERO气动代理模型 import load_vsp_surrogate


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "H6_三项风险闭环复验"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "动力倍率单因素敏感性_131GP_20261011"
SURROGATE_PATH = (
    ROOT
    / "气动代理模型数据"
    / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
)
DESIGN_LABELS = (
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
DISTURBANCE = LongitudinalDragDisturbance(
    phase_name="endurance_loiter",
    start_time_s=3600.0,
    duration_s=600.0,
    additional_drag_n=100.0,
    response_step_s=5.0,
    recovery_window_s=900.0,
    recovery_dwell_s=30.0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surrogate", type=Path, default=SURROGATE_PATH)
    parser.add_argument("--relative-step", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.relative_step <= 0.25:
        raise SystemExit("relative-step must be in (0, 0.25].")

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = candidate["decision"]
    design = AirframeDesign(**dict(zip(DESIGN_LABELS, decision[:10])))
    calibration = calibrate_endurance_task_baseline()
    assumptions = calibration.assumptions
    surrogate = load_vsp_surrogate(args.surrogate)
    aero = surrogate.corrected_aero(design, calibration.dimension_calibration)
    profile = load_endurance_profile()
    speeds = [float(segment["speed_target_mps"]) for segment in profile["segments"]]
    speeds[2] = float(decision[10])
    engine_caps = [float(value) for value in decision[11:17]]
    motor_assist = [float(value) for value in decision[17:23]]

    baseline_factor = float(assumptions.engine_rating_multiplier)
    if not (
        baseline_factor == assumptions.motor_rating_multiplier
        == assumptions.battery_energy_multiplier
    ):
        raise ValueError("The fixed-candidate baseline must use one unified multiplier.")
    factor_cases = [("baseline", "all", baseline_factor)]
    for label, field_name in (
        ("engine", "engine_rating_multiplier"),
        ("motor", "motor_rating_multiplier"),
        ("battery_energy", "battery_energy_multiplier"),
    ):
        baseline_value = float(getattr(assumptions, field_name))
        factor_cases.extend(
            (
                (f"{label}_minus", field_name, baseline_value * (1.0 - args.relative_step)),
                (f"{label}_plus", field_name, baseline_value * (1.0 + args.relative_step)),
            )
        )

    rows: list[dict[str, object]] = []
    for case_name, field_name, factor in factor_cases:
        active_assumptions = assumptions
        if field_name == "all":
            active_assumptions = replace(
                assumptions,
                engine_rating_multiplier=factor,
                motor_rating_multiplier=factor,
                battery_energy_multiplier=factor,
            )
        else:
            active_assumptions = replace(assumptions, **{field_name: factor})
        evaluation = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            calibration=calibration.dimension_calibration,
            assumptions=active_assumptions,
            aero_override=aero,
            dispatch_policy=MPCEnergyManagementCoordinator(),
            powertrain=HybridSystemPowertrain(
                config=HybridSystemAdapterConfig()
            ),
            disturbances=(DISTURBANCE,),
        )
        constraints = evaluation.result.constraints
        diagnostics = evaluation.result.diagnostics
        rows.append(
            {
                "case": case_name,
                "changed_multiplier": field_name,
                "factor": factor,
                "engine_multiplier": active_assumptions.engine_rating_multiplier,
                "motor_multiplier": active_assumptions.motor_rating_multiplier,
                "battery_energy_multiplier": active_assumptions.battery_energy_multiplier,
                "feasible": evaluation.result.feasible,
                "total_violation": evaluation.result.total_violation,
                "endurance_h": evaluation.achieved_duration_h,
                "fuel_used_kg": float(diagnostics["fuel_mass_kg"])
                - evaluation.simulation.terminal_fuel_kg,
                "terminal_fuel_kg": evaluation.simulation.terminal_fuel_kg,
                "terminal_soc": float(evaluation.simulation.profile["soc"].iloc[-1]),
                "minimum_soc": float(evaluation.simulation.profile["soc"].min()),
                "minimum_signed_constraint_margin": min(-float(value) for value in constraints.values()),
                "battery_discharge_energy_mj": diagnostics.get("battery_discharge_energy_mj"),
                "battery_charge_energy_mj": diagnostics.get("battery_charge_energy_mj"),
                "battery_bus_to_chemical_discharge_efficiency": diagnostics.get(
                    "battery_bus_to_chemical_discharge_efficiency"
                ),
                "battery_chemical_to_bus_charge_efficiency": diagnostics.get(
                    "battery_chemical_to_bus_charge_efficiency"
                ),
                "battery_round_trip_efficiency_estimate": diagnostics.get(
                    "battery_round_trip_efficiency_estimate"
                ),
                "battery_efficiency_diagnostic_basis": diagnostics.get(
                    "battery_efficiency_diagnostic_basis"
                ),
            }
        )
        print(
            f"{case_name}: feasible={evaluation.result.feasible}, "
            f"fuel={rows[-1]['fuel_used_kg']:.3f} kg, SOC={rows[-1]['terminal_soc']:.4f}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "动力倍率单因素对比.csv", index=False)
    metadata = {
        "candidate": str(args.candidate.resolve()),
        "aerodynamic_surrogate": str(args.surrogate.resolve()),
        "aerodynamic_training_samples": getattr(surrogate, "training_samples", None),
        "baseline_multipliers": {
            "engine": assumptions.engine_rating_multiplier,
            "motor": assumptions.motor_rating_multiplier,
            "battery_energy": assumptions.battery_energy_multiplier,
        },
        "relative_step": args.relative_step,
        "case_count": len(rows),
        "method": "same candidate, same controller, same disturbance; vary one sizing multiplier at a time around the 1.20 baseline",
        "limits": [
            "Sizing multipliers remain provisional assumptions, not hardware calibration.",
            "Battery port efficiencies are computed from simulated bus and battery-terminal powers; they are not measured cell/thermal/aging efficiency.",
            "No optimizer re-run was performed for each multiplier; this is a fixed-candidate robustness screen.",
        ],
    }
    (output_dir / "动力倍率敏感性说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
