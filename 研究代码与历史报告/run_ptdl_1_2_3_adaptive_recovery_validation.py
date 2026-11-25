"""Validate wind recovery and health-adaptive EMS in PTDL 1.2.2/1.2.3."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_80h_endurance_profile import calibrate_endurance_task_baseline, evaluate_endurance_task
from ptdl_flight_propulsion_control import FlightControlParameters, evaluate_coupled_mission
from ptdl_intelligent_energy_management import FaultTolerantFuzzyCoordinator, FuzzyAdaptiveCoordinator, HealthState
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from run_ptdl_1_2_3_robustness_validation import (
    DISTURBANCES,
    SCENARIOS,
    candidate_inputs,
    design_from_candidate,
    mission_for,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_1_2_3_adaptive_recovery_results"
PARETO = ROOT / "ptdl_moead_h6_robust_vsp_results" / "稳健H6_MOEAD可行Pareto候选.csv"
SURROGATE = ROOT / "ptdl_vsp_targeted_candidate_results" / "VSPAERO扩充代理模型.json"
COMBINED = next(item for item in DISTURBANCES if item.label == "保守组合")


def build_policy(name: str):
    if name == "基线模糊调度":
        return FuzzyAdaptiveCoordinator()
    return FaultTolerantFuzzyCoordinator(
        FuzzyAdaptiveCoordinator(),
        HealthState(engine_available_fraction=0.85, battery_available_energy_fraction=0.80),
    )


def local_recovery_record(scenario, policy_name: str, aircraft, candidate: pd.Series) -> dict[str, object]:
    caps, motors, _ = candidate_inputs(candidate)
    phase_index = {
        "main_climb_to_leveloff": 0,
        "low_speed_high_lift": 1,
        "endurance_loiter": 2,
        "long_descent": 3,
    }[scenario.phase_name]
    active_aircraft = replace(
        aircraft,
        cd0=aircraft.cd0 * COMBINED.cd0_factor,
        battery_capacity_mj=aircraft.battery_capacity_mj * COMBINED.battery_capacity_factor,
        engine_efficiency_values=tuple(value * COMBINED.engine_efficiency_factor for value in aircraft.engine_efficiency_values),
        motor_power_min_kw=0.0 if scenario.disable_motor else aircraft.motor_power_min_kw,
        motor_power_max_kw=0.0 if scenario.disable_motor else aircraft.motor_power_max_kw,
    )
    mission = mission_for(scenario, COMBINED, float(caps[phase_index]), float(motors[phase_index]))
    controller = FlightControlParameters(vertical_wind_compensation_gain=0.0 if policy_name == "基线模糊调度" else 1.0)
    simulation, _ = evaluate_coupled_mission(mission, active_aircraft, controller=controller, dispatch_policy=build_policy(policy_name))
    profile = simulation.profile
    h_error = abs(float(profile["h_m"].iloc[-1]) - scenario.altitude_target_m)
    v_error = abs(float(profile["v_mps"].iloc[-1]) - scenario.speed_target_mps)
    meets = (
        simulation.termination_reason == "mission_complete"
        and h_error <= scenario.altitude_tolerance_m
        and v_error <= scenario.speed_tolerance_mps
    )
    return {
        "scenario": scenario.label,
        "policy": policy_name,
        "tracking_requirement_met": meets,
        "terminal_altitude_error_m": h_error,
        "terminal_speed_error_mps": v_error,
        "peak_altitude_error_m": float(profile["e_h_m"].abs().max()),
        "peak_power_shortfall_kw": float(profile["power_shortfall_kw"].max()),
        "minimum_soc": float(profile["soc"].min()),
        "terminal_soc": float(profile["soc"].iloc[-1]),
    }


def endurance_record(policy_name: str, candidate: pd.Series, calibration, aero) -> dict[str, object]:
    caps, motors, speeds = candidate_inputs(candidate)
    assumptions = replace(
        calibration.assumptions,
        engine_efficiency_values=tuple(value * COMBINED.engine_efficiency_factor for value in calibration.assumptions.engine_efficiency_values),
        hybrid_system_specific_energy_wh_per_kg=calibration.assumptions.hybrid_system_specific_energy_wh_per_kg * COMBINED.battery_capacity_factor,
    )
    evaluation = evaluate_endurance_task(
        design_from_candidate(candidate), engine_caps_kw=caps, motor_assist_kw=motors, speeds_mps=speeds,
        target_total_duration_h=86.0, calibration=calibration.dimension_calibration, assumptions=assumptions,
        aero_override=replace(aero, cd0=aero.cd0 * COMBINED.cd0_factor), dispatch_policy=build_policy(policy_name),
    )
    profile = evaluation.simulation.profile
    return {
        "policy": policy_name,
        "achieved_duration_h": evaluation.achieved_duration_h,
        "margin_to_80h": evaluation.achieved_duration_h - 80.0,
        "meets_80h": evaluation.achieved_duration_h >= 80.0 - 1e-9,
        "minimum_soc": float(profile["soc"].min()),
        "terminal_soc": float(profile["soc"].iloc[-1]),
        "termination_reason": evaluation.simulation.termination_reason,
    }


def export_figures(local: pd.DataFrame, endurance: pd.DataFrame) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    scenario_names = list(local["scenario"].drop_duplicates())
    policies = list(local["policy"].drop_duplicates())
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    x = np.arange(len(scenario_names))
    width = 0.34
    for index, policy in enumerate(policies):
        subset = local[local["policy"] == policy].set_index("scenario").loc[scenario_names]
        axes[0].bar(x + (index - 0.5) * width, subset["terminal_altitude_error_m"], width, label=policy)
        axes[1].bar(x + (index - 0.5) * width, subset["minimum_soc"], width, label=policy)
    axes[0].set(title="组合扰动下的终端高度误差", ylabel="绝对误差 (m)")
    axes[1].set(title="组合扰动下的最低 SOC", ylabel="最低 SOC (-)")
    for axis in axes:
        axis.set_xticks(x, scenario_names, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(OUTPUT / "图20_风扰补偿与健康自适应调度局部恢复对比.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), constrained_layout=True)
    colors = ["#1769aa" if value else "#c2410c" for value in endurance["meets_80h"]]
    axes[0].bar(endurance["policy"], endurance["achieved_duration_h"], color=colors)
    axes[0].axhline(80.0, color="#c2410c", linestyle="--", label="80 h 硬约束")
    axes[0].set(title="组合退化任务的续航对比", ylabel="可达总时长 (h)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(frameon=False)
    axes[1].bar(endurance["policy"], endurance["terminal_soc"], color="#2a9d8f")
    axes[1].set(title="任务终端 SOC", ylabel="SOC (-)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(OUTPUT / "图21_健康自适应SOC参考与组合退化续航对比.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate = pd.read_csv(PARETO).sort_values("candidate_index").iloc[0]
    calibration = calibrate_endurance_task_baseline()
    aero = VSPAeroSurrogate.from_json(SURROGATE).corrected_aero(design_from_candidate(candidate), calibration.dimension_calibration)
    from ptdl_full_task_profile import preliminary_hybrid_aircraft
    aircraft = preliminary_hybrid_aircraft(design_from_candidate(candidate), calibration.dimension_calibration, calibration.assumptions, aero_override=aero)
    policies = ("基线模糊调度", "风扰补偿与健康自适应调度")
    local = pd.DataFrame([local_recovery_record(scenario, policy, aircraft, candidate) for scenario in SCENARIOS for policy in policies])
    endurance = pd.DataFrame([endurance_record(policy, candidate, calibration, aero) for policy in policies])
    local.to_csv(OUTPUT / "表8_组合扰动局部恢复对比.csv", index=False, encoding="utf-8-sig")
    endurance.to_csv(OUTPUT / "表9_组合退化全任务续航对比.csv", index=False, encoding="utf-8-sig")
    export_figures(local, endurance)
    (OUTPUT / "阶段审阅说明.md").write_text(
        "# 1.2.2 和 1.2.3 自适应恢复验证\n\n"
        "本试验以既有稳健候选为对象，在同一组合退化口径下比较基线模糊调度与风扰补偿、发动机降额识别及 SOC 参考自适应后的策略。"
        "垂直风为短时局部工况，完整任务续航仅施加阻力、发动机、电池和效率退化。结果用于比较当前参数化模型内的控制作用，不代表真实部件飞行试验结论。\n",
        encoding="utf-8",
    )
    print(local.to_string(index=False))
    print(endurance.to_string(index=False))


if __name__ == "__main__":
    main()
