"""Run PTDL 1.2.3 disturbance and degradation validation on a robust candidate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_80h_endurance_profile import calibrate_endurance_task_baseline, evaluate_endurance_task
from ptdl_airframe_design import AirframeDesign
from ptdl_flight_propulsion_control import MissionDefinition, PhaseCommand, evaluate_coupled_mission
from ptdl_intelligent_energy_management import FaultTolerantFuzzyCoordinator, FuzzyAdaptiveCoordinator, HealthState
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_1_2_3_robustness_validation_results"
PARETO = ROOT / "ptdl_moead_h6_robust_vsp_results" / "稳健H6_MOEAD可行Pareto候选.csv"
SURROGATE = ROOT / "ptdl_vsp_targeted_candidate_results" / "VSPAERO扩充代理模型.json"


@dataclass(frozen=True)
class Scenario:
    label: str
    initial_altitude_m: float
    initial_speed_mps: float
    phase_name: str
    duration_s: float
    altitude_target_m: float
    speed_target_mps: float
    max_gamma_deg: float
    time_step_s: float
    disable_motor: bool = False
    altitude_tolerance_m: float = 25.0
    speed_tolerance_mps: float = 1.5


@dataclass(frozen=True)
class Disturbance:
    label: str
    description: str
    vertical_wind_mps: float = 0.0
    cd0_factor: float = 1.0
    engine_available_fraction: float = 1.0
    battery_capacity_factor: float = 1.0
    engine_efficiency_factor: float = 1.0


SCENARIOS = (
    Scenario("起飞后初始爬升", 50.0, 38.0, "main_climb_to_leveloff", 600.0, 1000.0, 52.0, 8.0, 5.0, False, 30.0, 1.5),
    Scenario("低速高升力", 900.0, 30.0, "low_speed_high_lift", 600.0, 900.0, 30.0, 3.0, 5.0, False, 25.0, 1.5),
    Scenario("稳定巡航", 3000.0, 36.01, "endurance_loiter", 1800.0, 3000.0, 36.01, 1.0, 10.0, False, 25.0, 1.5),
    # Glide is intentionally unpowered; its terminal band is wider than the
    # powered stages and measures safe recovery rather than powered tracking.
    Scenario("无动力滑翔", 3000.0, 36.0, "long_descent", 900.0, 1800.0, 25.0, 4.0, 5.0, True, 100.0, 7.0),
)
DISTURBANCES = (
    Disturbance("名义", "名义条件"),
    Disturbance("上升风", "垂直上升风 +1.5 m/s", vertical_wind_mps=1.5),
    Disturbance("下沉风", "垂直下沉风 -1.5 m/s", vertical_wind_mps=-1.5),
    Disturbance("阻力偏差", "CD0 +10%", cd0_factor=1.10),
    Disturbance("发动机降额", "发动机可用功率 85%", engine_available_fraction=0.85),
    Disturbance("电池衰减", "电池可用容量 80%", battery_capacity_factor=0.80),
    Disturbance("保守组合", "下沉风、CD0 +10%、发动机 85%、电池 80%", -1.5, 1.10, 0.85, 0.80, 0.95),
)


def design_from_candidate(row: pd.Series) -> AirframeDesign:
    fields = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{field: float(row[field]) for field in fields})


def candidate_inputs(row: pd.Series) -> tuple[np.ndarray, np.ndarray, list[float]]:
    phases = (
        "main_climb_to_leveloff", "short_leveloff_to_isr_altitude", "endurance_loiter",
        "long_descent", "approach_descent", "flare",
    )
    caps = np.asarray([float(row[f"engine_cap_{phase}_kw"]) for phase in phases])
    motors = np.asarray([float(row[f"motor_assist_{phase}_kw"]) for phase in phases])
    speeds = [52.0, 36.0, float(row["isr_cruise_speed_mps"]), 42.0, 30.0, 21.0]
    return caps, motors, speeds


def policy_for(disturbance: Disturbance):
    base = FuzzyAdaptiveCoordinator()
    if disturbance.engine_available_fraction < 1.0 or disturbance.battery_capacity_factor < 1.0:
        return FaultTolerantFuzzyCoordinator(base, HealthState(
            engine_available_fraction=disturbance.engine_available_fraction,
            battery_available_energy_fraction=disturbance.battery_capacity_factor,
        ))
    return base


def mission_for(scenario: Scenario, disturbance: Disturbance, engine_cap_kw: float, motor_kw: float) -> MissionDefinition:
    return MissionDefinition(phases=(PhaseCommand(
        name=scenario.phase_name,
        duration_s=scenario.duration_s,
        altitude_target_m=scenario.altitude_target_m,
        speed_target_mps=scenario.speed_target_mps,
        engine_power_kw=0.0 if scenario.disable_motor else engine_cap_kw,
        motor_power_kw=0.0 if scenario.disable_motor else motor_kw,
        max_flight_path_angle_deg=scenario.max_gamma_deg,
        time_step_s=scenario.time_step_s,
        vertical_wind_mps=disturbance.vertical_wind_mps,
    ),), initial_altitude_m=scenario.initial_altitude_m, initial_speed_mps=scenario.initial_speed_mps)


def local_case(scenario: Scenario, disturbance: Disturbance, aircraft, candidate: pd.Series) -> dict[str, object]:
    phase_caps, phase_motors, _ = candidate_inputs(candidate)
    phase_index = {
        "main_climb_to_leveloff": 0, "low_speed_high_lift": 1, "endurance_loiter": 2, "long_descent": 3,
    }[scenario.phase_name]
    active_aircraft = replace(
        aircraft,
        cd0=aircraft.cd0 * disturbance.cd0_factor,
        battery_capacity_mj=aircraft.battery_capacity_mj * disturbance.battery_capacity_factor,
        motor_power_min_kw=0.0 if scenario.disable_motor else aircraft.motor_power_min_kw,
        motor_power_max_kw=0.0 if scenario.disable_motor else aircraft.motor_power_max_kw,
    )
    mission = mission_for(scenario, disturbance, float(phase_caps[phase_index]), float(phase_motors[phase_index]))
    simulation, _ = evaluate_coupled_mission(mission, active_aircraft, dispatch_policy=policy_for(disturbance))
    profile = simulation.profile
    h_error = abs(float(profile["h_m"].iloc[-1]) - scenario.altitude_target_m)
    v_error = abs(float(profile["v_mps"].iloc[-1]) - scenario.speed_target_mps)
    h_tol = scenario.altitude_tolerance_m
    v_tol = scenario.speed_tolerance_mps
    mission_executed = simulation.termination_reason == "mission_complete"
    tracking_requirement_met = mission_executed and h_error <= h_tol and v_error <= v_tol
    settled = (profile["e_h_m"].abs() <= h_tol) & (profile["e_v_mps"].abs() <= v_tol)
    settling_time = float(profile["time_s"].iloc[-1])
    for index in range(len(profile)):
        if bool(settled.iloc[index:].all()):
            settling_time = float(profile["time_s"].iloc[index])
            break
    return {
        "scenario": scenario.label,
        "disturbance": disturbance.label,
        "description": disturbance.description,
        "mission_executed": mission_executed,
        "tracking_requirement_met": tracking_requirement_met,
        "termination_reason": simulation.termination_reason,
        "terminal_altitude_error_m": h_error,
        "terminal_speed_error_mps": v_error,
        "settling_time_s": settling_time,
        "minimum_soc": float(profile["soc"].min()),
        "fuel_consumed_kg": float(np.trapezoid(profile["fuel_flow_kgps"], profile["time_s"])),
        "peak_power_shortfall_kw": float(profile["power_shortfall_kw"].max()),
        "peak_speed_error_mps": float(profile["e_v_mps"].abs().max()),
        "peak_altitude_error_m": float(profile["e_h_m"].abs().max()),
    }


def endurance_case(disturbance: Disturbance, candidate: pd.Series, calibration, aero) -> dict[str, object]:
    caps, motors, speeds = candidate_inputs(candidate)
    assumptions = replace(
        calibration.assumptions,
        engine_efficiency_values=tuple(value * disturbance.engine_efficiency_factor for value in calibration.assumptions.engine_efficiency_values),
        hybrid_system_specific_energy_wh_per_kg=calibration.assumptions.hybrid_system_specific_energy_wh_per_kg * disturbance.battery_capacity_factor,
    )
    evaluation = evaluate_endurance_task(
        design_from_candidate(candidate), engine_caps_kw=caps, motor_assist_kw=motors, speeds_mps=speeds,
        target_total_duration_h=86.0, calibration=calibration.dimension_calibration, assumptions=assumptions,
        aero_override=replace(aero, cd0=aero.cd0 * disturbance.cd0_factor), dispatch_policy=policy_for(disturbance),
    )
    profile = evaluation.simulation.profile
    return {
        "disturbance": disturbance.label,
        "description": disturbance.description,
        "achieved_duration_h": evaluation.achieved_duration_h,
        "margin_to_80h": evaluation.achieved_duration_h - 80.0,
        "meets_80h": evaluation.achieved_duration_h >= 80.0 - 1e-9,
        "minimum_soc": float(profile["soc"].min()),
        "terminal_soc": float(profile["soc"].iloc[-1]),
        "termination_reason": evaluation.simulation.termination_reason,
    }


def export_figures(local: pd.DataFrame, endurance: pd.DataFrame) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    matrix = local.pivot(index="scenario", columns="disturbance", values="tracking_requirement_met").astype(float)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), constrained_layout=True)
    image = axes[0].imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=28, ha="right")
    axes[0].set_yticks(range(len(matrix.index)), matrix.index)
    axes[0].set_title("四类标准工况终端跟踪要求满足情况")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axes[0].text(col, row, "满足" if matrix.iloc[row, col] else "超限", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axes[0], ticks=[0, 1], label="完成状态")
    ordered = endurance.sort_values("achieved_duration_h")
    colors = ["#1769aa" if value else "#c2410c" for value in ordered["meets_80h"]]
    axes[1].barh(ordered["disturbance"], ordered["achieved_duration_h"], color=colors)
    axes[1].axvline(80.0, color="#c2410c", linestyle="--", linewidth=1.5, label="80 h 硬约束")
    axes[1].set(xlabel="86 h 延伸仿真可达总时长 (h)", title="完整任务续航鲁棒性")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(OUTPUT / "图12_1_2_3任务完成率与80小时鲁棒性.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for scenario, group in local.groupby("scenario", sort=False):
        axes[0].plot(group["disturbance"], group["peak_altitude_error_m"], marker="o", label=scenario)
        axes[1].plot(group["disturbance"], group["minimum_soc"], marker="o", label=scenario)
    for axis, label, title in zip(axes, ("峰值高度误差 (m)", "最低 SOC (-)"), ("扰动下的飞行状态偏差", "扰动下的能量安全裕度")):
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=28)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(OUTPUT / "图13_1_2_3扰动响应与SOC裕度.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate = pd.read_csv(PARETO).sort_values("candidate_index").iloc[0]
    calibration = calibrate_endurance_task_baseline()
    aero = VSPAeroSurrogate.from_json(SURROGATE).corrected_aero(design_from_candidate(candidate), calibration.dimension_calibration)
    from ptdl_full_task_profile import preliminary_hybrid_aircraft
    aircraft = preliminary_hybrid_aircraft(design_from_candidate(candidate), calibration.dimension_calibration, calibration.assumptions, aero_override=aero)
    local = pd.DataFrame([local_case(s, d, aircraft, candidate) for s in SCENARIOS for d in DISTURBANCES])
    endurance = pd.DataFrame([endurance_case(d, candidate, calibration, aero) for d in DISTURBANCES])
    local.to_csv(OUTPUT / "表3_四类标准工况扰动鲁棒性结果.csv", index=False, encoding="utf-8-sig")
    endurance.to_csv(OUTPUT / "表4_全任务80小时扰动鲁棒性结果.csv", index=False, encoding="utf-8-sig")
    export_figures(local, endurance)
    completion_rate = float(local["tracking_requirement_met"].mean())
    endurance_rate = float(endurance["meets_80h"].mean())
    (OUTPUT / "阶段审阅说明.md").write_text(
        f"# 1.2.3 多工况鲁棒性验证\n\n以稳健 H=6 候选 {int(candidate['candidate_index'])} 为对象，完成四类标准工况与七类参数化扰动的 {len(local)} 组闭环试验。"
        f"局部终端跟踪要求满足率为 {completion_rate:.1%}；完整任务的 80 h 满足率为 {endurance_rate:.1%}。"
        "风扰通过垂直速度项进入实际高度积分，CD0 偏差作用于阻力，发动机降额通过可用功率边界作用于调度器，电池衰减同时作用于能量容量与 SOC 保护目标。"
        "这些均为待真实模型替换的参数化场景，不构成飞行试验结论。\n", encoding="utf-8"
    )
    print(local.groupby("disturbance")["tracking_requirement_met"].mean().to_string())
    print(endurance[["disturbance", "achieved_duration_h", "meets_80h"]].to_string(index=False))


if __name__ == "__main__":
    main()
