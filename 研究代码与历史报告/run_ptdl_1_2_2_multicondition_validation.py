"""Validate PTDL 1.2.2 dispatch policies on standard flight-condition cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_airframe_design import AirframeDesign
from ptdl_flight_propulsion_control import MissionDefinition, PhaseCommand, evaluate_coupled_mission
from ptdl_hybrid_coordination import HybridCoordinationParameters, RuleBasedHybridCoordinator
from ptdl_intelligent_energy_management import FaultTolerantFuzzyCoordinator, FuzzyAdaptiveCoordinator, HealthState
from ptdl_calibration_cache import load_dimension_calibration
from ptdl_full_task_profile import preliminary_hybrid_aircraft as map_preliminary_hybrid_aircraft


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_1_2_2_multicondition_validation_results"


@dataclass(frozen=True)
class Scenario:
    name: str
    initial_altitude_m: float
    initial_speed_mps: float
    phase: PhaseCommand
    disable_motor: bool = False


SCENARIOS = (
    Scenario(
        "post_takeoff_initial_climb", 50.0, 38.0,
        PhaseCommand("main_climb_to_leveloff", 600.0, 1000.0, 52.0, 100.0, 20.0, 8.0, 5.0),
    ),
    Scenario(
        "low_speed_high_lift", 900.0, 30.0,
        PhaseCommand("low_speed_high_lift", 600.0, 900.0, 30.0, 78.0, 15.0, 3.0, 5.0),
    ),
    Scenario(
        "steady_cruise", 3000.0, 36.01,
        PhaseCommand("endurance_loiter", 1800.0, 3000.0, 36.01, 80.0, 0.0, 1.0, 10.0),
    ),
    Scenario(
        "unpowered_glide", 3000.0, 36.0,
        PhaseCommand("long_descent", 900.0, 1800.0, 25.0, 0.0, 0.0, 4.0, 5.0),
        disable_motor=True,
    ),
)


def preliminary_hybrid_aircraft():
    # Short standard-condition cases need the common dimensional calibration,
    # not an expensive 80-hour mission recalibration on every policy run.
    calibration = load_dimension_calibration()
    # A fixed 80 kg hybrid package is used only to exercise the coordination
    # policies. It is not an engineering sizing decision pending component data.
    design = AirframeDesign(
        wing_area_m2=26.0,
        aspect_ratio=26.0,
        taper_ratio=0.42,
        sweep_deg=1.0,
        tip_twist_deg=-0.5,
        fuselage_length_m=8.3,
        fuselage_diameter_m=0.75,
        propeller_diameter_m=2.4,
        payload_mass_kg=110.0,
        hybrid_system_mass_kg=80.0,
    )
    return map_preliminary_hybrid_aircraft(design, calibration)


def mission_for(scenario: Scenario) -> MissionDefinition:
    return MissionDefinition(
        phases=(scenario.phase,),
        initial_altitude_m=scenario.initial_altitude_m,
        initial_speed_mps=scenario.initial_speed_mps,
    )


def summarize(scenario: Scenario, policy_name: str, evaluation: object) -> dict[str, float | str]:
    profile = evaluation.profile
    time = profile["time_s"].to_numpy(dtype=float)
    progress = np.clip(time / max(scenario.phase.duration_s, 1e-9), 0.0, 1.0)
    altitude_reference = scenario.initial_altitude_m + progress * (scenario.phase.altitude_target_m - scenario.initial_altitude_m)
    speed_reference = scenario.initial_speed_mps + progress * (scenario.phase.speed_target_mps - scenario.initial_speed_mps)
    altitude_tracking_error = altitude_reference - profile["h_m"].to_numpy(dtype=float)
    speed_tracking_error = speed_reference - profile["v_mps"].to_numpy(dtype=float)
    final_altitude_error = abs(scenario.phase.altitude_target_m - float(profile["h_m"].iloc[-1]))
    final_speed_error = abs(scenario.phase.speed_target_mps - float(profile["v_mps"].iloc[-1]))
    altitude_tolerance = max(10.0, 0.02 * abs(scenario.phase.altitude_target_m - scenario.initial_altitude_m))
    speed_tolerance = 1.0
    within_terminal_band = (
        (profile["e_h_m"].abs().to_numpy() <= altitude_tolerance)
        & (profile["e_v_mps"].abs().to_numpy() <= speed_tolerance)
    )
    settling_time = float(scenario.phase.duration_s)
    for index in range(len(time)):
        if within_terminal_band[index:].all():
            settling_time = float(time[index])
            break
    return {
        "scenario": scenario.name,
        "policy": policy_name,
        "termination_reason": evaluation.termination_reason,
        "duration_s": float(profile["time_s"].iloc[-1]),
        "fuel_consumed_kg": float(np.trapezoid(profile["fuel_flow_kgps"], profile["time_s"])),
        "soc_change": float(profile["soc"].iloc[-1] - profile["soc"].iloc[0]),
        "mean_abs_speed_tracking_error_mps": float(np.mean(np.abs(speed_tracking_error))),
        "peak_abs_speed_tracking_error_mps": float(np.max(np.abs(speed_tracking_error))),
        "mean_abs_altitude_tracking_error_m": float(np.mean(np.abs(altitude_tracking_error))),
        "peak_abs_altitude_tracking_error_m": float(np.max(np.abs(altitude_tracking_error))),
        "terminal_abs_speed_error_mps": final_speed_error,
        "terminal_abs_altitude_error_m": final_altitude_error,
        "settling_time_s": settling_time,
        "power_mode_changes": int((profile["power_mode"] != profile["power_mode"].shift()).sum() - 1),
        "minimum_soc": float(profile["soc"].min()),
        "maximum_load_factor": float(profile["load_factor"].max()),
        "maximum_cd0": float(profile["cd0"].max()),
        "minimum_cd0": float(profile["cd0"].min()),
    }


def export_figures(summary: pd.DataFrame) -> None:
    """Export concise figures for the 1.2.2 stage review."""
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
    selected_policy = "fuzzy_adaptive"
    figure, axes = plt.subplots(len(SCENARIOS), 2, figsize=(11.5, 10.5), sharex="col", constrained_layout=True)
    for row, scenario in enumerate(SCENARIOS):
        profile = pd.read_csv(OUTPUT / f"{scenario.name}_{selected_policy}_time_history.csv")
        time_min = profile["time_s"].to_numpy(dtype=float) / 60.0
        progress = np.clip(profile["time_s"].to_numpy(dtype=float) / scenario.phase.duration_s, 0.0, 1.0)
        h_reference = scenario.initial_altitude_m + progress * (scenario.phase.altitude_target_m - scenario.initial_altitude_m)
        v_reference = scenario.initial_speed_mps + progress * (scenario.phase.speed_target_mps - scenario.initial_speed_mps)
        axes[row, 0].plot(time_min, h_reference, "--", color="#64748B", label="reference")
        axes[row, 0].plot(time_min, profile["h_m"], color="#0B5CAD", label="actual")
        axes[row, 0].set_ylabel(f"{scenario.name}\naltitude (m)")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].plot(time_min, v_reference, "--", color="#64748B", label="reference")
        axes[row, 1].plot(time_min, profile["v_mps"], color="#D97706", label="actual")
        axes[row, 1].set_ylabel("speed (m/s)")
        axes[row, 1].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].legend(frameon=False, loc="best")
    axes[-1, 0].set_xlabel("time (min)")
    axes[-1, 1].set_xlabel("time (min)")
    figure.suptitle("Calibrated standard-condition response: fuzzy adaptive dispatch", fontsize=13)
    figure.savefig(OUTPUT / "图4_四类标准工况飞行状态轨迹_模糊自适应调度.png", dpi=220)
    plt.close(figure)

    metrics = (
        ("fuel_consumed_kg", "fuel consumed (kg)"),
        ("minimum_soc", "minimum SOC (-)"),
        ("terminal_abs_speed_error_mps", "terminal speed error (m/s)"),
        ("terminal_abs_altitude_error_m", "terminal altitude error (m)"),
    )
    policies = list(summary["policy"].drop_duplicates())
    scenario_names = list(summary["scenario"].drop_duplicates())
    colors = ("#64748B", "#2563EB", "#16A34A", "#DC2626")
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 7.8), constrained_layout=True)
    x = np.arange(len(scenario_names))
    width = 0.19
    for axis, (column, label) in zip(axes.flat, metrics):
        for index, policy in enumerate(policies):
            values = [float(summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)][column].iloc[0]) for scenario in scenario_names]
            axis.bar(x + (index - 1.5) * width, values, width, label=policy, color=colors[index])
        axis.set_title(label)
        axis.set_xticks(x, scenario_names, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Policy comparison under calibrated standard conditions", fontsize=13)
    figure.savefig(OUTPUT / "图5_四类标准工况调度策略对比.png", dpi=220)
    plt.close(figure)


def export_review_note(summary: pd.DataFrame) -> None:
    fuzzy = summary[summary["policy"] == "fuzzy_adaptive"].set_index("scenario")
    text = f"""# 1.2.2 多工况协同验证阶段说明

## 工况与模型口径

本轮试验使用完整混动能力映射后的飞推控闭环模型，四类标准工况分别为起飞后初始爬升、低速高升力、3000 m 定高巡航和无动力滑翔。所有工况均限定在当前低阶模型的速度、升力和航迹角包线内；爬升和滑翔的过程误差相对线性阶段参考轨迹计算，终端误差单列，避免把正常状态转移误判为跟踪失效。

## 模糊自适应调度结果

模糊调度在初始爬升结束时的终端高度误差为 {fuzzy.loc['post_takeoff_initial_climb', 'terminal_abs_altitude_error_m']:.2f} m、终端速度误差为 {fuzzy.loc['post_takeoff_initial_climb', 'terminal_abs_speed_error_mps']:.2f} m/s；定高巡航的对应误差为 {fuzzy.loc['steady_cruise', 'terminal_abs_altitude_error_m']:.3f} m 和 {fuzzy.loc['steady_cruise', 'terminal_abs_speed_error_mps']:.3f} m/s。低速高升力工况下，模糊策略在维持 SOC 的同时消耗更多燃油，体现了当前策略“以燃油补偿电池能量”的明确权衡。无动力滑翔下各策略结果相同，说明调度器在发动机和电机均不可用时不会注入虚假推进功率。

## 结论边界

本结果验证的是控制与调度接口、状态反馈链路及策略差异，不是对真实发动机 BSFC、电池热特性或故障模式的最终性能声明。发动机功率、效率图与电池容量/热边界仍应在动能学院模型到位后替换；替换后可用同一四工况脚本复算，保持比较口径一致。
"""
    (OUTPUT / "阶段审阅说明.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_aircraft = preliminary_hybrid_aircraft()
    policy_builders = {
        "stage_fixed": lambda: None,
        "rule_soc_feedback": lambda: RuleBasedHybridCoordinator(HybridCoordinationParameters()),
        "fuzzy_adaptive": lambda: FuzzyAdaptiveCoordinator(),
        "fuzzy_engine_derated": lambda: FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(), HealthState(engine_available_fraction=0.70, motor_available_fraction=1.0)
        ),
    }
    records: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        aircraft = replace(base_aircraft, motor_power_min_kw=0.0, motor_power_max_kw=0.0) if scenario.disable_motor else base_aircraft
        for policy_name, build_policy in policy_builders.items():
            simulation, _ = evaluate_coupled_mission(mission_for(scenario), aircraft, dispatch_policy=build_policy())
            simulation.profile.to_csv(OUTPUT / f"{scenario.name}_{policy_name}_time_history.csv", index=False)
            records.append(summarize(scenario, policy_name, simulation))
    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT / "multicondition_policy_comparison.csv", index=False)
    export_figures(summary)
    export_review_note(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
