"""Screen 80-hour feasibility against key provisional PTDL model parameters.

This is a deterministic one-factor-plus-conservative-combination screening.
It identifies which unvalidated assumptions most threaten the 80-hour hard
constraint before real propulsion-system data are available.  It is not a
probabilistic certification or a replacement for component-level validation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_80h_endurance_profile import calibrate_endurance_task_baseline, evaluate_endurance_task
from ptdl_airframe_design import AirframeDesign
from ptdl_full_task_profile import PreliminaryPropulsionAssumptions
from ptdl_intelligent_energy_management import FaultTolerantFuzzyCoordinator, FuzzyAdaptiveCoordinator, HealthState
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_80h_parameter_uncertainty_screen_results"
SURROGATE_PATH = ROOT / "ptdl_vsp_aero_surrogate_results" / "vsp_aero_surrogate.json"
PHASES = (
    "main_climb_to_leveloff", "short_leveloff_to_isr_altitude", "endurance_loiter",
    "long_descent", "approach_descent", "flare",
)
ENGINE_CAPS_KW = (145.0, 145.0, 115.0, 65.0, 80.0, 55.0)
MOTOR_ASSIST_KW = (0.0,) * len(PHASES)
SPEEDS_MPS = (52.0, 36.0, 30.1, 42.0, 30.0, 21.0)
REFERENCE_DESIGN = AirframeDesign(
    wing_area_m2=25.18,
    aspect_ratio=30.0,
    taper_ratio=0.45,
    sweep_deg=0.30,
    tip_twist_deg=-0.63,
    fuselage_length_m=8.25,
    fuselage_diameter_m=0.667,
    propeller_diameter_m=2.29,
    payload_mass_kg=110.0,
    hybrid_system_mass_kg=80.0,
)


CASES = (
    ("nominal", "nominal reference", 1.00, 1.00, 1.00, 1.00),
    ("cd0_minus_10pct", "CD0 -10%", 0.90, 1.00, 1.00, 1.00),
    ("cd0_plus_10pct", "CD0 +10%", 1.10, 1.00, 1.00, 1.00),
    ("engine_eff_minus_5pct", "engine efficiency -5%", 1.00, 0.95, 1.00, 1.00),
    ("engine_eff_plus_5pct", "engine efficiency +5%", 1.00, 1.05, 1.00, 1.00),
    ("battery_specific_energy_minus_20pct", "battery specific energy -20%", 1.00, 1.00, 0.80, 1.00),
    ("battery_specific_energy_plus_20pct", "battery specific energy +20%", 1.00, 1.00, 1.20, 1.00),
    ("engine_available_85pct", "engine available power 85%", 1.00, 1.00, 1.00, 0.85),
    ("conservative_combination", "CD0 +10%, efficiency -5%, battery -20%, engine 85%", 1.10, 0.95, 0.80, 0.85),
)


def _evaluate_case(
    label: str,
    description: str,
    cd0_factor: float,
    efficiency_factor: float,
    battery_energy_factor: float,
    engine_available_fraction: float,
) -> dict[str, object]:
    calibration = calibrate_endurance_task_baseline()
    base_aero = VSPAeroSurrogate.from_json(SURROGATE_PATH).corrected_aero(REFERENCE_DESIGN, calibration.dimension_calibration)
    aero = replace(base_aero, cd0=base_aero.cd0 * cd0_factor)
    assumptions = replace(
        calibration.assumptions,
        engine_efficiency_values=tuple(value * efficiency_factor for value in calibration.assumptions.engine_efficiency_values),
        hybrid_system_specific_energy_wh_per_kg=calibration.assumptions.hybrid_system_specific_energy_wh_per_kg * battery_energy_factor,
    )
    policy = FuzzyAdaptiveCoordinator()
    if engine_available_fraction < 1.0:
        policy = FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(), HealthState(engine_available_fraction=engine_available_fraction, motor_available_fraction=1.0)
        )
    evaluation = evaluate_endurance_task(
        REFERENCE_DESIGN,
        engine_caps_kw=ENGINE_CAPS_KW,
        motor_assist_kw=MOTOR_ASSIST_KW,
        speeds_mps=SPEEDS_MPS,
        target_total_duration_h=86.0,
        calibration=calibration.dimension_calibration,
        assumptions=assumptions,
        aero_override=aero,
        dispatch_policy=policy,
    )
    profile = evaluation.simulation.profile
    return {
        "case": label,
        "description": description,
        "cd0_factor": cd0_factor,
        "engine_efficiency_factor": efficiency_factor,
        "battery_specific_energy_factor": battery_energy_factor,
        "engine_available_fraction": engine_available_fraction,
        "achieved_duration_h": evaluation.achieved_duration_h,
        "endurance_margin_to_80h": evaluation.achieved_duration_h - 80.0,
        "meets_80h_hard_constraint": evaluation.achieved_duration_h >= 80.0 - 1e-9,
        "termination_reason": evaluation.simulation.termination_reason,
        "fuel_consumed_kg": float(np.trapezoid(profile["fuel_flow_kgps"], profile["time_s"])),
        "terminal_soc": float(profile["soc"].iloc[-1]),
        "minimum_soc": float(profile["soc"].min()),
        "endurance_constraint_g": float(evaluation.result.constraints["endurance_80h"]),
        "total_constraint_violation": float(evaluation.result.total_violation),
    }


def _export_figure(frame: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
    ordered = frame.sort_values("achieved_duration_h", ascending=True).reset_index(drop=True)
    colors = ["#16A34A" if value else "#DC2626" for value in ordered["meets_80h_hard_constraint"]]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)
    axes[0].barh(ordered["description"], ordered["achieved_duration_h"], color=colors)
    axes[0].axvline(80.0, color="#0F172A", linestyle="--", linewidth=1.5, label="80 h hard constraint")
    axes[0].set_xlabel("achieved total duration (h)")
    axes[0].set_title("80-hour feasibility under parameter screening")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="x", alpha=0.25)
    nominal = float(frame.loc[frame["case"] == "nominal", "achieved_duration_h"].iloc[0])
    deviations = ordered["achieved_duration_h"] - nominal
    axes[1].barh(ordered["description"], deviations, color=["#2563EB" if value >= 0 else "#D97706" for value in deviations])
    axes[1].axvline(0.0, color="#0F172A", linewidth=1.0)
    axes[1].set_xlabel("duration change relative to nominal (h)")
    axes[1].set_title("Sensitivity relative to nominal")
    axes[1].grid(axis="x", alpha=0.25)
    figure.savefig(OUTPUT / "图6_80小时任务关键参数不确定性筛选.png", dpi=220)
    plt.close(figure)


def _export_review_note(frame: pd.DataFrame) -> None:
    nominal = frame.loc[frame["case"] == "nominal"].iloc[0]
    failed = frame.loc[~frame["meets_80h_hard_constraint"]]
    status = "所有筛选工况均满足 80 小时约束。" if failed.empty else "未通过 80 小时约束的筛选工况为：" + "、".join(failed["description"].tolist()) + "。"
    note = f"""# 80 小时任务关键参数不确定性筛选说明

## 试验口径

固定构型、80 kg 混动系统质量、任务阶段功率上限与模糊自适应调度参数；在同一全任务剖面中将任务上限延长到 86 h，使燃油或 SOC 限制能够直接反映为可达总时长。逐一扰动 `CD0`、发动机效率、电池系统比能量及发动机可用功率，并设置保守组合工况。名义工况的可达总时长为 {nominal['achieved_duration_h']:.2f} h，对 80 h 的裕度为 {nominal['endurance_margin_to_80h']:.2f} h。

## 阶段结论

{status} 该结论用于识别当前低阶模型中最需要优先获得真实数据的参数，不构成概率鲁棒性证明或真实飞机性能承诺。

## 使用边界

发动机效率扰动目前作用于预估效率图，电池比能量扰动作用于系统级能量容量，发动机降额通过 1.2.2 的故障容错接口作用于实际功率边界。动能学院提供 BSFC 图、发动机动态与电池热/衰减模型后，应将本脚本的参数因子替换为其测量误差带，并补充蒙特卡洛抽样。
"""
    (OUTPUT / "阶段审阅说明.md").write_text(note, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = [_evaluate_case(*case) for case in CASES]
    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT / "80小时任务关键参数不确定性筛选结果.csv", index=False, encoding="utf-8-sig")
    _export_figure(frame)
    _export_review_note(frame)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
