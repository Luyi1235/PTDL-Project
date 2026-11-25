"""Compare energy-management policies on one shared 80 h mission profile."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import (
    build_endurance_mission,
    calibrate_endurance_task_baseline,
    evaluate_endurance_task,
    load_endurance_profile,
)
from 低阶地面滑跑模型 import simulate_ground_roll
from 飞推控闭环物理模型 import LongitudinalDragDisturbance
from 响应指标 import (
    RESPONSE_SPEED_TOLERANCE_MPS,
    apply_paired_response_metrics,
    paired_response_metrics,
    zero_amplitude_disturbances,
)
from 智能能量管理与容错调度 import (
    FaultTolerantFuzzyCoordinator,
    FuzzyAdaptiveCoordinator,
    HealthState,
    MPCEnergyManagementCoordinator,
    NeuralImitationCoordinator,
)
from 全任务剖面定义 import load_preliminary_propulsion_config, preliminary_hybrid_aircraft
from 机体参数化设计与气动阻力 import AirframeDesign
from 规则混动协同控制 import RuleBasedHybridCoordinator
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from VSPAERO气动代理模型 import load_vsp_surrogate


DEFAULT_CANDIDATE = (
    ROOT
    / "运行结果"
    / "H6_三项风险闭环复验"
    / "NSGA-II与MOEA-D择优可行候选.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "H6_非标定扩展验证" / "多工况协同控制"
SURROGATE_PATH = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
NEURAL_MODEL_PATH = (
    ROOT
    / "运行结果"
    / "神经网络MPC模仿策略训练测试_20261011"
    / "神经网络MPC模仿策略.json"
)

SCENARIOS = (
    {
        "name": "爬升段扰动",
        "phase": "main_climb_to_leveloff",
        "start_s": 900.0,
        "duration_s": 60.0,
        "drag_n": 80.0,
        "motor_unavailable": False,
        "unpowered_glide": False,
    },
    {
        "name": "低速高升力进近扰动",
        "phase": "approach_descent",
        "start_s": 120.0,
        "duration_s": 30.0,
        "drag_n": 80.0,
        "motor_unavailable": False,
        "unpowered_glide": False,
    },
    {
        "name": "长航时巡航扰动",
        "phase": "endurance_loiter",
        "start_s": 3600.0,
        "duration_s": 600.0,
        "drag_n": 100.0,
        "motor_unavailable": False,
        "unpowered_glide": False,
    },
    {
        "name": "长航时迎风阶跃",
        "phase": "endurance_loiter",
        "start_s": 3600.0,
        "duration_s": 60.0,
        "drag_n": 0.0,
        "headwind_step_mps": 8.0,
        "response_step_s": 2.0,
        "motor_unavailable": False,
        "unpowered_glide": False,
    },
    {
        "name": "下降段电机关闭",
        "phase": "long_descent",
        "start_s": 900.0,
        "duration_s": 120.0,
        "drag_n": 60.0,
        "motor_unavailable": True,
        "unpowered_glide": False,
    },
    {
        "name": "无动力滑翔扰动",
        "phase": "long_descent",
        "start_s": 900.0,
        "duration_s": 120.0,
        "drag_n": 60.0,
        "motor_unavailable": True,
        "unpowered_glide": True,
    },
    {
        "name": "拉平段扰动",
        "phase": "flare",
        "start_s": 120.0,
        "duration_s": 30.0,
        "drag_n": 40.0,
        "motor_unavailable": False,
        "unpowered_glide": False,
    },
)

POLICY_NAMES = (
    "定阶段功率基线",
    "规则SOC调度",
    "模糊自适应",
    "容错模糊-正常",
    "容错模糊-传感器失效",
    "容错模糊-发动机降额",
    "MPC滚动调度",
    "神经网络模仿MPC",
)


def _paired_response_metrics(
    disturbed: pd.DataFrame,
    nominal: pd.DataFrame,
    phase_name: str,
    phase_start_s: float,
    phase_end_s: float,
    event_start_s: float,
    event_duration_s: float,
    recovery_window_s: float,
    recovery_dwell_s: float,
    altitude_tolerance_m: float,
    speed_tolerance_mps: float = RESPONSE_SPEED_TOLERANCE_MPS,
) -> dict[str, float | bool]:
    """Measure disturbance response against the same-controller nominal trajectory."""
    return paired_response_metrics(
        disturbed,
        nominal,
        phase_name,
        event_start_s,
        event_duration_s,
        recovery_window_s,
        recovery_dwell_s,
        altitude_tolerance_m,
        speed_tolerance_mps,
    )


def _paired_touchdown_metrics(
    disturbed: pd.DataFrame,
    nominal: pd.DataFrame,
    aircraft: object,
    speed_tolerance_mps: float = RESPONSE_SPEED_TOLERANCE_MPS,
    altitude_tolerance_m: float = 10.0,
    alpha_tolerance_deg: float = 2.0,
) -> dict[str, object]:
    """Use paired touchdown state and rollout instead of post-landing recovery."""
    disturbed_rows = disturbed.loc[disturbed["phase"] == "flare"]
    nominal_rows = nominal.loc[nominal["phase"] == "flare"]
    if disturbed_rows.empty or nominal_rows.empty:
        missing = []
        if disturbed_rows.empty:
            missing.append("扰动仿真未进入拉平阶段")
        if nominal_rows.empty:
            missing.append("零扰动基线未进入拉平阶段")
        return {
            "paired_recovery_time_s": np.nan,
            "paired_recovery_success": None,
            "paired_peak_speed_deviation_mps": np.nan,
            "paired_peak_altitude_deviation_m": np.nan,
            "response_metric_valid": False,
            "terminal_phase_limited": True,
            "response_observation_window_s": 0.0,
            "response_metric_type": "paired_touchdown_endpoint",
            "terminal_response_metric_valid": False,
            "paired_touchdown_available": False,
            "paired_touchdown_unavailable_reason": "；".join(missing),
            "paired_touchdown_speed_delta_mps": np.nan,
            "paired_touchdown_altitude_delta_m": np.nan,
            "paired_touchdown_alpha_delta_deg": np.nan,
            "paired_touchdown_endpoint_pass": False,
            "disturbed_landing_roll_feasible": False,
            "disturbed_landing_roll_distance_m": np.nan,
            "nominal_landing_roll_feasible": False,
            "nominal_landing_roll_distance_m": np.nan,
            "landing_roll_distance_delta_m": np.nan,
        }
    disturbed_flare = disturbed_rows.iloc[-1]
    nominal_flare = nominal_rows.iloc[-1]
    speed_delta = float(disturbed_flare["v_mps"] - nominal_flare["v_mps"])
    altitude_delta = float(disturbed_flare["h_m"] - nominal_flare["h_m"])
    alpha_delta_deg = float(
        np.rad2deg(disturbed_flare["alpha_rad"] - nominal_flare["alpha_rad"])
    )
    disturbed_roll = simulate_ground_roll(
        aircraft,
        "landing",
        0.0,
        initial_speed_mps=float(disturbed_flare["v_mps"]),
        mass_kg=float(disturbed_flare["mass_kg"]),
    )
    nominal_roll = simulate_ground_roll(
        aircraft,
        "landing",
        0.0,
        initial_speed_mps=float(nominal_flare["v_mps"]),
        mass_kg=float(nominal_flare["mass_kg"]),
    )
    endpoint_pass = (
        abs(speed_delta) <= speed_tolerance_mps
        and abs(altitude_delta) <= altitude_tolerance_m
        and abs(alpha_delta_deg) <= alpha_tolerance_deg
    )
    return {
        "paired_recovery_time_s": np.nan,
        "paired_recovery_success": None,
        "paired_peak_speed_deviation_mps": abs(speed_delta),
        "paired_peak_altitude_deviation_m": abs(altitude_delta),
        "response_metric_valid": False,
        "terminal_phase_limited": True,
        "response_observation_window_s": 0.0,
        "response_metric_type": "paired_touchdown_endpoint",
        "terminal_response_metric_valid": True,
        "paired_touchdown_available": True,
        "paired_touchdown_unavailable_reason": "",
        "paired_touchdown_speed_delta_mps": speed_delta,
        "paired_touchdown_altitude_delta_m": altitude_delta,
        "paired_touchdown_alpha_delta_deg": alpha_delta_deg,
        "paired_touchdown_endpoint_pass": endpoint_pass,
        "disturbed_landing_roll_feasible": disturbed_roll.feasible,
        "disturbed_landing_roll_distance_m": disturbed_roll.distance_m,
        "nominal_landing_roll_feasible": nominal_roll.feasible,
        "nominal_landing_roll_distance_m": nominal_roll.distance_m,
        "landing_roll_distance_delta_m": (
            disturbed_roll.distance_m - nominal_roll.distance_m
        ),
    }


def _make_policy(name: str):
    if name == "定阶段功率基线":
        return None
    if name == "规则SOC调度":
        return RuleBasedHybridCoordinator()
    if name == "模糊自适应":
        return FuzzyAdaptiveCoordinator()
    if name == "容错模糊-正常":
        return FaultTolerantFuzzyCoordinator(FuzzyAdaptiveCoordinator(), HealthState())
    if name == "容错模糊-传感器失效":
        return FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(), HealthState(sensor_data_valid=False)
        )
    if name == "容错模糊-发动机降额":
        return FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(), HealthState(engine_available_fraction=0.70)
        )
    if name == "MPC滚动调度":
        return MPCEnergyManagementCoordinator()
    if name == "神经网络模仿MPC":
        return NeuralImitationCoordinator(NEURAL_MODEL_PATH)
    raise ValueError(f"未知控制策略: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于同一 H=6 择优设计开展全任务剖面多工况控制器验证。"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surrogate", type=Path, default=SURROGATE_PATH)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    surrogate = load_vsp_surrogate(args.surrogate)
    decision = np.asarray(candidate["decision"], dtype=float)
    calibration = calibrate_endurance_task_baseline()
    profile = load_endurance_profile()
    segments = profile["segments"]
    phase_names = tuple(segment["label"] for segment in segments)
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
    design = AirframeDesign(**dict(zip(design_labels, decision[:10])))
    aero = surrogate.corrected_aero(design, calibration.dimension_calibration)
    aircraft = preliminary_hybrid_aircraft(
        design,
        calibration.dimension_calibration,
        calibration.assumptions,
        aero_override=aero,
    )
    _, sizing_config = load_preliminary_propulsion_config()
    speeds = [float(segment["speed_target_mps"]) for segment in segments]
    speeds[phase_names.index("endurance_loiter")] = float(decision[10])
    engine_caps = decision[11:17].copy()
    motor_assist = decision[17:23].copy()
    mission_template = build_endurance_mission(profile, speeds_mps=speeds)
    phase_indices = {phase.name: index for index, phase in enumerate(mission_template.phases)}
    phase_offsets: dict[str, float] = {}
    elapsed = 0.0
    for phase in mission_template.phases:
        phase_offsets[phase.name] = elapsed
        elapsed += phase.duration_s
    powertrain_factory = partial(
        HybridSystemPowertrain, config=HybridSystemAdapterConfig()
    )
    rows: list[dict[str, object]] = []
    nominal_runs: dict[tuple[str, str, float, float, bool, bool], object] = {}

    for scenario in SCENARIOS:
        event = LongitudinalDragDisturbance(
            phase_name=scenario["phase"],
            start_time_s=scenario["start_s"],
            duration_s=scenario["duration_s"],
            additional_drag_n=scenario["drag_n"],
            response_step_s=float(scenario.get("response_step_s", 5.0)),
            recovery_window_s=900.0,
            recovery_dwell_s=30.0,
            headwind_step_mps=float(scenario.get("headwind_step_mps", 0.0)),
        )
        scenario_engine_caps = engine_caps.copy()
        scenario_motor_assist = motor_assist.copy()
        if scenario["motor_unavailable"]:
            index = phase_indices[scenario["phase"]]
            scenario_motor_assist[index] = 0.0

        for policy_name in POLICY_NAMES:
            nominal_key = (
                policy_name,
                str(scenario["phase"]),
                float(scenario["start_s"]),
                float(scenario["duration_s"]),
                bool(scenario["motor_unavailable"]),
                bool(scenario["unpowered_glide"]),
            )
            if nominal_key not in nominal_runs:
                nominal_runs[nominal_key] = evaluate_endurance_task(
                    design,
                    engine_caps_kw=scenario_engine_caps,
                    motor_assist_kw=scenario_motor_assist,
                    speeds_mps=speeds,
                    calibration=calibration.dimension_calibration,
                    assumptions=calibration.assumptions,
                    aero_override=aero,
                    dispatch_policy=_make_policy(policy_name),
                    powertrain=powertrain_factory(),
                    disturbances=zero_amplitude_disturbances((event,)),
                    unpowered_glide_phases=(
                        (scenario["phase"],) if scenario["unpowered_glide"] else ()
                    ),
                    phase_overrides=(
                        {
                            "long_descent": {
                                "altitude_target_m": 0.0,
                                "speed_target_mps": 50.0,
                                "time_step_s": 5.0,
                            }
                        }
                        if scenario["unpowered_glide"]
                        else None
                    ),
                )
                nominal_runs[nominal_key].simulation.profile.to_csv(
                    output_dir
                    / (
                        f"零幅扰动同采样基线_{policy_name}_"
                        f"{'无动力滑翔' if scenario['unpowered_glide'] else '下降段电机关闭' if scenario['motor_unavailable'] else '正常动力'}_时序.csv"
                    ),
                    index=False,
                )
            evaluation = evaluate_endurance_task(
                design,
                engine_caps_kw=scenario_engine_caps,
                motor_assist_kw=scenario_motor_assist,
                speeds_mps=speeds,
                calibration=calibration.dimension_calibration,
                assumptions=calibration.assumptions,
                aero_override=aero,
                dispatch_policy=_make_policy(policy_name),
                powertrain=powertrain_factory(),
                disturbances=(event,),
                unpowered_glide_phases=(
                    (scenario["phase"],) if scenario["unpowered_glide"] else ()
                ),
                phase_overrides=(
                    {
                        "long_descent": {
                            "altitude_target_m": 0.0,
                            "speed_target_mps": 50.0,
                            "time_step_s": 5.0,
                        }
                    }
                    if scenario["unpowered_glide"]
                    else None
                ),
            )
            simulation = evaluation.simulation
            frame = simulation.profile
            metrics = simulation.response_metrics
            phase = mission_template.phases[phase_indices[scenario["phase"]]]
            nominal_frame = nominal_runs[nominal_key].simulation.profile
            if scenario["phase"] == "flare":
                paired_metrics = _paired_touchdown_metrics(
                    frame,
                    nominal_frame,
                    aircraft,
                    altitude_tolerance_m=max(
                        10.0, 0.01 * max(abs(phase.altitude_target_m), 100.0)
                    ),
                )
            else:
                paired_metrics = apply_paired_response_metrics(
                    evaluation,
                    nominal_runs[nominal_key],
                    (event,),
                )
            stem = f"{scenario['name']}_{policy_name}".replace("/", "_")
            frame.to_csv(output_dir / f"{stem}_时序.csv", index=False)
            rows.append(
                {
                    "scenario": scenario["name"],
                    "phase": scenario["phase"],
                    "controller": policy_name,
                    "motor_unavailable_in_phase": scenario["motor_unavailable"],
                    "unpowered_glide_in_phase": scenario["unpowered_glide"],
                    "disturbance_drag_n": scenario["drag_n"],
                    "disturbance_headwind_step_mps": scenario.get("headwind_step_mps", 0.0),
                    "disturbance_duration_s": scenario["duration_s"],
                    "termination_reason": simulation.termination_reason,
                    "mission_feasible": evaluation.result.feasible,
                    "total_violation": evaluation.result.total_violation,
                    "endurance_h": evaluation.result.diagnostics["endurance_h"],
                    "fuel_used_kg": evaluation.result.diagnostics["fuel_used_kg"],
                    "terminal_soc": evaluation.result.diagnostics["terminal_soc"],
                    "J_energy_mj": evaluation.result.objectives["J_energy_mj"],
                    "J_stability": evaluation.result.objectives["J_stability"],
                    "J_response_s": (
                        np.nan
                        if scenario["phase"] == "flare"
                        else evaluation.result.objectives["J_response_s"]
                    ),
                    "legacy_J_response_s": evaluation.result.objectives["J_response_s"],
                    "settling_time_s": metrics.get("settling_time_s", 0.0),
                    "legacy_pre_event_baseline_recovery_time_s": metrics.get("recovery_time_s", 0.0),
                    "legacy_pre_event_baseline_recovery_success": metrics.get("recovery_success", 0.0) >= 1.0,
                    **paired_metrics,
                    "mode_switch_time_s": metrics.get("mode_switch_time_s", 0.0),
                    "max_alpha_deg": float(np.rad2deg(frame["alpha_rad"]).max()),
                    "max_load_factor": float(frame["load_factor"].max()),
                    "engine_rating_multiplier": calibration.assumptions.engine_rating_multiplier,
                    "motor_rating_multiplier": calibration.assumptions.motor_rating_multiplier,
                    "battery_energy_multiplier": calibration.assumptions.battery_energy_multiplier,
                    "sized_engine_limit_kw": aircraft.engine_electrical_power_max_kw,
                    "sized_motor_limit_kw": aircraft.motor_power_max_kw,
                    "sized_battery_capacity_mj": aircraft.battery_capacity_mj,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "多工况控制器对比.csv", index=False)
    metadata = {
        "study": "component-backed policy comparison on the same selected H=6 design and full 80 h mission",
        "candidate_source": str(args.candidate.resolve()),
        "selected_optimizer": candidate.get("optimizer"),
        "aerodynamic_model": "VSPAERO-corrected surrogate",
        "aerodynamic_surrogate_file": str(args.surrogate.resolve()),
        "aerodynamic_surrogate_training_samples": getattr(
            surrogate, "training_samples", None
        ),
        "neural_imitation_policy": {
            "model_file": str(NEURAL_MODEL_PATH.resolve()),
            "safety_projection": (
                "engine output is bounded by propulsion demand and charge acceptance; "
                "negative motor power for charging is permitted only below the phase SOC target"
            ),
            "interpretation": (
                "MPC behavior-cloning prototype; predictive accuracy is not evidence of "
                "closed-loop superiority"
            ),
        },
        "powertrain": "Hybrid_System components under the same provisional aggregate sizing assumptions",
        "engine_dynamics": {
            "output_slew_limit_kwps": powertrain_factory().config.engine_power_rate_limit_kwps,
            "glide_entry": (
                "explicit engine_shutdown_transition phase followed by zero engine/motor "
                "power during unpowered_glide"
            ),
            "unpowered_glide_max_integration_step_s": 5.0,
        },
        "ultra_reference_basis": sizing_config["reference_basis"],
        "propulsion_sizing_note": sizing_config["sizing_note"],
        "effective_propulsion_assumptions": {
            "engine_rating_multiplier": calibration.assumptions.engine_rating_multiplier,
            "motor_rating_multiplier": calibration.assumptions.motor_rating_multiplier,
            "battery_energy_multiplier": calibration.assumptions.battery_energy_multiplier,
            "engine_limit_kw": aircraft.engine_electrical_power_max_kw,
            "motor_limit_kw": aircraft.motor_power_max_kw,
            "battery_capacity_mj": aircraft.battery_capacity_mj,
        },
        "scenario_definitions": list(SCENARIOS),
        "controllers": list(POLICY_NAMES),
        "case_count": len(summary),
        "response_measurement": (
            "The PPT J_response_s objective is updated from the same-controller paired disturbance response "
            "against a same-controller zero-amplitude reference with identical event timing and integration steps. "
            "Recovery requires speed error <= 0.5 m/s and altitude error <= max(10 m, 0.2% phase target) "
            "for 30 consecutive seconds, within a 900 s post-disturbance window."
        ),
        "response_metric_fields": {
            "response_metric_valid": "是否至少有一个完整的恢复驻留窗口可观测",
            "terminal_phase_limited": "任务终点是否早于请求的恢复窗口终点",
            "response_observation_window_s": "扰动结束后的实际可观测秒数",
            "response_metric_type": "飞行阶段为配对恢复；拉平阶段为配对触地端点与着陆滑跑",
        },
        "flare_interpretation": (
            "拉平扰动不再以触地后的飞行恢复时间判定；改用扰动与同控制器零扰动基线的触地速度、高度、迎角差，以及对应着陆滑跑距离差。"
            "终端差异判据是项目筛查值，不替代跑道长度、下沉率、姿态和结构载荷的工程限值。"
        ),
        "limitations": [
            "Landing rollout is a separate low-order ground model initialized from the simulated flare terminal speed and mass.",
            "The low-speed approach case is not a stall-envelope or wind-tunnel validation.",
            "The motor-unavailable descent retains engine power; the separate glide case zeros both engine and motor output in the point-mass model.",
            "The tested glide schedule uses a 50 m/s target and 5 s maximum integration step; all eight controller simulations reached a terminal state in this low-order screen, but feasibility varies and the schedule/aerodynamics are not validated real-aircraft glide performance.",
            "Added drag and the 8 m/s headwind step are longitudinal stress screens, not validated atmospheric gust models.",
            "Component ratings and efficiencies remain provisional; no propulsion-model calibration was performed.",
            "The selected candidate has little endurance margin; several phase-specific stress cases deplete fuel before 80 h and must be treated as infeasible rather than used for controller ranking.",
        ],
    }
    (output_dir / "多工况验证说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
