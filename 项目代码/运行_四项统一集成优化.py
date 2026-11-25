"""Run the PTDL 1.2.1/1.2.2/2.2.1/2.2.2 component-backed integration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from MOEAD多目标联合优化 import (
    CandidateEvaluation,
    OBJECTIVE_LABELS,
    OptimizerRun,
    PTDLEnduranceJointProblem,
    pareto_indices,
    run_ptdl_80h_joint_moead,
)
from 八十小时续航任务模型 import calibrate_endurance_task_baseline, evaluate_endurance_task, load_endurance_profile
from 飞推控闭环物理模型 import LongitudinalDragDisturbance
from 指标体系与评价函数 import EvaluationResult, ObjectiveNormalizer, weighted_preference_score
from 响应指标 import (
    apply_paired_response_metrics,
    project_response_stress_scenario,
    zero_amplitude_disturbances,
)
from 全任务剖面定义 import preliminary_hybrid_aircraft
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 动能三层能量管理 import SourceThreeLayerCoordinator
from 规则混动协同控制 import RuleBasedHybridCoordinator
from 智能能量管理与容错调度 import (
    FaultTolerantFuzzyCoordinator,
    FuzzyAdaptiveCoordinator,
    HealthState,
    MPCParameters,
    MPCEnergyManagementCoordinator,
    NeuralImitationCoordinator,
)
from PTDL联合优化高保真接口 import (
    LiveOpenVSPAeroProvider,
    PTDL6DoFOptimizerValidator,
    PTDL6DoFFullMissionOptimizerValidator,
    StepwiseOpenVSPAeroCandidateValidator,
)
from VSPAERO气动代理模型 import FEATURE_LABELS, load_vsp_surrogate
from 运行_动力模型预估额定值闭环验证 import _write_visual


DEFAULT_SEED = ROOT / "验证测试" / "预估额定值基准候选.json"
DOE_VERSION = "Ew_物理选点_网格修复_20261006"
MODEL_VERSION = f"{DOE_VERSION}_边界适配"
DEFAULT_SURROGATE = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
DEFAULT_AERO_DOMAIN_DATA = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_131点_Ew_物理选点_网格修复_自适应补样_20261011.csv"
DEFAULT_OUTPUT = ROOT / "运行结果" / "四项统一集成"
DEFAULT_NEURAL_POLICY = (
    ROOT
    / "运行结果"
    / "神经网络MPC模仿策略训练测试_20261011"
    / "神经网络MPC模仿策略.json"
)
DEFAULT_VSP_PYTHON = ROOT.parent.parent / "tools" / "Python311" / "python.exe"
SOC_SELECTION_FLOOR = 0.25
FUEL_RESERVE_FRACTION = 0.05
PREFERENCE_WEIGHTS = {
    "energy_priority": {"J_energy_mj": 0.60, "J_stability": 0.20, "J_response_s": 0.20},
    "stability_priority": {"J_energy_mj": 0.20, "J_stability": 0.60, "J_response_s": 0.20},
    "response_priority": {"J_energy_mj": 0.20, "J_stability": 0.20, "J_response_s": 0.60},
    "balanced": {"J_energy_mj": 1.0 / 3.0, "J_stability": 1.0 / 3.0, "J_response_s": 1.0 / 3.0},
}
H6_RESPONSE_DISTURBANCES = project_response_stress_scenario()


def _candidate_decision(problem: PTDLEnduranceJointProblem, record: dict) -> np.ndarray:
    design_values = [record["design"][name] for name in problem.design_labels]
    return np.asarray(
        design_values
        + [record["isr_cruise_speed_mps"]]
        + record["engine_caps_kw"]
        + record["motor_assist_kw"],
        dtype=float,
    )


def _read_evaluation_archive(path: Path, problem: PTDLEnduranceJointProblem) -> list[CandidateEvaluation]:
    frame = pd.read_csv(path)
    evaluations: list[CandidateEvaluation] = []
    for _, row in frame.iterrows():
        objectives = {label: float(row[label]) for label in OBJECTIVE_LABELS}
        constraints = {
            column[2:]: float(row[column])
            for column in frame.columns
            if column.startswith("g_") and pd.notna(row[column])
        }
        diagnostics: dict[str, object] = {}
        for column in frame.columns:
            if column.startswith("diagnostic_") and pd.notna(row[column]):
                value = row[column]
                diagnostics[column.removeprefix("diagnostic_")] = (
                    str(value) if isinstance(value, str) else float(value)
                )
        evaluations.append(
            CandidateEvaluation(
                np.asarray([row[label] for label in problem.decision_labels], dtype=float),
                EvaluationResult(objectives, constraints, diagnostics),
            )
        )
    return evaluations


def _evaluation_summary(
    evaluation: object,
    aircraft: object,
    powertrain: HybridSystemPowertrain,
    controller_name: str,
    design: AirframeDesign,
    assumptions: object,
) -> dict[str, object]:
    frame = evaluation.simulation.profile
    state = powertrain.initial_state(aircraft, aircraft.initial_soc)
    return {
        "model_status": "preliminary equivalent component sizing; not validated hardware performance",
        "controller": controller_name,
        "feasible": bool(evaluation.result.feasible),
        "total_violation": evaluation.result.total_violation,
        "achieved_total_duration_h": evaluation.achieved_duration_h,
        "requested_total_duration_h": 80.0,
        "termination_reason": evaluation.simulation.termination_reason,
        "fuel_used_kg": aircraft.initial_fuel_kg - evaluation.simulation.terminal_fuel_kg,
        "terminal_fuel_kg": evaluation.simulation.terminal_fuel_kg,
        "terminal_soc": float(frame["soc"].iloc[-1]),
        "minimum_soc": float(frame["soc"].min()),
        "response_metrics": evaluation.simulation.response_metrics,
        "battery_diagnostics": {
            key: value
            for key, value in evaluation.result.diagnostics.items()
            if key.startswith("battery_")
        },
        "hybrid_system_mass_kg": design.hybrid_system_mass_kg,
        "estimated_engine_limit_kw": aircraft.engine_electrical_power_max_kw,
        "estimated_motor_limit_kw": aircraft.motor_power_max_kw,
        "estimated_battery_kwh": aircraft.battery_capacity_mj / 3.6,
        "native_generator_ac_kw": powertrain.generator_rated_ac_kw,
        "native_motor_total_kw": powertrain.motor_rated_mechanical_kw * powertrain.config.motor_branches,
        "native_battery_kwh": powertrain.battery_parameters["BAT_Energy_nominal_Wh"] / 1000.0,
        "rating_energy_multipliers": {
            "engine": assumptions.engine_rating_multiplier,
            "motor": assumptions.motor_rating_multiplier,
            "battery_energy": assumptions.battery_energy_multiplier,
        },
        "engine_rating_scale_over_native_generator": state.engine_rating_scale,
        "motor_branches": state.motor_branches,
        "battery_parallel_equivalent": state.battery_parallel_equivalent,
        "objectives": evaluation.result.objectives,
        "constraints": evaluation.result.constraints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统一运行飞推控、智能调度、评价函数、NSGA-II/MOEA-D及发动机/电池/电机模型。"
    )
    parser.add_argument("--seed-candidate", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--surrogate", type=Path, default=DEFAULT_SURROGATE)
    parser.add_argument("--aero-domain-data", type=Path, default=DEFAULT_AERO_DOMAIN_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO", "PTDL-自适应分块",
                 "PTDL-耦合代理辅助", "PTDL-耦合引导", "PTDL-耦合贡献学习",
                 "PTDL-耦合贡献学习保先验", "PTDL-去重贡献学习"),
        default=("NSGA-II", "MOEA/D"),
    )
    parser.add_argument(
        "--controller",
        choices=("mpc", "rule", "fuzzy", "fault-tolerant-fuzzy", "neural-imitation", "phase-mpc", "source-ems"),
        default="mpc",
    )
    parser.add_argument("--neural-policy-model", type=Path, default=DEFAULT_NEURAL_POLICY)
    parser.add_argument("--vsp-mesh-level", choices=("legacy", "medium", "fine"),
                        default="legacy")
    parser.add_argument("--response-definition", choices=("legacy", "recovery-integral"),
                        default="legacy")
    parser.add_argument("--phase-mpc-config", type=Path)
    parser.add_argument("--mpc-horizon", type=int, default=3)
    parser.add_argument("--mpc-action-levels", type=int, default=3)
    parser.add_argument("--mpc-soc-terminal-weight-mj", type=float, default=4.0)
    parser.add_argument("--mpc-engine-slew-penalty", type=float, default=0.002)
    parser.add_argument("--wind-step-mps", type=float, default=8.0)
    parser.add_argument("--wind-duration-s", type=float, default=60.0)
    parser.add_argument("--design-bounds-json", type=Path, default=None)
    parser.add_argument("--engine-map", type=Path, default=None)
    parser.add_argument("--reuse-optimizer-output-dir", type=Path, default=None)
    parser.add_argument(
        "--aero-mode",
        choices=("live-vspaero-finalists", "live-vspaero-all", "surrogate", "stepwise"),
        default="live-vspaero-finalists",
        help="代理搜索后实算 Pareto 候选、全候选实算、仅使用代理，或每个候选逐步调用 VSPAERO。",
    )
    parser.add_argument("--aero-step-duration-s", type=float, default=8.0)
    parser.add_argument("--aero-integration-step-s", type=float, default=0.25)
    parser.add_argument("--vsp-python", type=Path, default=DEFAULT_VSP_PYTHON)
    parser.add_argument("--sixdof-duration-s", type=float, default=8.0)
    parser.add_argument("--sixdof-integration-step-s", type=float, default=0.05)
    parser.add_argument(
        "--sixdof-mode",
        choices=("optimizer-loop", "pareto-screen", "stepwise", "full-mission"),
        default="optimizer-loop",
        help="短时逐候选检查、Pareto复评、短时逐步推进，或每个候选运行完整六阶段JSBSim任务。",
    )
    parser.add_argument("--sixdof-full-mission-hours", type=float, default=80.0)
    parser.add_argument(
        "--sixdof-full-mission-duration-basis",
        choices=("total-including-landing", "endurance-before-landing"),
        default="total-including-landing",
        help="80小时按含着陆的总任务时长计算，或要求先完成80小时飞行再下降着陆。",
    )
    parser.add_argument("--sixdof-full-mission-control-period-s", type=float, default=2.0)
    parser.add_argument("--sixdof-full-mission-terminal-control-period-s", type=float, default=0.5)
    parser.add_argument("--sixdof-full-mission-sample-period-s", type=float, default=600.0)
    parser.add_argument("--sixdof-full-mission-pitch-kp", type=float, default=4.0)
    parser.add_argument("--sixdof-full-mission-pitch-kd", type=float, default=4.0)
    parser.add_argument("--sixdof-full-mission-terminal-pitch-kp", type=float, default=2.5)
    parser.add_argument("--sixdof-full-mission-terminal-pitch-kd", type=float, default=4.0)
    parser.add_argument("--sixdof-full-mission-terminal-pitch-ki", type=float, default=0.15)
    parser.add_argument("--engine-rating-multiplier", type=float, default=None)
    parser.add_argument("--motor-rating-multiplier", type=float, default=None)
    parser.add_argument("--battery-energy-multiplier", type=float, default=None)
    parser.add_argument("--engine-baseline-power-kw", type=float)
    parser.add_argument("--engine-upgrade-kw-per-hybrid-kg", type=float)
    parser.add_argument("--engine-power-ceiling-kw", type=float)
    parser.add_argument("--hybrid-specific-energy-wh-per-kg", type=float)
    parser.add_argument("--motor-power-kw-per-hybrid-kg", type=float)
    parser.add_argument("--engine-efficiency-factor", type=float, default=1.0)
    parser.add_argument("--engine-speed-rpm", type=float, default=5000.0)
    parser.add_argument("--motor-speed-rpm", type=float, default=2100.0)
    parser.add_argument("--motor-branches", type=int, default=2)
    parser.add_argument("--engine-fuel-flow-scale", type=float, default=1.0)
    parser.add_argument("--engine-power-rate-limit-kwps", type=float, default=10.0)
    parser.add_argument("--engine-time-constant-s", type=float, default=0.0)
    parser.add_argument("--engine-command-delay-s", type=float, default=0.0)
    parser.add_argument("--battery-resistance-multiplier", type=float, default=1.0)
    parser.add_argument("--inverter-efficiency", type=float, default=0.95)
    parser.add_argument("--battery-soc-min", type=float, default=0.20)
    parser.add_argument("--battery-soc-max", type=float, default=1.00)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_file.resolve() if args.progress_file else None

    def report_progress(percent: float, stage: str, detail: str = "") -> None:
        if progress_path is None:
            return
        payload = {
            "percent": max(0, min(99, round(percent))),
            "stage": stage,
            "detail": detail,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        temporary_path = progress_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(progress_path)

    report_progress(1, "初始化", "正在加载任务、模型与约束")
    if args.mpc_horizon < 1 or args.mpc_horizon > 6:
        parser.error("--mpc-horizon must be between 1 and 6.")
    if args.mpc_action_levels < 2 or args.mpc_action_levels > 7:
        parser.error("--mpc-action-levels must be between 2 and 7.")
    if args.wind_duration_s <= 0 or args.wind_step_mps < 0:
        parser.error("Wind step must be non-negative and duration must be positive.")
    if args.mpc_soc_terminal_weight_mj < 0 or args.mpc_engine_slew_penalty < 0:
        parser.error("MPC weights and penalties must be non-negative.")
    if not 0.5 <= args.engine_efficiency_factor <= 1.5:
        parser.error("--engine-efficiency-factor must be between 0.5 and 1.5.")
    if not 0.1 <= args.aero_step_duration_s <= 60.0:
        parser.error("--aero-step-duration-s must be between 0.1 and 60 seconds.")
    if (
        args.aero_mode == "stepwise"
        and args.sixdof_mode == "stepwise"
        and max(args.aero_step_duration_s, args.sixdof_duration_s) > 10.0
    ):
        parser.error(
            "同时启用逐步 VSPAERO 与逐步 JSBSim 联算时，候选轨迹时长不得超过 10 秒。"
        )
    if not 1.0 / 120.0 <= args.aero_integration_step_s <= 0.25:
        parser.error("--aero-integration-step-s must be between 1/120 and 0.25 seconds.")
    if not 5.0 <= args.sixdof_duration_s <= 60.0:
        parser.error("--sixdof-duration-s must be between 5 and 60 seconds.")
    if not 1.0 / 120.0 <= args.sixdof_integration_step_s <= 0.25:
        parser.error("--sixdof-integration-step-s must be between 1/120 and 0.25 seconds.")
    if args.sixdof_mode == "full-mission":
        if args.sixdof_full_mission_hours < 3.0:
            parser.error("--sixdof-full-mission-hours must be at least 3 hours.")
        if args.sixdof_full_mission_control_period_s < args.sixdof_integration_step_s:
            parser.error("Full-mission control period must be >= the JSBSim integration step.")
        if args.sixdof_full_mission_sample_period_s <= 0:
            parser.error("Full-mission sample period must be positive.")
    seed_record = json.loads(args.seed_candidate.read_text(encoding="utf-8"))
    calibration = calibrate_endurance_task_baseline()
    assumptions = calibration.assumptions
    overrides = {
        "engine_rating_multiplier": args.engine_rating_multiplier,
        "motor_rating_multiplier": args.motor_rating_multiplier,
        "battery_energy_multiplier": args.battery_energy_multiplier,
    }
    sizing_overrides = {
        "baseline_engine_power_kw": args.engine_baseline_power_kw,
        "engine_upgrade_kw_per_hybrid_kg": args.engine_upgrade_kw_per_hybrid_kg,
        "engine_power_ceiling_kw": args.engine_power_ceiling_kw,
        "hybrid_system_specific_energy_wh_per_kg": args.hybrid_specific_energy_wh_per_kg,
        "motor_power_kw_per_hybrid_kg": args.motor_power_kw_per_hybrid_kg,
    }
    assumptions = replace(
        assumptions,
        **{
            name: value
            for name, value in {**overrides, **sizing_overrides}.items()
            if value is not None
        },
        engine_efficiency_values=tuple(
            float(np.clip(value * args.engine_efficiency_factor, 0.05, 0.60))
            for value in assumptions.engine_efficiency_values
        ),
    )
    surrogate = load_vsp_surrogate(args.surrogate)
    aero_domain_frame = pd.read_csv(args.aero_domain_data)
    missing_aero_features = set(FEATURE_LABELS).difference(aero_domain_frame.columns)
    if missing_aero_features:
        raise ValueError(f"Aerodynamic DOE is missing features: {sorted(missing_aero_features)}")
    aero_domain_bounds = {
        label: (float(aero_domain_frame[label].min()), float(aero_domain_frame[label].max()))
        for label in FEATURE_LABELS
    }
    if args.design_bounds_json:
        requested_bounds = json.loads(args.design_bounds_json.read_text(encoding="utf-8"))
        unknown = set(requested_bounds).difference(aero_domain_bounds)
        if unknown:
            raise ValueError(f"Unknown aerodynamic design-bound features: {sorted(unknown)}")
        for label, values in requested_bounds.items():
            if len(values) != 2:
                raise ValueError(f"Design bound for {label} must contain [minimum, maximum].")
            low, high = map(float, values)
            domain_low, domain_high = aero_domain_bounds[label]
            if low < domain_low or high > domain_high or low >= high:
                raise ValueError(
                    f"{label} bounds [{low}, {high}] must be increasing and inside "
                    f"the DOE range [{domain_low}, {domain_high}]."
                )
            aero_domain_bounds[label] = (low, high)
    disturbances = tuple(
        replace(
            event,
            headwind_step_mps=args.wind_step_mps,
            duration_s=args.wind_duration_s,
        )
        for event in H6_RESPONSE_DISTURBANCES
    )
    sixdof_validator = PTDL6DoFOptimizerValidator(
        duration_s=args.sixdof_duration_s,
        integration_step_s=(
            args.sixdof_integration_step_s
            if args.sixdof_mode == "stepwise"
            else None
        ),
    )
    surrogate_provider = lambda design, active_calibration: surrogate.corrected_aero(
        design, active_calibration
    )
    live_aero_provider = None
    stepwise_aero_validator = None
    if args.aero_mode in {"live-vspaero-finalists", "live-vspaero-all"}:
        live_aero_provider = LiveOpenVSPAeroProvider(
            python_executable=args.vsp_python,
            worker_script=ROOT / "运行脚本" / "运行_VSPAERO单构型评估.py",
            output_dir=output_dir / "OpenVSP_VSPAERO实算构型",
            calibration=calibration.dimension_calibration,
            mesh_level=args.vsp_mesh_level,
            timeout_s={"legacy": 180., "medium": 900., "fine": 1800.}[args.vsp_mesh_level],
        )
    elif args.aero_mode == "stepwise":
        stepwise_aero_validator = StepwiseOpenVSPAeroCandidateValidator(
            python_executable=args.vsp_python,
            output_dir=output_dir / "VSPAERO逐积分步候选复评",
            duration_s=args.aero_step_duration_s,
            integration_step_s=args.aero_integration_step_s,
            couple_with_jsbsim=args.sixdof_mode == "stepwise",
            jsbsim_duration_s=args.sixdof_duration_s,
            jsbsim_integration_step_s=args.sixdof_integration_step_s,
            mesh_level=args.vsp_mesh_level,
        )
    if args.aero_mode == "live-vspaero-all":
        aero_provider = live_aero_provider
        aero_model_label = "live OpenVSP/VSPAERO for every optimizer candidate; component-build-up CD0"
    elif args.aero_mode == "stepwise":
        aero_provider = None
        aero_model_label = (
            "live VSPAERO coupled to JSBSim at every integration step of a short "
            "candidate trajectory; returned aerodynamic coefficients feed the 80 h "
            "low-order mission"
            if args.sixdof_mode == "stepwise"
            else "live VSPAERO at every integration step of a short low-order point-mass "
            "candidate trajectory; mean coefficients feed the 80 h low-order mission"
        )
    else:
        aero_provider = surrogate_provider
        aero_model_label = (
            "VSPAERO-trained surrogate search with live OpenVSP/VSPAERO Pareto finalist re-evaluation"
            if args.aero_mode == "live-vspaero-finalists"
            else "VSPAERO-trained geometry surrogate only"
        )
    powertrain_factory = partial(
        HybridSystemPowertrain,
        config=HybridSystemAdapterConfig(
            engine_map_path=args.engine_map,
            engine_speed_rpm=args.engine_speed_rpm,
            motor_speed_rpm=args.motor_speed_rpm,
            motor_branches=args.motor_branches,
            engine_fuel_flow_scale=args.engine_fuel_flow_scale,
            engine_power_rate_limit_kwps=args.engine_power_rate_limit_kwps,
            battery_resistance_multiplier=args.battery_resistance_multiplier,
            inverter_efficiency=args.inverter_efficiency,
            battery_soc_min=args.battery_soc_min,
            battery_soc_max=args.battery_soc_max,
            engine_time_constant_s=args.engine_time_constant_s,
            engine_command_delay_s=args.engine_command_delay_s,
            engine_efficiency_multiplier=args.engine_efficiency_factor,
        ),
    )
    controller_factories = {
        "mpc": partial(
            MPCEnergyManagementCoordinator,
            MPCParameters(
                horizon_steps=args.mpc_horizon,
                engine_action_levels=args.mpc_action_levels,
                soc_terminal_weight_mj=args.mpc_soc_terminal_weight_mj,
                engine_slew_penalty_mj_per_kw=args.mpc_engine_slew_penalty,
            ),
        ),
        "rule": RuleBasedHybridCoordinator,
        "source-ems": partial(SourceThreeLayerCoordinator, inverter_efficiency=args.inverter_efficiency),
        "fuzzy": FuzzyAdaptiveCoordinator,
        "fault-tolerant-fuzzy": lambda: FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(), HealthState()
        ),
        "neural-imitation": lambda: NeuralImitationCoordinator(args.neural_policy_model),
    }
    (output_dir / "动力模型版本与参数.json").write_text(
        json.dumps(powertrain_factory().metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.controller == "phase-mpc":
        from 协同自适应方法 import PhaseScheduledMPC
        if args.phase_mpc_config is None:
            parser.error("--phase-mpc-config is required for phase-mpc.")
        phase_parameters = json.loads(args.phase_mpc_config.read_text(encoding="utf-8"))
        controller_factories["phase-mpc"] = lambda: PhaseScheduledMPC(phase_parameters)
    dispatch_factory = controller_factories[args.controller]
    if args.sixdof_mode == "full-mission":
        sixdof_validator = PTDL6DoFFullMissionOptimizerValidator(
            output_dir=output_dir / "JSBSim完整任务逐候选",
            mission_hours=args.sixdof_full_mission_hours,
            mission_duration_basis=args.sixdof_full_mission_duration_basis,
            integration_step_s=args.sixdof_integration_step_s,
            control_period_s=args.sixdof_full_mission_control_period_s,
            terminal_control_period_s=args.sixdof_full_mission_terminal_control_period_s,
            sample_period_s=args.sixdof_full_mission_sample_period_s,
            pitch_kp=args.sixdof_full_mission_pitch_kp,
            pitch_kd=args.sixdof_full_mission_pitch_kd,
            terminal_pitch_kp=args.sixdof_full_mission_terminal_pitch_kp,
            terminal_pitch_kd=args.sixdof_full_mission_terminal_pitch_kd,
            terminal_pitch_ki=args.sixdof_full_mission_terminal_pitch_ki,
            battery_energy_multiplier=assumptions.battery_energy_multiplier,
            engine_efficiency_values=assumptions.engine_efficiency_values,
            dispatch_policy_factory=dispatch_factory,
            minimum_soc=args.battery_soc_min,
            powertrain_config={
                "engine_speed_rpm": args.engine_speed_rpm,
                "motor_speed_rpm": args.motor_speed_rpm,
                "motor_branches": args.motor_branches,
                "engine_map_path": (
                    str(args.engine_map.resolve()) if args.engine_map else None
                ),
                "engine_fuel_flow_scale": args.engine_fuel_flow_scale,
                "engine_power_rate_limit_kwps": args.engine_power_rate_limit_kwps,
                "engine_time_constant_s": args.engine_time_constant_s,
                "engine_command_delay_s": args.engine_command_delay_s,
                "engine_efficiency_multiplier": args.engine_efficiency_factor,
                "battery_resistance_multiplier": args.battery_resistance_multiplier,
                "inverter_efficiency": args.inverter_efficiency,
                "battery_soc_min": args.battery_soc_min,
                "battery_soc_max": args.battery_soc_max,
            },
        )
    selected_algorithms = tuple(dict.fromkeys(args.algorithms))
    algorithm_paths = {
        "NSGA-II": ("NSGA-II", "ptdl_80h_joint_nsga_ii"),
        "MOEA/D": ("MOEA-D", "ptdl_80h_joint_moead"),
        "MOPSO": ("MOPSO", "ptdl_80h_joint_mopso"),
        "Hybrid NSGA-II/MOPSO": ("Hybrid-NSGA-II-MOPSO", "ptdl_80h_joint_hybrid_nsga_ii_mopso"),
        "PTDL-自适应分块": ("PTDL-Block-Adaptive", "ptdl_80h_joint_block_adaptive"),
        "PTDL-耦合代理辅助": ("PTDL-Coupled-Surrogate", "ptdl_80h_joint_coupled_surrogate"),
        "PTDL-耦合引导": ("PTDL-Coupled-Guided", "ptdl_80h_joint_coupled_guided"),
        "PTDL-耦合贡献学习": ("PTDL-Coupled-Credit", "ptdl_80h_joint_coupled_credit"),
        "PTDL-耦合贡献学习保先验": ("PTDL-Coupled-Credit-Prior", "ptdl_80h_joint_coupled_credit_prior"),
        "PTDL-去重贡献学习": ("PTDL-Coupled-Unique", "ptdl_80h_joint_coupled_unique"),
    }

    seed_problem = PTDLEnduranceJointProblem(
        aero_provider=aero_provider,
        powertrain_factory=powertrain_factory,
        dispatch_policy_factory=dispatch_factory,
        disturbances=disturbances,
        design_feature_bounds=aero_domain_bounds,
        assumptions=assumptions,
        response_definition=args.response_definition,
        sixdof_validator=(
            sixdof_validator
            if args.sixdof_mode in {"optimizer-loop", "full-mission"}
            or (
                args.sixdof_mode == "stepwise"
                and args.aero_mode != "stepwise"
            )
            else None
        ),
        stepwise_aero_validator=stepwise_aero_validator,
    )
    optimizations: dict[str, dict[str, object]] = {}
    source_root = args.reuse_optimizer_output_dir.resolve() if args.reuse_optimizer_output_dir else output_dir
    if args.reuse_optimizer_output_dir:
        for algorithm in selected_algorithms:
            directory, prefix = algorithm_paths[algorithm]
            source_dir = source_root / directory
            destination_dir = output_dir / directory
            destination_dir.mkdir(parents=True, exist_ok=True)
            for filename in (
                f"{prefix}_convergence.csv",
                f"{prefix}_feasible_pareto.csv",
                f"{prefix}_evaluated_candidates.csv",
                f"{prefix}_generation_feasible_pareto.csv",
                f"{prefix}_final_population_feasible_pareto.csv",
            ):
                shutil.copy2(source_dir / filename, destination_dir / filename)
            archived_items = _read_evaluation_archive(
                source_dir / f"{prefix}_evaluated_candidates.csv", seed_problem
            )
            archived_final = _read_evaluation_archive(
                source_dir / f"{prefix}_final_population.csv", seed_problem
            )
            reevaluated: dict[tuple[float, ...], CandidateEvaluation] = {}
            for item in archived_items + archived_final:
                key = tuple(np.round(item.decision, 12))
                if key not in reevaluated:
                    reevaluated[key] = seed_problem.evaluate(item.decision)
            archive_items = [
                reevaluated[tuple(np.round(item.decision, 12))]
                for item in archived_items
            ]
            final_items = [
                reevaluated[tuple(np.round(item.decision, 12))]
                for item in archived_final
            ]
            history = pd.read_csv(source_dir / f"{prefix}_convergence.csv")
            run = OptimizerRun(
                algorithm,
                np.vstack([item.decision for item in final_items]),
                final_items,
                history,
                archive_items,
            )
            optimizations[algorithm] = {
                "run": run,
                "feasible_archive": OptimizerRun(
                    algorithm,
                    np.vstack([item.decision for item in archive_items]),
                    archive_items,
                    history,
                    archive_items,
                ),
                "problem": seed_problem,
            }
    else:
        seed_decision = _candidate_decision(seed_problem, seed_record)
        for algorithm_index, algorithm in enumerate(selected_algorithms):
            directory, _ = algorithm_paths[algorithm]

            def on_generation(
                generation: int,
                total: int,
                *,
                index: int = algorithm_index,
                name: str = algorithm,
            ) -> None:
                fraction = (index + generation / max(total, 1)) / len(selected_algorithms)
                report_progress(
                    5 + 65 * fraction,
                    "多目标优化",
                    f"{name}：第 {generation}/{total} 代",
                )

            report_progress(
                5 + 65 * algorithm_index / len(selected_algorithms),
                "多目标优化",
                f"正在运行 {algorithm}，共 {args.generations} 代",
            )
            optimizations[algorithm] = run_ptdl_80h_joint_moead(
                output_dir / directory,
                resolution=args.resolution,
                generations=args.generations,
                seed=args.seed,
                aero_provider=aero_provider,
                aero_model_label=aero_model_label,
                powertrain_factory=powertrain_factory,
                dispatch_policy_factory=dispatch_factory,
                seed_decisions=[seed_decision],
                algorithm=algorithm,
                disturbances=disturbances,
                design_feature_bounds=aero_domain_bounds,
                assumptions=assumptions,
                sixdof_validator=(
                    sixdof_validator
                    if args.sixdof_mode in {"optimizer-loop", "full-mission"}
                    or (
                        args.sixdof_mode == "stepwise"
                        and args.aero_mode != "stepwise"
                    )
                    else None
                ),
                stepwise_aero_validator=stepwise_aero_validator,
                progress_callback=on_generation,
                response_definition=args.response_definition,
            )
    report_progress(72, "Pareto 复评", "正在整理各算法的历代可行 Pareto 档案")
    all_candidates: list[tuple[str, int, object, object]] = []
    algorithm_rows: list[dict[str, object]] = []
    for algorithm, result in optimizations.items():
        run = result["run"]
        archive = result["feasible_archive"]
        result_dir = source_root / algorithm_paths[algorithm][0]
        for index in pareto_indices(archive.evaluations):
            all_candidates.append((algorithm, index, archive.evaluations[index], result["problem"]))
        feasible_final = [item for item in run.evaluations if item.feasible]
        feasible_history = [
            item for item in (run.evaluated_archive or run.evaluations) if item.feasible
        ]
        historical_pareto = [
            archive.evaluations[index] for index in pareto_indices(archive.evaluations)
        ]
        final_pareto = [
            run.evaluations[index] for index in pareto_indices(run.evaluations)
        ]
        historical_objectives = [item.result.objectives for item in historical_pareto]
        final_objectives = [item.result.objectives for item in final_pareto]
        algorithm_rows.append(
            {
                "algorithm": algorithm,
                "population_size": len(run.evaluations),
                "generations": args.generations,
                "comparison_metric_basis": "历代累计可行 Pareto 外部档案",
                "final_population_feasible_count": len(feasible_final),
                "final_population_pareto_count": len(final_pareto),
                "final_population_pareto_min_J_energy_mj": min(
                    (item["J_energy_mj"] for item in final_objectives), default=np.nan
                ),
                "final_population_pareto_min_J_stability": min(
                    (item["J_stability"] for item in final_objectives), default=np.nan
                ),
                "final_population_pareto_min_J_response_s": min(
                    (item["J_response_s"] for item in final_objectives), default=np.nan
                ),
                "historical_evaluated_feasible_count": len(feasible_history),
                "historical_feasible_pareto_count": len(historical_pareto),
                "historical_pareto_min_J_energy_mj": min(
                    (item["J_energy_mj"] for item in historical_objectives), default=np.nan
                ),
                "historical_pareto_min_J_stability": min(
                    (item["J_stability"] for item in historical_objectives), default=np.nan
                ),
                "historical_pareto_min_J_response_s": min(
                    (item["J_response_s"] for item in historical_objectives), default=np.nan
                ),
                "runtime_s": json.loads(
                    (result_dir / f"{algorithm_paths[algorithm][1]}_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )["optimizer"]["runtime_s"],
            }
        )
    algorithm_comparison = pd.DataFrame(algorithm_rows)
    algorithm_comparison.to_csv(output_dir / "NSGA-II与MOEA-D同模型算法比较.csv", index=False)
    if not all_candidates:
        message = (
            "优化已完成，当前动力参数和任务约束下未找到可行候选。"
            "各算法的约束违背最少候选和收敛记录已保存。"
        )
        (output_dir / "四项统一集成验证摘要.json").write_text(
            json.dumps({
                "optimization_completed": True,
                "candidate_feasible_after_independent_replay": False,
                "outcome": "no_feasible_candidate",
                "message": message,
                "powertrain_source": powertrain_factory().metadata,
                "optimizers": list(selected_algorithms),
                "controller_used_by_optimizer": args.controller,
                "algorithm_comparison": algorithm_rows,
                "optimization": {
                    "resolution": args.resolution, "generations": args.generations,
                    "seed": args.seed,
                },
                "response_disturbance": {"objective_definition": args.response_definition},
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_progress(100, "完成，未找到可行候选", message)
        print(message)
        return
    if args.aero_mode == "live-vspaero-finalists":
        assert live_aero_provider is not None
        report_progress(76, "VSPAERO 复评", "正在对 Pareto 候选进行 OpenVSP/VSPAERO 实算")
        finalist_problem = PTDLEnduranceJointProblem(
            profile_path=seed_problem.profile_path,
            aero_provider=live_aero_provider,
            powertrain_factory=powertrain_factory,
            dispatch_policy_factory=dispatch_factory,
            disturbances=disturbances,
            design_feature_bounds=aero_domain_bounds,
            assumptions=assumptions,
            sixdof_validator=(
                sixdof_validator
                if args.sixdof_mode in {"optimizer-loop", "full-mission"}
                or (
                    args.sixdof_mode == "stepwise"
                    and args.aero_mode != "stepwise"
                )
                else None
            ),
            stepwise_aero_validator=stepwise_aero_validator,
        )
        live_candidates: list[tuple[str, int, object, object]] = []
        for candidate_index, (algorithm, index, item, _) in enumerate(all_candidates):
            live_evaluation = finalist_problem.evaluate(item.decision)
            live_candidates.append((algorithm, index, live_evaluation, finalist_problem))
            report_progress(
                76 + 15 * (candidate_index + 1) / max(len(all_candidates), 1),
                "VSPAERO 复评",
                f"候选 {candidate_index + 1}/{len(all_candidates)}",
            )
        all_candidates = live_candidates
        aero_provider = live_aero_provider
        aero_model_label = (
            "VSPAERO surrogate search; finalists re-evaluated with live OpenVSP/VSPAERO "
            "and candidate-specific JSBSim constraints"
        )
    if args.sixdof_mode == "pareto-screen":
        report_progress(91, "六自由度筛查", "正在对各算法 Pareto 候选执行 JSBSim 短时复评")
        screen_problem = PTDLEnduranceJointProblem(
            profile_path=seed_problem.profile_path,
            aero_provider=aero_provider,
            powertrain_factory=powertrain_factory,
            dispatch_policy_factory=dispatch_factory,
            disturbances=disturbances,
            design_feature_bounds=aero_domain_bounds,
            assumptions=assumptions,
            sixdof_validator=sixdof_validator,
            stepwise_aero_validator=stepwise_aero_validator,
        )
        screened_candidates: list[tuple[str, int, object, object]] = []
        for candidate_index, (algorithm, index, item, _) in enumerate(all_candidates):
            evaluation = screen_problem.evaluate(item.decision)
            if evaluation.result.feasible:
                screened_candidates.append(
                    (algorithm, index, evaluation, screen_problem)
                )
            report_progress(
                91 + 2 * (candidate_index + 1) / max(len(all_candidates), 1),
                "六自由度筛查",
                f"Pareto 候选 {candidate_index + 1}/{len(all_candidates)}",
            )
        all_candidates = screened_candidates
        if not all_candidates:
            raise SystemExit(
                "Pareto 候选经六自由度筛查后均不可行；请改用优化闭环模式或调整候选/任务设置。"
            )
        pd.DataFrame(
            [
                {"algorithm": algorithm, **item.result.objectives}
                for algorithm, _, item, _ in all_candidates
            ]
        ).to_csv(
            output_dir / "六自由度筛查后Pareto目标.csv",
            index=False,
        )
    objective_frame = pd.DataFrame(
        [item.result.objectives for _, _, item, _ in all_candidates],
        columns=OBJECTIVE_LABELS,
    )
    normalizer = ObjectiveNormalizer.from_samples(
        objective_frame,
        labels=OBJECTIVE_LABELS,
        worst_quantile=0.95,
    )
    margin_candidates: list[tuple[str, int, object, object, float, float, float]] = []
    scan_rows: list[dict[str, object]] = []
    for algorithm, index, candidate, candidate_problem in all_candidates:
        diagnostics = candidate.result.diagnostics
        initial_fuel = float(diagnostics.get("fuel_mass_kg", 0.0))
        fuel_remaining = initial_fuel - float(diagnostics.get("fuel_used_kg", 0.0))
        terminal_soc = float(diagnostics.get("terminal_soc", float("nan")))
        fuel_margin = fuel_remaining - FUEL_RESERVE_FRACTION * initial_fuel
        response_success = float(diagnostics.get("response_recovery_success", 1.0)) >= 1.0
        margin_pass = (
            terminal_soc >= SOC_SELECTION_FLOOR
            and fuel_remaining > 0.0
            and response_success
        )
        normalized = normalizer.transform(candidate.result.objectives)
        for preference, weights in PREFERENCE_WEIGHTS.items():
            score = weighted_preference_score(normalized, weights)
            scan_rows.append(
                {
                    "algorithm": algorithm,
                    "archive_candidate_index": index,
                    "preference": preference,
                    **{f"w_{key.removeprefix('J_')}": value for key, value in weights.items()},
                    **{f"normalized_{key}": value for key, value in normalized.items()},
                    "weighted_score": score,
                    **candidate.result.objectives,
                    "terminal_soc": terminal_soc,
                    "fuel_remaining_kg": fuel_remaining,
                    "fuel_reserve_required_kg": FUEL_RESERVE_FRACTION * initial_fuel,
                    "fuel_margin_kg": fuel_margin,
                    "response_recovery_success": response_success,
                    "margin_screen_pass": margin_pass,
                    "five_percent_fuel_target_met": fuel_margin >= 0.0,
                    "selected": False,
                    "selected_as_primary": False,
                }
            )
        if margin_pass:
            margin_candidates.append(
                (algorithm, index, candidate, candidate_problem, terminal_soc, fuel_remaining, fuel_margin)
            )
    report_progress(93, "方案筛选", "正在完成目标归一化、裕度筛选与独立回放")
    if not margin_candidates:
        pd.DataFrame(scan_rows).to_csv(output_dir / "多目标权重扫描与安全裕度筛选.csv", index=False)
        raise SystemExit(
            "现有硬约束可行历史候选中，没有同时满足 SOC、燃油余量和响应恢复筛选的候选。"
            "未生成独立回放，请扩展设计空间或重新评估任务要求。"
        )
    screened_front = [
        margin_candidates[index]
        for index in pareto_indices([entry[2] for entry in margin_candidates])
    ]
    selected_by_preference: dict[str, tuple[str, int, object, object, float, float, float]] = {}
    for preference, weights in PREFERENCE_WEIGHTS.items():
        selected_by_preference[preference] = min(
            screened_front,
            key=lambda entry: weighted_preference_score(
                normalizer.transform(entry[2].result.objectives), weights
            ),
        )
        chosen_row = selected_by_preference[preference]
        for row in scan_rows:
            if (
                row["preference"] == preference
                and row["algorithm"] == chosen_row[0]
                and row["archive_candidate_index"] == chosen_row[1]
            ):
                row["selected"] = True
    pd.DataFrame(scan_rows).to_csv(output_dir / "多目标权重扫描与安全裕度筛选.csv", index=False)
    reserve_choice = max(margin_candidates, key=lambda entry: entry[5])
    for row in scan_rows:
        if row["algorithm"] == reserve_choice[0] and row["archive_candidate_index"] == reserve_choice[1]:
            row["selected_as_primary"] = True
    pd.DataFrame(scan_rows).to_csv(output_dir / "多目标权重扫描与安全裕度筛选.csv", index=False)
    chosen_algorithm, chosen_index, chosen, problem, chosen_terminal_soc, chosen_fuel_remaining, chosen_fuel_margin = reserve_choice
    decision = chosen.decision
    design = problem.design_from_decision(decision)
    phase_count = len(problem.phase_labels)
    engine_caps = decision[11 : 11 + phase_count].tolist()
    motor_assist = decision[11 + phase_count :].tolist()
    speeds = [phase.speed_target_mps for phase in problem.base_mission.phases]
    loiter_index = problem.phase_labels.index("endurance_loiter")
    speeds[loiter_index] = float(decision[10])
    if getattr(aero_provider, "accepts_operating_point", False):
        aero = aero_provider(
            design,
            calibration.dimension_calibration,
            speed_mps=float(decision[10]),
            altitude_m=float(problem.base_mission.phases[loiter_index].altitude_target_m),
        )
    else:
        aero = aero_provider(design, calibration.dimension_calibration)
    aircraft = preliminary_hybrid_aircraft(
        design,
        calibration.dimension_calibration,
        assumptions,
        aero_override=aero,
    )

    policies = {
        "1.2.1_rule_based": RuleBasedHybridCoordinator,
        "1.2.2_fuzzy_adaptive": FuzzyAdaptiveCoordinator,
        "1.2.2_fault_tolerant_fuzzy": lambda: FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(), HealthState()
        ),
        "1.2.2_mpc": controller_factories["mpc"],
        "1.2.2_neural_imitation": controller_factories["neural-imitation"],
    }
    if args.controller == "phase-mpc":
        policies["1.2.2_phase_mpc"] = controller_factories["phase-mpc"]
    if args.controller == "source-ems":
        policies["1.2.2_source_ems"] = controller_factories["source-ems"]
    selected_policy_name = {
        "mpc": "1.2.2_mpc",
        "rule": "1.2.1_rule_based",
        "fuzzy": "1.2.2_fuzzy_adaptive",
        "fault-tolerant-fuzzy": "1.2.2_fault_tolerant_fuzzy",
        "neural-imitation": "1.2.2_neural_imitation",
        "phase-mpc": "1.2.2_phase_mpc",
        "source-ems": "1.2.2_source_ems",
    }[args.controller]
    comparisons: list[dict[str, object]] = []
    selected_summary: dict[str, object] | None = None
    for controller_name, policy_factory in policies.items():
        powertrain = powertrain_factory()
        evaluation = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            calibration=calibration.dimension_calibration,
            assumptions=assumptions,
            aero_override=aero,
            dispatch_policy=policy_factory(),
            powertrain=powertrain,
            disturbances=disturbances,
        )
        nominal_evaluation = evaluate_endurance_task(
            design,
            engine_caps_kw=engine_caps,
            motor_assist_kw=motor_assist,
            speeds_mps=speeds,
            calibration=calibration.dimension_calibration,
            assumptions=assumptions,
            aero_override=aero,
            dispatch_policy=policy_factory(),
            powertrain=powertrain_factory(),
            disturbances=zero_amplitude_disturbances(disturbances),
        )
        apply_paired_response_metrics(
            evaluation,
            nominal_evaluation,
            disturbances,
            definition=args.response_definition,
        )
        summary = _evaluation_summary(
            evaluation, aircraft, powertrain, controller_name, design, assumptions
        )
        comparisons.append(
            {
                "controller": controller_name,
                "feasible": summary["feasible"],
                "total_violation": summary["total_violation"],
                "endurance_h": summary["achieved_total_duration_h"],
                "fuel_used_kg": summary["fuel_used_kg"],
                "terminal_soc": summary["terminal_soc"],
                **summary["objectives"],
            }
        )
        if controller_name == selected_policy_name:
            selected_summary = summary
            evaluation.simulation.profile.to_csv(
                output_dir / f"择优候选_{args.controller}_时序.csv",
                index=False,
            )
            _write_visual(
                evaluation.simulation.profile,
                summary,
                output_dir / f"择优候选_{args.controller}.png",
            )

    if selected_summary is None or not selected_summary["feasible"]:
        raise SystemExit(
            f"所选控制器 {args.controller} 独立回放未通过 2.2.1 可行性复核，拒绝将候选标记为可行。"
        )

    aero_records = getattr(aero_provider, "records", [])
    if stepwise_aero_validator is not None:
        aero_records = stepwise_aero_validator.records
    if aero_records:
        pd.DataFrame(aero_records).to_csv(
            output_dir
            / (
                "VSPAERO逐积分步候选评价.csv"
                if stepwise_aero_validator is not None
                else "OpenVSP_VSPAERO候选气动实算记录.csv"
            ),
            index=False,
        )
    if sixdof_validator.records:
        pd.DataFrame(sixdof_validator.records).to_csv(
            output_dir
            / (
                "JSBSim完整任务逐候选优化记录.csv"
                if args.sixdof_mode == "full-mission"
                else "JSBSim六自由度优化候选筛查.csv"
            ),
            index=False,
        )

    candidate_output = {
        "provenance": f"selected from component-backed PTDL feasible archives; selected algorithm: {chosen_algorithm}",
        "optimizer": chosen_algorithm,
        "controller": args.controller,
        "decision_labels": list(problem.decision_labels),
        "decision": [float(value) for value in decision],
        "design": {
            name: float(value)
            for name, value in zip(problem.design_labels, decision[: len(problem.design_labels)])
        },
        "isr_cruise_speed_mps": float(decision[10]),
        "engine_caps_kw": engine_caps,
        "motor_assist_kw": motor_assist,
        "optimization_feasible": bool(chosen.feasible),
        "selection_preference": "reserve_priority_max_remaining_fuel",
        "shared_normalized_objectives": normalizer.transform(chosen.result.objectives),
        "selection_margins": {
            "terminal_soc_floor": SOC_SELECTION_FLOOR,
            "terminal_soc": chosen_terminal_soc,
            "fuel_reserve_fraction": FUEL_RESERVE_FRACTION,
            "fuel_remaining_kg": chosen_fuel_remaining,
            "fuel_margin_over_screen_kg": chosen_fuel_margin,
            "response_recovery_success": bool(
                chosen.result.diagnostics.get("response_recovery_success", 1.0) >= 1.0
            ),
            "five_percent_target_met": chosen_fuel_margin >= 0.0,
            "status": "SOC, positive fuel, and configured disturbance recovery screened; 5% fuel value remains advisory",
        },
        "high_fidelity_candidate_screen": {
            "aerodynamic_constraints": {
                name: float(value)
                for name, value in chosen.result.constraints.items()
                if name.startswith("vsp_stepwise_")
            },
            "aerodynamic_diagnostics": {
                name: value
                for name, value in chosen.result.diagnostics.items()
                if name.startswith("vsp_stepwise_")
            },
            "sixdof_constraints": {
                name: float(value)
                for name, value in chosen.result.constraints.items()
                if name.startswith("sixdof_")
            },
            "sixdof_diagnostics": {
                name: value
                for name, value in chosen.result.diagnostics.items()
                if name.startswith("sixdof_")
            },
            "aerodynamic_coefficients_used": {
                "cd0_component_build_up": aero.cd0,
                "cl_alpha_per_rad": aero.cl_alpha_per_rad,
                "oswald_efficiency": aero.oswald_efficiency,
                "cl_max": aero.cl_max,
                "propulsive_efficiency": aero.propulsive_efficiency,
            },
            "power_limit_basis": (
                "median closed-loop propulsor input power in endurance_loiter; "
                "minimum across uncertainty scenarios"
            ),
            "status": (
                "candidate-specific software-level screen; short stepwise runs feed the "
                "low-order 80 h mission through reduced aerodynamic coefficients; not full-task "
                "6-DoF or flight validation"
            ),
        },
        "independent_replay": selected_summary,
    }
    (output_dir / "NSGA-II与MOEA-D择优可行候选.json").write_text(
        json.dumps(candidate_output, indent=2),
        encoding="utf-8",
    )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "控制器部件模型统一评价.csv", index=False)
    metadata = {
        "integrated_sections": ["1.2.1", "1.2.2", "2.2.1", "2.2.2"],
        "powertrain_components": ["engine_map_genset", "generator_gearbox", "battery_rint", "motor_pmsm", "power_electronics_ems"],
        "powertrain_source": powertrain_factory().metadata,
        "optimizers": list(selected_algorithms),
        "controller_used_by_optimizer": args.controller,
        "controller_parameters": {
            "mpc_horizon_steps": args.mpc_horizon,
            "mpc_engine_action_levels": args.mpc_action_levels,
            "mpc_soc_terminal_weight_mj": args.mpc_soc_terminal_weight_mj,
            "mpc_engine_slew_penalty_mj_per_kw": args.mpc_engine_slew_penalty,
        },
        "powertrain_adapter_parameters": {
            "engine_speed_rpm": args.engine_speed_rpm,
            "motor_speed_rpm": args.motor_speed_rpm,
            "motor_branches": args.motor_branches,
            "engine_fuel_flow_scale": args.engine_fuel_flow_scale,
            "engine_efficiency_multiplier": args.engine_efficiency_factor,
            "engine_power_rate_limit_kwps": args.engine_power_rate_limit_kwps,
            "battery_resistance_multiplier": args.battery_resistance_multiplier,
            "inverter_efficiency": args.inverter_efficiency,
            "battery_soc_min": args.battery_soc_min,
            "battery_soc_max": args.battery_soc_max,
            "engine_map_path": str(args.engine_map) if args.engine_map else "包内默认 EngineMap_60Case.csv",
        },
        "aero_model": aero_model_label,
        "propulsion_sizing_assumptions": {
            "baseline_engine_power_kw": assumptions.baseline_engine_power_kw,
            "engine_upgrade_kw_per_hybrid_kg": assumptions.engine_upgrade_kw_per_hybrid_kg,
            "engine_power_ceiling_kw": assumptions.engine_power_ceiling_kw,
            "engine_rating_multiplier": assumptions.engine_rating_multiplier,
            "motor_rating_multiplier": assumptions.motor_rating_multiplier,
            "battery_energy_multiplier": assumptions.battery_energy_multiplier,
            "hybrid_system_specific_energy_wh_per_kg": assumptions.hybrid_system_specific_energy_wh_per_kg,
            "motor_power_kw_per_hybrid_kg": assumptions.motor_power_kw_per_hybrid_kg,
            "engine_efficiency_factor": args.engine_efficiency_factor,
            "efficiency_scale_from_1p20_baseline": calibration.efficiency_scale,
            "interpretation": "PTDL equivalent sizing; component fuel uses the 1005 altitude map directly, independent of the historical baseline efficiency curve",
        },
        "aero_training_samples": surrogate.training_samples,
        "aero_domain": {
            "dataset": str(args.aero_domain_data.resolve()),
            "featurewise_min_max": {
                label: {"minimum": bounds[0], "maximum": bounds[1]}
                for label, bounds in aero_domain_bounds.items()
            },
            "scope": "optimizer geometry bounds only; featurewise ranges do not prove interpolation inside the DOE convex hull",
        },
        "candidate_feasible_after_independent_replay": True,
        "sixdof_candidate_constraint": {
            "model": "PTDL_EST_6DOF with candidate geometry, mass, inertia, and aerodynamic coefficients",
            "duration_s": (
                args.sixdof_full_mission_hours * 3600.0
                if args.sixdof_mode == "full-mission"
                else args.sixdof_duration_s
            ),
            "duration_basis": (
                args.sixdof_full_mission_duration_basis
                if args.sixdof_mode == "full-mission"
                else None
            ),
            "integration_step_s": (
                args.sixdof_integration_step_s
                if args.sixdof_mode in {"stepwise", "full-mission"}
                else None
            ),
            "evaluation_scope": (
                (
                    "each optimizer candidate must complete the requested airborne endurance before powered descent, approach, flare, and touchdown"
                    if args.sixdof_full_mission_duration_basis
                    == "endurance-before-landing"
                    else "full six-phase JSBSim mission for every optimizer candidate; candidate phase-power schedule is passed into the 6-DoF model"
                )
                if args.sixdof_mode == "full-mission"
                else
                "per-optimizer-candidate, explicit JSBSim timestep trajectory"
                if args.sixdof_mode == "stepwise"
                else "per-optimizer-candidate constraint"
                if args.sixdof_mode == "optimizer-loop"
                else "post-optimization Pareto-candidate screen"
            ),
            "control": (
                f"{args.controller} propulsion dispatch with the six-phase attitude, route, and altitude controller"
                if args.sixdof_mode == "full-mission"
                else "estimated three-axis attitude/rate feedback plus sequential small surface pulses"
            ),
            "evaluated_candidate_count": (
                len(aero_records)
                if args.aero_mode == "stepwise" and args.sixdof_mode == "stepwise"
                else len(sixdof_validator.records)
            ),
            "evidence_csv": (
                "VSPAERO逐积分步候选评价.csv"
                if args.aero_mode == "stepwise" and args.sixdof_mode == "stepwise"
                else "JSBSim完整任务逐候选优化记录.csv"
                if args.sixdof_mode == "full-mission"
                else "JSBSim六自由度优化候选筛查.csv"
            ),
            "coupled_effective_duration_s": (
                max(args.aero_step_duration_s, args.sixdof_duration_s)
                if args.aero_mode == "stepwise" and args.sixdof_mode == "stepwise"
                else None
            ),
            "coupled_effective_integration_step_s": (
                min(args.aero_integration_step_s, args.sixdof_integration_step_s)
                if args.aero_mode == "stepwise" and args.sixdof_mode == "stepwise"
                else None
            ),
            "status": (
                "estimated-parameter full-mission optimizer feedback; software evidence only"
                if args.sixdof_mode == "full-mission"
                else "estimated software-level time-domain screen; not engineering validation"
            ),
        },
        "openvsp_vspaero_candidate_aerodynamics": {
            "mode": args.aero_mode,
            "evaluated_geometry_count": len(aero_records),
            "integration_step_s": (
                args.aero_integration_step_s
                if args.aero_mode == "stepwise"
                else None
            ),
            "duration_s": (
                args.aero_step_duration_s
                if args.aero_mode == "stepwise"
                else None
            ),
            "effective_duration_s": (
                max(args.aero_step_duration_s, args.sixdof_duration_s)
                if args.aero_mode == "stepwise" and args.sixdof_mode == "stepwise"
                else args.aero_step_duration_s
                if args.aero_mode == "stepwise"
                else None
            ),
            "effective_integration_step_s": (
                min(args.aero_integration_step_s, args.sixdof_integration_step_s)
                if args.aero_mode == "stepwise" and args.sixdof_mode == "stepwise"
                else args.aero_integration_step_s
                if args.aero_mode == "stepwise"
                else None
            ),
            "vsp_python": (
                str(args.vsp_python.resolve())
                if args.aero_mode in {"live-vspaero-finalists", "live-vspaero-all", "stepwise"}
                else None
            ),
            "evidence_csv": (
                "VSPAERO逐积分步候选评价.csv"
                if args.aero_mode == "stepwise"
                else "OpenVSP_VSPAERO候选气动实算记录.csv"
                if args.aero_mode in {"live-vspaero-finalists", "live-vspaero-all"}
                else None
            ),
            "scope": "live VSPAERO VLM supplies CL-alpha and induced-drag trends; CD0 remains a component-build-up estimate",
        },
        "feasible_archive_size": len(all_candidates),
        "margin_screen_candidate_count": len(margin_candidates),
        "five_percent_fuel_target_candidate_count": sum(row["five_percent_fuel_target_met"] for row in scan_rows) // len(PREFERENCE_WEIGHTS),
        "primary_selection_rule": "maximize remaining fuel among original-hard-feasible candidates with terminal SOC >= 25%, positive fuel, and successful configured disturbance recovery; the three-objective Pareto front remains available separately",
        "normalization_reference": {
            "labels": list(normalizer.labels),
            "ideal": normalizer.ideal.tolist(),
            "reference_worst_p95": normalizer.reference_worst.tolist(),
            "population_basis": "union of selected algorithms' all-generation feasible evaluated candidates",
        },
        "preference_weights": PREFERENCE_WEIGHTS,
        "provisional_margin_screen": {
            "terminal_soc_min": SOC_SELECTION_FLOOR,
            "fuel_reserve_fraction": FUEL_RESERVE_FRACTION,
            "response_recovery_required": True,
            "purpose": "SOC and successful configured disturbance recovery are additional selection screens; 5% fuel is advisory because it is not a PPT hard constraint",
        },
        "response_disturbance": {
            "objective_definition": args.response_definition,
            "integral_definition": "integral sqrt((delta_v/0.5)^2+(delta_h/height_band)^2) dt; seconds",
            "type": "equivalent headwind step in the low-order longitudinal model; a stress-test assumption, not measured wind data",
            "phase": "endurance_loiter",
            "start_after_phase_entry_s": 3600.0,
            "duration_s": args.wind_duration_s,
            "headwind_step_mps": args.wind_step_mps,
            "recovery_window_s": 900.0,
            "recovery_dwell_s": 30.0,
            "response_method": "paired disturbed-versus-zero-amplitude trajectory under the same candidate and controller",
            "measured_response_metrics": selected_summary["response_metrics"] if selected_summary else None,
        },
        "selected_algorithm": chosen_algorithm,
        "chosen_candidate_archive_index": chosen_index,
        "optimization": {
            "resolution": args.resolution,
            "generations": args.generations,
            "seed": args.seed,
        },
        "selected_candidate_summary": selected_summary,
        "controller_comparison": comparisons,
        "algorithm_comparison": algorithm_rows,
        "metric_basis": (
            "three PPT objective categories retained; response uses recovery plus normalized paired-error integral; hard constraints unchanged"
            if args.response_definition == "recovery-integral" else
            "historical PPT J_E, J_stab, J_resp definitions and hard constraints retained"
        ),
        "model_boundary": "preliminary equivalent sizing; not validated hardware performance",
    }
    (output_dir / "四项统一集成验证摘要.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    report_progress(99, "完成", "优化、复评和择优候选独立回放已完成")
    print(comparison_frame.to_string(index=False))
    print("\nSame-model optimizer comparison:")
    print(algorithm_comparison.to_string(index=False))
    print(f"\nFeasible selected candidate: {output_dir / 'NSGA-II与MOEA-D择优可行候选.json'}")
    print(f"Validation chart: {output_dir / 'NSGA-II与MOEA-D择优候选_MPC.png'}")


if __name__ == "__main__":
    main()
