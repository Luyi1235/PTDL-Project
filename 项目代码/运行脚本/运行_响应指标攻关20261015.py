"""Paired wind, plant delay and step-size comparisons without changing objectives."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 八十小时续航任务模型 import evaluate_endurance_task, calibrate_endurance_task_baseline
from 飞推控闭环物理模型 import FlightControlParameters
from MOEAD多目标联合优化 import PTDLEnduranceJointProblem
from VSPAERO气动代理模型 import load_vsp_surrogate
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 规则混动协同控制 import RuleBasedHybridCoordinator
from 智能能量管理与容错调度 import (
    MPCEnergyManagementCoordinator, FuzzyAdaptiveCoordinator, NeuralImitationCoordinator,
)
from 协同自适应方法 import PhaseScheduledMPC
from 响应指标 import project_response_stress_scenario, zero_amplitude_disturbances, apply_paired_response_metrics

OUT = ROOT / "运行结果" / "响应指标与控制参数_可行基准_20261015"
PHASE = ROOT / "运行结果" / "蓝色攻关_控制与轨迹_20261013" / "分阶段MPC配置.json"
NEURAL = ROOT / "运行结果" / "神经网络MPC模仿策略训练测试_20261011" / "神经网络MPC模仿策略.json"


def worker(job):
    policy, wind, delay, step = job
    name = f"{policy}_wind{wind:g}_delay{delay:g}_dt{step:g}"
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "结果.json"
    if done.exists():
        saved = json.loads(done.read_text(encoding="utf8"))
        if saved.get("numerical_rate_tolerance") == "20261015":
            return saved
    factories = dict(rule=RuleBasedHybridCoordinator, fuzzy=FuzzyAdaptiveCoordinator,
                     mpc=MPCEnergyManagementCoordinator,
                     scheduled=lambda: PhaseScheduledMPC(json.loads(PHASE.read_text(encoding="utf8"))),
                     neural=lambda: NeuralImitationCoordinator(NEURAL))
    model = load_vsp_surrogate(ROOT / "气动代理模型数据" / "气动代理模型_211构型_分目标GP_20261013.json")
    seed = json.loads((OUT / "比较基准.json").read_text(encoding="utf8"))["decision"]
    problem = PTDLEnduranceJointProblem()
    x = np.clip(seed, problem.lower_bounds, problem.upper_bounds)
    design = problem.design_from_decision(x)
    count = len(problem.phase_labels)
    calibration = calibrate_endurance_task_baseline()
    aero = model.corrected_aero(design, calibration.dimension_calibration)
    speeds = [p.speed_target_mps for p in problem.base_mission.phases]
    speeds[problem.phase_labels.index("endurance_loiter")] = float(x[10])
    event = replace(project_response_stress_scenario(wind)[0], start_time_s=3601.3,
                    duration_s=61.7, response_step_s=step)
    config = HybridSystemAdapterConfig(engine_command_delay_s=delay)
    common = dict(
        design=design, engine_caps_kw=x[11:11+count], motor_assist_kw=x[11+count:11+2*count],
        speeds_mps=speeds, calibration=calibration.dimension_calibration,
        assumptions=calibration.assumptions, aero_override=aero,
    )
    disturbed = evaluate_endurance_task(**common, dispatch_policy=factories[policy](),
                                      powertrain=HybridSystemPowertrain(config=config), disturbances=(event,))
    nominal = evaluate_endurance_task(**common, dispatch_policy=factories[policy](),
                                    powertrain=HybridSystemPowertrain(config=config),
                                    disturbances=zero_amplitude_disturbances((event,)))
    metrics = apply_paired_response_metrics(disturbed, nominal, (event,), definition="recovery-integral")
    f = disturbed.simulation.profile.copy()
    n = nominal.simulation.profile
    f["delta_speed_mps"] = f.v_mps - np.interp(f.time_s, n.time_s, n.v_mps)
    f["delta_altitude_m"] = f.h_m - np.interp(f.time_s, n.time_s, n.h_m)
    phase = f[f.phase == event.phase_name]
    before = f[f.time_s < phase.time_s.min()]
    entry = float(before.time_s.max()) if len(before) else 0.
    start, end = entry + event.start_time_s, entry + event.start_time_s + event.duration_s
    segment = f[f.time_s.between(start-30, end+event.recovery_window_s)].copy()
    segment["time_from_event_start_s"] = segment.time_s - start
    segment.to_csv(folder / "响应轨迹.csv", index=False)
    row = dict(policy=policy, wind_mps=wind, engine_delay_s=delay, response_step_s=step,
               **disturbed.result.objectives, **metrics, feasible=disturbed.result.feasible,
               fuel_kg=float(disturbed.result.diagnostics["fuel_used_kg"]),
               terminal_soc=float(f.soc.iloc[-1]), folder=str(folder),
               event_start_s=event.start_time_s, event_duration_s=event.duration_s)
    row["numerical_rate_tolerance"] = "20261015"
    row["constraint_excesses"] = {k: float(v) for k, v in disturbed.result.constraints.items() if v > 0}
    done.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8")
    return row


def control_gain_study():
    model = load_vsp_surrogate(ROOT / "气动代理模型数据" / "气动代理模型_211构型_分目标GP_20261013.json")
    from functools import partial
    from PTDL分块自适应优化 import PTDLBlockAdaptive
    problem = PTDLEnduranceJointProblem(
        aero_provider=model.corrected_aero,
        dispatch_policy_factory=RuleBasedHybridCoordinator,
        powertrain_factory=partial(HybridSystemPowertrain, config=HybridSystemAdapterConfig()),
        disturbances=project_response_stress_scenario(12),
        control_parameter_bounds=dict(speed_tracking_time_constant_s=(15., 60.),
                                      gamma_tracking_gain=(.02, .08),
                                      alpha_time_constant_s=(.8, 3.)),
        seed_decisions=[json.loads((OUT / "比较基准.json").read_text(encoding="utf8"))["decision"]],
        response_definition="recovery-integral",
    )
    run = PTDLBlockAdaptive(resolution=3, generations=5).run(problem)
    pd.DataFrame([dict(feasible=e.feasible, violation=e.violation, **e.result.objectives,
                       **e.result.diagnostics, **dict(zip(problem.decision_labels, e.decision)))
                  for e in run.evaluated_archive]).to_csv(OUT / "飞控参数联合搜索.csv", index=False)
    run.history.to_csv(OUT / "飞控参数联合搜索收敛.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "比较基准.json").exists():
        source = ROOT / "运行结果" / "七算法同预算_先导_20261015"
        frames = pd.concat([pd.read_csv(p) for p in source.glob("a*/*/全部评价.csv")], ignore_index=True)
        eligible = frames[frames.feasible & (frames.battery_discharge_energy_mj > 20)]
        baseline = eligible.sort_values("J_energy_mj").iloc[0]
        labels = PTDLEnduranceJointProblem().decision_labels
        (OUT / "比较基准.json").write_text(json.dumps(dict(
            decision=[float(baseline[k]) for k in labels],
            selection="lowest-energy feasible pilot candidate with battery discharge >20 MJ",
            source=str(source), metrics={k: float(baseline[k]) for k in
                                       ("J_energy_mj", "J_stability", "battery_discharge_energy_mj")},
        ), ensure_ascii=False, indent=2), encoding="utf8")
    jobs = [(policy, wind, delay, 1.) for policy in ("rule", "fuzzy", "mpc", "scheduled", "neural")
            for wind in (4., 8., 12.) for delay in (0., 3.)]
    jobs += [("rule", 8., 0., dt) for dt in (.5, 2.)]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for f in as_completed(futures):
            try:
                rows.append(f.result())
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            except Exception as e:
                errors.append(dict(job=futures[f], error=repr(e)))
                print(errors[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / "响应与能耗汇总.csv", index=False)
            (OUT / "进度.json").write_text(json.dumps(dict(completed=len(rows), total=len(jobs),
                                                           errors=errors), indent=2), encoding="utf8")
    control_gain_study()
    if errors:
        raise RuntimeError("Some response comparisons failed")
