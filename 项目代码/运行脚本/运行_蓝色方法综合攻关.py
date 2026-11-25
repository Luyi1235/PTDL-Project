"""Phase tuning, observable health estimation and glide-guidance experiments."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"核心模型"))
from 协同自适应方法 import PhaseScheduledMPC, EstimatedHealthCoordinator, identify_delayed_response
from 智能能量管理与容错调度 import (
    MPCParameters, MPCEnergyManagementCoordinator, FuzzyAdaptiveCoordinator,
    NeuralImitationCoordinator, HealthState, NEURAL_PHASES,
)
from 规则混动协同控制 import RuleBasedHybridCoordinator
from PTDL六自由度参考模型 import PTDL6DoFParameters
from 运行_PTDL估计六自由度全任务闭环 import run


class TimedCoordinator:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.samples = {}

    def reset(self):
        self.coordinator.reset()
        self.samples = {}

    def dispatch(self, observation):
        started = time.perf_counter()
        result = self.coordinator.dispatch(observation)
        self.samples.setdefault(observation.phase_name, []).append(time.perf_counter()-started)
        return result

    def update_feedback(self, **values):
        if hasattr(self.coordinator, "update_feedback"):
            self.coordinator.update_feedback(**values)

    def statistics(self):
        return {phase: {"calls": len(times), "median_ms": float(np.median(times)*1000),
                        "p95_ms": float(np.quantile(times, .95)*1000),
                        "max_ms": float(max(times)*1000)}
                for phase, times in self.samples.items()}


def job_run(job):
    output, name, policy, config, scenario=job
    folder=Path(output)/name
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"运行日志.txt").open("w",encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        factories={
            "rule": RuleBasedHybridCoordinator, "fuzzy": FuzzyAdaptiveCoordinator,
            "mpc":lambda:MPCEnergyManagementCoordinator(MPCParameters(**config)),
            "scheduled":lambda:PhaseScheduledMPC(config),
            "neural":lambda:NeuralImitationCoordinator(config["model"]),
        }
        coordinator=factories[policy]()
        fault=None
        if scenario=="battery_health":
            coordinator=EstimatedHealthCoordinator(coordinator)
            fault=lambda phase,elapsed,t: HealthState(
                battery_available_energy_fraction=.6 if phase=="main_climb_to_leveloff" and 100<=elapsed<600 else 1.)
        overrides={}
        if scenario=="slow_approach":
            overrides={"approach_descent":{"speed_mps":30.0},"flare":{"speed_mps":29.0}}
        if scenario.startswith("glide"):
            target=float(scenario.split("_")[-1])
            overrides={phase:{"speed_mps":target} for phase in ("long_descent","approach_descent","flare")}
        if scenario=="recovery":
            overrides={"long_descent":{"power_off_duration_s":900,"engine_cap_kw":40,"speed_mps":32}}
        if scenario in {"adaptive_recovery","fixed_recovery"}:
            overrides={"long_descent":{"power_off_duration_s":1300,"engine_cap_kw":25,"speed_mps":32}}
        timed = TimedCoordinator(coordinator)
        started=time.perf_counter()
        try:
            summary=run(folder,integration_step_s=.05,control_period_s=2,
                        terminal_control_period_s=.5,sample_period_s=2,
                        mission_hours=4,mission_mode="full",pitch_kp=4,pitch_kd=4,
                        dispatch_policy=timed,phase_overrides=overrides,
                        energy_glide_guidance=scenario.startswith("glide"),
                        recovery_guidance=scenario=="adaptive_recovery",
                        plant_health_provider=fault,
                        powertrain_config=(
                            {"engine_time_constant_s":2.5,"engine_command_delay_s":1.0}
                            if scenario=="dynamic_delay" else None
                        ),
                        speed_command_event=(
                            (scenario.removeprefix("response_"),30.,60.,3.)
                            if scenario.startswith("response_") else None
                        ))
            phase_metrics=summary["phase_metrics_sampled"]
            account=summary["battery_energy_accounting"]
            record={
                "name":name,"policy":policy,"scenario":scenario,"config":config,
                "completed":summary["mission_completed"],
                "minimum_speed":summary["minimum_speed_mps_sampled"],
                "fuel_kg":summary["fuel_used_kg_estimated"],
                "minimum_soc":summary["minimum_soc"],
                "battery_loss_mj":account["internal_loss_mj"],
                "battery_efficiency":account["conversion_efficiency"],
                "pitch_max":summary["maximum_abs_pitch_deg_sampled"],
                "altitude_error_mean":float(np.mean([v["altitude_error_abs_mean_m"] for v in phase_metrics.values()])),
                "speed_error_max":max(v["speed_error_abs_max_mps"] for v in phase_metrics.values()),
                "wall_s":time.perf_counter()-started,
                "path":str(folder),"phase_metrics":phase_metrics,
                "dispatch_timing": timed.statistics(),
            }
            trace=pd.read_csv(folder/"PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
            reference=PTDL6DoFParameters()
            density=1.225*np.maximum(1-2.25577e-5*trace.altitude_m,1e-3)**4.2559
            required=1.1*np.sqrt(2*trace.mass_kg*9.80665/(
                density*reference.reference_wing_area_m2*1.42*np.maximum(np.cos(np.radians(trace.bank_deg)),.1)))
            record["local_speed_margin_min_mps"]=float((trace.speed_mps-required).min())
            record["touchdown_sink_mps"]=-summary["terminal_vertical_speed_up_mps"]
            record["maximum_alpha_deg"]=float(trace.alpha_deg.max())
            record["screen_pass"]=bool(record["completed"] and record["local_speed_margin_min_mps"]>=0
                                       and record["pitch_max"]<=25 and record["minimum_soc"]>=.199999
                                       and record["touchdown_sink_mps"]<=2 and record["maximum_alpha_deg"]<=12)
            if hasattr(coordinator,"estimator"):
                record["health_estimator"]=asdict(coordinator.estimator)
        except Exception as error:
            record={"name":name,"policy":policy,"scenario":scenario,"config":config,
                    "completed":False,"screen_pass":False,"error":repr(error),"path":str(folder)}
        (folder/"试验记录.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
        return record


def execute(jobs,workers,output):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(job_run,j) for j in jobs]):
            row=future.result()
            rows.append(row)
            print(row["name"],row["screen_pass"],row.get("error",""),flush=True)
            (output/"试验汇总.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    return rows


def identify(output):
    rows=[]
    rng=np.random.default_rng(20261013)
    for index,phase in enumerate(NEURAL_PHASES):
        dt=.5
        tau=1.5+index*.4
        delay=1+index%3
        u=np.repeat(rng.uniform(5,70,80),8)
        y=np.zeros(len(u))
        a=np.exp(-dt/tau)
        for k in range(delay,len(u)-1):
            y[k+1]=a*y[k]+(1-a)*u[k-delay]
        measured=y+rng.normal(0,.02,len(y))
        result=identify_delayed_response(u,measured,dt,6)
        rows.append({"phase":phase,"assumed_tau_s":tau,"assumed_delay_s":delay*dt,**result})
    pd.DataFrame(rows).to_csv(output/"六阶段动态辨识.csv",index=False)
    (output/"辨识数据口径.json").write_text(json.dumps({
        "source":"synthetic first-order delayed plant with known reference parameters",
        "purpose":"method verification with independent temporal validation",
        "not_done":"No real plant identification; coefficients are not silently substituted into component maps",
    },indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    identify(out)
    configs=[{"horizon_steps":h,"soc_terminal_weight_mj":w,"engine_slew_penalty_mj_per_kw":s}
             for h in (2,3,4) for w in (4,20,80) for s in (.002,.02)]
    jobs=[(str(out),f"tune_{i:02d}","mpc",cfg,"nominal") for i,cfg in enumerate(configs)]
    rows=execute(jobs,args.workers,out)
    valid=[r for r in rows if r["screen_pass"]]
    if not valid:
        raise RuntimeError("No feasible MPC tuning configuration")
    selected={}
    for phase in NEURAL_PHASES:
        available=[r for r in valid if phase in r["phase_metrics"]]
        winner=min(available,key=lambda r: (
            r["phase_metrics"][phase]["speed_error_abs_max_mps"]/10
            +r["phase_metrics"][phase]["altitude_error_abs_mean_m"]/500
            +r["fuel_kg"]/50+r["battery_loss_mj"]/20
            +max(0,.3-r["phase_metrics"][phase]["soc_end"])*10
            +r["wall_s"]/5000))
        selected[phase]=winner["config"]
    (out/"分阶段MPC配置.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    validation=[]
    for scenario in ("nominal","slow_approach","recovery","battery_health","glide_28","glide_30","glide_32","glide_34"):
        for policy in ("rule","fuzzy","mpc","scheduled"):
            validation.append((str(out),scenario+"_"+policy,policy,
                               selected if policy=="scheduled" else {},scenario))
    rows+=execute(validation,args.workers,out)
    (out/"试验汇总.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame([{k:v for k,v in r.items() if k not in ("config","phase_metrics","health_estimator")} for r in rows]).to_csv(out/"方法工况比较.csv",index=False)
    print("Finished",len(rows),flush=True)


if __name__=="__main__":
    main()
