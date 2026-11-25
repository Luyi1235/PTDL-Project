"""Update post-fix cases and combine method evidence without overwriting runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from 运行_蓝色方法综合攻关 import execute

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "运行结果"
STUDY = RESULTS / "蓝色攻关_阶段响应统一验收_20261013"
BENCH = RESULTS / "蓝色攻关_六自由度80h算法比较_20261013"
OBJECTIVES = ["J_energy_mj", "J_stability", "J_response_s"]


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, list):
            return [clean(v) for v in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def corrected_cases(output):
    selected = json.loads((RESULTS / "蓝色攻关_控制与轨迹_20261013/分阶段MPC配置.json").read_text(encoding="utf-8"))
    neural = RESULTS / "蓝色攻关_神经增量_20261013/增量神经策略.json"
    jobs = []
    for policy in ("rule", "fuzzy", "mpc", "scheduled", "neural"):
        config = selected if policy == "scheduled" else {"model": str(neural)} if policy == "neural" else {}
        jobs.append((str(output), "dynamic_delay_" + policy, policy, config, "dynamic_delay"))
    fresh = execute(jobs, 5, output)
    previous = json.loads((STUDY / "试验汇总.json").read_text(encoding="utf-8"))
    rows = [r for r in previous if r["scenario"] != "dynamic_delay"] + fresh
    write_json(output / "统一70例汇总.json", rows)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("config", "phase_metrics", "health_estimator")} for r in rows]).to_csv(
        output / "统一70例比较.csv", index=False)
    health = []
    for row in rows:
        if row["scenario"] != "battery_health":
            continue
        trace = pd.read_csv(Path(row["path"]) / "PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
        fault = trace[(trace.time_s >= 200) & (trace.time_s < 590)]
        health.append({
            "policy": row["policy"], "assumed_available_fraction": .6,
            "estimated_fraction_mean": float(fault.estimated_battery_available_fraction.mean()),
            "mae": float((fault.estimated_battery_available_fraction - .6).abs().mean()),
            "observed_samples": len(fault),
            "engine_near_rating_observations": row["health_estimator"]["engine_observations"],
        })
    pd.DataFrame(health).to_csv(output / "在线容量估计.csv", index=False)


def decision_archive(output):
    progress = json.loads((BENCH / "进度.json").read_text(encoding="utf-8"))
    if progress["completed"] != progress["total"] or progress["errors"]:
        raise RuntimeError("Full benchmark must finish before final decision analysis.")
    frames = []
    for path in BENCH.glob("*/*/*_feasible_pareto.csv"):
        if "_generation_" in path.name or "_final_population_" in path.name:
            continue
        frame = pd.read_csv(path)
        frame["source_archive"] = str(path)
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    design_columns = list(pooled.columns[pooled.columns.get_loc("wing_area_m2"):pooled.columns.get_loc("J_energy_mj")])
    pooled.to_csv(output / "历代档案来源记录.csv", index=False)
    pooled = pooled.drop_duplicates(subset=design_columns).reset_index(drop=True)
    numeric = pooled[OBJECTIVES].to_numpy(float)
    finite = np.isfinite(numeric).all(axis=1)
    pooled = pooled[finite].copy().reset_index(drop=True)
    pooled["candidate_id"] = np.arange(len(pooled))
    energy_rows, phase_rows = [], []
    for row in pooled.itertuples():
        summary = json.loads(Path(row.diagnostic_sixdof_full_mission_summary_path).read_text(encoding="utf-8"))
        ledger = summary["battery_energy_accounting"]
        energy_rows.append({
            "sixdof_conversion_efficiency": ledger["conversion_efficiency"],
            "sixdof_battery_loss_mj": ledger["internal_loss_mj"],
            "sixdof_net_chemical_withdrawal_mj": ledger["net_chemical_withdrawal_mj"],
        })
        phase_rows += [
            {"candidate_id": row.candidate_id, "phase": phase, **values}
            for phase, values in summary["phase_battery_energy_accounting"].items()
        ]
    pooled = pd.concat([pooled, pd.DataFrame(energy_rows)], axis=1)
    normal = json.loads((BENCH / "归一化基准.json").read_text(encoding="utf-8"))
    ideal, nadir = np.array(normal["ideal"]), np.array(normal["nadir"])
    # Fixed reserve requirements precede preference scoring.
    mask = (
        (pooled["total_violation"] <= 1e-8)
        & (pooled["diagnostic_minimum_soc"] >= .30)
        & (pooled["diagnostic_sixdof_full_mission_minimum_soc"] >= .30)
        & (pooled["diagnostic_sixdof_full_mission_fuel_margin_kg"] >= 10)
    )
    eligible = pooled[mask].copy()
    weights = [(i/10, j/10, (10-i-j)/10) for i in range(11) for j in range(11-i)]
    choices = []
    for efficiency_min in (None, .990, .995, .998):
        pool = eligible
        if efficiency_min is not None:
            pool = pool[pool["sixdof_conversion_efficiency"].notna()
                        & (pool["sixdof_conversion_efficiency"] >= efficiency_min)]
        for w in weights:
            record = dict(zip(["weight_energy", "weight_stability", "weight_response"], w))
            record.update(efficiency_min=efficiency_min, eligible_candidates=len(pool))
            if len(pool):
                score = ((pool[OBJECTIVES].to_numpy(float)-ideal)/np.maximum(nadir-ideal, 1e-9)) @ np.array(w)
                winner = pool.iloc[int(np.argmin(score))]
                record.update(candidate_id=int(winner.candidate_id), score=float(np.min(score)),
                              fuel_kg=float(winner.diagnostic_fuel_used_kg),
                              battery_efficiency=float(winner.sixdof_conversion_efficiency),
                              battery_loss_mj=float(winner.sixdof_battery_loss_mj),
                              net_chemical_withdrawal_mj=float(winner.sixdof_net_chemical_withdrawal_mj))
            choices.append(record)
    pooled.to_csv(output / "六自由度约束可行方案池.csv", index=False)
    eligible.to_csv(output / "同储备候选池.csv", index=False)
    pd.DataFrame(choices).to_csv(output / "正式三目标权重与效率门槛扫描.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(output / "候选六阶段电池能量分项.csv", index=False)
    write_json(output / "决策条件.json", {
        "source": str(BENCH), "pooled_candidates": len(pooled), "eligible_candidates": len(eligible),
        "minimum_soc_both_models": .30, "sixdof_fuel_reserve_kg": 10,
        "normalization": normal, "weight_count": len(weights),
        "efficiency_thresholds": [None, .990, .995, .998],
        "efficiency_basis": "Six-DoF control-interval terminal/chemical energy ledger",
        "efficiency_sensitivity_design": "Exploratory thresholds on this batch; not pre-registered acceptance limits.",
        "formal_objectives": OBJECTIVES,
        "scope": "Battery efficiency is a diagnostic preference filter; formal PPT objectives remain unchanged.",
        "empty_pool_policy": "No selection; thresholds are never weakened automatically.",
    })
    runs = pd.read_csv(BENCH / "多种子运行比较.csv")
    statistics = runs.groupby("algorithm").agg(
        seeds=("seed", "count"), feasibility_mean=("final_feasible_ratio", "mean"),
        feasibility_std=("final_feasible_ratio", "std"), hv_mean=("final_hypervolume", "mean"),
        hv_std=("final_hypervolume", "std"), runtime_mean_s=("runtime_s", "mean"),
        evaluations=("evaluation_calls", "sum"),
    ).reset_index()
    statistics.to_csv(output / "六自由度算法配置依据.csv", index=False)
    controls = json.loads((RESULTS / "蓝色攻关_恢复与配置库_20261013/控制算法配置库.json").read_text(encoding="utf-8"))
    write_json(output / "协同设计方法配置库.json", {
        "controls": controls, "optimizers": json.loads(statistics.to_json(orient="records")),
        "optimizer_conditions": {"population": 6, "generations": 3, "seeds_per_algorithm": 8,
                                 "mission": "80 hours before descent and landing", "controller": "rule",
                                 "aerodynamic_model": "131 point ARD RBF GP"},
        "selection_policy": "Use feasibility, archive quality and cost under the recorded conditions; no universal ranking.",
        "decision_rules": json.loads((output / "决策条件.json").read_text(encoding="utf-8")),
    })
    print(statistics.to_string(index=False), flush=True)
    print("Decision pool", len(pooled), "reserve-qualified", len(eligible), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["cases", "decisions"], required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (corrected_cases if args.mode == "cases" else decision_archive)(args.output_dir)


if __name__ == "__main__":
    main()
