"""Three-seed repeatability study for robust fuzzy-joint H=6 MOEA/D."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_2_2_2_optimizer import MOEAD, OBJECTIVE_LABELS, pareto_indices
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from run_ptdl_h6_robust_fuzzy_joint_moead import RobustFuzzyJointProblem, SURROGATE_PATH, records


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_moead_h6_robust_fuzzy_repeatability_results"
SEEDS = (20260981, 20260982, 20260983)
H = 6
GENERATIONS = 12


def unique_pareto(frame: pd.DataFrame) -> pd.DataFrame:
    feasible = frame.loc[frame["feasible"].astype(bool)].copy()
    if feasible.empty:
        return feasible
    values = feasible[list(OBJECTIVE_LABELS)].to_numpy(dtype=float)
    keep = []
    for index in range(len(feasible)):
        dominated = np.any(np.all(values <= values[index], axis=1) & np.any(values < values[index], axis=1))
        if not dominated:
            keep.append(index)
    return feasible.iloc[keep].drop_duplicates(subset=list(OBJECTIVE_LABELS)).reset_index(drop=True)


def export_figures(history: pd.DataFrame, statistics: pd.DataFrame, pareto: pd.DataFrame) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    aggregate = history.groupby("generation")["feasible_ratio"].agg(["mean", "min", "max"]).reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].plot(aggregate["generation"], aggregate["mean"], color="#1769aa", marker="o", ms=3, label="三种子均值")
    axes[0].fill_between(aggregate["generation"], aggregate["min"], aggregate["max"], color="#93c5fd", alpha=0.55, label="种子范围")
    axes[0].set(title="H=6 联合优化可行率收敛带", xlabel="迭代代数", ylabel="可行解比例", ylim=(-0.03, 1.03))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].bar(statistics["seed"].astype(str), statistics["unique_pareto_count"], color="#0891b2", label="唯一 Pareto 候选数")
    axes[1].set(title="不同随机种子的可行 Pareto 候选数", xlabel="随机种子", ylabel="候选数")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(OUTPUT / "图16_H6联合优化三随机种子收敛与可行性统计.png", dpi=220)
    plt.close(figure)

    figure = plt.figure(figsize=(8.0, 6.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    if not pareto.empty:
        codes = pd.Categorical(pareto["seed"]).codes
        points = axis.scatter(pareto["J_energy_mj"], pareto["J_stability"], pareto["J_response_s"], c=codes, cmap="viridis", s=54, edgecolor="#1f2937", linewidth=0.35)
        figure.colorbar(points, ax=axis, shrink=0.72, pad=0.12, label="随机种子编码")
    else:
        axis.text2D(0.2, 0.5, "未获得可行 Pareto 候选", transform=axis.transAxes)
    axis.set(xlabel="J_energy (MJ)", ylabel="J_stability (-)", zlabel="J_response (s)", title="三随机种子合并稳健 Pareto 解集")
    figure.savefig(OUTPUT / "图17_H6联合优化三随机种子Pareto解集.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    populations, histories, pareto_sets, rows = [], [], [], []
    surrogate = VSPAeroSurrogate.from_json(SURROGATE_PATH)
    for seed in SEEDS:
        problem = RobustFuzzyJointProblem(surrogate)
        run = MOEAD(resolution=H, generations=GENERATIONS, neighborhood_size=5, seed=seed).run(problem)
        population = records(problem, run)
        population["seed"] = seed
        history = run.history.copy()
        history["seed"] = seed
        pareto = unique_pareto(population)
        populations.append(population)
        histories.append(history)
        pareto_sets.append(pareto)
        rows.append({
            "seed": seed,
            "final_feasible_count": int(population["feasible"].sum()),
            "feasible_ratio": float(population["feasible"].mean()),
            "unique_pareto_count": len(pareto),
            "minimum_energy_mj": float(pareto["J_energy_mj"].min()) if not pareto.empty else float("nan"),
            "minimum_stability": float(pareto["J_stability"].min()) if not pareto.empty else float("nan"),
            "minimum_response_s": float(pareto["J_response_s"].min()) if not pareto.empty else float("nan"),
        })
    merged = pd.concat(populations, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    pareto = pd.concat(pareto_sets, ignore_index=True)
    statistics = pd.DataFrame(rows)
    merged.to_csv(OUTPUT / "三随机种子最终种群.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUTPUT / "三随机种子收敛历史.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(OUTPUT / "表6_H6三随机种子可行Pareto统计.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(OUTPUT / "表7_H6三随机种子重复试验汇总.csv", index=False, encoding="utf-8-sig")
    export_figures(history, statistics, pareto)
    metadata = {
        "scope": "three-seed repeatability check for robust fuzzy-joint optimization",
        "surrogate": str(SURROGATE_PATH.relative_to(ROOT)),
        "hard_constraint": "T_endurance >= 80 h under nominal and CD0 +10% cases",
        "moead": {"resolution_H": H, "weights_per_seed": 28, "generations": GENERATIONS, "seeds": list(SEEDS)},
        "aggregate_feasible_ratio_mean": float(statistics["feasible_ratio"].mean()),
        "aggregate_feasible_ratio_std": float(statistics["feasible_ratio"].std(ddof=0)),
        "total_unique_seed_pareto_records": int(len(pareto)),
    }
    (OUTPUT / "运行说明.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    note = f"""# H=6 稳健模糊联合优化重复性试验

采用三个随机种子、每个种子 28 个 MOEA/D 权重子问题和 {GENERATIONS} 代迭代，且使用 23 点 VSPAERO 扩充代理模型。名义与 CD0 +10% 保守工况均须满足 80 h 约束。

三次试验的平均最终可行率为 {statistics['feasible_ratio'].mean():.1%}，标准差为 {statistics['feasible_ratio'].std(ddof=0):.1%}；各种子的 Pareto 数量、目标极值见表6和表7，收敛带与合并 Pareto 解集见图16和图17。该试验用于判断算法对随机初值的敏感性，尚不是最终收敛性证明；下一阶段可在已确认的稳定参数范围内提高迭代代数并增加高保真部件模型。
"""
    (OUTPUT / "阶段审阅说明.md").write_text(note, encoding="utf-8")
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
