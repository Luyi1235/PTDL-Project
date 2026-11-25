"""Perform a reproducible stall-endurance repair screen for combined PTDL degradation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from run_ptdl_h6_combined_degradation_moead import CombinedDegradationRobustProblem, SURROGATE_PATH


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ptdl_moead_h6_combined_degradation_results_v2" / "组合退化MOEAD最终种群.csv"
OUTPUT = ROOT / "ptdl_combined_degradation_refinement_results"
WING_AREAS_M2 = (24.50, 24.55, 24.58, 24.60, 24.65, 24.70)
HYBRID_MASSES_KG = (60.0, 65.0, 70.0, 75.0, 80.0)


def record(problem: CombinedDegradationRobustProblem, decision: np.ndarray, wing_area: float, hybrid_mass: float) -> dict[str, float | bool]:
    evaluation = problem.evaluate(decision)
    result = evaluation.result
    row: dict[str, float | bool] = {
        "wing_area_m2": wing_area,
        "hybrid_system_mass_kg": hybrid_mass,
        "feasible": evaluation.feasible,
        "total_violation": evaluation.violation,
        "J_energy_mj": result.objectives["J_energy_mj"],
        "J_stability": result.objectives["J_stability"],
        "J_response_s": result.objectives["J_response_s"],
        "combined_achieved_total_duration_h": result.diagnostics.get("combined_achieved_total_duration_h", np.nan),
        "combined_margin_to_80h": result.diagnostics.get("combined_margin_to_80h", np.nan),
        "stall_speed_violation_mps": result.constraints.get("stall_speed", np.nan),
    }
    row.update({label: float(value) for label, value in zip(problem.decision_labels, decision)})
    return row


def export_figure(screen: pd.DataFrame) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    pivot = screen.pivot(index="hybrid_system_mass_kg", columns="wing_area_m2", values="total_violation")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
    image = axes[0].imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    axes[0].set_xticks(range(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns])
    axes[0].set_yticks(range(len(pivot.index)), [f"{value:.0f}" for value in pivot.index])
    axes[0].set(xlabel="翼面积 (m²)", ylabel="混动系统质量 (kg)", title="组合退化下的总违反度")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            axes[0].text(column, row, "可行" if value <= 1e-9 else f"{value:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes[0], label="总违反度")
    for mass, group in screen.groupby("hybrid_system_mass_kg", sort=True):
        axes[1].plot(group["wing_area_m2"], group["stall_speed_violation_mps"], marker="o", label=f"{mass:.0f} kg")
    axes[1].axhline(0.0, color="#1f2937", linewidth=1.0)
    axes[1].set(xlabel="翼面积 (m²)", ylabel="失速速度约束 g (m/s)", title="翼面积对失速约束的修复作用")
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="混动质量", frameon=False, fontsize=8)
    figure.savefig(OUTPUT / "图22_组合退化续航与失速约束定向修复筛选.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    problem = CombinedDegradationRobustProblem(VSPAeroSurrogate.from_json(SURROGATE_PATH))
    base = pd.read_csv(SOURCE).sort_values("total_violation").iloc[0]
    base_decision = np.asarray([float(base[label]) for label in problem.decision_labels], dtype=float)
    payload_index = problem.decision_labels.index("payload_mass_kg")
    wing_index = problem.decision_labels.index("wing_area_m2")
    hybrid_index = problem.decision_labels.index("hybrid_system_mass_kg")
    rows = []
    for hybrid_mass in HYBRID_MASSES_KG:
        for wing_area in WING_AREAS_M2:
            decision = base_decision.copy()
            decision[wing_index] = wing_area
            decision[payload_index] = 100.0
            decision[hybrid_index] = hybrid_mass
            rows.append(record(problem, decision, wing_area, hybrid_mass))
    screen = pd.DataFrame(rows)
    feasible = screen[screen["feasible"]].sort_values(["J_energy_mj", "J_stability", "J_response_s"])
    if feasible.empty:
        raise RuntimeError("No feasible repair point in the declared screen.")
    seed = feasible.iloc[[0]].copy()
    screen.to_csv(OUTPUT / "表10_组合退化续航与失速约束定向筛选.csv", index=False, encoding="utf-8-sig")
    seed.to_csv(OUTPUT / "组合退化可行修复种子.csv", index=False, encoding="utf-8-sig")
    export_figure(screen)
    (OUTPUT / "阶段审阅说明.md").write_text(
        "# 组合退化约束定向修复筛选\n\n"
        "以完整组合退化 MOEA/D 的最小违反度候选为基准，在固定任务功率和模糊调度参数条件下，"
        "扫描翼面积和混动系统质量，并将有效载荷保持在当前研究下限 100 kg。"
        "筛选结果作为 MOEA/D 的可行锚点，不改变既有硬约束；真实载荷下限和混动质量参数到位后应重新计算。\n",
        encoding="utf-8",
    )
    print(seed[["wing_area_m2", "hybrid_system_mass_kg", "payload_mass_kg", "combined_achieved_total_duration_h", "stall_speed_violation_mps", "J_energy_mj"]].to_string(index=False))


if __name__ == "__main__":
    main()
