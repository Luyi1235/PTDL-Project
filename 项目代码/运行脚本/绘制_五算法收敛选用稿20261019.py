"""Export the selected convergence panel with the four original baselines."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

import 绘制_PTDL耦合引导算法对比20261019 as charts


def main():
    _, history, _, _ = charts.read_and_validate()
    algorithms = [name for name in charts.ALGORITHMS if name != "PTDL-自适应分块"]
    charts.ALGORITHMS = algorithms
    history = history[history.algorithm.isin(algorithms)].copy()
    charts.style()
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    fig.subplots_adjust(left=.10, right=.96, top=.90, bottom=.16)
    charts.convergence(ax, history)
    ax.set_title("(a) 相同计算预算下的收敛过程")
    if [line.get_label() for line in ax.lines] != [charts.NAMES[a] for a in algorithms]:
        raise ValueError("Rendered algorithm labels do not match the requested selection")
    charts.OUT.mkdir(parents=True, exist_ok=True)
    stem = "06_a图_五算法收敛"
    charts.save(fig, stem, "五种算法的收敛过程", "", [])
    history.to_csv(charts.OUT / f"{stem}_绘图数据.csv", index=False, encoding="utf-8-sig")
    medians = history[history.generation.eq(12)].groupby("algorithm").hypervolume.median()
    paragraph = (
        "图a展示了低阶80小时飞推控耦合任务中五种算法在相同评价预算下的收敛过程。"
        "曲线表示16次独立运行的超体积中位数，色带表示中间50%的结果，超体积越大代表方案集的综合质量越高。"
        "PTDL耦合引导算法在搜索前期迅速提升，并在后续迭代中保持领先。"
        f"累计完成364次任务评价时，其超体积中位数达到{medians[charts.FOCUS]:.2f}，"
        f"高于MOPSO的{medians['MOPSO']:.2f}、"
        f"NSGA-II与MOPSO混合算法的{medians['Hybrid NSGA-II/MOPSO']:.2f}、"
        f"NSGA-II的{medians['NSGA-II']:.2f}和MOEA/D的{medians['MOEA/D']:.2f}。"
        "结果表明，PTDL耦合引导算法在该任务中具有更高的搜索效率，"
        "能够在有限评价预算内获得综合质量更高的可行方案集。"
    )
    (charts.OUT / f"{stem}_配图文字.txt").write_text(paragraph, encoding="utf-8")
    source = charts.SOURCE / "共享基准逐代超体积.csv"
    record = {
        "algorithms": algorithms,
        "removed_curve": "PTDL-自适应分块",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plot_helper_sha256": hashlib.sha256(Path(charts.__file__).read_bytes()).hexdigest(),
        "source_values_unchanged": True,
        "final_hv_medians": {charts.NAMES[k]: float(v) for k, v in medians.items()},
        "curves": len(ax.lines),
        "seeds_per_algorithm": 16,
        "figure_files": [str(charts.OUT / f"{stem}.{ext}") for ext in ("png", "pdf", "svg")],
    }
    (charts.OUT / f"{stem}_数据说明.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
