"""Score every method from all counted evaluations using a common archive policy."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from 运行_算法方法扩展20261015 import summarize, OBJECTIVE_LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "运行结果" / "耦合算法独立八种子H6_20261015"


def main():
    plan = json.loads((OUT / "运行前冻结配置.json").read_text(encoding="utf8"))
    records = [json.loads(path.read_text(encoding="utf8")) for path in OUT.glob("a*/*/统计.json")]
    expected = {(name, seed) for name in plan["algorithms"] for seed in plan["seeds"]}
    actual = {(r["algorithm"], r["seed"]) for r in records}
    if expected != actual:
        raise RuntimeError(f"Incomplete benchmark: {len(actual)}/{len(expected)}")
    code_versions = {record["code_sha256"] for record in records}
    if len(code_versions) != 1:
        raise RuntimeError("Benchmark mixed code versions")
    for name, digest in plan["source_manifest"].items():
        frozen_source = OUT / "冻结源代码" / name
        source = frozen_source if frozen_source.exists() else ROOT / "核心模型" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen source changed")
    for record in records:
        folder = Path(record["folder"])
        frame = pd.read_csv(folder / "全部评价.csv")
        if len(frame) != record["population"] * (record["generations"] + 1):
            raise RuntimeError("Counted evaluation budget mismatch")
        record["evaluated_feasible_ratio"] = float(frame.feasible.mean())
        record["archive_policy"] = "All counted feasible non-dominated evaluations, no archive cap"
        columns = [name for name in frame if name not in ("evaluation", "feasible", "violation")]
        rows = []
        for generation in range(record["generations"] + 1):
            evaluated = frame.iloc[:record["population"] * (generation + 1)]
            feasible = evaluated.loc[evaluated.feasible]
            values = feasible[list(OBJECTIVE_LABELS)].to_numpy(float)
            if not len(values):
                continue
            dominance = (np.all(values[:, None, :] <= values[None, :, :], axis=2)
                         & np.any(values[:, None, :] < values[None, :, :], axis=2))
            front = feasible.iloc[np.flatnonzero(~dominance.any(axis=0))].copy()
            front = front.drop_duplicates(list(OBJECTIVE_LABELS))
            front["generation"] = generation
            rows.append(front[["generation"] + columns])
        pd.concat(rows, ignore_index=True).to_csv(folder / "逐代档案.csv", index=False)
    summarize(OUT, records)
    runs = pd.read_csv(OUT / "多种子算法结果.csv")
    pairs = pd.read_csv(OUT / "配对比较与消融.csv")
    focus = runs[runs.algorithm == "PTDL-耦合代理辅助"]
    baselines = ["NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO", "PTDL-自适应分块"]
    comparisons = []
    for name in baselines:
        other = runs[runs.algorithm == name]
        pair = pairs[pairs.comparator == name].iloc[0]
        comparisons.append(dict(
            baseline=name, baseline_mean_hv=float(other.hypervolume.mean()),
            candidate_mean_hv=float(focus.hypervolume.mean()),
            relative_mean_gain=float(focus.hypervolume.mean() / other.hypervolume.mean() - 1),
            wins=int(pair.wins), losses=int(pair.losses), p_holm=float(pair.p_holm),
        ))
    decision = dict(
        independent_seeds=len(plan["seeds"]), truth_evaluations=int(runs.evaluations.sum()),
        comparisons=comparisons,
        all_baseline_means_exceeded=all(r["relative_mean_gain"] > 0 for r in comparisons),
        all_baselines_significantly_exceeded=all(
            r["relative_mean_gain"] > 0 and r["p_holm"] < .05 for r in comparisons),
        statistical_policy="Two-sided paired Wilcoxon; Holm across all comparisons including ablation",
        archive_policy="All counted feasible historical evaluations; uncapped common scoring",
        dynamics="Component-backed low-order 80 h task, same physical GP and recovery-integral objective",
        timing_note="Worker wall times observed in parallel batches; not a controlled serial speed comparison",
    )
    guided = runs[runs.algorithm == "PTDL-取消在线代理"].set_index("seed")
    decision["guided_configuration"] = dict(
        workbench_name="PTDL-耦合引导", benchmark_name="PTDL-取消在线代理",
        mean_hv=float(guided.hypervolume.mean()),
        selected_after_comparison=True,
        comparisons=[
            dict(baseline=name, relative_mean_gain=float(
                guided.hypervolume.mean() / runs.loc[runs.algorithm == name, "hypervolume"].mean() - 1))
            for name in baselines
        ],
        selection_note="Same tested search steps with online screening disabled; post-comparison configuration selection.",
    )
    (OUT / "强基线验收.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
