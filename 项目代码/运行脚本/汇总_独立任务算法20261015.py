"""Uncapped counted-evaluation archives and paired comparisons for any focus."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from 运行_算法方法扩展20261015 import summarize, OBJECTIVE_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus", required=True)
    parser.add_argument("--report-stem", default="主方法")
    args = parser.parse_args()
    out = args.output_dir
    plan = json.loads((out / "运行前冻结配置.json").read_text(encoding="utf8"))
    records = [json.loads(p.read_text(encoding="utf8")) for p in out.glob("a*/*/统计.json")]
    expected = {(a, s) for a in plan["algorithms"] for s in plan["seeds"]}
    if {(r["algorithm"], r["seed"]) for r in records} != expected:
        raise RuntimeError("Incomplete benchmark")
    expected_digest = None
    if (out / "冻结源代码").exists():
        sources = sorted(out / "冻结源代码" / name for name in plan["source_manifest"])
        for source in sources:
            if hashlib.sha256(source.read_bytes()).hexdigest() != plan["source_manifest"][source.name]:
                raise RuntimeError("Frozen source file changed")
        # Match the worker's platform-aware Path ordering, including Windows case folding.
        expected_digest = hashlib.sha256(b"".join(
            source.name.encode("utf8")+source.read_bytes() for source in sources)).hexdigest()
    versions = {r["code_sha256"] for r in records}
    if len(versions) != 1 or (expected_digest and versions != {expected_digest}):
        raise RuntimeError("Mixed or changed code versions")
    for record in records:
        folder = Path(record["folder"])
        frame = pd.read_csv(folder / "全部评价.csv")
        if len(frame) != record["population"]*(record["generations"]+1):
            raise RuntimeError("Evaluation budget mismatch")
        record["evaluated_feasible_ratio"] = float(frame.feasible.mean())
        rows = []
        columns = [k for k in frame if k not in ("evaluation", "feasible", "violation")]
        for generation in range(record["generations"]+1):
            feasible = frame.iloc[:record["population"]*(generation+1)].query("feasible")
            values = feasible[list(OBJECTIVE_LABELS)].to_numpy(float)
            if not len(values):
                continue
            dominates = (np.all(values[:, None, :] <= values[None, :, :], axis=2)
                         & np.any(values[:, None, :] < values[None, :, :], axis=2))
            front = feasible.iloc[np.flatnonzero(~dominates.any(0))].copy()
            front = front.drop_duplicates(list(OBJECTIVE_LABELS))
            front["generation"] = generation
            rows.append(front[["generation"]+columns])
        if rows:
            pd.concat(rows).to_csv(folder / "逐代档案.csv", index=False)
    summarize(out, records)
    runs = pd.read_csv(out / "多种子算法结果.csv")
    focus = runs[runs.algorithm == args.focus].set_index("seed")
    tests = []
    rng = np.random.default_rng(2026101549)
    for name in plan["algorithms"]:
        if name == args.focus:
            continue
        other = runs[runs.algorithm == name].set_index("seed")
        delta = focus.loc[plan["seeds"], "hypervolume"].to_numpy() - other.loc[
            plan["seeds"], "hypervolume"].to_numpy()
        bootstrap = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
        tests.append(dict(
            comparator=name, paired_seeds=len(delta), wins=int((delta > 1e-12).sum()),
            mean_hv=float(other.hypervolume.mean()), focus_hv=float(focus.hypervolume.mean()),
            relative_gain=float(focus.hypervolume.mean()/other.hypervolume.mean()-1),
            delta_mean=float(delta.mean()),
            delta_ci_low=float(np.quantile(bootstrap, .025)),
            delta_ci_high=float(np.quantile(bootstrap, .975)),
            p_raw=1. if np.all(abs(delta) < 1e-12) else float(wilcoxon(delta).pvalue)))
    adjusted = 0.
    for i, row in enumerate(sorted(tests, key=lambda r: r["p_raw"])):
        adjusted = max(adjusted, min(1., (len(tests)-i)*row["p_raw"]))
        row["p_holm"] = adjusted
    pd.DataFrame(tests).to_csv(out / f"{args.report_stem}配对比较.csv", index=False)
    decision = dict(
        focus=args.focus, runs=len(records), seeds=len(plan["seeds"]),
        true_evaluations=int(runs.evaluations.sum()), comparisons=tests,
        all_means_exceeded=all(r["relative_gain"] > 0 for r in tests),
        all_significantly_exceeded=all(r["relative_gain"] > 0 and r["p_holm"] < .05 for r in tests),
        archive_policy="All counted feasible evaluations; no archive cap in scoring",
        statistics="Paired two-sided Wilcoxon with Holm; paired bootstrap mean-difference interval",
    )
    specification = out / "研究算子版本.json"
    if specification.exists():
        baseline_names = [name for name in json.loads(specification.read_text(encoding="utf8"))["baseline_names"]
                          if name != args.focus]
        baselines = [r for r in tests if r["comparator"] in baseline_names]
        if len(baselines) != len(baseline_names):
            raise RuntimeError("Missing declared strong baseline")
        decision.update(
            strong_baselines=baseline_names,
            all_baseline_means_exceeded=all(r["relative_gain"] > 0 for r in baselines),
            all_baselines_significantly_exceeded=all(
                r["relative_gain"] > 0 and r["p_holm"] < .05 for r in baselines))
    (out / f"{args.report_stem}验收.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
