"""Merge fixed, non-overlapping completed runs from the split CPU schedule."""

import argparse
import json
from pathlib import Path

from 运行_六自由度多种子闭环比较 import ROOT, ALGORITHMS, summarize

BASE=ROOT/"运行结果"/"扩大预算_H4G7_六自由度80h_20261013"
EXTRA=ROOT/"运行结果"/"扩大预算_H4G7_混合算法补齐_20261013"
SEEDS=list(range(20261013,20261017))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--completed-only",action="store_true")
    args=parser.parse_args()
    records=[]
    sources=[]
    pending=[]
    for algorithm,prefix in ALGORITHMS.items():
        root=EXTRA if algorithm=="Hybrid NSGA-II/MOPSO" else BASE
        for seed in SEEDS:
            path=root/prefix/str(seed)/"单次统计.json"
            if not path.is_file() and args.completed_only:
                pending.append(dict(algorithm=algorithm,seed=seed,folder=str(path.parent)))
                continue
            row=json.loads(path.read_text(encoding="utf-8"))
            assert row["algorithm"]==algorithm and row["seed"]==seed
            assert row["population"]==15 and row["generations"]==7
            assert row["evaluation_calls"]==120 and row["sixdof_endurance_hours"]==80
            records.append(row)
            sources.append(str(path))
    assert len({(r["algorithm"],r["seed"]) for r in records})==len(records)
    assert len(records)+len(pending)==16
    output=ROOT/"运行结果"/"扩大预算_已完成12组归档_20261013" if args.completed_only else BASE
    output.mkdir(parents=True,exist_ok=True)
    summarize(output,records)
    scheduling=dict(selection="Three algorithms from original batch; Hybrid from dedicated four-worker batch; source chosen before results",
                    sources=sources,pending=pending,requested_runs=16,completed_runs=len(records),
                    stop_reason="User requested stopping computation and prioritizing report writing" if pending else None,
                    duplicate_work="Original queued/running Hybrid jobs are redundant and excluded; retained files are not additional independent seeds",
                    scope="Identical 131-point proxy, H4 G7, four seeds, 80h endurance-before-landing",
                    runtime="Concurrent wall time; scheduling changes contention, so runtime is not a controlled CPU-efficiency comparison")
    (output/"并行分批归档说明.json").write_text(json.dumps(scheduling,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"进度.json").write_text(json.dumps(dict(completed=len(records),total=len(records),errors=[],
                                                  originally_requested=16,pending=pending,
                                                  schedule="completed_subset_user_stopped" if pending else "split_parallel_completed"),
                                               ensure_ascii=False,indent=2),encoding="utf-8")
    if pending:
        for root in (BASE,EXTRA):
            progress_path=root/"进度.json"
            progress=(json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file()
                      else dict(completed=0,total=len(SEEDS),errors=[]))
            progress["status"]="stopped_by_user_for_report"
            progress["stop_reason"]=scheduling["stop_reason"]
            progress_path.write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Merged {len(records)} completed runs, {sum(r['evaluation_calls'] for r in records)} evaluation calls")


if __name__=="__main__":
    main()
