"""Merge frozen equal-budget experiments with identical evaluator sources."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "运行结果" / name for name in (
    "耦合贡献独立十六种子H6_20261016", "去重贡献独立十六种子H6_20261016")]
OUT = ROOT / "运行结果" / "九算法独立十六种子_20261016"


def main():
    plans = [json.loads((p / "运行前冻结配置.json").read_text(encoding="utf8")) for p in SOURCES]
    for key in ("seeds", "resolution", "generations", "model_sha256",
                "response_definition", "source_manifest"):
        if plans[0][key] != plans[1][key]:
            raise RuntimeError(f"Shared experiment condition differs: {key}")
    OUT.mkdir(parents=True, exist_ok=True)
    algorithms = plans[0]["algorithms"]+plans[1]["algorithms"]
    merged_plan = dict(plans[0], algorithms=algorithms, source_runs=[str(p) for p in SOURCES])
    (OUT / "运行前冻结配置.json").write_text(json.dumps(merged_plan, ensure_ascii=False, indent=2), encoding="utf8")
    snapshot = OUT / "冻结源代码"
    snapshot.mkdir(exist_ok=True)
    for name, digest in plans[0]["source_manifest"].items():
        source = SOURCES[0] / "冻结源代码" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen source mismatch")
        shutil.copy2(source, snapshot / name)
    for source, plan in zip(SOURCES, plans):
        records = [json.loads(p.read_text(encoding="utf8")) for p in source.glob("a*/*/统计.json")]
        expected = {(a, s) for a in plan["algorithms"] for s in plan["seeds"]}
        if {(r["algorithm"], r["seed"]) for r in records} != expected:
            raise RuntimeError(f"Experiment incomplete: {source}")
        for record in records:
            original = Path(record["folder"])
            folder = OUT / f"a{algorithms.index(record['algorithm'])}" / str(record["seed"])
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("全部评价.csv", "逐代档案.csv", "收敛.csv", "进度.json"):
                shutil.copy2(original / name, folder / name)
            record["source_folder"] = str(original)
            record["folder"] = str(folder)
            record["raw_evaluation_sha256"] = hashlib.sha256((folder / "全部评价.csv").read_bytes()).hexdigest()
            (folder / "统计.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
    operator = Path(__file__).parent / "耦合搜索去重试验20261016.py"
    shutil.copy2(operator, OUT / operator.name)
    (OUT / "研究算子版本.json").write_text(json.dumps(dict(
        file=operator.name, sha256=hashlib.sha256(operator.read_bytes()).hexdigest(),
        evaluator_source_identical=True, normalization="Recomputed once across all nine methods",
        primary_method="PTDL-去重贡献学习",
        baseline_names=["NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO",
                        "PTDL-自适应分块", "PTDL-取消在线代理"],
        ablations=["PTDL-耦合贡献学习", "PTDL-耦合贡献学习保先验"],
    ), ensure_ascii=False, indent=2), encoding="utf8")
    subprocess.run([sys.executable, str(Path(__file__).parent / "汇总_独立任务算法20261015.py"),
                    "--output-dir", str(OUT), "--focus", "PTDL-去重贡献学习"], check=True)


if __name__ == "__main__":
    main()
