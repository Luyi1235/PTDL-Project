"""Convert a paired multi-seed benchmark into a reusable algorithm catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 优化算法配置库 import summarize_equal_budget_benchmark


DEFAULT_BENCHMARK = ROOT / "运行结果" / "H10四算法等预算_8代_3种子_131GP_20261012"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    catalog = summarize_equal_budget_benchmark(
        benchmark_dir / "算法多随机种子统计.csv",
        benchmark_dir / "多随机种子算法对比说明.json",
    )
    output = args.output or benchmark_dir / "PTDL优化算法配置库.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"算法配置库已生成：{output}")
    for item in catalog["algorithm_profiles"]:
        print(
            f"{item['algorithm']}: 可行率排名 {item['feasibility_rank']}, "
            f"超体积排名 {item['hypervolume_rank']}, "
            f"耗时排名 {item['runtime_rank']}"
        )


if __name__ == "__main__":
    main()
