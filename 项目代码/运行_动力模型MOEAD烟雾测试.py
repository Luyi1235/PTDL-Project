"""Run a small PTDL MOEA/D integration test with Hybrid_System components."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from MOEAD多目标联合优化 import run_ptdl_80h_joint_moead
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain


def main() -> None:
    parser = argparse.ArgumentParser(description="使用发动机、电池和电机部件模型运行 PTDL 飞推控优化。")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "动力模型MOEAD")
    parser.add_argument("--resolution", type=int, default=6, help="MOEA/D 单纯形分辨率 H")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--engine-map", type=Path, default=None, help="可选发动机图 CSV 路径")
    parser.add_argument("--smoke", action="store_true", help="运行 H=1、1 代的快速集成检查")
    args = parser.parse_args()

    factory = partial(
        HybridSystemPowertrain,
        config=HybridSystemAdapterConfig(
            engine_map_path=args.engine_map,
        ),
    )
    run_ptdl_80h_joint_moead(
        args.output_dir,
        resolution=1 if args.smoke else args.resolution,
        generations=1 if args.smoke else args.generations,
        seed=args.seed,
        aero_model_label="PTDL low-order/VSPAERO provider",
        powertrain_factory=factory,
    )


if __name__ == "__main__":
    main()
