"""Standalone smoke runner for the PTDL-Hybrid_System powertrain integration."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from MOEAD多目标联合优化 import run_ptdl_80h_joint_moead
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain


def main() -> None:
    parser = argparse.ArgumentParser(description="用现有动力部件运行 PTDL 80 小时优化链路烟雾测试")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "Hybrid_System动力接入烟雾测试",
    )
    parser.add_argument("--resolution", type=int, default=6, help="MOEA/D 权重分辨率 H")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--smoke", action="store_true", help="仅运行 H=1、1 代")
    args = parser.parse_args()

    factory = partial(
        HybridSystemPowertrain,
        config=HybridSystemAdapterConfig(),
    )
    run_ptdl_80h_joint_moead(
        args.output_dir,
        resolution=1 if args.smoke else args.resolution,
        generations=1 if args.smoke else args.generations,
        seed=args.seed,
        aero_model_label="当前 PTDL 气动模型",
        powertrain_factory=factory,
    )


if __name__ == "__main__":
    main()
