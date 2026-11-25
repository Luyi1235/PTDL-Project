"""Workbench entry point for current surrogate formats and joint optimization."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "核心模型"))

from 工作台气动代理加载 import load_workbench_surrogate
import 运行_四项统一集成优化 as optimizer


def main():
    optimizer.load_vsp_surrogate = load_workbench_surrogate
    optimizer.main()


if __name__ == "__main__":
    main()
