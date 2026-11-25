"""Create an empty provenance-aware CSV template for future bench measurements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 标定数据处理 import write_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "运行结果" / "标定数据模板_待实测" / "component_measurements.csv",
    )
    args = parser.parse_args()
    write_template(args.output)
    print(f"已创建空模板（未含任何测量值）：{args.output.resolve()}")


if __name__ == "__main__":
    main()
