"""Run the current SOC 0.40-0.90, descent-off three-layer EMS mission."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 动力模型.full_system.cli import main


if __name__ == "__main__":
    main(3)
