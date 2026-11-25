"""Re-plot saved hybrid-system results without re-running the simulation.

Usage:
    python -m full_system.replot_mode 3
    python -m full_system.replot_mode --csv results/dynamic/mode3/hybrid_system_mode3_outputs.csv
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 动力模型.full_system.plotting import plot_results

TEXT_COLUMNS = {"Phase_name", "Solver"}


def load_rows(csv_file):
    with csv_file.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"CSV has no data rows: {csv_file}")
    for row in rows:
        for name, value in row.items():
            if name not in TEXT_COLUMNS and value != "":
                row[name] = float(value)
    return rows


def locate_csv(results_directory, solver, mode):
    """dynamic runs live in results/<solver>/modeN, fast runs in results/modeN."""
    candidates = (
        results_directory/solver/f"mode{mode}"/f"hybrid_system_mode{mode}_outputs.csv",
        results_directory/f"mode{mode}"/f"hybrid_system_mode{mode}_outputs.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("No results CSV found; tried:\n  " + "\n  ".join(map(str, candidates)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", type=int, choices=(3,),
                        help="EMS mode whose saved results should be re-plotted")
    parser.add_argument("--csv", type=Path, help="explicit outputs CSV; overrides mode lookup")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT/"results")
    parser.add_argument("--solver", choices=("mission_fast", "dynamic"), default="dynamic")
    args = parser.parse_args()
    if args.csv is None and args.mode is None:
        parser.error("give mode 3 or --csv")

    csv_file = args.csv or locate_csv(args.results_dir, args.solver, args.mode)
    if not csv_file.is_file():
        raise SystemExit(f"No such file: {csv_file}")
    rows = load_rows(csv_file)
    # Fall back to the filename when only --csv was given.
    mode = args.mode
    if mode is None:
        digits = "".join(c for c in csv_file.stem if c.isdigit())
        mode = int(digits[0]) if digits else 0

    for path in plot_results(rows=rows, output_directory=csv_file.parent, mode=mode):
        print(path)


if __name__ == "__main__":
    main()
