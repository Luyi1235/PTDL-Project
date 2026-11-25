"""Shared command-line options for the two official mission entry points."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

from .simulation import run_mission

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _format_time(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ConsoleProgress:
    """Flushed lines work in PowerShell, IDE consoles and redirected logs."""

    def __init__(self):
        self.started = None

    def __call__(self, stage, current, total):
        if self.started is None:
            self.started = perf_counter()
        elapsed = perf_counter()-self.started
        if stage == "simulate":
            fraction = min(current/total, 1.0) if total > 0 else 1.0
            if fraction >= 1.0:
                remaining = "00:00:00"
            elif fraction > 0.0:
                remaining = _format_time(elapsed*(1.0-fraction)/fraction)
            else:
                remaining = "估算中"
            print(f"仿真进度 {100*fraction:6.2f}% | 仿真时间 {current:.3f}/{total:g} s"
                  f" | 已用 {_format_time(elapsed)} | 预计剩余 {remaining}", flush=True)
        elif stage == "save":
            print("仿真计算完成，正在保存 CSV 和 JSON...", flush=True)
        elif stage == "plot":
            print("结果已保存，正在生成图片...", flush=True)
        elif stage == "done":
            print(f"运行完成，总耗时 {_format_time(elapsed)}。", flush=True)


def main(mode):
    # Use UTF-8 for Chinese status lines in Windows terminals and piped logs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=f"Hybrid-system EMS mode {mode}")
    parser.add_argument("--solver", choices=("mission_fast", "dynamic"), default="dynamic")
    parser.add_argument("--step", type=float, default=0.05, help="CSV interval (s), independent of integration step")
    parser.add_argument("--duration", type=float, default=None, help="Stop time, 0..4811 s")
    parser.add_argument("--initial-soc", type=float, default=0.90)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress display")
    parser.add_argument("--cold-start", action="store_true", help="Dynamic motor starts at rest; differs from default MATLAB mission")
    parser.add_argument("--output-dir", type=Path, help="Optional result directory")
    args = parser.parse_args()
    if args.cold_start and args.solver != "dynamic":
        parser.error("--cold-start requires --solver dynamic")
    target = args.output_dir
    if target is None:
        target = PROJECT_ROOT / "results"
        if args.solver == "dynamic":
            target /= "dynamic"
        target /= f"mode{mode}"
    progress = None if args.no_progress else ConsoleProgress()
    if progress is not None:
        print(f"开始运行模式 {mode}，求解器 {args.solver}。", flush=True)
    result = run_mission(mode, target, args.step, args.duration, not args.no_plots,
                         args.solver, not args.cold_start,
                         progress_callback=progress, initial_soc=args.initial_soc)
    print(f"Solver: {args.solver}")
    print(f"CSV: {result['csv_path']}")
    print(f"Summary: {result['summary_path']}")
    print(f"Fuel used: {result['summary']['fuel_used_kg']:.12f} kg")
    print(f"Final SOC: {result['summary']['final_soc']:.12f}")
