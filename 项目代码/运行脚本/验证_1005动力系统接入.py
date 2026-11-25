"""Check packaged 1005 components against the supplied source and save provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))

from 动力模型.full_system.simulation import HybridSystemSimulation, summarize, write_results
from ptdl_hybrid_system_adapter import HybridSystemPowertrain


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"C:\Users\pc\Desktop\1005\python"))
    parser.add_argument("--output", type=Path, default=ROOT / "运行结果" / "动力系统更新_20261023")
    parser.add_argument("--full-native", action="store_true")
    parser.add_argument("--native-solver", choices=("mission_fast", "dynamic"), default="dynamic")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    packaged = ROOT / "核心模型" / "动力模型"
    manifest = []
    for module in ("battery_rint", "engine_map_genset", "generator_gearbox",
                   "motor_pmsm", "power_electronics_ems", "propeller", "full_system"):
        for origin in sorted((args.source / module).glob("*")):
            if origin.suffix not in {".py", ".csv"}:
                continue
            relative = origin.relative_to(args.source)
            destination = packaged / relative
            source_hash, package_hash = sha256(origin), sha256(destination)
            manifest.append({
                "file": relative.as_posix(),
                "supplied_sha256": source_hash,
                "packaged_sha256": package_hash,
                "identical": source_hash == package_hash,
                "adaptation": "" if source_hash == package_hash else "package-relative imports",
            })
    (args.output / "源文件与包内文件清单.json").write_text(
        json.dumps({"source": str(args.source), "files": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checks = []
    for solver, duration, output_step in (("mission_fast", 30.0, .5), ("dynamic", .5, .05)):
        source_program = (
            "import json\n"
            "from full_system.simulation import HybridSystemSimulation\n"
            f"rows=HybridSystemSimulation(solver={solver!r}).run("
            f"duration_s={duration},output_step_s={output_step})\n"
            "print(json.dumps(rows))\n"
        )
        original = json.loads(subprocess.check_output(
            [sys.executable, "-B", "-X", "utf8", "-c", source_program],
            cwd=args.source,
            env=dict(os.environ, PYTHONPATH=str(args.source)),
            text=True, encoding="utf-8", timeout=120,
        ))
        actual = HybridSystemSimulation(solver=solver).run(duration_s=duration, output_step_s=output_step)
        assert len(original) == len(actual)
        largest = 0.0
        for expected_row, row in zip(original, actual):
            assert set(expected_row) == set(row)
            for key, value in expected_row.items():
                if isinstance(value, (int, float)):
                    largest = max(largest, abs(value - row[key]))
                else:
                    assert value == row[key], key
        assert largest < 1e-9, (solver, largest)
        checks.append({"solver": solver, "duration_s": duration,
                       "sample_count": len(actual), "max_absolute_difference": largest,
                       "passed": True})
    report = {"powertrain": HybridSystemPowertrain().metadata, "source_parity": checks}
    (args.output / "接入核对结果.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.full_native:
        simulation = HybridSystemSimulation(solver=args.native_solver)
        rows = simulation.run(output_step_s=.05, progress_callback=lambda stage, current, total:
                              print(f"{stage} {current:.1f}/{total:.1f} s", flush=True))
        _, _, native_summary = write_results(rows, args.output / "新版原生4811秒任务", 3)
        report["native_mission"] = native_summary
        reference_csv = args.source / "results/mode3/hybrid_system_mode3_outputs.csv"
        if args.native_solver == "dynamic" and reference_csv.is_file():
            largest = 0.0
            count = 0
            with reference_csv.open(encoding="utf-8-sig", newline="") as stream:
                for count, reference_row in enumerate(csv.DictReader(stream), 1):
                    if count > len(rows):
                        raise AssertionError("Reference contains more rows than the replay")
                    row = rows[count - 1]
                    for key, value in reference_row.items():
                        if isinstance(row[key], (int, float)):
                            largest = max(largest, abs(float(value) - row[key]))
                        elif value != str(row[key]):
                            raise AssertionError((key, value, row[key]))
            assert count == len(rows)
            report["supplied_full_trajectory_comparison"] = {
                "sample_count": count, "maximum_absolute_difference": largest,
                "matching_within_1e_8": largest < 1e-8,
            }
    (args.output / "接入核对结果.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
