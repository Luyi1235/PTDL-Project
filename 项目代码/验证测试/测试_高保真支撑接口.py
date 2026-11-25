"""Regression tests for calibration data, HIL protocol, and JSBSim-independent helpers."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 标定数据处理 import REQUIRED_COLUMNS, split_run_ids, validate_calibration_csv, write_template
from OpenVSP_VSPAERO接口 import load_vspaero_polar
from MOEAD多目标联合优化 import PTDLEnduranceJointProblem
from PTDL联合优化高保真接口 import LiveOpenVSPAeroProvider
from 八十小时续航任务模型 import calibrate_endurance_task_baseline
from 机体参数化设计与气动阻力 import AirframeDesign, aero_from_design
from ptdl_hil_protocol import ActuatorLimits, CommandWatchdog, ControlCommand, validate_command


class CalibrationDataTests(unittest.TestCase):
    def test_template_contains_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "template.csv"
            write_template(path)
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows, [list(REQUIRED_COLUMNS)])

    def test_validates_measured_schema_and_monotonic_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "measurements.csv"
            rows = [
                ["run_a", "0.0", "engine", "power_kw", "20.0", "kW", "bench", "sea_level", "valid"],
                ["run_a", "0.1", "engine", "power_kw", "21.0", "kW", "bench", "sea_level", "valid"],
            ]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(REQUIRED_COLUMNS)
                writer.writerows(rows)
            self.assertEqual(validate_calibration_csv(path), [])

    def test_rejects_non_monotonic_run_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "measurements.csv"
            rows = [
                ["run_a", "0.1", "engine", "power_kw", "20", "kW", "bench", "sea_level", "valid"],
                ["run_a", "0.0", "engine", "power_kw", "21", "kW", "bench", "sea_level", "valid"],
            ]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(REQUIRED_COLUMNS)
                writer.writerows(rows)
            self.assertTrue(any("严格递增" in issue for issue in validate_calibration_csv(path)))

    def test_splits_whole_runs_reproducibly(self) -> None:
        train, validation = split_run_ids(["a", "b", "c", "d"], seed=7)
        train_again, validation_again = split_run_ids(["a", "b", "c", "d"], seed=7)
        self.assertEqual((train, validation), (train_again, validation_again))
        self.assertFalse(set(train).intersection(validation))
        self.assertEqual(set(train).union(validation), {"a", "b", "c", "d"})


class HilProtocolTests(unittest.TestCase):
    def test_command_bounds_and_sequence(self) -> None:
        command = ControlCommand(3, 1.0, 0.5, 20.0, 0.0, 0.0, 0.0)
        self.assertEqual(validate_command(command, expected_sequence=3), [])
        errors = validate_command(
            ControlCommand(4, 1.0, 1.1, 200.0, 0.0, 0.0, 0.0),
            ActuatorLimits(),
            expected_sequence=3,
        )
        self.assertGreaterEqual(len(errors), 3)

    def test_watchdog_returns_neutral_command_after_timeout(self) -> None:
        now = [10.0]
        watchdog = CommandWatchdog(timeout_s=0.1, clock=lambda: now[0])
        self.assertIsNotNone(watchdog.failsafe_command(0, 0.0))
        watchdog.accept()
        self.assertIsNone(watchdog.failsafe_command(1, 0.1))
        now[0] += 0.2
        failsafe = watchdog.failsafe_command(2, 0.3)
        self.assertEqual((failsafe.throttle, failsafe.motor_power_kw), (0.0, 0.0))


class VspaeroPolarTests(unittest.TestCase):
    def test_reads_current_vspaero_cltot_header(self) -> None:
        header = (
            "Beta Mach AoA Re/1e6 CLtot CDi E CLiw CDiw Ew\n"
            "0 0.1 -4 2  -0.4 0.004 0.8 -0.4 0.004 0.7\n"
            "0 0.1 -2 2  -0.2 0.001 0.8 -0.2 0.001 0.7\n"
            "0 0.1  0 2   0.0 0.000 0.8  0.0 0.000 0.7\n"
            "0 0.1  2 2   0.2 0.001 0.8  0.2 0.001 0.7\n"
            "0 0.1  4 2   0.4 0.004 0.8  0.4 0.004 0.7\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.polar"
            path.write_text(header, encoding="utf-8")
            result = load_vspaero_polar(path, AirframeDesign(), "estimated_geometry.vsp3")
        self.assertEqual(len(result.data), 5)
        self.assertGreater(result.cl_alpha_per_rad, 0.0)
        self.assertGreater(result.oswald_efficiency, 0.0)

    def test_reports_nan_solver_rows_as_invalid_points(self) -> None:
        polar_text = (
            "Beta Mach AoA Re/1e6 CLtot CDi E CLiw CDiw Ew\n"
            "0 0.1 -3 2 nan nan nan nan nan nan\n"
            "0 0.1 -1 2 nan nan nan nan nan nan\n"
            "0 0.1  1 2 nan nan nan nan nan nan\n"
            "0 0.1  3 2 nan nan nan nan nan nan\n"
            "0 0.1  5 2 nan nan nan nan nan nan\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diverged.polar"
            path.write_text(polar_text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "0 valid of 5 data rows"):
                load_vspaero_polar(path, AirframeDesign(), "estimated_geometry.vsp3")


class VspaeroOptimizationFailureTests(unittest.TestCase):
    def test_live_provider_falls_back_and_records_solver_failure(self) -> None:
        task_calibration = calibrate_endurance_task_baseline()
        calibration = task_calibration.dimension_calibration
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LiveOpenVSPAeroProvider(
                sys.executable,
                ROOT / "运行脚本" / "运行_VSPAERO单构型评估.py",
                temp_dir,
                calibration,
            )
            provider._baselines[(36.0, 3000.0)] = (5.0, 0.8)

            def fail_candidate(*args, **kwargs):
                raise RuntimeError("synthetic VSPAERO divergence")

            provider._run_polar = fail_candidate
            aero = provider(
                AirframeDesign(aspect_ratio=25.0),
                calibration,
                speed_mps=36.0,
                altitude_m=3000.0,
            )

        self.assertIsNotNone(provider.last_failure)
        self.assertIn("synthetic VSPAERO divergence", provider.last_failure)
        self.assertEqual(provider.records[-1]["status"], "vspaero_failed_low_order_fallback_infeasible")
        self.assertGreater(aero.cl_alpha_per_rad, 0.0)

    def test_diverged_candidate_is_infeasible_without_aborting_evaluation(self) -> None:
        class DivergingProvider:
            accepts_operating_point = True
            last_failure = None

            def __call__(self, design, calibration, **kwargs):
                self.last_failure = "synthetic VSPAERO divergence"
                return aero_from_design(design, calibration, **kwargs)

        provider = DivergingProvider()
        problem = PTDLEnduranceJointProblem(aero_provider=provider)
        decision = problem.sample(1, np.random.default_rng(41))[0]

        evaluation = problem.evaluate(decision)

        self.assertEqual(evaluation.result.constraints["vspaero_candidate_failed"], 1.0)
        self.assertFalse(evaluation.feasible)


if __name__ == "__main__":
    unittest.main()
