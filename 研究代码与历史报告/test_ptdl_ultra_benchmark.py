"""Regression tests for the open-data ULTRA endurance benchmark."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptdl_ultra_benchmark import (
    KNOT_TO_MPS,
    calibrate_ultra_baseline,
    endurance_at_speed,
    required_cd0_factor_for_target,
    required_engine_efficiency_for_target,
    run_ultra_benchmark,
)


class UltraBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = calibrate_ultra_baseline()

    def test_inverse_calibration_reproduces_72_hour_baseline(self) -> None:
        self.assertAlmostEqual(self.calibration.baseline_endurance_h, 72.0, delta=0.05)
        self.assertGreater(self.calibration.cd0, 0.015)
        self.assertLess(self.calibration.cd0, 0.040)

    def test_speed_only_change_does_not_reach_80_hours(self) -> None:
        endurance_h = endurance_at_speed(self.calibration, 68.52 * KNOT_TO_MPS)
        self.assertGreater(endurance_h, self.calibration.baseline_endurance_h)
        self.assertLess(endurance_h, 80.0)

    def test_80_hour_energy_requirements_are_explicit(self) -> None:
        cd0_factor = required_cd0_factor_for_target(self.calibration)
        engine_efficiency = required_engine_efficiency_for_target(self.calibration)

        self.assertGreater(cd0_factor, 0.80)
        self.assertLess(cd0_factor, 0.95)
        self.assertGreater(engine_efficiency, 0.30)
        self.assertLess(engine_efficiency, 0.36)

    def test_benchmark_exports_traceable_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_ultra_benchmark(Path(directory) / "ultra")
            output_dir = result["output_dir"]
            self.assertTrue((output_dir / "ultra_80h_feasibility_summary.csv").exists())
            self.assertTrue((output_dir / "ultra_speed_endurance_scan.csv").exists())
            self.assertTrue((output_dir / "ultra_model_metadata.json").exists())
            self.assertEqual(len(result["summary"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
