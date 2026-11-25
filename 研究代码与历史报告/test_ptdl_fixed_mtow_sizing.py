"""Regression tests for fixed-MTOW preliminary sizing."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ptdl_airframe_design import BASELINE
from ptdl_fixed_mtow_sizing import calibrate_dimension_baseline, evaluate_fixed_mtow_design, run_fixed_mtow_screen


class FixedMtowSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = calibrate_dimension_baseline()

    def test_baseline_uses_mass_closure_and_the_calibrated_loiter_case(self) -> None:
        evaluation = evaluate_fixed_mtow_design(BASELINE, 36.01, self.calibration, duration_cap_h=82.0)
        self.assertAlmostEqual(evaluation.metrics.mtow_margin_kg, 0.0, delta=1e-8)
        self.assertAlmostEqual(evaluation.metrics.fuel_mass_kg, self.calibration.reference.fuel_mass_kg, delta=1e-6)
        self.assertGreater(evaluation.achieved_endurance_h, 71.5)
        self.assertIn("span_limit", evaluation.result.constraints)
        self.assertIn("endurance_80h", evaluation.result.constraints)

    def test_hybrid_mass_is_paid_for_by_residual_fuel_at_fixed_mtow(self) -> None:
        hybrid = replace(BASELINE, hybrid_system_mass_kg=80.0)
        baseline = evaluate_fixed_mtow_design(BASELINE, 36.01, self.calibration, duration_cap_h=2.0)
        evaluation = evaluate_fixed_mtow_design(hybrid, 36.01, self.calibration, duration_cap_h=2.0)
        self.assertAlmostEqual(
            baseline.metrics.fuel_mass_kg - evaluation.metrics.fuel_mass_kg,
            80.0,
            delta=1e-6,
        )

    def test_screen_exports_the_traceable_design_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "screen"
            frame = run_fixed_mtow_screen(output_dir, sample_count=3, seed=20260904)
            self.assertEqual(len(frame), 4)
            self.assertTrue((output_dir / "fixed_mtow_sizing_screen.csv").exists())
            self.assertTrue((output_dir / "fixed_mtow_sizing_metadata.json").exists())
            self.assertIn("fuel_mass_kg", frame.columns)
            self.assertIn("g_endurance_80h", frame.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
