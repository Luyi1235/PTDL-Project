"""Tests for adapting the supplied ULTRA checkout profile."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptdl_airframe_design import AirframeDesign
from ptdl_full_task_profile import (
    build_full_task_mission,
    evaluate_full_task_profile,
    export_full_task_baseline,
    load_source_profile,
    source_profile_duration_h,
)


class FullTaskProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_source_profile()

    def test_source_profile_is_a_5000km_checkout_not_an_80h_definition(self) -> None:
        self.assertEqual(self.profile["reference"]["target_total_range_m"], 5_000_000)
        self.assertGreater(source_profile_duration_h(self.profile), 20.0)
        self.assertLess(source_profile_duration_h(self.profile), 22.0)

    def test_80h_adapter_extends_only_the_isr_cruise_phase(self) -> None:
        mission = build_full_task_mission(self.profile, target_total_duration_h=80.0)
        self.assertEqual(mission.phases[0].name, "main_climb_to_leveloff")
        self.assertEqual(mission.phases[-1].name, "flare")
        self.assertAlmostEqual(sum(phase.duration_s for phase in mission.phases) / 3600.0, 80.0, delta=1e-8)
        cruise = next(phase for phase in mission.phases if phase.name == "isr_cruise_alpha0_5000km")
        self.assertGreater(cruise.duration_s, 250_000.0)

    def test_full_profile_evaluation_exports_an_audit_even_when_infeasible(self) -> None:
        design = AirframeDesign(payload_mass_kg=100.0, hybrid_system_mass_kg=120.0)
        with tempfile.TemporaryDirectory() as directory:
            evaluation = export_full_task_baseline(Path(directory), design)
            self.assertGreater(len(evaluation.simulation.profile), 1)
            self.assertIn("endurance_80h", evaluation.result.constraints)
            self.assertTrue((Path(directory) / "full_task_baseline_time_history.csv").exists())
            self.assertTrue((Path(directory) / "full_task_baseline_metadata.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
