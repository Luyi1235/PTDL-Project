"""Tests for the PTDL-specific 80 h endurance mission profile."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptdl_80h_endurance_profile import (
    build_endurance_mission,
    export_endurance_baseline,
    export_endurance_extension_verification,
    load_endurance_profile,
)
from ptdl_airframe_design import AirframeDesign


class EnduranceProfileTests(unittest.TestCase):
    def test_profile_builds_an_exact_80h_airborne_mission(self) -> None:
        profile = load_endurance_profile()
        mission = build_endurance_mission(profile)
        self.assertAlmostEqual(sum(phase.duration_s for phase in mission.phases) / 3600.0, 80.0, delta=1e-8)
        self.assertEqual(mission.phases[2].name, "endurance_loiter")
        self.assertAlmostEqual(mission.phases[2].altitude_target_m, 3000.0)
        self.assertAlmostEqual(mission.phases[2].speed_target_mps, 36.01)

    def test_endurance_baseline_exports_a_complete_audit(self) -> None:
        design = AirframeDesign(payload_mass_kg=100.0, hybrid_system_mass_kg=80.0)
        with tempfile.TemporaryDirectory() as directory:
            evaluation = export_endurance_baseline(Path(directory), design)
            self.assertGreater(len(evaluation.simulation.profile), 1)
            self.assertIn("endurance_80h", evaluation.result.constraints)
            self.assertTrue((Path(directory) / "ptdl_80h_endurance_baseline_time_history.csv").exists())
            self.assertTrue((Path(directory) / "ptdl_80h_endurance_baseline_metadata.json").exists())

    def test_extension_verification_requires_more_than_80h(self) -> None:
        design = AirframeDesign(payload_mass_kg=100.0, hybrid_system_mass_kg=0.0)
        with tempfile.TemporaryDirectory() as directory:
            evaluation = export_endurance_extension_verification(
                Path(directory),
                design,
                target_total_duration_h=81.0,
                speeds_mps=[52.0, 36.01, 30.0, 42.0, 30.0, 21.0],
            )
            self.assertGreaterEqual(evaluation.achieved_duration_h, 80.0)
            self.assertTrue((Path(directory) / "ptdl_endurance_extension_evaluation.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
