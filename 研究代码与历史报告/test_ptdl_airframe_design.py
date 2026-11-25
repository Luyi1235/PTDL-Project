"""Tests for the dimension-driven PTDL airframe model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptdl_airframe_design import BASELINE, AirframeDesign, aero_from_design, aircraft_from_design, design_metrics, geometry_constraints, write_openvsp_geometry
from ptdl_flight_propulsion_control import isa_density, parasite_drag_coefficient
from ptdl_ultra_benchmark import calibrate_ultra_baseline
from ptdl_vspaero_adapter import load_vspaero_polar


class AirframeDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = calibrate_ultra_baseline()

    def test_baseline_dimensions_reconstruct_the_public_span_scale(self) -> None:
        metrics = design_metrics(BASELINE)
        self.assertAlmostEqual(metrics.wing_span_m, 24.99, delta=0.05)
        self.assertLessEqual(geometry_constraints(BASELINE)["span_limit"], 0.0)
        self.assertLessEqual(geometry_constraints(BASELINE)["stall_speed"], 0.0)

    def test_aerodynamic_coefficients_are_derived_not_decision_variables(self) -> None:
        aero = aero_from_design(BASELINE, self.calibration)
        aircraft = aircraft_from_design(BASELINE, self.calibration)
        self.assertAlmostEqual(aero.cd0, self.calibration.cd0, delta=1e-8)
        self.assertGreater(aero.oswald_efficiency, 0.65)
        self.assertAlmostEqual(aircraft.wing_span_m, design_metrics(BASELINE).wing_span_m)

    def test_component_build_up_cd0_varies_with_geometry_and_mission_reynolds(self) -> None:
        enlarged_fuselage = AirframeDesign(fuselage_length_m=9.8, fuselage_diameter_m=0.92)
        baseline_aero = aero_from_design(BASELINE, self.calibration)
        enlarged_aero = aero_from_design(enlarged_fuselage, self.calibration)
        self.assertGreater(enlarged_aero.cd0, baseline_aero.cd0)

        aircraft = aircraft_from_design(BASELINE, self.calibration)
        reference_cd0 = parasite_drag_coefficient(
            aircraft,
            isa_density(self.calibration.reference.calibration_altitude_m),
            self.calibration.reference.cruise_speed_mps,
        )
        low_speed_cd0 = parasite_drag_coefficient(aircraft, isa_density(8000.0), 18.0)
        self.assertAlmostEqual(reference_cd0, aircraft.cd0, delta=1e-8)
        self.assertGreater(low_speed_cd0, reference_cd0)

    def test_openvsp_geometry_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ultra.vsp3"
            write_openvsp_geometry(BASELINE, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

    def test_saved_vspaero_polar_is_parseable(self) -> None:
        polar_path = Path("ptdl_airframe_results/vspaero_baseline/airframe_DegenGeom.polar")
        if not polar_path.exists():
            self.skipTest("VSPAERO baseline output has not been generated.")
        result = load_vspaero_polar(polar_path, BASELINE, polar_path.with_suffix(".vsp3"))
        self.assertGreater(result.cl_alpha_per_rad, 5.0)
        self.assertGreater(result.oswald_efficiency, 0.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
