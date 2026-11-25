"""Software smoke tests for the PTDL estimated JSBSim 6-DoF model."""

from __future__ import annotations

import importlib.util
import math
import csv
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
SCRIPT = ROOT / "运行脚本" / "运行_PTDL估计六自由度验证.py"
sys.path.insert(0, str(CORE))

from PTDL六自由度参考模型 import PTDL6DoFParameters
from PTDL联合优化高保真接口 import PTDL6DoFOptimizerValidator
from 八十小时续航任务模型 import calibrate_endurance_task_baseline
from 机体参数化设计与气动阻力 import AirframeDesign, aero_from_design

spec = importlib.util.spec_from_file_location("ptdl_6dof_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runner)


class PTDL6DoFEstimateTests(unittest.TestCase):
    def test_estimated_mass_closes_to_public_scale_mtow(self) -> None:
        params = PTDL6DoFParameters()
        self.assertLess(abs(params.mass_closure_error_kg), 0.02)

    def test_model_loads_and_runs_with_three_axis_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary = runner.run(40.0, 0.1, Path(temp))
            self.assertTrue(summary["finite_state_history"])
            self.assertTrue(summary["software_smoke_envelope_passed"])
            self.assertGreater(summary["sample_count"], 300)
            self.assertLess(abs(summary["initial_speed_mps"] - 36.011), 0.2)
            self.assertLess(abs(summary["initial_altitude_m"] - 3000.0), 1.0)
            with open(summary["outputs"][0], encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                header = reader.fieldnames or []
            self.assertIn("elevator_rad", header)
            self.assertIn("aileron_rad", header)
            self.assertIn("rudder_rad", header)
            for surface in ("elevator_rad", "aileron_rad", "rudder_rad"):
                self.assertGreater(max(abs(float(row[surface])) for row in rows), 1e-5)
            self.assertTrue(all(math.isfinite(float(row["speed_mps"])) for row in rows))

    def test_candidate_specific_sixdof_check_returns_optimizer_constraints(self) -> None:
        calibration = calibrate_endurance_task_baseline().dimension_calibration
        design = AirframeDesign(
            wing_area_m2=31.988386,
            aspect_ratio=28.921683,
            taper_ratio=0.322985,
            sweep_deg=3.026379,
            tip_twist_deg=0.216925,
            fuselage_length_m=8.241671,
            fuselage_diameter_m=0.707050,
            propeller_diameter_m=2.152803,
            payload_mass_kg=100.0,
            hybrid_system_mass_kg=0.0,
        )
        result = PTDL6DoFOptimizerValidator(duration_s=5.0)(
            design,
            aero_from_design(design, calibration),
            calibration,
            cruise_speed_mps=36.0,
            cruise_altitude_m=6000.0,
            available_power_kw=100.0,
        )

        self.assertEqual(result["constraints"]["sixdof_finite_state"], 0.0)
        self.assertLessEqual(result["constraints"]["sixdof_bank_deg"], 0.0)
        self.assertLessEqual(result["constraints"]["sixdof_control_saturation"], 0.0)
        self.assertEqual(result["diagnostics"]["sixdof_status"], "estimated_JSBSim_candidate_envelope_check")


if __name__ == "__main__":
    unittest.main()
