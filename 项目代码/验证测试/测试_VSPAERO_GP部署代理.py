from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (  # noqa: E402
    ArdRbfGaussianProcessVSPAeroSurrogate,
    FEATURE_LABELS,
    GaussianProcessVSPAeroSurrogate,
    load_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402


MODEL = ROOT / "气动代理模型数据" / "气动代理模型_扩充DOE131点_二次趋势Matern32GP_20261011.json"
ARD_RBF_MODEL = (
    ROOT
    / "气动代理模型数据"
    / "气动代理模型_扩充DOE131点_ARD_RBF_GP_20261012.json"
)
DATA = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_131点_Ew_物理选点_网格修复_自适应补样_20261011.csv"


class GaussianProcessDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_vsp_surrogate(MODEL)
        cls.data = pd.read_csv(DATA)

    def test_loader_selects_serialized_gp_without_training_dependency(self) -> None:
        self.assertIsInstance(self.model, GaussianProcessVSPAeroSurrogate)
        self.assertEqual(self.model.training_samples, 131)

    def test_deployed_model_reproduces_all_training_labels(self) -> None:
        predictions = np.asarray(
            [
                self.model.predict(
                    AirframeDesign(
                        **{name: float(row[name]) for name in FEATURE_LABELS}
                    )
                )
                for _, row in self.data.iterrows()
            ]
        )
        targets = self.data.loc[
            :, ["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]
        ].to_numpy(float)
        self.assertLess(float(np.max(np.abs(predictions - targets))), 1e-3)

    def test_corrected_aero_is_finite_at_baseline_and_candidate(self) -> None:
        from 八十小时续航任务模型 import calibrate_endurance_task_baseline

        calibration = calibrate_endurance_task_baseline().dimension_calibration
        for design in (
            AirframeDesign(),
            AirframeDesign(
                wing_area_m2=30.0,
                aspect_ratio=27.0,
                taper_ratio=0.50,
                sweep_deg=3.0,
                tip_twist_deg=-1.0,
                fuselage_length_m=8.8,
                fuselage_diameter_m=0.82,
            ),
        ):
            aero = self.model.corrected_aero(design, calibration)
            self.assertTrue(np.isfinite(aero.cl_alpha_per_rad))
            self.assertTrue(np.isfinite(aero.oswald_efficiency))
            self.assertGreater(aero.cl_alpha_per_rad, 0.0)
            self.assertGreater(aero.oswald_efficiency, 0.0)


class ArdRbfGaussianProcessDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_vsp_surrogate(ARD_RBF_MODEL)

    def test_loader_selects_deployable_ard_rbf_without_sklearn(self) -> None:
        self.assertIsInstance(self.model, ArdRbfGaussianProcessVSPAeroSurrogate)
        self.assertEqual(self.model.training_samples, 131)

    def test_predictions_and_uncertainty_are_finite(self) -> None:
        for design in (
            AirframeDesign(),
            AirframeDesign(
                wing_area_m2=30.0,
                aspect_ratio=27.0,
                taper_ratio=0.50,
                sweep_deg=3.0,
                tip_twist_deg=-1.0,
                fuselage_length_m=8.8,
                fuselage_diameter_m=0.82,
            ),
        ):
            prediction = self.model.predict_with_uncertainty(design)
            self.assertTrue(np.isfinite(prediction).all())
            self.assertGreaterEqual(prediction[2], 0.0)
            self.assertGreaterEqual(prediction[3], 0.0)

    def test_corrected_aero_remains_compatible_with_optimizer_interface(self) -> None:
        from 八十小时续航任务模型 import calibrate_endurance_task_baseline

        calibration = calibrate_endurance_task_baseline().dimension_calibration
        aero = self.model.corrected_aero(AirframeDesign(), calibration)
        self.assertTrue(np.isfinite(aero.cl_alpha_per_rad))
        self.assertTrue(np.isfinite(aero.oswald_efficiency))
        self.assertGreater(aero.cl_alpha_per_rad, 0.0)
        self.assertGreater(aero.oswald_efficiency, 0.0)


if __name__ == "__main__":
    unittest.main()
