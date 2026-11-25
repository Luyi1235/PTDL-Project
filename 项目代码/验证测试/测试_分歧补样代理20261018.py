from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 推进_分歧补样4096代理20261018 as study


class DisagreementSamplingTests(unittest.TestCase):
    def test_native_batch_prediction_matches_portable_reference(self):
        data = pd.read_csv(study.SOURCE)
        x = data[study.FEATURES].to_numpy(float)[::311]
        native = study.old_batch_prediction(x)
        model = study.TargetwiseAeroGP.from_json(study.BASE / "均匀覆盖候选代理.json")
        design_class = study.previous.expansion.balanced.previous.prior.original.prior.AirframeDesign
        portable = np.array([model.predict(design_class(**dict(zip(study.FEATURES, row))))
                             for row in x])
        np.testing.assert_allclose(native, portable, atol=1e-9, rtol=1e-9)

    def test_frozen_plan_is_feasible_disjoint_and_uses_training_acquisition(self):
        manifest = study.previous.read(study.OUT / "配置冻结.json")
        jobs = manifest["jobs"]
        counts = Counter((job["role"], job["basis"]) for job in jobs)
        self.assertEqual(counts, {
            ("train", "uniform_feasible_coverage"): 512,
            ("train", "training_models_disagreement"): 256,
            ("train", "training_oof_error_neighborhood"): 256,
            ("test", "fresh_sealed_uniform"): 192})
        self.assertFalse(manifest["external_error_labels_used"])
        x = np.array([[job["design"][k] for k in study.FEATURES] for job in jobs])
        self.assertEqual(len(np.unique(x, axis=0)), 1216)
        training = pd.read_csv(study.SOURCE)
        forbidden = study.forbidden_geometries(training)
        span = np.ptp(training[study.FEATURES].to_numpy(float), axis=0)
        for start in range(0, len(x), 32):
            distances = np.linalg.norm(
                (x[start:start+32, None]-forbidden[None])/span, axis=2)
            self.assertGreater(distances.min(), 1e-5)
        original = study.previous.expansion.balanced.previous.prior.original
        for job in jobs:
            constraints = original.prior.geometry_constraints(
                original.prior.AirframeDesign(**job["design"]))
            self.assertLessEqual(max(constraints.values()), 1e-9)


if __name__ == "__main__":
    unittest.main()
