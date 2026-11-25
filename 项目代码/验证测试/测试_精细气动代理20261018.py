import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 研究_精细树与核代理20261018 as study
from 精细气动代理 import FineAeroSurrogate, component_prediction, kernel_matrix, shape_features


class FineAeroTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(188)
        self.x = rng.uniform([24., 19., .32, 0., -3., 7.6, .66],
                             [39., 30., .65, 7., 1., 9.8, .91], (72, 7))
        self.y = .8 + .03*np.sin(self.x[:, 3]) + .02*self.x[:, 2]

    def test_kernel_matches_reference_and_matern_is_symmetric(self):
        z = shape_features(self.x)
        np.testing.assert_allclose(kernel_matrix(z[:9], z, "rbf", .3),
                                   rbf_kernel(z[:9], z, gamma=.3), atol=1e-12)
        k = kernel_matrix(z, z, "matern32", .2)
        np.testing.assert_allclose(k, k.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(k), 1., atol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-8)

    def test_tree_and_both_kernel_exports_predict_unseen_inputs(self):
        configs = [dict(study.CONFIGS[0], iterations=20, depth=3),
                   dict(method="krr", kernel="rbf", gamma=.1, alpha=.03),
                   dict(method="krr", kernel="matern32", gamma=.3, alpha=.03),
                   dict(method="svr", kernel="rbf", gamma=.1, C=5., epsilon=.05)]
        probe = (self.x[:15] + self.x[15:30])/2
        for config in configs:
            with self.subTest(config=config):
                model = study.fit(config, self.x, self.y)
                portable = study.portable(model)
                np.testing.assert_allclose(component_prediction(portable, probe),
                                           study.predict(model, probe), atol=1e-8, rtol=1e-8)

    def test_json_roundtrip_and_invalid_scales_rejected(self):
        model = study.fit(dict(method="krr", kernel="rbf", gamma=.1, alpha=.03),
                          self.x, self.y)
        component = dict(study.portable(model), weight=1.)
        payload = dict(schema="ptdl-fine-aero-v1", training_samples=72,
                       baseline_targets=[6., .8],
                       targets=[dict(components=[copy.deepcopy(component)], interval_std=.05)
                                for _ in range(2)])
        obj = FineAeroSurrogate(payload)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            obj.to_json(path)
            restored = FineAeroSurrogate.from_json(path)
            np.testing.assert_allclose(obj.predict_many(self.x[:8]),
                                       restored.predict_many(self.x[:8]), atol=1e-12)
        for mutation in ("weight", "scale", "baseline"):
            bad = copy.deepcopy(payload)
            if mutation == "weight":
                bad["targets"][0]["components"][0]["weight"] = .5
            elif mutation == "scale":
                bad["targets"][0]["components"][0]["scale"][0] = 0.
            else:
                bad["baseline_targets"][0] = float("nan")
            with self.assertRaises(ValueError):
                FineAeroSurrogate(bad).validate()


if __name__ == "__main__":
    unittest.main()
