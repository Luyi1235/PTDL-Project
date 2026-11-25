import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 研究_尺度约束与总体误差20261017 as study
from 尺度约束气动代理 import ScalePopulationAero, geometry_transform
from 局部几何气动代理 import numeric_tree_predict


class ScalePopulationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.x = rng.uniform([24., 19., .32, 0., -3., 7.6, .66],
                             [39., 30., .65, 7., 1., 9.8, .91], (100, 7))
        self.y = .8 + .05*np.sin(self.x[:, 3])+.03*self.x[:, 2]

    def test_transform_is_invariant_and_nonmutating(self):
        scaled = self.x.copy()
        scaled[:, 0] *= 4
        scaled[:, 5:7] *= 2
        original = self.x.copy()
        np.testing.assert_allclose(geometry_transform(self.x, "canonical-area"),
                                   geometry_transform(scaled, "canonical-area"), rtol=1e-12)
        np.testing.assert_array_equal(self.x, original)
        for mode in ("unknown",):
            with self.assertRaises(ValueError):
                geometry_transform(self.x, mode)
        with self.assertRaises(ValueError):
            geometry_transform([[0., 24., .5, 0., -1., 8., .8]], "canonical-area")

    def test_weights_retain_every_sample_and_uniform_is_emphasized(self):
        weights = study.weights(["old", "uniform_feasible_coverage", "gp_uncertainty"], "uniform4")
        self.assertTrue(np.all(weights > 0))
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertAlmostEqual(weights[1]/weights[0], 4.)

    def test_selection_uses_only_designated_training_population(self):
        y = np.arange(60)/100
        population = np.arange(60) < 40
        a = np.tile(y, (2, 1))
        a[:, population] += .1
        b = np.tile(y, (2, 1))
        b[:, ~population] += 10
        _, selected, _ = study.select([a, b], y, population)
        self.assertEqual(selected, [(1, 1.)])

    def test_normalized_tree_export_and_interface_roundtrip(self):
        config = dict(study.CONFIGS[0], transform="canonical-area", iterations=15, depth=3)
        fitted = study.fit(config, self.x, self.y, np.repeat("old", len(self.y)))
        exported = study.prior.export(fitted["estimator"], "interference")
        np.testing.assert_allclose(
            numeric_tree_predict(exported, geometry_transform(self.x, "canonical-area")),
            study.predict(fitted, self.x), rtol=1e-10, atol=1e-10)
        baseline = json.loads(study.prior.prior.REFERENCE.read_text(encoding="utf8"))
        payload = dict(schema="ptdl-scale-population-aero-v1", baseline=baseline,
                       training_samples=100, wake_interval_std=.05,
                       wake_components=[dict(kind="tree", transform="canonical-area",
                                             weight=1., model=exported)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.json"
            ScalePopulationAero(payload).to_json(path)
            restored = ScalePopulationAero.from_json(path)
            np.testing.assert_allclose(restored.predict_many(self.x[:4])[:, 1],
                                       study.predict(fitted, self.x[:4]), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
