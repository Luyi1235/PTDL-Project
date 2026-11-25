import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 研究_局部细分尾迹代理20261017 as study
from 局部几何气动代理 import (
    LocalGeometryAero, model_features, numeric_tree_predict, validate_numeric_model,
    wake_path_features,
)


class LocalWakeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(453)
        self.x = rng.uniform(
            [24., 19., .32, 0., -3., 7.6, .66],
            [39., 30., .65, 7., 1., 9.8, .91], (90, 7))
        self.y = .8 + .05*np.sin(self.x[:, 3]) + .1*self.x[:, 2]

    def test_dimensionless_offsets_and_invalid_geometry(self):
        original = wake_path_features(self.x)
        scaled = self.x.copy()
        scaled[:, 0] *= 9
        scaled[:, 5:7] *= 3
        np.testing.assert_allclose(original[:, 28:], wake_path_features(scaled)[:, 28:],
                                   rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            wake_path_features([[0., 20., .5, 0., -1., 8., .8]])

    def test_all_tree_policies_export_and_boundary_predictions(self):
        for policy in ("SymmetricTree", "Depthwise", "Lossguide"):
            config = dict(study.CONFIGS[0], policy=policy, iterations=12, depth=3,
                          min_data=4, mode="wake_path")
            fitted = study.fit(config, self.x, self.y, np.repeat("uniform", len(self.y)))
            payload = study.export(fitted, "wake_path")
            validate_numeric_model(payload)
            np.testing.assert_allclose(
                numeric_tree_predict(payload, self.x),
                fitted.predict(model_features(self.x, "wake_path")), atol=1e-10, rtol=1e-10)
        split = dict(split_type="FloatFeature", float_feature_index=0, border=30.)
        model = dict(features="interference", scale=1., bias=0.,
                     trees=[dict(split=split, left={"value": 1.}, right={"value": 2.})])
        probes = self.x[:3].copy()
        probes[:, 0] = [30., np.nextafter(np.float32(30.), np.float32(np.inf)), 29.]
        np.testing.assert_array_equal(numeric_tree_predict(model, probes), [1., 2., 1.])

    def test_reject_invalid_numeric_trees(self):
        model = dict(features="wake_path", scale=1., bias=0., trees=[{"value": float("nan")}])
        with self.assertRaises(ValueError):
            validate_numeric_model(model)
        bad = dict(features="wake_path", scale=1., bias=0., trees=[
            dict(split=dict(split_type="FloatFeature", float_feature_index=200, border=1.),
                 left={"value": 1.}, right={"value": 2.})])
        with self.assertRaises(ValueError):
            validate_numeric_model(bad)

    def test_interface_roundtrip_and_invalid_weights(self):
        baseline = json.loads(study.prior.REFERENCE.read_text(encoding="utf8"))
        payload = dict(schema="ptdl-local-geometry-aero-v1", baseline=baseline,
                       training_samples=1512, wake_interval_std=.05,
                       wake_components=[dict(kind="reference", weight=1.,
                                             model=baseline["targets"][1])])
        source = LocalGeometryAero(payload)
        design = study.prior.previous.prior.original.prior.AirframeDesign(
            **dict(zip(study.prior.FEATURES, self.x[0])))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.json"
            source.to_json(path)
            restored = LocalGeometryAero.from_json(path)
            np.testing.assert_allclose(source.predict(design), restored.predict(design))
            self.assertEqual(len(restored.predict_with_uncertainty(design)), 4)
            invalid = copy.deepcopy(payload)
            invalid["wake_components"][0]["weight"] = -1.
            path.write_text(json.dumps(invalid), encoding="utf8")
            with self.assertRaises(ValueError):
                LocalGeometryAero.from_json(path)


if __name__ == "__main__":
    unittest.main()
