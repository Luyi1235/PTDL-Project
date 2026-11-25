"""Geometry-only descriptor checks and preservation of the lift predictor."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 尾迹射线网格特征 import FEATURE_NAMES, read_wake_mesh, ray_descriptors
from 均匀风险尾迹代理 import UniformRiskWakeAero, features
from 精细气动代理 import FineAeroSurrogate

MESH = ROOT / "运行结果/分歧补样4096代理_20261018/solver/adaptive4096_test_1163/case/airframe.vspgeom"
BASELINE = ROOT / "运行结果/分歧补样4096代理_20261018/精细训练/精细气动候选代理.json"


class WakeRayTests(unittest.TestCase):
    def test_split_horizontal_tail_wake_is_supported(self):
        path = ROOT / "运行结果/千构型原定义代理_20261017/solver/stable_train_0018/case/airframe.vspgeom"
        mesh = read_wake_mesh(path)
        self.assertEqual(set(mesh.sheets), {2, 3, 4, 5, 6})
        value = ray_descriptors(mesh, 30., 24.)
        self.assertEqual(value.shape, (len(FEATURE_NAMES),))
        self.assertEqual(value[3], 5.)
        self.assertTrue(np.isfinite(value).all())

    def test_wrapped_trailing_edge_nodes_are_read(self):
        mesh = read_wake_mesh(MESH)
        self.assertEqual({k: len(v) for k, v in mesh.sheets.items()},
                         {2: 6, 3: 6, 4: 11, 6: 6})
        value = ray_descriptors(mesh, 28.52810, 26.45707)
        self.assertEqual(value.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(value).all())

    def test_translation_and_similar_geometry_invariance(self):
        mesh = read_wake_mesh(MESH)
        expected = ray_descriptors(mesh, 28.52810, 26.45707)
        transformed = replace(mesh, xyz=1.7*mesh.xyz+np.array([7., -4., 3.]))
        actual = ray_descriptors(transformed, 28.52810*1.7**2, 26.45707)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)

    def test_missing_geometry_is_rejected_when_model_requires_it(self):
        config = dict(mesh=True, transform="canonical-area")
        with self.assertRaisesRegex(ValueError, "descriptors are required"):
            features([[30., 24., .5, 3., -1., 8., .8]], None, config)

    def test_serialization_preserves_lift_and_wake_predictions(self):
        x = np.array([[30., 24., .5, 3., -1., 8., .8]])
        config = dict(mesh=False, transform="canonical-area")
        baseline = json.loads(BASELINE.read_text(encoding="utf8"))
        estimator = DummyRegressor(strategy="constant", constant=.8).fit(
            features(x, None, config), [.8])
        payload = dict(schema="ptdl-uniform-risk-wake-v1", baseline=baseline,
            wake_components=[dict(kind="tree", config=config, estimator=estimator, weight=1.)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "尾迹模型.joblib"
            joblib.dump(payload, path)
            model = UniformRiskWakeAero.load(path)
            p = model.predict_many(x)
            np.testing.assert_allclose(p[:, 0], FineAeroSurrogate(baseline).predict_many(x)[:, 0])
            np.testing.assert_allclose(p[:, 1], [.8])
            self.assertFalse(model.requires_mesh)


if __name__ == "__main__":
    unittest.main()
