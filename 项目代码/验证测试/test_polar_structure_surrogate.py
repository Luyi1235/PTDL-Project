"""Force reconstruction, auxiliary-target fitting, and artifact regression checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 研究_极曲线结构代理20261019 as study
from 极曲线结构代理 import (
    BASIS, PolarStructureAero, fitted_prediction, force_coefficients, reconstruct,
)
from 精细气动代理 import FineAeroSurrogate


class PolarStructureTests(unittest.TestCase):
    def test_quadratic_forces_preserve_original_efficiency_definition(self):
        coefficients = np.array([
            [[.1, .5, 0.], [.08, .49, 0.], [.04, .02, .3]],
            [[.2, .4, 0.], [.18, .39, 0.], [.05, -.01, .2]],
        ])
        curves = np.einsum("ak,nck->nac", BASIS, coefficients)
        aspect = np.array([20., 30.])
        raw = curves.copy()
        raw[:, :, 2] /= np.pi*aspect[:, None]
        fitted = force_coefficients(raw, aspect)
        np.testing.assert_allclose(fitted, coefficients.reshape(2, 9), atol=1e-12)
        expected = []
        for row in curves:
            eligible = (abs(row[:, 1]) >= .15) & (row[:, 2] > 0)
            expected.append(np.median(row[eligible, 0]**2/row[eligible, 2]))
        np.testing.assert_allclose(reconstruct(fitted, [0., 0.]), expected, atol=1e-12)

    def test_working_angle_filter_and_missing_drag_fallback(self):
        coefficients = np.zeros((2, 3, 3))
        coefficients[:, 0, 0] = .4
        coefficients[:, 1, 0] = .1
        coefficients[:, 2, 0] = -.2
        np.testing.assert_array_equal(reconstruct(coefficients.reshape(2, 9), [.7, .8]),
                                      [.7, .8])
        with self.assertRaises(ValueError):
            force_coefficients(np.zeros((2, 4, 3)), [20., 30.])
        with self.assertRaises(ValueError):
            reconstruct(np.full((2, 9), np.nan), [.7, .8])

    def test_multitarget_fit_and_roundtrip_preserve_lift(self):
        with np.load(study.OUT / "极曲线训练数据.npz") as archive:
            indices = np.arange(64)
            x, y, mesh, coefficients = (
                archive[key][indices] for key in ("x", "y", "mesh", "coefficients"))
        config = dict(study.CONFIGS[4], iterations=12, depth=3, selected=8)
        fitted = study.fit(config, x[:48], y[:48], mesh[:48], coefficients[:48], 2)
        base_count = study.previous.features(x[:1], None, dict(config, mesh=False)).shape[1]
        self.assertEqual(len(fitted["columns"]), base_count+8)
        self.assertTrue(set(range(base_count)).issubset(fitted["columns"]))
        for mode in study.MODES:
            p = fitted_prediction(fitted, x[48:], mesh[48:], mode)
            self.assertEqual(p.shape, (16,))
            self.assertTrue(np.isfinite(p).all())
        baseline = json.loads(study.previous.BASELINE.read_text(encoding="utf8"))
        payload = dict(schema="ptdl-polar-structure-v1", baseline=baseline,
            wake_components=[dict(kind="profile", weight=1., mode="direct", fitted=fitted)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.joblib"
            joblib.dump(payload, path)
            model = PolarStructureAero.load(path)
            p = model.predict_many(x[48:], mesh[48:])
            np.testing.assert_allclose(p[:, 0],
                FineAeroSurrogate(baseline).predict_many(x[48:])[:, 0], atol=1e-12)
            np.testing.assert_allclose(p[:, 1],
                fitted_prediction(fitted, x[48:], mesh[48:], "direct"), atol=1e-12)
            self.assertTrue(model.requires_mesh)
            with self.assertRaises(ValueError):
                model.predict_many(x[48:])

    def test_invalid_artifact_mixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.joblib"
            joblib.dump(dict(schema="ptdl-polar-structure-v1",
                wake_components=[dict(weight=-.2)]), path)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                PolarStructureAero.load(path)


if __name__ == "__main__":
    unittest.main()
