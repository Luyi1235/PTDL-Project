"""Check numerical inference, geometry invariance, and fresh-label isolation."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
sys.path.insert(0, str(ROOT / "核心模型"))
import 研究_局部线性与非对称树20261018 as study
from 局部线性树气动代理 import LocalLinearTreeAero, tree_features


class LocalTreeTests(unittest.TestCase):
    def test_similar_geometry_has_identical_shape_features(self):
        x = np.array([[30., 24., .5, 3., -1., 8., .8]])
        scaled = x.copy()
        scaled[:, 0] *= 1.7**2
        scaled[:, 5:] *= 1.7
        for mode in ("shape", "interference", "wake_path"):
            np.testing.assert_allclose(tree_features(x, mode), tree_features(scaled, mode),
                                       atol=1e-12, rtol=1e-12)

    def test_linear_tree_roundtrip_and_nonfinite_rejection(self):
        rng = np.random.default_rng(42)
        x = rng.uniform([23, 19, .3, 0, -3, 7.5, .65], [39, 30, .65, 8, 1, 10, .95],
                        size=(100, 7))
        config = next(c for c in study.CONFIGS
                      if c["family"] == "lightgbm" and c["linear"])
        fitted = study.fit(dict(config, iterations=20), x, .8 + .02*np.sin(x[:, 3]))
        payload = dict(schema="ptdl-local-linear-tree-aero-v1", training_samples=100,
                       baseline_targets=[6., .8], targets=[
                           dict(components=[dict(fitted, weight=1.)], interval_std=.02),
                           dict(components=[dict(fitted, weight=1.)], interval_std=.03)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "中文模型.joblib"
            joblib.dump(payload, path)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                model = LocalLinearTreeAero.load(path)
                p = model.predict_many(x[:5])
                np.testing.assert_allclose(p, LocalLinearTreeAero(payload).predict_many(x[:5]))
                self.assertEqual(p.shape, (5, 2))
                np.testing.assert_allclose(p[:1], model.predict_many(x[:1]))
            bad = x[:1].copy()
            bad[0, 2] = np.nan
            with self.assertRaises(ValueError):
                model.predict_many(bad)
            invalid = copy.deepcopy(payload)
            invalid["targets"][0]["components"][0]["weight"] = -1.
            joblib.dump(invalid, path)
            with self.assertRaises(ValueError):
                LocalLinearTreeAero.load(path)

    def test_predictions_cannot_be_registered_after_test_computation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, native = root / "new", root / "native"
            output.mkdir()
            native.mkdir()
            jobs = [dict(role="test", sample=f"test_{i}") for i in range(192)]
            (native / "配置冻结.json").write_text(json.dumps(dict(jobs=jobs)), encoding="utf8")
            (output / "模型选择.json").write_text(
                json.dumps(dict(model_hashes={}, model_paths={})), encoding="utf8")
            (native / "solver" / jobs[0]["sample"]).mkdir(parents=True)
            with patch.object(study.batch, "OUT", native):
                with self.assertRaisesRegex(RuntimeError, "before test computation"):
                    study.lock_predictions(output)

    def test_frozen_artifact_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "artifact.json"
            path.write_text("original", encoding="utf8")
            hashes = {str(path): study.digest(path)}
            study.check_hashes(hashes)
            path.write_text("changed", encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                study.check_hashes(hashes)


if __name__ == "__main__":
    unittest.main()
