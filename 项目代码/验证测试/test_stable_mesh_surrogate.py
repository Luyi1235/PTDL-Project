"""Numerical descriptor cleanup and stable artifact inference checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 研究_稳定网格尾迹代理20261019 as study
import 评价_稳定网格尾迹代理20261019 as evaluation
from 稳定网格尾迹代理 import StableMeshWakeAero, features
from 尾迹射线网格特征 import extract
from 精细气动代理 import FineAeroSurrogate


class StableDescriptorTests(unittest.TestCase):
    def test_shared_holdout_requires_prediction_registration_before_solving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "registered"
            out.mkdir()
            native = root / "expanded"
            (native / "solver" / "test_0").mkdir(parents=True)
            with (patch.object(evaluation, "OUT", out),
                  patch.object(evaluation.expansion, "OUT", native),
                  patch.object(evaluation.batch.previous, "read",
                               return_value=dict(hashes={}, jobs=[
                                   dict(sample="test_0", role="test")])),
                  patch.object(evaluation.expansion.native, "prepare") as prepare):
                with self.assertRaisesRegex(RuntimeError, "before stable-model prediction"):
                    evaluation.register(None, root / "unused.json")
                prepare.assert_not_called()

    def test_reexported_geometry_has_same_float32_model_input(self):
        original = pd.read_csv(study.study.batch.OUT / "独立评价逐点.csv").set_index("sample")
        saved = pd.read_csv(study.study.OUT / "192构型诊断复评/诊断逐点.csv").set_index("sample")
        config = study.CONFIGS[1]
        for name in ("adaptive4096_test_1060", "adaptive4096_test_1091"):
            inputs = []
            for frame in (original, saved):
                row = frame.loc[name]
                mesh = extract(Path(row.geometry_file).with_suffix(".vspgeom"),
                               row.wing_area_m2, row.aspect_ratio)
                x = row[study.profile.FEATURES].to_numpy(float)[None]
                inputs.append(features(x, mesh[None], config).astype(np.float32))
            np.testing.assert_array_equal(*inputs)

    def test_only_numerical_scale_values_are_zeroed(self):
        with np.load(study.profile.OUT / "极曲线训练数据.npz") as archive:
            x, mesh = archive["x"][:8], archive["mesh"][:8]
        config = study.CONFIGS[0]
        raw = study.study.features(x, mesh, config)
        clean = features(x, mesh, config)
        small = abs(raw) < config["zero_tolerance"]
        self.assertTrue(((raw != 0) & small).any())
        np.testing.assert_array_equal(clean[small], 0.)
        np.testing.assert_array_equal(clean[~small], raw[~small])
        with self.assertRaises(ValueError):
            features(x, mesh, dict(config, zero_tolerance=0.))

    def test_fit_export_and_lift_preservation(self):
        with np.load(study.profile.OUT / "极曲线训练数据.npz") as archive:
            x, y, mesh = (archive[k][:64] for k in ("x", "y", "mesh"))
        config = dict(study.CONFIGS[1], iterations=12)
        estimator = study.fit(config, x[:48], y[:48], mesh[:48], np.ones(48, bool))
        baseline = json.loads(study.study.BASELINE.read_text(encoding="utf8"))
        payload = dict(schema="ptdl-stable-mesh-wake-v1", baseline=baseline,
            wake_components=[dict(kind="tree", config=config, estimator=estimator, weight=1.)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stable.joblib"
            joblib.dump(payload, path)
            model = StableMeshWakeAero.load(path)
            prediction = model.predict_many(x[48:], mesh[48:])
            np.testing.assert_allclose(prediction[:, 0],
                FineAeroSurrogate(baseline).predict_many(x[48:])[:, 0], atol=1e-12, rtol=0)
            np.testing.assert_allclose(prediction[:, 1],
                estimator.predict(features(x[48:], mesh[48:], config), thread_count=2))
            self.assertTrue(model.requires_mesh)


if __name__ == "__main__":
    unittest.main()
