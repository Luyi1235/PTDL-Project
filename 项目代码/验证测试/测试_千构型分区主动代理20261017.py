import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 推进_千构型主动补样20261017 as active
from 分目标气动高斯过程 import TargetwiseAeroGP, target_mean
from 机体参数化设计与气动阻力 import AirframeDesign


class ActiveSurrogateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(55)
        cls.x = np.array([26., 24., .45, 1., 0., 8.38, .75])+rng.normal(size=(75, 7))*[
            2., 2., .04, .3, .8, .4, .03]
        cls.probe = cls.x[:6]+[.3, .2, .01, .02, .03, .04, .001]
        cls.polar = np.zeros((len(cls.x), 5, 4))
        cl = .105*active.direct.ANGLES[None, :]+.06*cls.x[:, 4:5]
        cls.polar[:, :, 0] = cl
        cls.polar[:, :, 1] = cl*.99
        cls.polar[:, :, 2] = cl**2/(np.pi*cls.x[:, 1:2]*.9)+.0001
        cls.polar[:, :, 3] = cl**2/(np.pi*cls.x[:, 1:2]*cls.polar[:, :, 2])
        cls.y = np.median(cls.polar[:, [0, 3, 4], 3], axis=1)

    def test_portable_models_match_unseen_sklearn_predictions(self):
        specs = [
            dict(family="direct", settings=dict(method="boost", mode="physics", leaves=7, penalty=1.)),
            dict(family="physical", settings=dict(method="gp_restart", mode="physics",
                                                  kernel="matern52", noise="learn")),
            dict(family="physical", settings=dict(method="polar", mode="physics", leaves=7,
                                                  penalty=1., reconstruction="polynomial")),
            dict(family="physical", settings=dict(method="polar", mode="physics", leaves=7,
                                                  penalty=1., reconstruction="pointwise")),
            dict(family="regime", settings=dict(method="regime", mode="physics", expert="gp", blend=.5)),
        ]
        for spec in specs:
            with self.subTest(method=spec["settings"]["method"]):
                fitted = active.fit(spec, self.x, self.y, self.polar)
                portable = active.portable_scalar(fitted, .1)
                got = [target_mean(portable, row[None, :]) for row in self.probe]
                np.testing.assert_allclose(got, active.predict(spec, fitted, self.probe),
                                           rtol=1e-9, atol=1e-9)
                payload = dict(schema="ptdl_targetwise_gp_v1", feature_labels=active.direct.FEATURES,
                               training_samples=len(self.x), baseline_targets=[6., .8],
                               targets=[portable, portable])
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.json"
                    TargetwiseAeroGP(payload).to_json(path)
                    restored = TargetwiseAeroGP.from_json(path)
                    self.assertTrue(np.isfinite(restored.predict_with_uncertainty(AirframeDesign())).all())

    def test_missing_and_duplicate_oof_rows_rejected(self):
        records = [dict(config=0, split=s, validation=[0, 1], prediction=[1., 2.])
                   for s in ("random", "spatial")]
        np.testing.assert_allclose(active.records_matrix(records, 2, 0), [[1, 2], [1, 2]])
        with self.assertRaises(RuntimeError):
            active.records_matrix(records+[records[0]], 2, 0)
        with self.assertRaises(RuntimeError):
            active.records_matrix(records, 3, 0)

    def test_mixture_uses_both_validation_distributions(self):
        y = np.array([0., 1., 2.])
        predictions = [np.vstack([y, y+2]), np.vstack([y+2, y])]
        score, components, prediction = active.select_mixture(predictions, y)
        self.assertEqual(len(components), 1)
        self.assertEqual(score, 1.)
        # A complementary-error mixture should improve both distributions.
        predictions = [np.vstack([y+1, y+1]), np.vstack([y-1, y-1])]
        score, components, prediction = active.select_mixture(predictions, y)
        self.assertEqual(score, 0.)
        self.assertEqual(components, [(0, .5), (1, .5)])

    def test_regime_gate_shape_and_range_checked(self):
        spec = dict(family="regime", settings=dict(method="regime", mode="physics", expert="boost", blend=.5))
        portable = active.portable_scalar(active.fit(spec, self.x, self.y, self.polar), .1)
        payload = dict(schema="ptdl_targetwise_gp_v1", feature_labels=active.direct.FEATURES,
                       training_samples=len(self.x), baseline_targets=[6., .8], targets=[portable, portable])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            broken = copy.deepcopy(payload)
            broken["targets"][0]["gate"]["dual"][0].pop()
            path.write_text(json.dumps(broken), encoding="utf8")
            with self.assertRaises(ValueError):
                TargetwiseAeroGP.from_json(path)
            broken = copy.deepcopy(payload)
            broken["targets"][0]["blend"] = 1.1
            path.write_text(json.dumps(broken), encoding="utf8")
            with self.assertRaises(ValueError):
                TargetwiseAeroGP.from_json(path)


if __name__ == "__main__":
    unittest.main()
