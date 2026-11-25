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
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 研究_几何交互代理20261017 as study
from 导出_几何交互代理20261017 import portable
from 分目标气动高斯过程 import TargetwiseAeroGP, feature_map, target_mean


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(610)
        cls.x = np.array([26., 24., .45, 1., 0., 8.38, .75])+rng.normal(size=(90, 7))*[
            2, 2, .04, .4, .5, .3, .03]
        cls.y = .8+.03*cls.x[:, 4]-.01*cls.x[:, 3]+.01*np.sin(cls.x[:, 1])
        cls.probe = cls.x[:15]+[.1, .1, .001, .01, .01, .01, .001]

    def test_interaction_features_match_geometry_and_scaling(self):
        f = feature_map(self.x, "interference")
        self.assertEqual(f.shape, (90, 28))
        scaled = self.x.copy()
        scaled[:, 0] *= 9
        scaled[:, 5:7] *= 3
        np.testing.assert_allclose(f[:, 7:], feature_map(scaled, "interference")[:, 7:], rtol=1e-12)
        from 机体参数化设计与气动阻力 import AirframeDesign, design_metrics
        design = AirframeDesign(**dict(zip(study.FEATURES, self.x[0])))
        metrics = design_metrics(design)
        self.assertAlmostEqual(f[0, 17], metrics.horizontal_tail_area_m2/self.x[0, 0])

    def test_catboost_export_and_roundtrip_all_transforms(self):
        for mode, transform in (("raw", "identity"), ("physics", "identity"),
                                ("interference", "identity"), ("interference", "log"),
                                ("interference", "inverse")):
            with self.subTest(mode=mode, transform=transform):
                config = dict(method="cat", mode=mode, transform=transform, iterations=80,
                              depth=4, penalty=3.)
                fitted = study.fit(config, self.x, self.y)
                target = portable(fitted, .05)
                np.testing.assert_allclose(
                    [target_mean(target, row[None, :]) for row in self.probe],
                    study.predict(fitted, self.probe), rtol=1e-10, atol=1e-10)
                payload = dict(schema="ptdl_targetwise_gp_v1", feature_labels=study.FEATURES,
                               baseline_targets=[6., .8], training_samples=len(self.x),
                               targets=[target, target])
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "surrogate.json"
                    TargetwiseAeroGP(payload).to_json(path)
                    restored = TargetwiseAeroGP.from_json(path)
                    np.testing.assert_allclose(restored.predict(
                        study.prior.original.prior.AirframeDesign()), [
                            target_mean(target, np.array([[26., 24.02, .45, 1., 0., 8.38, .75]]))]*2)

    def test_float32_border_behavior_matches_native(self):
        estimator = CatBoostRegressor(iterations=10, depth=3, verbose=False, thread_count=1,
                                       allow_writing_files=False, random_seed=5)
        estimator.fit(self.x, self.y)
        config = dict(method="cat", mode="raw", transform="identity")
        target = portable(dict(config=config, estimator=estimator), .05)
        tree = next(t for t in target["trees"] if t["columns"])
        column, threshold = tree["columns"][0], tree["thresholds"][0]
        probe = np.tile(self.x[0], (4, 1))
        probe[:, column] = [threshold, np.nextafter(threshold, np.inf),
                           float(np.nextafter(np.float32(threshold), np.float32(np.inf))),
                           float(np.nextafter(np.float32(threshold), np.float32(-np.inf)))]
        np.testing.assert_allclose([target_mean(target, row[None, :]) for row in probe],
                                   estimator.predict(probe), rtol=1e-12, atol=1e-12)

    def test_malformed_cat_tree_rejected(self):
        config = dict(method="cat", mode="raw", transform="identity", iterations=15, depth=3, penalty=3.)
        target = portable(study.fit(config, self.x, self.y), .05)
        payload = dict(schema="ptdl_targetwise_gp_v1", feature_labels=study.FEATURES,
                       baseline_targets=[6., .8], training_samples=len(self.x), targets=[target, target])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for mutation in ("bad_column", "bad_leaves", "bad_transform"):
                broken = copy.deepcopy(payload)
                if mutation == "bad_column":
                    broken["targets"][0]["trees"][0]["columns"][0] = .5
                elif mutation == "bad_leaves":
                    broken["targets"][0]["trees"][0]["leaves"].pop()
                else:
                    broken["targets"][0]["transform"] = "clip"
                path.write_text(json.dumps(broken), encoding="utf8")
                with self.assertRaises(ValueError):
                    TargetwiseAeroGP.from_json(path)


if __name__ == "__main__":
    unittest.main()
