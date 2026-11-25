"""Check weighting and residual fitting without accessing external test labels."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 研究_密度均衡与残差代理20261017 as study
from 导出_密度均衡与残差代理20261017 import portable
from 分目标气动高斯过程 import TargetwiseAeroGP, target_mean


class BalancedResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(331)
        cls.x = np.array([28., 24., .45, 2., 0., 8.4, .75])+rng.normal(size=(54, 7))*[
            1., 1., .025, .3, .2, .2, .02]
        cls.y = .75+.02*cls.x[:, 4]+.004*np.sin(cls.x[:, 1])+.01*cls.x[:, 3]
        cls.acquisition = np.array(["space_filling"]*40+["gp_uncertainty"]*14)

    def test_weights_keep_all_samples_and_use_geometry_scale(self):
        for mode in ("none", "density", "acquisition"):
            w = study.sample_weights(self.x, self.acquisition, mode)
            self.assertTrue(np.isfinite(w).all())
            self.assertTrue((w > 0).all())
            self.assertAlmostEqual(w.mean(), 1.)
        scaled = self.x*np.array([1e4, 1., .1, 100., 5., 1e3, 1e3])
        np.testing.assert_allclose(
            study.sample_weights(self.x, self.acquisition, "density"),
            study.sample_weights(scaled, self.acquisition, "density"), rtol=1e-11)
        w = study.sample_weights(self.x, self.acquisition, "acquisition")
        self.assertAlmostEqual(w[-1]/w[0], .35)

    def test_ordered_and_robust_models_predict(self):
        for ordered, loss in ((True, "RMSE"), (False, "Huber")):
            config = dict(method="tree", mode="physics", depth=3, penalty=5.,
                          weighting="density", loss=loss, ordered=ordered, iterations=15)
            fitted = study.fit(config, self.x, self.y, self.acquisition, study.TARGETS[1])
            self.assertTrue(np.isfinite(study.predict(fitted, self.x[:3])).all())

    def test_correction_labels_are_inner_out_of_fold_errors(self):
        config = dict(method="residual", mode="dimensionless", kernel="matern32",
                      shrinkage=.35, inner_folds=3, iterations=12)
        fitted = study.fit(config, self.x, self.y, self.acquisition, study.TARGETS[1])
        a, b = next(KFold(3, shuffle=True, random_state=2026101729).split(self.x))
        base_config = copy.deepcopy(fitted["base"]["config"])
        independent = study.fit_tree(base_config, self.x[a], self.y[a], self.acquisition[a])
        np.testing.assert_allclose(fitted["inner_oof"][b], study.tree_predict(independent, self.x[b]))
        residual = self.y-fitted["inner_oof"]
        expected = (residual-fitted["estimator"]._y_train_mean)/fitted["estimator"]._y_train_std
        np.testing.assert_allclose(fitted["estimator"].y_train_, expected)
        self.assertTrue(np.isfinite(study.predict(fitted, self.x[:4])).all())

    def test_residual_export_and_validation(self):
        config = dict(method="residual", mode="physics", kernel="rbf",
                      shrinkage=.35, inner_folds=3, iterations=12)
        fitted = study.fit(config, self.x, self.y, self.acquisition, study.TARGETS[1])
        target = portable(fitted, .05)
        probe = (self.x[:8]+self.x[8:16])/2
        np.testing.assert_allclose(
            [target_mean(target, row[None, :]) for row in probe],
            study.predict(fitted, probe), rtol=1e-9, atol=1e-9)
        payload = dict(schema="ptdl_targetwise_gp_v1", feature_labels=study.FEATURES,
                       baseline_targets=[6., .8], training_samples=len(self.x), targets=[target, target])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf8")
            TargetwiseAeroGP.from_json(path)
            for invalid in (-.1, 0., 1.2, float("nan")):
                broken = copy.deepcopy(payload)
                broken["targets"][0]["shrinkage"] = invalid
                path.write_text(json.dumps(broken), encoding="utf8")
                with self.assertRaises(ValueError):
                    TargetwiseAeroGP.from_json(path)


if __name__ == "__main__":
    unittest.main()
