import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
from 分目标气动高斯过程 import TargetwiseAeroGP, feature_map, kernel_cross
from 多精度气动高斯过程 import fit_autoregressive, autoregressive_prediction
from 机体参数化设计与气动阻力 import AirframeDesign
from VSPAERO气动代理模型 import FEATURE_LABELS
from 运行_分目标气动代理攻关 import fit_target
from 训练_主动增量混合代理20261016 import portable_target_prediction


class MixedAerodynamicTests(unittest.TestCase):
    def test_saved_models_equal_independent_training_predictor(self):
        frame = pd.read_csv(ROOT / "运行结果" / "主动增量代理_20261015" / "合并留出验证.csv")
        x = frame[list(FEATURE_LABELS)].to_numpy(float)
        for path in (
            ROOT / "运行结果" / "主动增量代理_20261015" / "增量构型气动代理.json",
            ROOT / "运行结果" / "物理趋势残差代理_20261016" / "物理趋势候选代理.json",
        ):
            model = TargetwiseAeroGP.from_json(path)
            expected = np.column_stack([portable_target_prediction(t, x)
                                        for t in model.payload["targets"]])
            actual = np.array([model.predict(AirframeDesign(**dict(zip(FEATURE_LABELS, row))))
                               for row in x])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            for row, means in zip(x[:3], actual[:3]):
                uncertain = model.predict_with_uncertainty(AirframeDesign(**dict(zip(FEATURE_LABELS, row))))
                np.testing.assert_allclose(uncertain[:2], means, rtol=1e-12, atol=1e-12)
                self.assertTrue(np.all(np.isfinite(uncertain)))
                self.assertTrue(np.all(np.array(uncertain[2:]) > 0))

    def test_invalid_ensemble_weights_and_tree_cycle_rejected(self):
        original = json.loads((ROOT / "运行结果" / "主动增量代理_20261015" /
                               "增量构型气动代理.json").read_text(encoding="utf8"))
        for alteration in ("weights", "tree"):
            payload = copy.deepcopy(original)
            if alteration == "weights":
                payload["targets"][0]["weights"][0] = -1.
            else:
                boosting = next(t for t in payload["targets"][0]["models"]
                                if t.get("kind") == "hist-boosting")
                boosting["trees"][0]["left"][0] = 0
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(payload), encoding="utf8")
                with self.assertRaises(ValueError):
                    TargetwiseAeroGP.from_json(path)

    def test_conditional_cokriging_matches_joint_gaussian_solution(self):
        rng = np.random.default_rng(23)
        lo = np.array([25., 20., .3, -2., -2., 7., .7])
        hi = np.array([35., 30., .7, 6., 1., 10., .9])
        low_x, high_x, query = (rng.uniform(lo, hi, size=(n, 7)) for n in (35, 9, 3))

        def low_truth(x):
            return 6+.15*np.sin(x[:, 1]/4)+.08*x[:, 4]+.2*x[:, 6]

        low_y = low_truth(low_x)
        high_y = .95*low_truth(high_x)+.02*high_x[:, 3]
        low = fit_target(low_x, low_y, "physics", "matern32")[1]
        model = fit_autoregressive(low, high_x, high_y)
        zl = np.asarray(low["train"])
        zh = np.asarray(model["high_z"])
        zq = (feature_map(query, "physics")-low["feature_mean"])/low["feature_scale"]
        kl = lambda a, b: kernel_cross(a, b, np.asarray(low["length"]), low["amplitude"], low["kernel"])
        kd = lambda a, b: kernel_cross(a, b, np.asarray(model["length"]), model["amplitude"], model["kernel"])
        rho = model["rho"]
        ll = kl(zl, zl)+np.eye(len(zl))*(low["noise"]+1e-8)
        lh = rho*kl(zl, zh)
        hh = rho**2*kl(zh, zh)+kd(zh, zh)+np.eye(len(zh))*(model["noise"]+1e-8)
        joint = np.block([[ll, lh], [lh.T, hh]])
        observations = np.r_[(low_y-low["mean"])/low["scale"],
                             (high_y-low["mean"])/low["scale"]-model["bias"]]
        for x, z in zip(query, zq):
            cross = np.r_[rho*kl(z[None, :], zl)[0],
                           (rho**2*kl(z[None, :], zh)+kd(z[None, :], zh))[0]]
            expected_mean = low["mean"]+low["scale"]*(model["bias"]+cross@np.linalg.solve(joint, observations))
            expected_var = rho**2*low["amplitude"]+model["amplitude"]+model["noise"]+1e-8
            expected_var -= cross@np.linalg.solve(joint, cross)
            mean, std = autoregressive_prediction(model, x[None, :])
            self.assertAlmostEqual(mean, expected_mean, places=6)
            self.assertAlmostEqual(std, low["scale"]*np.sqrt(max(0., expected_var)), places=6)
            self.assertAlmostEqual(mean, autoregressive_prediction(model, x[None, :], False), places=12)


if __name__ == "__main__":
    unittest.main()
