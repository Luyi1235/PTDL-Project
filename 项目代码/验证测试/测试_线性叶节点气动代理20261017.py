from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 研究_线性叶节点代理20261017 as study
from 线性叶节点气动代理 import predict_trees


class LinearLeafTests(unittest.TestCase):
    def test_native_export_constant_and_linear_leaves(self):
        rng = np.random.default_rng(47)
        x = rng.uniform([24., 19., .32, 0., -3., 7.6, .66],
                        [39., 30., .65, 7., 1., 9.8, .91], (130, 7))
        y = .7 + .1*x[:, 2] + .03*np.sin(x[:, 3])
        probe = np.vstack((x[:20], (x[20:40]+x[40:60])/2))
        for mode in ("raw", "interference"):
            for linear in (False, True):
                config = dict(study.CONFIGS[0], iterations=20, mode=mode, linear=linear)
                model = study.fit(config, x, y)
                np.testing.assert_allclose(predict_trees(study.portable(model), probe),
                                           study.predict(model, probe), atol=1e-10, rtol=1e-10)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "中文模型.txt"
                    path.write_text(model["estimator"].booster_.model_to_string(), encoding="utf8")
                    restored = study.lgb.Booster(model_str=path.read_text(encoding="utf8"))
                    f = (study.features(probe, mode)-model["center"])/model["scale"]
                    np.testing.assert_allclose(restored.predict(f, num_threads=1),
                                               study.predict(model, probe), atol=1e-10, rtol=1e-10)

    def test_explicit_linear_leaf_and_split_boundary(self):
        leaves = dict(features="raw", center=[0.]*7, scale=[1.]*7,
                      trees=[dict(split_feature=0, threshold=30., decision_type="<=",
                                  left_child=dict(leaf_value=0., leaf_const=.2,
                                                  leaf_features=[0], leaf_coeff=[.01]),
                                  right_child=dict(leaf_value=.8))])
        x = np.array([[30., 24., .5, 3., -1., 8., .8],
                      [31., 24., .5, 3., -1., 8., .8]])
        np.testing.assert_allclose(predict_trees(leaves, x), [.5, .8])


if __name__ == "__main__":
    unittest.main()
