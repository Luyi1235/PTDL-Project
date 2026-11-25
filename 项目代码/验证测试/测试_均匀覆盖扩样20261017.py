"""Check acquisition independence and expanded model export."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 推进_均匀覆盖3072代理20261017 as expansion
import 训练_均匀覆盖代理20261017 as train
from 分目标气动高斯过程 import target_mean


class UniformExpansionTests(unittest.TestCase):
    def test_expansion_separates_training_and_new_holdout(self):
        plan = expansion.plan(1536, 8)
        self.assertEqual(len(plan["jobs"]), 32)
        self.assertEqual(sum(j["role"] == "train" for j in plan["jobs"]), 24)
        rows = np.array([[j["design"][k] for k in train.FEATURES] for j in plan["jobs"]])
        self.assertEqual(len(np.unique(rows, axis=0)), len(rows))
        data = pd.read_csv(train.balanced.previous.DATA)[train.FEATURES].to_numpy(float)
        scale = np.ptp(data, axis=0)
        for row, job in zip(rows, plan["jobs"]):
            self.assertGreater(np.linalg.norm((data-row)/scale, axis=1).min(), 1e-5)
            violations = train.balanced.previous.prior.original.prior.geometry_constraints(
                train.balanced.previous.prior.original.prior.AirframeDesign(**job["design"]))
            self.assertLessEqual(max(violations.values()), 1e-9)
        self.assertEqual(plan["wake_nodes"], 8)
        self.assertEqual(plan["protocol"], "free-wake-legacy-v1")
        self.assertFalse(plan["automatic_production_promotion"])

    def test_expanded_candidate_families_export_exactly(self):
        rng = np.random.default_rng(333)
        x = np.array([28., 24., .45, 2., 0., 8.4, .75])+rng.normal(size=(65, 7))*[
            1., 1., .02, .3, .3, .2, .02]
        y = .8+.015*x[:, 4]+.003*np.sin(x[:, 1])
        acquisition = np.array(["uniform_feasible_coverage"]*len(x))
        for config in (
            dict(method="tree", mode="physics", depth=3, penalty=3., weighting="density",
                 loss="RMSE", ordered=True, iterations=15),
            dict(method="boost", mode="interference", leaves=4, penalty=5.,
                 transform="identity", iterations=20),
        ):
            model = train.fit(config, x, y, acquisition)
            exported = train.portable(model, .05)
            probe = (x[:8]+x[8:16])/2
            np.testing.assert_allclose(
                [target_mean(exported, row[None, :]) for row in probe],
                train.predict(model, probe), rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
