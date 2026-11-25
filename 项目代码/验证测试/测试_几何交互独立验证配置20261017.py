"""Ensure the running independent batch has frozen models and unseen geometries."""

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 研究_几何交互代理20261017 as study


class FrozenValidationTests(unittest.TestCase):
    def test_unseen_geometries(self):
        frozen = json.loads((ROOT / "运行结果/几何交互独立验证_20261017/冻结配置.json").read_text(
            encoding="utf8"))
        x = np.array([[job["design"][key] for key in study.FEATURES] for job in frozen["jobs"]])
        train = pd.read_csv(study.DATA)[study.FEATURES].to_numpy(float)
        self.assertEqual(x.shape, (128, 7))
        self.assertEqual(len(np.unique(x, axis=0)), 128)
        scale = np.ptp(train, axis=0)
        for folder in (study.prior.SOURCE, study.prior.OUT):
            history = pd.read_csv(folder / "中网格独立测试标签.csv")[study.FEATURES].to_numpy(float)
            for row in x:
                self.assertGreater(np.linalg.norm((history-row)/scale, axis=1).min(), 1e-5)
        for row, job in zip(x, frozen["jobs"]):
            self.assertGreater(np.linalg.norm((train-row)/scale, axis=1).min(), 1e-5)
            violation = study.prior.original.prior.geometry_constraints(
                study.prior.original.prior.AirframeDesign(**job["design"]))
            self.assertLessEqual(max(violation.values()), 1e-9)

    def test_frozen_model_and_relative_baseline(self):
        frozen = json.loads((ROOT / "运行结果/几何交互独立验证_20261017/冻结配置.json").read_text(
            encoding="utf8"))
        self.assertTrue(frozen["model_selection_frozen_before_test"])
        self.assertFalse(frozen["automatic_production_promotion"])
        for name, path in frozen["models"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                             frozen["model_hashes"][name])
        new = json.loads(Path(frozen["models"]["geometry_interaction"]).read_text(encoding="utf8"))
        previous = json.loads(Path(frozen["models"]["previous1512"]).read_text(encoding="utf8"))
        np.testing.assert_array_equal(new["baseline_targets"], previous["baseline_targets"])


if __name__ == "__main__":
    unittest.main()
