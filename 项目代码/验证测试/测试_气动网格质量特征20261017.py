"""Check geometry-only descriptors against stored native mesh files."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 气动网格质量特征 import FEATURE_NAMES, mesh_quality


class MeshQualityTests(unittest.TestCase):
    def test_native_counts_and_panel_partition(self):
        path = ROOT / "运行结果/千构型原定义代理_20261017/solver/stable_train_0821/case/airframe.vspgeom"
        values = mesh_quality(path, 30., 25.)
        self.assertEqual(values.shape, (len(FEATURE_NAMES),))
        np.testing.assert_array_equal(values[:3], [834, 736, 4])
        self.assertEqual(sum(values[5::8]), 736)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values[5::8] > 0).all())

    def test_topology_change_is_visible(self):
        first = ROOT / "运行结果/千构型原定义代理_20261017/solver/stable_train_0821/case/airframe.vspgeom"
        second = ROOT / "运行结果/微小几何扰动_20261017/solver/local_03_027/case/airframe.vspgeom"
        a, b = mesh_quality(first, 30., 25.), mesh_quality(second, 30., 25.)
        np.testing.assert_array_equal(b[:3], [835, 736, 4])
        self.assertGreater(np.max(np.abs(a-b)), .5)


if __name__ == "__main__":
    unittest.main()
