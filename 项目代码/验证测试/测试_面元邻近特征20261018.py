from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 气动面元邻近特征 import panel_proximity, FEATURE_NAMES


class PanelProximityTests(unittest.TestCase):
    def mesh(self, path, scale=1., translation=0.):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                           [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
        points = scale*points+translation
        lines = ["# vspgeom v3", "1", "8 2 2"]
        lines += [" ".join(map(str, p)) for p in points]
        lines += ["2", "4 1 2 3 4", "4 5 6 7 8", "1", "2"]
        path.write_text("\n".join(lines), encoding="utf8")

    def test_features_are_translation_and_scale_invariant(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mesh.vspgeom"
            self.mesh(path)
            first = panel_proximity(path, 30., 24.)
            self.mesh(path, scale=2., translation=7.)
            second = panel_proximity(path, 120., 24.)
            np.testing.assert_allclose(first, second, atol=1e-12)
            self.assertEqual(len(first), len(FEATURE_NAMES))
            self.assertEqual(first[0], 1.)
            self.assertAlmostEqual(first[FEATURE_NAMES.index("pair_1_2_distance_min")],
                                   1/np.sqrt(30/24))
            self.assertAlmostEqual(first[FEATURE_NAMES.index("pair_1_2_normal_alignment_near")], 1.)
            self.assertTrue(np.isfinite(first).all())

    def test_invalid_connectivity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mesh.vspgeom"
            self.mesh(path)
            path.write_text(path.read_text().replace("4 1 2 3 4", "3 1 2 3 4"))
            with self.assertRaises(ValueError):
                panel_proximity(path, 30., 24.)


if __name__ == "__main__":
    unittest.main()
