import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
sys.path.insert(0, str(ROOT / "验证测试"))
from PTDL耦合代理辅助优化 import PTDLCoupledSurrogateSearch
from 测试_20261015算法与响应 import ToyProblem


class CoupledSearchTests(unittest.TestCase):
    def test_counted_budget_fixed_bounds_and_complete_generation_archives(self):
        for assisted in (True, False):
            run = PTDLCoupledSurrogateSearch(resolution=2, generations=5, seed=7,
                                             candidate_pool=8, use_surrogate=assisted).run(ToyProblem())
            self.assertEqual(len(run.evaluated_archive), 36)
            self.assertEqual(len(run.evaluations), 6)
            self.assertEqual(set(run.generation_pareto_archive), {0, 1, 2, 3, 4, 5})
            self.assertTrue(np.all(run.decisions[:, 2] == 0))
            self.assertEqual(run.history.iloc[-1].surrogate_screen_active, float(assisted))
            for snapshot in run.generation_pareto_archive.values():
                self.assertTrue(all(any(e is saved for saved in run.evaluated_archive) for e in snapshot))

    def test_identical_seed_reproduces_online_screening(self):
        a = PTDLCoupledSurrogateSearch(resolution=2, generations=5, seed=11, candidate_pool=8).run(ToyProblem())
        b = PTDLCoupledSurrogateSearch(resolution=2, generations=5, seed=11, candidate_pool=8).run(ToyProblem())
        np.testing.assert_array_equal(a.decisions, b.decisions)


if __name__ == "__main__":
    unittest.main()
