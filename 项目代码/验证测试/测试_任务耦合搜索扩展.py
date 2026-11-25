import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "验证测试")]
from PTDL耦合代理辅助优化 import PTDLCoupledSurrogateSearch
from PTDL任务耦合自适应优化 import PTDLTaskAdaptiveSearch
from 测试_20261015算法与响应 import ToyProblem


class TaskSearchTests(unittest.TestCase):
    def test_learned_search_budget_bounds_seed_and_archives(self):
        for policy in ("credit", "credit-prior", "local-linear"):
            def run():
                return PTDLCoupledSurrogateSearch(
                    resolution=2, generations=5, seed=103, use_surrogate=False,
                    candidate_pool=8, proposal_policy=policy).run(ToyProblem())
            first, repeated = run(), run()
            self.assertEqual(len(first.evaluated_archive), 36)
            self.assertEqual(set(first.generation_pareto_archive), set(range(6)))
            np.testing.assert_array_equal(first.decisions, repeated.decisions)
            self.assertTrue(np.all(first.decisions[:, 2] == 0))
            for evaluation in first.evaluated_archive:
                self.assertTrue(np.all(evaluation.decision >= ToyProblem.lower_bounds))
                self.assertTrue(np.all(evaluation.decision <= ToyProblem.upper_bounds))

    def test_persistent_particle_search_same_initial_candidates(self):
        for adaptive in (True, False):
            run = PTDLTaskAdaptiveSearch(resolution=2, generations=4, seed=9,
                                         adaptive=adaptive).run(ToyProblem())
            baseline = PTDLCoupledSurrogateSearch(resolution=2, generations=4, seed=9,
                                                 use_surrogate=False).run(ToyProblem())
            self.assertEqual(len(run.evaluated_archive), 30)
            np.testing.assert_array_equal(
                np.vstack([e.decision for e in run.evaluated_archive[:6]]),
                np.vstack([e.decision for e in baseline.evaluated_archive[:6]]))

    def test_invalid_combination_rejected(self):
        with self.assertRaises(ValueError):
            PTDLCoupledSurrogateSearch(use_surrogate=True, proposal_policy="credit")

    def test_integrated_dedup_matches_research_wrapper(self):
        from 耦合候选去重 import UniqueCoupledProblem
        options = dict(resolution=2, generations=5, seed=37, use_surrogate=False,
                       proposal_policy="credit-prior", candidate_pool=8)
        integrated = PTDLCoupledSurrogateSearch(**options, deduplicate=True).run(ToyProblem())
        wrapped = PTDLCoupledSurrogateSearch(**options).run(
            UniqueCoupledProblem(ToyProblem(), 37, 6))
        self.assertEqual(len(integrated.evaluated_archive), 36)
        np.testing.assert_array_equal(
            np.vstack([e.decision for e in integrated.evaluated_archive]),
            np.vstack([e.decision for e in wrapped.evaluated_archive]))


if __name__ == "__main__":
    unittest.main()
