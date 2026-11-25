"""Budget, block ablation, portable physics GP, and event-boundary regression."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from MOEAD多目标联合优化 import CandidateEvaluation
from PTDL分块自适应优化 import PTDLBlockAdaptive
from 指标体系与评价函数 import EvaluationResult, _add_rate_constraint
from 分目标气动高斯过程 import feature_map
from 飞推控闭环物理模型 import (
    AircraftParameters, PhaseCommand, MissionDefinition,
    LongitudinalDragDisturbance, simulate_mission,
)


class ToyProblem:
    lower_bounds = np.zeros(3)
    upper_bounds = np.array([1., 1., 0.])
    decision_labels = ("geometry", "power", "fixed")
    design_labels = ("geometry",)

    def sample(self, n, rng):
        return rng.uniform(self.lower_bounds, self.upper_bounds, size=(n, 3))

    def evaluate(self, x):
        if x[2] != 0 or np.any(x < self.lower_bounds) or np.any(x > self.upper_bounds):
            raise AssertionError("Invalid design")
        return CandidateEvaluation(np.asarray(x), EvaluationResult(
            objectives=dict(J_energy_mj=float(x[0]**2 + x[1]**2),
                            J_stability=float((x[0]-1)**2),
                            J_response_s=float((x[1]-.7)**2)),
            constraints=dict(capacity=float(.4-x.sum())), diagnostics={}))


class MethodIncrementTests(unittest.TestCase):
    def test_rate_roundoff_cannot_turn_a_saturated_actuator_infeasible(self):
        import pandas as pd
        constraints = {}
        _add_rate_constraint(constraints, "rate", pd.Series([0., 10.+1e-14]),
                             np.array([0., 1.]), 10.)
        self.assertEqual(constraints["rate_max"], 0.)
        _add_rate_constraint(constraints, "rate", pd.Series([0., 10.+1e-5]),
                             np.array([0., 1.]), 10.)
        self.assertGreater(constraints["rate_max"], 0.)
    def test_all_ablation_variants_keep_exact_budget_and_fixed_decisions(self):
        for block, credit in ((True, True), (False, True), (True, False)):
            run = PTDLBlockAdaptive(resolution=2, generations=4, seed=41,
                                    block_search=block, adaptive_credit=credit).run(ToyProblem())
            self.assertEqual(len(run.evaluated_archive), 6*5)
            self.assertEqual(len(run.evaluations), 6)
            self.assertTrue(np.all(run.decisions[:, 2] == 0))
            self.assertEqual(set(run.generation_pareto_archive), set(range(5)))
            probabilities = run.history.filter(regex="probability").dropna().to_numpy()
            np.testing.assert_allclose(probabilities.sum(1), 1.)
            self.assertTrue(np.all(probabilities >= .1))

    def test_algorithm_repeatable_from_seed(self):
        a = PTDLBlockAdaptive(resolution=2, generations=3, seed=4).run(ToyProblem())
        b = PTDLBlockAdaptive(resolution=2, generations=3, seed=4).run(ToyProblem())
        np.testing.assert_array_equal(a.decisions, b.decisions)

    def test_physics_features_invariant_under_uniform_length_scaling(self):
        x = np.array([[26, 24, .45, 1, 0, 8.38, .75]], dtype=float)
        y = x.copy()
        y[:, 0] *= 4
        y[:, 5:7] *= 2
        np.testing.assert_allclose(feature_map(x, "physics"), feature_map(y, "physics"))
        self.assertEqual(feature_map(x, "physics").shape, (1, 9))

    def test_wind_edges_preserved_when_not_on_nominal_time_grid(self):
        event = LongitudinalDragDisturbance(
            phase_name="cruise", start_time_s=32.5, duration_s=7.3,
            additional_drag_n=100., response_step_s=2., recovery_window_s=20.)
        mission = MissionDefinition(
            (PhaseCommand("cruise", 100., 1000., 40., 90., 0., time_step_s=20.),),
            initial_altitude_m=1000., initial_speed_mps=40., disturbances=(event,))
        f = simulate_mission(mission, AircraftParameters()).profile
        times = f.time_s.to_numpy()
        self.assertLess(np.min(abs(times-32.5)), 1e-8)
        self.assertLess(np.min(abs(times-39.8)), 1e-8)
        active = f[f.additional_drag_n > 0] if "additional_drag_n" in f else None
        if active is not None and "dt_s" in active:
            self.assertAlmostEqual(active.dt_s.sum(), 7.3)


if __name__ == "__main__":
    unittest.main()
