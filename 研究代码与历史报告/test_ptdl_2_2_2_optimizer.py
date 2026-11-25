"""Regression tests for the PTDL 2.2.2 optimization prototype."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ptdl_2_2_2_optimizer import (
    CoupledPTDLControlProblem,
    LOWER_BOUNDS,
    MOEAD,
    UPPER_BOUNDS,
    OBJECTIVE_LABELS,
    PTDLControlProblem,
    pareto_indices,
    run_coupled_moead_baseline,
    run_delivery_experiment,
    run_ptdl_80h_joint_moead,
    run_source_profile_moead_audit,
    simulate_surrogate_mission,
)


class SurrogateAndOptimizerTests(unittest.TestCase):
    def test_surrogate_profile_matches_the_2_2_1_evaluator_contract(self) -> None:
        decision = 0.5 * (LOWER_BOUNDS + UPPER_BOUNDS)
        profile, response = simulate_surrogate_mission(decision)
        result = PTDLControlProblem().evaluate(decision).result

        required = {
            "time_s", "fuel_flow_kgps", "battery_discharge_kw", "battery_charge_kw",
            "soc", "h_m", "v_mps", "p_engine_kw", "p_motor_kw", "e_h_m",
            "e_v_mps", "e_theta_rad", "q_radps",
        }
        self.assertTrue(required.issubset(profile.columns))
        self.assertTrue(np.all(np.diff(profile["time_s"].to_numpy()) > 0.0))
        self.assertTrue(all(np.isfinite(value) for value in result.objectives.values()))
        self.assertEqual(set(response), {"settling_time_s", "recovery_time_s", "mode_switch_time_s"})

    def test_delivery_experiment_exports_complete_result_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "delivery"
            delivery = run_delivery_experiment(
                output_dir,
                seed=20260831,
                nsga_population=18,
                nsga_generations=8,
                moead_resolution=3,
                moead_generations=12,
            )
            expected = {
                "nsga_ii_convergence.csv",
                "moea_d_convergence.csv",
                "nsga_ii_pareto_candidates.csv",
                "moea_d_pareto_candidates.csv",
                "algorithm_comparison.csv",
                "representative_solutions.csv",
                "candidate_control_sequences.csv",
                "candidate_state_trajectories.csv",
                "convergence_feasibility.svg",
                "convergence_feasibility.png",
                "run_metadata.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output_dir.iterdir()}))

            comparison = pd.read_csv(output_dir / "algorithm_comparison.csv")
            representatives = pd.read_csv(output_dir / "representative_solutions.csv")
            self.assertEqual(set(comparison["algorithm"]), {"NSGA-II", "MOEA/D"})
            self.assertTrue((comparison["feasible_count"] > 0).all())
            self.assertEqual(set(representatives["preference"]), {
                "energy_priority", "stability_priority", "response_priority", "balanced",
            })
            self.assertTrue(set(OBJECTIVE_LABELS).issubset(representatives.columns))
            self.assertEqual(len(delivery["runs"]), 2)

    def test_final_populations_return_feasible_pareto_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            delivery = run_delivery_experiment(
                Path(directory) / "delivery",
                seed=20260831,
                nsga_population=18,
                nsga_generations=6,
                moead_resolution=3,
                moead_generations=10,
            )
            for run in delivery["runs"]:
                indices = pareto_indices(run.evaluations)
                self.assertGreater(len(indices), 0)
                for index in indices:
                    candidate = run.evaluations[index]
                    self.assertTrue(candidate.feasible)
                    self.assertLessEqual(candidate.result.constraints["endurance_80h"], 0.0)

    def test_coupled_moead_exports_a_feasibility_audit_without_claiming_80_hour_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "coupled"
            delivery = run_coupled_moead_baseline(output_dir, resolution=2, generations=2, seed=20260904)

            self.assertEqual(delivery["run"].algorithm, "MOEA/D")
            self.assertEqual(len(delivery["run"].evaluations), 6)
            self.assertTrue((output_dir / "moea_d_final_population.csv").exists())
            self.assertTrue((output_dir / "least_violating_candidate.csv").exists())
            self.assertTrue((output_dir / "least_violating_time_history.csv").exists())
            self.assertTrue((output_dir / "moea_d_feasible_pareto.csv").exists())
            self.assertGreater(delivery["least_violating"].result.constraints["endurance_80h"], 0.0)
            self.assertFalse(delivery["least_violating"].feasible)

    def test_coupled_problem_uses_eight_physical_phase_parameters(self) -> None:
        problem = CoupledPTDLControlProblem()
        candidate = problem.evaluate(0.5 * (problem.lower_bounds + problem.upper_bounds))

        self.assertEqual(len(problem.decision_labels), 8)
        self.assertEqual(candidate.decision.shape, (8,))
        self.assertIn("endurance_80h", candidate.result.constraints)
        self.assertTrue(all(np.isfinite(value) for value in candidate.result.objectives.values()))

    def test_source_profile_moead_exports_an_80h_feasibility_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "source_profile"
            delivery = run_source_profile_moead_audit(output_dir, resolution=1, generations=1, seed=20260905)
            self.assertEqual(delivery["run"].algorithm, "MOEA/D")
            self.assertEqual(len(delivery["run"].evaluations), 3)
            self.assertTrue((output_dir / "source_profile_moead_final_population.csv").exists())
            self.assertTrue((output_dir / "source_profile_moead_least_violating.csv").exists())
            self.assertTrue((output_dir / "source_profile_moead_least_violating_time_history.csv").exists())
            self.assertGreater(delivery["least_violating"].result.constraints["endurance_80h"], 0.0)

    def test_joint_endurance_moead_exports_dimension_and_control_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "joint_endurance"
            delivery = run_ptdl_80h_joint_moead(output_dir, resolution=1, generations=1, seed=20260906)
            self.assertEqual(delivery["run"].algorithm, "MOEA/D")
            self.assertEqual(len(delivery["run"].evaluations), 3)
            self.assertEqual(len(delivery["problem"].decision_labels), 23)
            self.assertTrue((output_dir / "ptdl_80h_joint_moead_final_population.csv").exists())
            self.assertTrue((output_dir / "ptdl_80h_joint_moead_least_violating.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
