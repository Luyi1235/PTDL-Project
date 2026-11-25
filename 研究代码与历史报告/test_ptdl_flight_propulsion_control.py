"""Regression tests for the reconstructed PTDL coupled reference model."""

from __future__ import annotations

import unittest

import numpy as np

from ptdl_flight_propulsion_control import (
    evaluate_coupled_mission,
    reference_endurance_mission,
    simulate_mission,
)


class CoupledModelTests(unittest.TestCase):
    def test_short_reference_mission_has_valid_profile_and_phase_closure(self) -> None:
        simulation = simulate_mission(reference_endurance_mission(loiter_duration_h=2.0))
        profile = simulation.profile
        required = {
            "time_s", "fuel_flow_kgps", "battery_discharge_kw", "battery_charge_kw",
            "soc", "h_m", "v_mps", "p_engine_kw", "p_motor_kw", "e_h_m",
            "e_v_mps", "e_theta_rad", "q_radps",
        }
        endpoints = profile.groupby("phase").last()

        self.assertEqual(simulation.termination_reason, "mission_complete")
        self.assertTrue(required.issubset(profile.columns))
        self.assertTrue(np.all(np.diff(profile["time_s"].to_numpy()) > 0.0))
        self.assertTrue(np.isfinite(profile.loc[:, list(required)].to_numpy(dtype=float)).all())
        self.assertLess(abs(endpoints.loc["main_climb", "h_m"] - 7600.0), 5.0)
        self.assertLess(abs(endpoints.loc["endurance_loiter", "v_mps"] - 70.0), 0.1)
        self.assertLess(abs(endpoints.loc["approach", "h_m"] - 30.0), 5.0)

    def test_reference_parameters_do_not_artificially_pass_80_hour_requirement(self) -> None:
        simulation, evaluation = evaluate_coupled_mission(reference_endurance_mission(loiter_duration_h=80.0))

        self.assertIn(simulation.termination_reason, {"fuel_depleted", "battery_soc_safe_limit"})
        self.assertLess(float(simulation.profile["time_s"].iloc[-1]) / 3600.0, 80.0)
        self.assertGreater(evaluation.constraints["endurance_80h"], 0.0)
        self.assertFalse(evaluation.feasible)

    def test_coupled_model_is_compatible_with_2_2_1_evaluator(self) -> None:
        simulation, evaluation = evaluate_coupled_mission(reference_endurance_mission(loiter_duration_h=2.0))

        self.assertEqual(set(evaluation.objectives), {"J_energy_mj", "J_stability", "J_response_s"})
        self.assertIn("endurance_80h", evaluation.constraints)
        self.assertIn("terminal_soc", evaluation.constraints)
        self.assertGreater(simulation.range_m, 0.0)
        self.assertTrue(all(np.isfinite(value) for value in evaluation.objectives.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
