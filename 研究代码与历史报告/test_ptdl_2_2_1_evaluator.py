"""Automated verification for the PTDL 2.2.1 evaluator.

The profiles in this file are synthetic test data. They validate evaluator
logic only and are not PTDL aircraft design or mission data.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ptdl_2_2_1_evaluator import (
    ObjectiveNormalizer,
    demo_configuration,
    evaluate_mission,
    generate_simplex_weights,
    weighted_preference_score,
)


def make_profile(
    duration_h: float = 80.5,
    fuel_flow_kgps: float = 0.00017,
    terminal_soc: float = 0.25,
    points: int = 5,
) -> pd.DataFrame:
    """Return a nominal, feasible cruise profile for deterministic tests."""
    time_s = np.linspace(0.0, duration_h * 3600.0, points)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "fuel_flow_kgps": np.full(points, fuel_flow_kgps),
            "battery_discharge_kw": np.full(points, 2.0),
            "battery_charge_kw": np.zeros(points),
            "soc": np.linspace(0.90, terminal_soc, points),
            "h_m": np.full(points, 2500.0),
            "v_mps": np.full(points, 35.0),
            "p_engine_kw": np.full(points, 65.0),
            "p_motor_kw": np.full(points, 2.0),
            "e_h_m": np.zeros(points),
            "e_v_mps": np.zeros(points),
            "e_theta_rad": np.zeros(points),
            "q_radps": np.zeros(points),
        }
    )


class EvaluateMissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = demo_configuration()

    def test_nominal_profile_is_feasible_and_matches_energy_hand_calculation(self) -> None:
        profile = make_profile()
        result = evaluate_mission(
            profile,
            self.config,
            {"settling_time_s": 20.0, "recovery_time_s": 5.0, "mode_switch_time_s": 3.0},
        )

        duration_s = 80.5 * 3600.0
        expected_fuel_kg = 0.00017 * duration_s
        expected_battery_mj = (2.0 / 0.95) * duration_s / 1000.0
        expected_energy_mj = expected_fuel_kg * 43.0 + expected_battery_mj

        self.assertTrue(result.feasible)
        self.assertEqual(result.total_violation, 0.0)
        self.assertTrue(all(value <= 0.0 for value in result.constraints.values()))
        self.assertAlmostEqual(result.objectives["J_energy_mj"], expected_energy_mj, places=8)
        self.assertEqual(result.objectives["J_stability"], 0.0)
        self.assertEqual(result.objectives["J_response_s"], 28.0)
        self.assertAlmostEqual(result.constraints["endurance_80h"], -0.5, places=10)

    def test_endurance_shortfall_is_detected(self) -> None:
        result = evaluate_mission(make_profile(duration_h=79.5), self.config)

        self.assertFalse(result.feasible)
        self.assertAlmostEqual(result.constraints["endurance_80h"], 0.5, places=10)
        self.assertAlmostEqual(result.total_violation, 0.5, places=10)

    def test_terminal_and_minimum_soc_violations_are_detected(self) -> None:
        result = evaluate_mission(make_profile(terminal_soc=0.10), self.config)

        self.assertFalse(result.feasible)
        self.assertAlmostEqual(result.constraints["terminal_soc"], 0.10, places=10)
        self.assertAlmostEqual(result.constraints["minimum_soc"], 0.05, places=10)

    def test_power_rate_and_battery_limit_violations_are_detected(self) -> None:
        profile = make_profile(points=3)
        profile.loc[:, "time_s"] = [0.0, 80.5 * 3600.0 - 1.0, 80.5 * 3600.0]
        profile.loc[:, "p_engine_kw"] = [65.0, 65.0, 71.0]
        profile.loc[:, "p_motor_kw"] = [2.0, 2.0, 11.0]
        profile.loc[:, "battery_discharge_kw"] = 81.0
        profile.loc[:, "battery_charge_kw"] = 41.0
        profile.loc[:, "battery_current_a"] = 251.0
        profile.loc[:, "battery_temperature_c"] = 51.0
        result = evaluate_mission(profile, self.config)

        self.assertAlmostEqual(result.constraints["engine_power_rate_max"], 1.0, places=10)
        self.assertAlmostEqual(result.constraints["motor_power_rate_max"], 1.0, places=10)
        self.assertAlmostEqual(result.constraints["battery_discharge_power_max"], 1.0, places=10)
        self.assertAlmostEqual(result.constraints["battery_charge_power_max"], 1.0, places=10)
        self.assertAlmostEqual(result.constraints["battery_current_max"], 1.0, places=10)
        self.assertAlmostEqual(result.constraints["battery_temperature_max"], 1.0, places=10)
        self.assertFalse(result.feasible)

    def test_objectives_increase_when_the_corresponding_performance_worsens(self) -> None:
        baseline = evaluate_mission(make_profile(), self.config, {"settling_time_s": 10.0})
        higher_energy = evaluate_mission(make_profile(fuel_flow_kgps=0.00020), self.config)
        larger_error_profile = make_profile()
        larger_error_profile.loc[:, "e_h_m"] = 30.0
        lower_stability = evaluate_mission(larger_error_profile, self.config)
        slower_response = evaluate_mission(make_profile(), self.config, {"settling_time_s": 20.0})

        self.assertGreater(higher_energy.objectives["J_energy_mj"], baseline.objectives["J_energy_mj"])
        self.assertGreater(lower_stability.objectives["J_stability"], baseline.objectives["J_stability"])
        self.assertGreater(slower_response.objectives["J_response_s"], baseline.objectives["J_response_s"])

    def test_invalid_profiles_are_rejected(self) -> None:
        missing_column = make_profile().drop(columns="soc")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            evaluate_mission(missing_column, self.config)

        duplicate_time = make_profile(points=3)
        duplicate_time.loc[:, "time_s"] = [0.0, 0.0, 80.5 * 3600.0]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            evaluate_mission(duplicate_time, self.config)

        nonfinite_value = make_profile()
        nonfinite_value.loc[0, "fuel_flow_kgps"] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            evaluate_mission(nonfinite_value, self.config)


class OptimizationHelperTests(unittest.TestCase):
    def test_fixed_normalization_and_preference_score(self) -> None:
        samples = pd.DataFrame(
            {
                "J_energy_mj": [10.0, 20.0, 30.0],
                "J_stability": [1.0, 2.0, 3.0],
                "J_response_s": [100.0, 200.0, 300.0],
            }
        )
        normalizer = ObjectiveNormalizer.from_samples(samples, worst_quantile=1.0)
        normalized = normalizer.transform(
            {"J_energy_mj": 20.0, "J_stability": 2.0, "J_response_s": 200.0}
        )
        score = weighted_preference_score(
            normalized,
            {"J_energy_mj": 0.50, "J_stability": 0.30, "J_response_s": 0.20},
        )

        self.assertEqual(tuple(normalizer.ideal), (10.0, 1.0, 100.0))
        self.assertEqual(tuple(normalizer.reference_worst), (30.0, 3.0, 300.0))
        self.assertEqual(normalized, {"J_energy_mj": 0.5, "J_stability": 0.5, "J_response_s": 0.5})
        self.assertAlmostEqual(score, 0.5, places=10)

    def test_moead_simplex_weights_have_expected_count_and_sum(self) -> None:
        weights = generate_simplex_weights(resolution=4)

        self.assertEqual(len(weights), 15)
        self.assertTrue((weights >= 0.0).all().all())
        self.assertTrue(np.allclose(weights.sum(axis=1).to_numpy(), 1.0))
        self.assertTrue(((weights == [1.0, 0.0, 0.0]).all(axis=1)).any())
        self.assertTrue(((weights == [0.0, 1.0, 0.0]).all(axis=1)).any())
        self.assertTrue(((weights == [0.0, 0.0, 1.0]).all(axis=1)).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
