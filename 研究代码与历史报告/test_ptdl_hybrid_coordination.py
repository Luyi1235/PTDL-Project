"""Tests for the PTDL 1.2.1 hybrid flight-control coordination model."""

from __future__ import annotations

import unittest

from ptdl_flight_propulsion_control import DispatchObservation, simulate_mission, reference_endurance_mission
from ptdl_hybrid_coordination import RuleBasedHybridCoordinator


def observation(*, soc: float, speed_error_mps: float = 0.0) -> DispatchObservation:
    return DispatchObservation(
        phase_name="main_climb_to_leveloff",
        time_step_s=1.0,
        soc=soc,
        altitude_error_m=0.0,
        speed_error_mps=speed_error_mps,
        required_propulsion_kw=70.0,
        engine_power_cap_kw=120.0,
        motor_power_min_kw=-30.0,
        motor_power_max_kw=80.0,
    )


class HybridCoordinationTests(unittest.TestCase):
    def test_low_soc_requests_generator_charge_margin(self) -> None:
        coordinator = RuleBasedHybridCoordinator()
        dispatch = coordinator.dispatch(observation(soc=0.45))

        self.assertEqual(dispatch.mode, "battery_charge")
        self.assertGreater(dispatch.engine_target_kw, 70.0)

    def test_high_soc_and_tracking_error_request_motor_assist(self) -> None:
        coordinator = RuleBasedHybridCoordinator()
        dispatch = coordinator.dispatch(observation(soc=0.90, speed_error_mps=4.0))

        self.assertEqual(dispatch.mode, "motor_assist")
        self.assertLess(dispatch.engine_target_kw, 70.0)

    def test_model_exports_coordination_states(self) -> None:
        coordinator = RuleBasedHybridCoordinator()
        simulation = simulate_mission(reference_endurance_mission(loiter_duration_h=1.0), dispatch_policy=coordinator)

        self.assertTrue({"p_engine_target_kw", "power_mode", "soc_target"}.issubset(simulation.profile.columns))
        self.assertTrue(simulation.profile["power_mode"].notna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
