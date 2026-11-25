"""Tests for the initial PTDL 1.2.2 fuzzy adaptive coordinator."""

from __future__ import annotations

import unittest

from ptdl_flight_propulsion_control import DispatchObservation
from ptdl_intelligent_energy_management import FaultTolerantFuzzyCoordinator, FuzzyAdaptiveCoordinator, HealthState


def observation(*, soc: float, speed_error_mps: float = 0.0, required_kw: float = 70.0) -> DispatchObservation:
    return DispatchObservation(
        phase_name="main_climb_to_leveloff",
        time_step_s=1.0,
        soc=soc,
        altitude_error_m=0.0,
        speed_error_mps=speed_error_mps,
        required_propulsion_kw=required_kw,
        engine_power_cap_kw=120.0,
        motor_power_min_kw=-30.0,
        motor_power_max_kw=80.0,
    )


class IntelligentEnergyManagementTests(unittest.TestCase):
    def test_low_soc_requests_fuzzy_charge(self) -> None:
        dispatch = FuzzyAdaptiveCoordinator().dispatch(observation(soc=0.45))
        self.assertEqual(dispatch.mode, "fuzzy_battery_charge")
        self.assertGreater(dispatch.engine_target_kw, 70.0)

    def test_tracking_error_requests_motor_assist(self) -> None:
        dispatch = FuzzyAdaptiveCoordinator().dispatch(observation(soc=0.85, speed_error_mps=4.0))
        self.assertEqual(dispatch.mode, "fuzzy_motor_assist")
        self.assertLess(dispatch.engine_target_kw, 70.0)

    def test_tracking_reserve_is_released_when_error_grows(self) -> None:
        coordinator = FuzzyAdaptiveCoordinator()
        quiet = coordinator.dispatch(observation(soc=0.80, speed_error_mps=0.0))
        coordinator.reset()
        recovering = coordinator.dispatch(observation(soc=0.80, speed_error_mps=5.0))
        self.assertIsNotNone(quiet.motor_power_max_kw)
        self.assertIsNotNone(recovering.motor_power_max_kw)
        self.assertLess(quiet.motor_power_max_kw, recovering.motor_power_max_kw)
        self.assertAlmostEqual(recovering.motor_power_max_kw, 80.0)

    def test_sensor_fault_enters_conservative_fallback(self) -> None:
        coordinator = FaultTolerantFuzzyCoordinator(FuzzyAdaptiveCoordinator(), HealthState(sensor_data_valid=False))
        dispatch = coordinator.dispatch(observation(soc=0.50))
        self.assertEqual(dispatch.mode, "sensor_fallback")
        self.assertEqual(dispatch.engine_target_kw, 70.0)

    def test_component_derating_is_exported_to_the_physical_powertrain(self) -> None:
        coordinator = FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(),
            HealthState(engine_available_fraction=0.60, motor_available_fraction=0.50),
        )
        dispatch = coordinator.dispatch(observation(soc=0.85, speed_error_mps=4.0))
        self.assertAlmostEqual(dispatch.engine_power_cap_kw, 72.0)
        self.assertAlmostEqual(dispatch.motor_power_min_kw, -15.0)
        self.assertAlmostEqual(dispatch.motor_power_max_kw, 40.0)

    def test_engine_derating_monitor_increases_soc_reserve_target(self) -> None:
        coordinator = FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(),
            HealthState(engine_available_fraction=0.85, battery_available_energy_fraction=0.80),
        )
        baseline = FuzzyAdaptiveCoordinator().dispatch(observation(soc=0.74))
        for _ in range(20):
            dispatch = coordinator.dispatch(observation(soc=0.74))
        self.assertLess(coordinator.estimated_engine_available_fraction, 0.93)
        self.assertGreater(dispatch.soc_target, baseline.soc_target)
        self.assertTrue(dispatch.mode.startswith("health_adaptive_"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
