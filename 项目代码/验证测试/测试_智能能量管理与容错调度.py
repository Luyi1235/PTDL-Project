"""Tests for the initial PTDL 1.2.2 fuzzy adaptive coordinator."""

from __future__ import annotations

import unittest
import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 飞推控闭环物理模型 import DispatchObservation, MissionFaultEvent
from 智能能量管理与容错调度 import (
    FaultTolerantCoordinator,
    FaultTolerantFuzzyCoordinator,
    FuzzyAdaptiveCoordinator,
    HealthState,
    MPCEnergyManagementCoordinator,
    NEURAL_PHASES,
    NeuralImitationCoordinator,
    ScheduledFaultCoordinator,
)


def observation(
    *,
    soc: float,
    speed_error_mps: float = 0.0,
    required_kw: float = 70.0,
    phase_elapsed_s: float = 0.0,
) -> DispatchObservation:
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
        phase_elapsed_s=phase_elapsed_s,
    )


class IntelligentEnergyManagementTests(unittest.TestCase):
    def _constant_neural_policy(self) -> tuple[NeuralImitationCoordinator, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        feature_count = len(NEURAL_PHASES) + 7
        payload = {
            "schema": "ptdl_mpc_imitation_mlp_v1",
            "phase_labels": list(NEURAL_PHASES),
            "feature_mean": [0.0] * feature_count,
            "feature_scale": [1.0] * feature_count,
            "target_mean": 0.0,
            "target_scale": 1.0,
            "coefs": [[[0.0] for _ in range(feature_count)]],
            "intercepts": [[0.9]],
            "phase_durations_s": {"main_climb_to_leveloff": 100.0},
            "phase_soc_targets": {"main_climb_to_leveloff": 0.62},
        }
        model_path = Path(temporary.name) / "constant-policy.json"
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        return NeuralImitationCoordinator(model_path), temporary

    def test_low_soc_requests_fuzzy_charge(self) -> None:
        dispatch = FuzzyAdaptiveCoordinator().dispatch(observation(soc=0.45))
        self.assertEqual(dispatch.mode, "fuzzy_battery_charge")
        self.assertGreater(dispatch.engine_target_kw, 70.0)

    def test_tracking_error_requests_motor_assist(self) -> None:
        dispatch = FuzzyAdaptiveCoordinator().dispatch(observation(soc=0.85, speed_error_mps=4.0))
        self.assertEqual(dispatch.mode, "fuzzy_motor_assist")
        self.assertLess(dispatch.engine_target_kw, 70.0)

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

    def test_combined_sensor_engine_motor_and_battery_faults_keep_reserve(self) -> None:
        coordinator = FaultTolerantFuzzyCoordinator(
            FuzzyAdaptiveCoordinator(),
            HealthState(
                engine_available_fraction=0.60,
                motor_available_fraction=0.50,
                battery_available_energy_fraction=0.50,
                sensor_data_valid=False,
            ),
        )
        dispatch = coordinator.dispatch(observation(soc=0.30, required_kw=100.0))

        self.assertEqual(dispatch.mode, "sensor_fallback")
        self.assertAlmostEqual(dispatch.engine_power_cap_kw, 72.0)
        self.assertAlmostEqual(dispatch.motor_power_min_kw, -15.0)
        self.assertAlmostEqual(dispatch.motor_power_max_kw, 40.0)
        self.assertAlmostEqual(dispatch.engine_target_kw, 72.0)
        self.assertAlmostEqual(dispatch.soc_target, 0.5266666667)

    def test_scheduled_fault_activates_and_clears_at_phase_local_times(self) -> None:
        event = MissionFaultEvent(
            phase_name="main_climb_to_leveloff",
            start_phase_elapsed_s=10.0,
            duration_s=5.0,
            engine_available_fraction=0.50,
            motor_available_fraction=0.50,
            battery_available_energy_fraction=0.50,
            sensor_data_valid=False,
        )
        coordinator = ScheduledFaultCoordinator(FuzzyAdaptiveCoordinator(), (event,))

        before = coordinator.dispatch(observation(soc=0.7, required_kw=100.0, phase_elapsed_s=9.0))
        active = coordinator.dispatch(observation(soc=0.7, required_kw=100.0, phase_elapsed_s=10.0))
        after = coordinator.dispatch(observation(soc=0.7, required_kw=100.0, phase_elapsed_s=15.0))

        self.assertNotEqual(before.mode, "sensor_fallback")
        self.assertEqual(active.mode, "sensor_fallback")
        self.assertAlmostEqual(active.engine_power_cap_kw, 60.0)
        self.assertAlmostEqual(active.motor_power_max_kw, 40.0)
        self.assertNotEqual(after.mode, "sensor_fallback")

    def test_mpc_dispatch_respects_engine_and_motor_feasible_power_split(self) -> None:
        controller = MPCEnergyManagementCoordinator()
        obs = observation(soc=0.70, required_kw=70.0)
        dispatch = controller.dispatch(obs)
        motor_kw = obs.required_propulsion_kw - dispatch.engine_target_kw
        self.assertGreaterEqual(dispatch.engine_target_kw, 0.0)
        self.assertLessEqual(dispatch.engine_target_kw, obs.engine_power_cap_kw)
        self.assertGreaterEqual(motor_kw, obs.motor_power_min_kw)
        self.assertLessEqual(motor_kw, obs.motor_power_max_kw)
        self.assertIn(dispatch.mode, {"mpc_motor_assist", "mpc_battery_charge", "mpc_engine_follow"})

    def test_mpc_reset_clears_power_demand_memory(self) -> None:
        controller = MPCEnergyManagementCoordinator()
        controller.dispatch(observation(soc=0.70, required_kw=70.0))
        controller.reset()
        self.assertIsNone(controller._previous_demand_kw)
        self.assertIsNone(controller._previous_engine_target_kw)
        self.assertEqual(controller._demand_slope_kwps, 0.0)

    def test_neural_policy_does_not_charge_above_soc_target(self) -> None:
        controller, temporary = self._constant_neural_policy()
        self.addCleanup(temporary.cleanup)
        obs = observation(soc=0.90, required_kw=16.0)

        dispatch = controller.dispatch(obs)

        self.assertLessEqual(dispatch.engine_target_kw, obs.required_propulsion_kw)
        self.assertGreaterEqual(
            obs.required_propulsion_kw - dispatch.engine_target_kw, 0.0
        )

    def test_neural_policy_charge_request_respects_motor_charge_limit(self) -> None:
        controller, temporary = self._constant_neural_policy()
        self.addCleanup(temporary.cleanup)
        obs = observation(soc=0.50, required_kw=16.0)

        dispatch = controller.dispatch(obs)
        motor_power = obs.required_propulsion_kw - dispatch.engine_target_kw

        self.assertGreaterEqual(motor_power, obs.motor_power_min_kw)
        self.assertLessEqual(motor_power, obs.motor_power_max_kw)

    def test_generic_fault_supervisor_wraps_neural_policy(self) -> None:
        controller, temporary = self._constant_neural_policy()
        self.addCleanup(temporary.cleanup)
        supervised = FaultTolerantCoordinator(
            controller, HealthState(engine_available_fraction=0.50)
        )

        dispatch = supervised.dispatch(observation(soc=0.70, required_kw=70.0))

        self.assertLessEqual(dispatch.engine_target_kw, 60.0)
        self.assertAlmostEqual(dispatch.engine_power_cap_kw, 60.0)
        self.assertLessEqual(
            observation(soc=0.70, required_kw=70.0).required_propulsion_kw
            - dispatch.engine_target_kw,
            80.0,
        )

    def test_scheduled_sensor_fault_wraps_mpc_and_clears(self) -> None:
        event = MissionFaultEvent(
            phase_name="main_climb_to_leveloff",
            start_phase_elapsed_s=10.0,
            duration_s=5.0,
            engine_available_fraction=0.60,
            sensor_data_valid=False,
        )
        coordinator = ScheduledFaultCoordinator(
            MPCEnergyManagementCoordinator(), (event,)
        )

        active = coordinator.dispatch(
            observation(soc=0.65, required_kw=100.0, phase_elapsed_s=10.0)
        )
        cleared = coordinator.dispatch(
            observation(soc=0.65, required_kw=70.0, phase_elapsed_s=15.0)
        )

        self.assertEqual(active.mode, "sensor_fallback")
        self.assertAlmostEqual(active.engine_power_cap_kw, 72.0)
        self.assertNotEqual(cleared.mode, "sensor_fallback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
