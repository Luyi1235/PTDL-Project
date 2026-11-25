"""Tests for the physical flight-power coupling and engine replacement seam."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 飞推控闭环物理模型 import (
    AircraftParameters,
    DispatchObservation,
    EngineGeneratorOutput,
    EngineGeneratorRequest,
    FlightControlParameters,
    LongitudinalDragDisturbance,
    MissionDefinition,
    PhaseCommand,
    PowerDispatch,
    simulate_mission,
)


class ZeroOutputEngine:
    """Stand-in for an unavailable engine-generator used to test the interface."""

    def initial_state(self, aircraft: AircraftParameters) -> int:
        return 0

    def step(
        self,
        state: int,
        request: EngineGeneratorRequest,
        dt_s: float,
        aircraft: AircraftParameters,
    ) -> EngineGeneratorOutput:
        return EngineGeneratorOutput(state=state + 1, electrical_power_kw=0.0, fuel_flow_kgps=0.0)


class CaptureDispatchPolicy:
    def __init__(self) -> None:
        self.observation: DispatchObservation | None = None

    def reset(self) -> None:
        self.observation = None

    def dispatch(self, observation: DispatchObservation) -> PowerDispatch:
        self.observation = observation
        return PowerDispatch(
            engine_target_kw=observation.engine_power_cap_kw,
            soc_target=observation.soc,
            mode="capture",
        )


class ClosedLoopDynamicsTests(unittest.TestCase):
    def _climb_mission(self) -> MissionDefinition:
        return MissionDefinition(
            phases=(
                PhaseCommand(
                    name="replacement_interface_climb",
                    duration_s=600.0,
                    altitude_target_m=1800.0,
                    speed_target_mps=52.0,
                    engine_power_kw=120.0,
                    motor_power_kw=0.0,
                    max_flight_path_angle_deg=8.0,
                    time_step_s=10.0,
                ),
            ),
            initial_speed_mps=42.0,
        )

    def test_actual_engine_output_changes_the_flight_state(self) -> None:
        engine_only_aircraft = AircraftParameters(motor_power_min_kw=0.0, motor_power_max_kw=0.0)
        nominal = simulate_mission(self._climb_mission(), aircraft=engine_only_aircraft)
        unavailable = simulate_mission(
            self._climb_mission(), aircraft=engine_only_aircraft, engine_model=ZeroOutputEngine()
        )

        nominal_final = nominal.profile.iloc[-1]
        unavailable_final = unavailable.profile.iloc[-1]
        self.assertGreater(nominal_final["p_engine_kw"], 0.0)
        self.assertEqual(unavailable_final["p_engine_kw"], 0.0)
        self.assertGreater(unavailable.profile["power_shortfall_kw"].max(), 1.0)
        self.assertGreater(nominal_final["h_m"], unavailable_final["h_m"] + 20.0)
        self.assertGreater(nominal_final["v_mps"], unavailable_final["v_mps"])

    def test_profile_records_requested_and_actual_propulsion_power(self) -> None:
        simulation = simulate_mission(self._climb_mission())
        profile = simulation.profile
        self.assertIn("p_propulsion_request_kw", profile.columns)
        self.assertIn("p_propulsor_input_kw", profile.columns)
        self.assertGreater(profile["p_propulsion_request_kw"].max(), 0.0)
        self.assertGreater(profile["p_propulsor_input_kw"].max(), 0.0)

    def test_speed_target_uses_fixed_tracking_time_constant_in_long_phase(self) -> None:
        def run_target(target_speed: float):
            mission = MissionDefinition(
                phases=(
                    PhaseCommand(
                        name="long_speed_tracking",
                        duration_s=120.0,
                        altitude_target_m=1000.0,
                        speed_target_mps=target_speed,
                        engine_power_kw=160.0,
                        motor_power_kw=0.0,
                        max_flight_path_angle_deg=3.0,
                        time_step_s=2.0,
                    ),
                ),
                initial_altitude_m=1000.0,
                initial_speed_mps=30.0,
            )
            return simulate_mission(mission)

        low_target = run_target(30.0).profile
        high_target = run_target(45.0).profile

        low_speed = float(low_target.loc[low_target["time_s"] <= 60.0, "v_mps"].iloc[-1])
        high_speed = float(high_target.loc[high_target["time_s"] <= 60.0, "v_mps"].iloc[-1])
        self.assertGreater(high_speed, low_speed + 2.0)

    def test_unpowered_glide_speed_target_changes_path_angle_command(self) -> None:
        controller = FlightControlParameters(
            glide_speed_path_angle_gain_rad_per_mps=0.001
        )

        def run_target(target_speed: float):
            mission = MissionDefinition(
                phases=(
                    PhaseCommand(
                        name="unpowered_glide",
                        duration_s=60.0,
                        altitude_target_m=900.0,
                        speed_target_mps=target_speed,
                        engine_power_kw=120.0,
                        motor_power_kw=30.0,
                        max_flight_path_angle_deg=8.0,
                        time_step_s=1.0,
                        unpowered_glide=True,
                    ),
                ),
                initial_altitude_m=1000.0,
                initial_speed_mps=30.0,
            )
            simulation = simulate_mission(mission, controller=controller)
            return simulation.profile.loc[
                simulation.profile["phase"] == "unpowered_glide"
            ]

        slow_target = run_target(30.0)
        fast_target = run_target(45.0)

        self.assertLess(
            float(fast_target["gamma_command_rad"].iloc[0]),
            float(slow_target["gamma_command_rad"].iloc[0]),
        )
        for profile in (slow_target, fast_target):
            self.assertTrue(np.allclose(profile["p_engine_kw"], 0.0))
            self.assertTrue(np.allclose(profile["p_motor_kw"], 0.0))

    def test_dispatch_observes_optimizer_phase_motor_assist_cap(self) -> None:
        policy = CaptureDispatchPolicy()
        mission = MissionDefinition(
            phases=(
                PhaseCommand(
                    "bounded_assist",
                    10.0,
                    0.0,
                    42.0,
                    120.0,
                    2.0,
                    8.0,
                    10.0,
                ),
            ),
            initial_speed_mps=42.0,
        )
        simulate_mission(mission, dispatch_policy=policy)

        self.assertIsNotNone(policy.observation)
        self.assertEqual(policy.observation.motor_power_max_kw, 2.0)

    def test_motor_disabled_phase_blocks_assist_and_regeneration(self) -> None:
        policy = CaptureDispatchPolicy()
        mission = MissionDefinition(
            phases=(
                PhaseCommand(
                    "engine_only_descent",
                    30.0,
                    900.0,
                    42.0,
                    60.0,
                    10.0,
                    5.0,
                    1.0,
                    motor_disabled=True,
                ),
            ),
            initial_altitude_m=1000.0,
            initial_speed_mps=42.0,
        )
        simulation = simulate_mission(mission, dispatch_policy=policy)

        self.assertIsNotNone(policy.observation)
        self.assertEqual(policy.observation.motor_power_min_kw, 0.0)
        self.assertEqual(policy.observation.motor_power_max_kw, 0.0)
        self.assertTrue(np.allclose(simulation.profile["p_motor_kw"], 0.0))

    def test_disturbance_is_recorded_and_recovery_is_measured(self) -> None:
        mission = MissionDefinition(
            phases=(
                PhaseCommand(
                    "endurance_loiter",
                    4000.0,
                    1000.0,
                    70.0,
                    58.0,
                    0.0,
                    3.0,
                    120.0,
                ),
            ),
            initial_altitude_m=1000.0,
            initial_speed_mps=70.0,
            disturbances=(
                LongitudinalDragDisturbance(
                    "endurance_loiter",
                    1200.0,
                    600.0,
                    500.0,
                    response_step_s=5.0,
                    recovery_window_s=900.0,
                ),
            ),
        )
        simulation = simulate_mission(mission)
        disturbed = simulation.profile.loc[simulation.profile["disturbance_drag_n"] > 0.0]

        self.assertGreater(len(disturbed), 20)
        self.assertTrue(np.allclose(disturbed["disturbance_drag_n"], 500.0))
        self.assertGreater(float(disturbed["e_v_mps"].abs().max()), 0.8)
        self.assertGreater(simulation.response_metrics["recovery_time_s"], 0.0)
        self.assertLessEqual(simulation.response_metrics["recovery_time_s"], 900.0)

    def test_unpowered_glide_uses_rate_limited_shutdown_then_zero_power(self) -> None:
        mission = MissionDefinition(
            phases=(
                PhaseCommand(
                    name="powered_descent",
                    duration_s=60.0,
                    altitude_target_m=900.0,
                    speed_target_mps=55.0,
                    engine_power_kw=120.0,
                    motor_power_kw=0.0,
                    max_flight_path_angle_deg=8.0,
                    time_step_s=5.0,
                ),
                PhaseCommand(
                    name="unpowered_glide",
                    duration_s=120.0,
                    altitude_target_m=500.0,
                    speed_target_mps=55.0,
                    engine_power_kw=120.0,
                    motor_power_kw=30.0,
                    max_flight_path_angle_deg=8.0,
                    time_step_s=1.0,
                    unpowered_glide=True,
                ),
            ),
            initial_altitude_m=1000.0,
            initial_speed_mps=60.0,
        )
        simulation = simulate_mission(mission)
        profile = simulation.profile
        glide = profile.loc[profile["phase"] == "unpowered_glide"]
        shutdown = profile.loc[profile["phase"] == "unpowered_glide_engine_shutdown"]

        self.assertGreater(float(shutdown["p_engine_kw"].max()), 0.0)
        self.assertLessEqual(float(shutdown["p_engine_kw"].iloc[-1]), 10.0)
        self.assertTrue((glide["power_mode"] == "unpowered_glide").all())
        self.assertTrue(np.allclose(glide["p_engine_kw"], 0.0))
        self.assertTrue(np.allclose(glide["p_motor_kw"], 0.0))
        self.assertTrue(np.allclose(glide["p_propulsor_input_kw"], 0.0))
        rates = np.abs(np.diff(profile["p_engine_kw"].to_numpy()) / np.diff(profile["time_s"].to_numpy()))
        self.assertLessEqual(float(rates.max()), 10.0 + 1e-6)
        self.assertGreater(float(profile["range_m"].iloc[-1]), 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
