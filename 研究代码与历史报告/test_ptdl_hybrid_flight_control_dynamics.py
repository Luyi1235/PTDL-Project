"""Tests for the physical flight-power coupling and engine replacement seam."""

from __future__ import annotations

import unittest

from ptdl_flight_propulsion_control import (
    AircraftParameters,
    EngineGeneratorOutput,
    EngineGeneratorRequest,
    MissionDefinition,
    PhaseCommand,
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

    def test_vertical_wind_disturbance_enters_altitude_state(self) -> None:
        calm = self._climb_mission()
        disturbed = MissionDefinition(
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
                    vertical_wind_mps=-1.5,
                ),
            ),
            initial_speed_mps=42.0,
        )
        calm_result = simulate_mission(calm)
        disturbed_result = simulate_mission(disturbed)
        self.assertLess(
            float(disturbed_result.profile["h_m"].iloc[-1]),
            float(calm_result.profile["h_m"].iloc[-1]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
