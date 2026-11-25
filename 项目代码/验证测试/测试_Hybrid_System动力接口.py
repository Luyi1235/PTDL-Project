"""Hybrid_System component and PTDL optimizer integration checks."""

from __future__ import annotations

import sys
import unittest
import math
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from 飞推控闭环物理模型 import AircraftParameters, HybridPowertrainRequest
from MOEAD多目标联合优化 import PTDLEnduranceJointProblem
from 全任务剖面定义 import (
    PreliminaryPropulsionAssumptions,
    load_preliminary_propulsion_config,
    preliminary_hybrid_aircraft,
)
from ULTRA基准标定 import UltraCalibration, UltraOpenData
from 机体参数化设计与气动阻力 import AirframeDesign


class HybridSystemAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plant = HybridSystemPowertrain()
        self.aircraft = AircraftParameters()

    def test_component_ratings_and_map_quality_are_exposed(self) -> None:
        metadata = self.plant.metadata
        self.assertEqual(metadata["engine_map_total_points"], 60)
        self.assertIsNone(metadata["engine_map_converged_points"])
        self.assertAlmostEqual(metadata["generator_rated_ac_kw"], 60.0)
        self.assertAlmostEqual(metadata["battery_nominal_energy_wh"], 16678.305)
        self.assertEqual(metadata["source_version"], "1005-20261023")
        self.assertEqual(metadata["battery_series_cells"], 81)
        self.assertEqual(metadata["battery_parallel_cells"], 3)

    def test_engine_speed_config_is_applied_to_generator_gear_ratio(self) -> None:
        plant = HybridSystemPowertrain(HybridSystemAdapterConfig(engine_speed_rpm=6100.0))
        design_generator_rpm = plant.generator_parameters["GEN_N_design_rpm"]
        self.assertAlmostEqual(
            plant.generator_parameters["GEN_GearRatio"] * design_generator_rpm,
            6100.0,
        )

    def test_engine_power_output_respects_configured_slew_rate(self) -> None:
        plant = HybridSystemPowertrain(
            HybridSystemAdapterConfig(engine_power_rate_limit_kwps=10.0)
        )
        state = plant.initial_state(self.aircraft, 0.85)
        powered = plant.step(
            state,
            HybridPowertrainRequest("powered", 120.0, 120.0, 120.0, 0.0, 0.0),
            1.0,
            self.aircraft,
        )
        shutdown = plant.step(
            state,
            HybridPowertrainRequest("shutdown", 0.0, 120.0, 0.0, 0.0, 0.0),
            1.0,
            self.aircraft,
        )
        self.assertLessEqual(powered.engine_power_kw, 10.0 + 1e-6)
        self.assertLessEqual(
            abs(shutdown.engine_power_kw - powered.engine_power_kw),
            10.0 + 1e-6,
        )

    def test_component_ratings_follow_ptdl_aircraft_estimates(self) -> None:
        aircraft = AircraftParameters(
            engine_electrical_power_max_kw=120.0,
            motor_power_max_kw=78.0,
            battery_capacity_mj=61.8,
        )
        state = self.plant.initial_state(aircraft, 0.85)
        self.assertAlmostEqual(state.engine_rating_scale, 2.0)
        self.assertAlmostEqual(state.battery_parallel_equivalent, 61.8e6 / 3600 / 16678.305)
        self.assertAlmostEqual(state.motor_shaft_rating_kw, 198.0)
        self.assertEqual(state.motor_branches, 9)
        self.assertAlmostEqual(state.motor.parameters["MOT_P_rated_W"], 24146.28113549115)
        request = HybridPowertrainRequest("rating_check", 100.0, 120.0, 60.0, 0.0, 78.0)
        output = self.plant.step(state, request, 10.0, aircraft)
        self.assertGreater(output.engine_power_kw, 60.0)
        self.assertGreater(output.propulsor_input_power_kw, 10.9)
        self.assertLessEqual(output.propulsor_input_power_kw, 60.0 + 1e-6)

    def test_preliminary_propulsion_multipliers_scale_ratings_and_energy(self) -> None:
        calibration = UltraCalibration(
            cd0=0.025,
            baseline_endurance_h=72.0,
            baseline_fuel_used_kg=367.0,
            cruise_power_kw=40.0,
            reference=UltraOpenData(),
        )
        design = AirframeDesign(
            wing_area_m2=26.0,
            aspect_ratio=20.0,
            taper_ratio=0.45,
            sweep_deg=2.0,
            tip_twist_deg=-2.0,
            fuselage_length_m=10.0,
            fuselage_diameter_m=1.0,
            propeller_diameter_m=2.0,
            payload_mass_kg=100.0,
            hybrid_system_mass_kg=100.0,
        )
        assumptions = PreliminaryPropulsionAssumptions(
            engine_rating_multiplier=1.3,
            motor_rating_multiplier=1.4,
            battery_energy_multiplier=1.5,
        )

        aircraft = preliminary_hybrid_aircraft(design, calibration, assumptions)

        self.assertAlmostEqual(aircraft.engine_electrical_power_max_kw, 105.0 * 1.3)
        self.assertAlmostEqual(aircraft.motor_power_max_kw, 100.0 * 1.4)
        self.assertAlmostEqual(aircraft.battery_capacity_mj, 100.0 * 220.0 * 3.6e-3 * 1.5)

    def test_ultra_reference_config_is_the_runtime_sizing_source(self) -> None:
        assumptions, config = load_preliminary_propulsion_config()
        self.assertEqual(config["schema"], "ptdl_ultra_referenced_propulsion_sizing_v1")
        self.assertEqual(assumptions.engine_rating_multiplier, 1.20)
        self.assertEqual(assumptions.motor_rating_multiplier, 1.20)
        self.assertEqual(assumptions.battery_energy_multiplier, 1.20)
        self.assertIn("not calibrated hardware data", config["sizing_note"])

    def test_battery_discharge_uses_ptdl_sign_and_soc_units(self) -> None:
        state = self.plant.initial_state(self.aircraft, 0.85)
        output = self.plant.step(
            state,
            HybridPowertrainRequest("test", 0.0, 120.0, 3.0, 0.0, 30.0),
            10.0,
            self.aircraft,
        )
        self.assertGreater(output.motor_power_kw, 0.0)
        self.assertGreater(output.battery_current_a, 0.0)
        self.assertLess(output.soc, 0.85)
        self.assertLessEqual(output.propulsor_input_power_kw, 3.0)

    def test_battery_cannot_discharge_below_configured_soc_floor(self) -> None:
        state = self.plant.initial_state(self.aircraft, self.plant.config.battery_soc_min)
        output = self.plant.step(
            state,
            HybridPowertrainRequest("soc_floor", 0.0, 0.0, 3.0, 0.0, 30.0),
            10.0,
            self.aircraft,
        )
        self.assertAlmostEqual(output.soc, self.plant.config.battery_soc_min)
        self.assertAlmostEqual(output.battery_current_a, 0.0)
        self.assertAlmostEqual(output.motor_power_kw, 0.0)

    def test_engine_surplus_charges_battery(self) -> None:
        state = self.plant.initial_state(self.aircraft, 0.70)
        output = self.plant.step(
            state,
            HybridPowertrainRequest("test", 4.8, 120.0, 0.5, -30.0, 30.0),
            10.0,
            self.aircraft,
        )
        self.assertLess(output.motor_power_kw, 0.0)
        self.assertLess(output.battery_current_a, 0.0)
        self.assertGreater(output.soc, 0.70)

    def test_optimizer_problem_accepts_a_fresh_component_factory(self) -> None:
        problem = PTDLEnduranceJointProblem(powertrain_factory=HybridSystemPowertrain)
        decision = problem.sample(1, __import__("numpy").random.default_rng(11))[0]
        candidate = problem.evaluate(decision)
        # Feasibility depends on the supplied fuel map; this is an interface test.
        self.assertTrue(all(math.isfinite(float(v)) for v in candidate.objectives))
        self.assertGreaterEqual(candidate.result.total_violation, 0.0)
        self.assertEqual(candidate.feasible, candidate.result.total_violation <= 1e-9)

    def test_fuel_uses_supplied_map_without_old_efficiency_substitution(self):
        state = self.plant.initial_state(self.aircraft, 0.85)
        power, fuel = self.plant._engine_output(state, 120, 120, 3000)
        native = state.engine_map.operating_point_for_power(
            120000 / state.engine_rating_scale / 0.96, 5000, 3000
        )
        self.assertAlmostEqual(fuel, native["FuelFlow_kg_s"] * state.engine_rating_scale)
        other = self.plant.initial_state(replace(self.aircraft, engine_efficiency_values=(0.1, 0.1)), 0.85)
        self.assertEqual((power, fuel), self.plant._engine_output(other, 120, 120, 3000))

    def test_altitude_is_forwarded_and_outside_map_is_counted(self):
        request = HybridPowertrainRequest("cruise", 100, 120, 50, -30, 30, altitude_m=3000)
        state = self.plant.initial_state(self.aircraft, 0.85)
        output = self.plant.step(state, request, 30, self.aircraft)
        _, expected = self.plant._engine_output(state, output.engine_power_kw, 120, 3000)
        self.assertAlmostEqual(output.fuel_flow_kgps, expected)
        self.assertNotEqual(expected, self.plant._engine_output(state, output.engine_power_kw, 120, 0)[1])
        self.plant.step(state, replace(request, altitude_m=6000), 30, self.aircraft)
        self.assertEqual(state.engine_map_altitude_clamped_steps, 1)

    def test_pack_resistance_is_scaled_once(self):
        p = self.plant.battery_parameters
        self.assertAlmostEqual(p["BAT_Rdis_pack_Ohm"][3], 81 / 3 * p["BAT_Cell_Rdis_Ohm"][3])
        changed = HybridSystemPowertrain(HybridSystemAdapterConfig(battery_resistance_multiplier=2))
        self.assertAlmostEqual(changed.battery_parameters["BAT_Rdis_pack_Ohm"][3], 2*p["BAT_Rdis_pack_Ohm"][3])

    def test_motor_returns_shaft_power_and_obeys_dc_input_limit(self):
        state = self.plant.initial_state(self.aircraft, 0.85)
        shaft, dc = self.plant._motor_output(state, 30, 300, 1000)
        self.assertAlmostEqual(shaft, 30)
        self.assertGreater(dc, shaft)
        self.assertLessEqual(dc, state.motor_branches * 40)
        self.assertEqual(self.plant._motor_output(state, 30, 300, 0), (0, 0))
        self.assertEqual(self.plant._motor_output(state, 0, 300, 100), (0, 0))

    def test_native_ems_shutdown_restart_and_dc_command_contract(self):
        from 动能三层能量管理 import SourceThreeLayerCoordinator
        from 飞推控闭环物理模型 import DispatchObservation
        policy = SourceThreeLayerCoordinator()
        obs = DispatchObservation("long_descent", 1, .8, 0, 0, 10, 60, -30, 30,
                                  required_motor_dc_kw=12)
        self.assertEqual(policy.dispatch(obs).engine_target_kw, 0)
        self.assertGreater(policy.dispatch(replace(obs, soc=.3)).engine_target_kw, 0)
        self.assertGreater(policy.dispatch(replace(obs, phase_name="flare")).engine_target_kw, 0)
        self.assertTrue(policy.dispatch(obs).engine_target_is_dc)
        state = self.plant.initial_state(self.aircraft, .8)
        output = self.plant.step(state, HybridPowertrainRequest(
            "cruise", 20, 120, 30, -30, 50, engine_target_is_dc=True
        ), 10, self.aircraft)
        self.assertAlmostEqual(output.engine_power_kw, 20)
        self.assertLess(output.bus_power_deficit_kw, 1e-6)

    def test_component_soc_bounds_are_applied_to_state_equations(self):
        plant = HybridSystemPowertrain(HybridSystemAdapterConfig(battery_soc_min=.3, battery_soc_max=.9))
        state = plant.initial_state(self.aircraft, .3)
        self.assertEqual(state.battery.parameters["BAT_SOC_min"], .3)
        self.assertEqual(state.battery.parameters["BAT_SOC_max"], .9)
        output = plant.step(state, HybridPowertrainRequest("floor", 0, 0, 20, 0, 40), 120, self.aircraft)
        self.assertAlmostEqual(output.soc, .3)

    def test_out_of_range_engine_speed_is_rejected(self):
        with self.assertRaises(ValueError):
            HybridSystemPowertrain(HybridSystemAdapterConfig(engine_speed_rpm=6600))

    def test_ground_takeoff_integrates_supplied_components(self):
        from 低阶地面滑跑模型 import simulate_ground_roll, GroundRollAssumptions
        result = simulate_ground_roll(
            self.aircraft, "takeoff", 120,
            assumptions=GroundRollAssumptions(maximum_roll_time_s=3),
            engine_power_fraction=.7, initial_soc=.85,
            powertrain=self.plant,
        )
        self.assertGreater(result.estimated_fuel_used_kg, 0)
        self.assertLess(result.terminal_soc_estimated, .85)
        self.assertEqual(result.assumptions["powertrain_source"]["source_version"], "1005-20261023")
        landing = simulate_ground_roll(
            self.aircraft, "landing", 0, initial_soc=.63,
            initial_speed_mps=10, powertrain=self.plant,
        )
        self.assertTrue(landing.feasible)
        self.assertAlmostEqual(landing.terminal_soc_estimated, .63)
        self.assertEqual(landing.estimated_fuel_used_kg, 0)

    def test_optimizer_passes_installed_sizing_to_full_mission(self):
        class Validator:
            full_mission = True
            accepts_powertrain_sizing = True

            def __call__(self, **kwargs):
                self.sizing = kwargs["powertrain_sizing"]
                return {"constraints": {}, "diagnostics": {}}

        validator = Validator()
        problem = PTDLEnduranceJointProblem(
            powertrain_factory=HybridSystemPowertrain, sixdof_validator=validator
        )
        candidate = problem.evaluate(problem.sample(1, __import__("numpy").random.default_rng(11))[0])
        diagnostics = candidate.result.diagnostics
        self.assertEqual(validator.sizing, {
            "engine_electrical_power_max_kw": diagnostics["engine_power_max_kw"],
            "motor_power_max_kw": diagnostics["motor_power_max_kw"],
            "battery_capacity_mj": diagnostics["battery_capacity_mj"],
        })

    def test_optimizer_bounds_follow_configured_propulsion_multipliers(self) -> None:
        assumptions = PreliminaryPropulsionAssumptions(
            engine_rating_multiplier=1.35,
            motor_rating_multiplier=1.45,
        )
        problem = PTDLEnduranceJointProblem(assumptions=assumptions)
        phase_count = len(problem.phase_labels)
        engine_upper = problem.upper_bounds[11 : 11 + phase_count]
        motor_upper = problem.upper_bounds[11 + phase_count :]
        base_profile = problem.profile["segments"]

        self.assertAlmostEqual(
            engine_upper[0],
            base_profile[0]["engine_power_bounds_kw"][1] * 1.35,
        )
        self.assertAlmostEqual(
            motor_upper[0],
            base_profile[0]["motor_assist_bounds_kw"][1] * 1.45,
        )


if __name__ == "__main__":
    unittest.main()
