"""Tests for the paired-trajectory response measurement used by scenario runs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "运行脚本"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from 运行_多工况协同控制验证 import _paired_response_metrics
from 运行_多随机种子算法对比 import _hypervolume_3d
from MOEAD多目标联合优化 import CandidateEvaluation, HybridNSGAIIMOPSO, MOPSO
from 低阶地面滑跑模型 import GroundRollAssumptions, simulate_ground_roll
from 八十小时续航任务模型 import build_endurance_mission, load_endurance_profile
from 智能能量管理与容错调度 import FuzzyAdaptiveCoordinator, ScheduledFaultCoordinator
from VSPAERO气动代理模型 import FEATURE_LABELS, fit_kriging_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign
from 响应指标 import (
    apply_paired_response_metrics,
    project_response_stress_scenario,
    zero_amplitude_disturbances,
)
from 指标体系与评价函数 import EvaluationResult, demo_configuration, evaluate_mission
from 飞推控闭环物理模型 import (
    AircraftParameters,
    LongitudinalDragDisturbance,
    MissionDefinition,
    MissionFaultEvent,
    PhaseCommand,
    simulate_mission,
)


class ExtendedValidationTests(unittest.TestCase):
    def test_kriging_surrogate_returns_mean_and_predictive_spread(self) -> None:
        data_path = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_23点.csv"
        data = pd.read_csv(data_path).head(12)
        designs = [
            AirframeDesign(
                **{label: float(row[label]) for label in FEATURE_LABELS}
            )
            for _, row in data.iterrows()
        ]
        model = fit_kriging_vsp_surrogate(
            designs,
            data["vspaero_cl_alpha_per_rad"].to_numpy(float),
            data["vspaero_oswald_efficiency"].to_numpy(float),
        )
        prediction = model.predict_with_uncertainty(designs[0])
        self.assertEqual(model.training_samples, 12)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(prediction[2], 0.0)
        self.assertGreaterEqual(prediction[3], 0.0)

    def test_endurance_mission_accepts_ground_roll_handoff_speed(self) -> None:
        mission = build_endurance_mission(
            load_endurance_profile(),
            target_total_duration_h=80.0,
            initial_speed_mps=37.5,
            initial_soc=0.73,
        )
        self.assertEqual(mission.initial_speed_mps, 37.5)
        self.assertEqual(mission.initial_soc, 0.73)

    def test_mission_fault_event_is_logged_only_inside_configured_window(self) -> None:
        event = MissionFaultEvent(
            phase_name="short_test",
            start_phase_elapsed_s=2.0,
            duration_s=3.0,
            engine_available_fraction=0.5,
            motor_available_fraction=0.5,
            sensor_data_valid=False,
        )
        phase = PhaseCommand("short_test", 8.0, 0.0, 42.0, 100.0, 20.0, time_step_s=1.0)
        mission = MissionDefinition((phase,), fault_events=(event,))
        simulation = simulate_mission(
            mission,
            AircraftParameters(),
            dispatch_policy=ScheduledFaultCoordinator(FuzzyAdaptiveCoordinator(), (event,)),
        )
        fault_rows = simulation.profile.loc[simulation.profile["fault_active"]]
        self.assertEqual(fault_rows["phase_elapsed_s"].tolist(), [3.0, 4.0, 5.0])
        self.assertTrue(fault_rows["power_mode"].astype(str).str.contains("sensor_fallback").all())
        after_rows = simulation.profile.loc[simulation.profile["phase_elapsed_s"] >= 6.0]
        self.assertFalse(after_rows["power_mode"].astype(str).str.contains("sensor_fallback").any())

    def test_paired_response_uses_nominal_path_and_dwell_band(self) -> None:
        times = np.arange(0.0, 101.0, 5.0)
        nominal = pd.DataFrame(
            {
                "time_s": times,
                "phase": "test_phase",
                "h_m": 1000.0,
                "v_mps": 30.0,
            }
        )
        disturbed = nominal.copy()
        disturbed.loc[disturbed["time_s"].between(30.0, 45.0), "h_m"] += 5.0
        disturbed.loc[disturbed["time_s"].between(30.0, 45.0), "v_mps"] += 0.6

        metrics = _paired_response_metrics(
            disturbed,
            nominal,
            "test_phase",
            phase_start_s=0.0,
            phase_end_s=100.0,
            event_start_s=20.0,
            event_duration_s=10.0,
            recovery_window_s=70.0,
            recovery_dwell_s=10.0,
            altitude_tolerance_m=10.0,
        )

        self.assertTrue(metrics["paired_recovery_success"])
        self.assertEqual(metrics["paired_recovery_time_s"], 20.0)
        self.assertAlmostEqual(metrics["paired_peak_speed_deviation_mps"], 0.6)
        self.assertEqual(metrics["paired_peak_altitude_deviation_m"], 5.0)
        self.assertTrue(metrics["response_excitation_exceeds_band"])

    def test_optimizer_response_uses_paired_baseline_and_updates_score(self) -> None:
        times = np.arange(0.0, 101.0, 5.0)
        nominal = pd.DataFrame(
            {"time_s": times, "phase": "test_phase", "h_m": 1000.0, "v_mps": 30.0}
        )
        disturbed = nominal.copy()
        disturbed.loc[disturbed["time_s"].between(30.0, 45.0), "h_m"] += 5.0
        disturbed.loc[disturbed["time_s"].between(30.0, 45.0), "v_mps"] += 0.6
        event = LongitudinalDragDisturbance(
            "test_phase", 20.0, 10.0, 100.0, recovery_window_s=70.0, recovery_dwell_s=10.0
        )
        nominal_evaluation = SimpleNamespace(
            simulation=SimpleNamespace(profile=nominal),
            mission=SimpleNamespace(phases=()),
            result=SimpleNamespace(objectives={}, diagnostics={}),
        )
        disturbed_evaluation = SimpleNamespace(
            simulation=SimpleNamespace(
                profile=disturbed,
                response_metrics={"mode_switch_time_s": 4.0},
            ),
            mission=SimpleNamespace(
                phases=(SimpleNamespace(name="test_phase", altitude_target_m=1000.0, duration_s=100.0),)
            ),
            result=SimpleNamespace(objectives={"J_response_s": 999.0}, diagnostics={}),
        )

        metrics = apply_paired_response_metrics(
            disturbed_evaluation, nominal_evaluation, (event,)
        )

        self.assertEqual(metrics["recovery_time_s"], 20.0)
        self.assertEqual(disturbed_evaluation.result.objectives["J_response_s"], 24.0)
        self.assertEqual(disturbed_evaluation.result.diagnostics["response_recovery_success"], 1.0)
        self.assertEqual(metrics["response_excitation_exceeds_band"], 1.0)

    def test_zero_amplitude_event_preserves_timing_but_removes_wind_and_drag(self) -> None:
        event = LongitudinalDragDisturbance(
            "loiter",
            10.0,
            5.0,
            100.0,
            headwind_step_mps=8.0,
            headwind_gust_amplitude_mps=3.0,
        )

        nominal_event = zero_amplitude_disturbances((event,))[0]

        self.assertEqual(nominal_event.start_time_s, event.start_time_s)
        self.assertEqual(nominal_event.duration_s, event.duration_s)
        self.assertEqual(nominal_event.additional_drag_n, 0.0)
        self.assertEqual(nominal_event.headwind_step_mps, 0.0)
        self.assertEqual(nominal_event.headwind_gust_amplitude_mps, 0.0)

    def test_shared_response_stress_scenario_has_explicit_sampling_and_window(self) -> None:
        event, = project_response_stress_scenario()

        self.assertEqual(event.phase_name, "endurance_loiter")
        self.assertEqual(event.headwind_step_mps, 8.0)
        self.assertEqual(event.duration_s, 60.0)
        self.assertEqual(event.response_step_s, 2.0)

    def test_failed_recovery_cannot_score_better_than_successful_recovery(self) -> None:
        profile = pd.DataFrame(
            {
                "time_s": [0.0, 100.0],
                "fuel_flow_kgps": [0.0, 0.0],
                "battery_discharge_kw": [0.0, 0.0],
                "battery_charge_kw": [0.0, 0.0],
                "soc": [0.8, 0.8],
                "h_m": [1000.0, 1000.0],
                "v_mps": [30.0, 30.0],
                "p_engine_kw": [0.0, 0.0],
                "p_motor_kw": [0.0, 0.0],
                "e_h_m": [0.0, 0.0],
                "e_v_mps": [0.0, 0.0],
                "e_theta_rad": [0.0, 0.0],
                "q_radps": [0.0, 0.0],
            }
        )
        successful = evaluate_mission(
            profile,
            demo_configuration(),
            {
                "settling_time_s": 8270.0,
                "recovery_time_s": 12.0,
                "mode_switch_time_s": 3.0,
                "recovery_success": 1.0,
                "failure_penalty_s": 0.0,
            },
        )
        failed = evaluate_mission(
            profile,
            demo_configuration(),
            {
                "settling_time_s": 2100.0,
                "recovery_time_s": 900.0,
                "recovery_success": 0.0,
                "failure_penalty_s": 10000.0,
            },
        )

        self.assertEqual(successful.objectives["J_response_s"], 15.0)
        self.assertEqual(successful.diagnostics["response_phase_settling_time_s"], 8270.0)
        self.assertGreater(failed.objectives["J_response_s"], successful.objectives["J_response_s"])

    def test_shared_three_objective_hypervolume(self) -> None:
        volume = _hypervolume_3d(
            np.asarray([[0.5, 0.5, 0.5]]),
            np.asarray([1.0, 1.0, 1.0]),
        )
        self.assertAlmostEqual(volume, 0.125)

    def test_ground_roll_model_reports_takeoff_and_landing(self) -> None:
        aircraft = AircraftParameters()
        takeoff = simulate_ground_roll(aircraft, "takeoff", 180.0)
        landing = simulate_ground_roll(aircraft, "landing", 0.0)
        self.assertTrue(takeoff.feasible)
        self.assertTrue(landing.feasible)
        self.assertGreater(takeoff.distance_m, 0.0)
        self.assertGreater(takeoff.shaft_energy_mj, 0.0)
        self.assertGreater(takeoff.mean_shaft_power_kw, 0.0)
        self.assertLess(takeoff.mean_shaft_power_kw, 180.0)
        self.assertGreater(landing.distance_m, 0.0)
        self.assertAlmostEqual(landing.terminal_speed_mps, 0.0)

    def test_ground_roll_slope_is_configurable_and_inputs_are_checked(self) -> None:
        aircraft = AircraftParameters()
        baseline = simulate_ground_roll(aircraft, "takeoff", 180.0)
        uphill = simulate_ground_roll(
            aircraft,
            "takeoff",
            180.0,
            assumptions=GroundRollAssumptions(runway_slope_deg=2.0),
        )
        self.assertGreater(uphill.distance_m, baseline.distance_m)
        fast_power_response = simulate_ground_roll(
            aircraft,
            "takeoff",
            180.0,
            assumptions=GroundRollAssumptions(
                shaft_power_response_time_constant_s=0.01,
                shaft_power_rate_limit_kwps=10000.0,
            ),
        )
        self.assertLess(fast_power_response.distance_m, baseline.distance_m)
        with self.assertRaises(ValueError):
            simulate_ground_roll(
                aircraft,
                "takeoff",
                180.0,
                assumptions=GroundRollAssumptions(rolling_friction_coefficient=-0.1),
            )

    def test_headwind_step_is_logged_and_changes_longitudinal_response(self) -> None:
        phase = PhaseCommand("cruise", 60.0, 1000.0, 40.0, 80.0, 0.0, time_step_s=5.0)
        baseline = simulate_mission(MissionDefinition((phase,)), AircraftParameters())
        disturbed = simulate_mission(
            MissionDefinition(
                (phase,),
                disturbances=(
                    LongitudinalDragDisturbance(
                        "cruise", 15.0, 20.0, 0.0, response_step_s=1.0, headwind_step_mps=8.0
                    ),
                ),
            ),
            AircraftParameters(),
        )
        self.assertEqual(float(disturbed.profile["equivalent_headwind_mps"].max()), 8.0)
        self.assertNotAlmostEqual(
            float(disturbed.profile["v_mps"].iloc[-1]),
            float(baseline.profile["v_mps"].iloc[-1]),
        )

    def test_sinusoidal_gust_is_repeatable_and_logged(self) -> None:
        phase = PhaseCommand("cruise", 30.0, 1000.0, 40.0, 80.0, 0.0, time_step_s=1.0)
        event = LongitudinalDragDisturbance(
            "cruise",
            5.0,
            15.0,
            0.0,
            response_step_s=0.5,
            headwind_gust_amplitude_mps=6.0,
            headwind_gust_period_s=5.0,
        )
        mission = MissionDefinition((phase,), disturbances=(event,))
        first = simulate_mission(mission, AircraftParameters()).profile
        second = simulate_mission(mission, AircraftParameters()).profile
        self.assertTrue(
            np.allclose(
                first["equivalent_headwind_mps"],
                second["equivalent_headwind_mps"],
            )
        )
        self.assertGreater(float(first["equivalent_headwind_mps"].abs().max()), 5.0)

    def test_mopso_runs_on_a_constrained_multiobjective_problem(self) -> None:
        class ToyProblem:
            lower_bounds = np.zeros(2)
            upper_bounds = np.ones(2)

            def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
                return rng.uniform(self.lower_bounds, self.upper_bounds, (count, 2))

            def evaluate(self, decision: np.ndarray) -> CandidateEvaluation:
                x = np.asarray(decision, dtype=float)
                result = EvaluationResult(
                    {
                        "J_energy_mj": float(x[0]),
                        "J_stability": float(1.0 - x[0]),
                        "J_response_s": float((x[1] - 0.5) ** 2),
                    },
                    {"box": 0.0},
                    {},
                )
                return CandidateEvaluation(x.copy(), result)

        run = MOPSO(resolution=2, generations=3, seed=7).run(ToyProblem())
        self.assertEqual(len(run.evaluations), 6)
        self.assertEqual(len(run.history), 4)
        self.assertEqual(set(run.generation_pareto_archive), {0, 1, 2, 3})
        self.assertTrue(all(item.feasible for item in run.evaluated_archive))

    def test_hybrid_nsga_mopso_preserves_elitist_population_and_archive(self) -> None:
        class ToyProblem:
            lower_bounds = np.zeros(2)
            upper_bounds = np.ones(2)

            def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
                return rng.uniform(self.lower_bounds, self.upper_bounds, (count, 2))

            def evaluate(self, decision: np.ndarray) -> CandidateEvaluation:
                x = np.asarray(decision, dtype=float)
                result = EvaluationResult(
                    {
                        "J_energy_mj": float(x[0]),
                        "J_stability": float(1.0 - x[0]),
                        "J_response_s": float((x[1] - 0.5) ** 2),
                    },
                    {"box": 0.0},
                    {},
                )
                return CandidateEvaluation(x.copy(), result)

        run = HybridNSGAIIMOPSO(resolution=2, generations=3, seed=19).run(ToyProblem())
        self.assertEqual(run.algorithm, "Hybrid NSGA-II/MOPSO")
        self.assertEqual(len(run.evaluations), 6)
        self.assertEqual(len(run.history), 4)
        self.assertEqual(set(run.generation_pareto_archive), {0, 1, 2, 3})
        self.assertTrue(all(item.feasible for item in run.evaluations))
        self.assertTrue(
            run.history["hybrid_ga_probability"].between(0.2, 0.8).all()
        )
        self.assertIn("hybrid_ga_improvement_rate", run.history)
        self.assertIn("hybrid_pso_improvement_rate", run.history)


if __name__ == "__main__":
    unittest.main(verbosity=2)
