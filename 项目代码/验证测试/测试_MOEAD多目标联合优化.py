"""Regression tests for the PTDL 2.2.2 optimization prototype."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
RUNNERS = ROOT / "运行脚本"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
if str(RUNNERS) not in sys.path:
    sys.path.insert(0, str(RUNNERS))

from MOEAD多目标联合优化 import (
    CoupledPTDLControlProblem,
    LOWER_BOUNDS,
    MOEAD,
    UPPER_BOUNDS,
    OBJECTIVE_LABELS,
    PTDLControlProblem,
    PTDLEnduranceJointProblem,
    pareto_indices,
    run_coupled_moead_baseline,
    run_delivery_experiment,
    run_ptdl_80h_joint_moead,
    run_source_profile_moead_audit,
    simulate_surrogate_mission,
)
from PTDL联合优化高保真接口 import (
    LiveOpenVSPAeroProvider,
    PTDL6DoFOptimizerValidator,
)
from 指标体系与评价函数 import ObjectiveNormalizer
from ULTRA基准标定 import calibrate_ultra_baseline
from 机体参数化设计与气动阻力 import BASELINE, aero_from_design, design_metrics
from 运行_PTDL估计六自由度全任务闭环 import (
    flare_target_vertical_speed_up_mps,
    mission_completion_status,
    phase_schedule,
    select_landing_guidance_phase,
)


class SurrogateAndOptimizerTests(unittest.TestCase):
    def test_objective_normalizer_handles_constant_large_scale_axes(self) -> None:
        samples = pd.DataFrame(
            {
                "J_energy_mj": [20767.1465, 20767.1465, 20767.1465],
                "J_stability": [112.67618, 112.67618, 112.67618],
                "J_response_s": [0.0, 0.0, 0.0],
            }
        )
        normalizer = ObjectiveNormalizer.from_samples(samples)
        result = normalizer.transform(
            {
                "J_energy_mj": 20767.1465,
                "J_stability": 112.67618,
                "J_response_s": 0.0,
            }
        )

        self.assertTrue(all(math.isfinite(value) for value in result.values()))
        self.assertEqual(list(result.values()), [0.0, 0.0, 0.0])

    def test_full_mission_schedule_keeps_requested_duration_before_landing_grace(self) -> None:
        schedule = phase_schedule(80.0, 2200.0, 2400.0, 32.0, "full")

        self.assertAlmostEqual(float(schedule[-1]["end_s"]), 80.0 * 3600.0)
        self.assertAlmostEqual(float(schedule[-2]["duration_s"]), 1200.0)
        self.assertAlmostEqual(float(schedule[-1]["duration_s"]), 900.0)
        self.assertAlmostEqual(
            float(schedule[-1]["end_s"]) - float(schedule[-2]["start_s"]),
            2100.0,
        )

    def test_phase_schedule_supports_bounded_power_loss_and_recovery(self) -> None:
        schedule = phase_schedule(
            6.0,
            2200.0,
            2400.0,
            32.0,
            "full",
            {"long_descent": {"power_off_duration_s": 300.0}},
        )
        long_descent = next(
            item for item in schedule if item["name"] == "long_descent"
        )

        self.assertEqual(float(long_descent["power_off_duration_s"]), 300.0)
        self.assertLess(
            float(long_descent["power_off_duration_s"]),
            float(long_descent["duration_s"]),
        )
        self.assertEqual(float(long_descent["engine_cap_kw"]), 70.0)
        self.assertEqual(float(long_descent["motor_cap_kw"]), 0.0)

    def test_endurance_then_land_preserves_80h_before_descent(self) -> None:
        schedule = phase_schedule(
            80.0, 2200.0, 2400.0, 32.0, "endurance_then_land"
        )
        endurance_end_s = float(
            next(item for item in schedule if item["name"] == "endurance_loiter")[
                "end_s"
            ]
        )
        profile_end_s = float(schedule[-1]["end_s"])

        self.assertAlmostEqual(endurance_end_s, 80.0 * 3600.0)
        self.assertGreater(profile_end_s, endurance_end_s)
        self.assertAlmostEqual(profile_end_s - endurance_end_s, 8100.0)
        self.assertFalse(
            mission_completion_status(
                mission_mode="endurance_then_land",
                flight_profile_completed=True,
                fuel_feasible=True,
                touchdown=True,
                elapsed_s=79.99 * 3600.0,
                requested_duration_s=80.0 * 3600.0,
            )
        )
        self.assertTrue(
            mission_completion_status(
                mission_mode="endurance_then_land",
                flight_profile_completed=True,
                fuel_feasible=True,
                touchdown=True,
                elapsed_s=82.25 * 3600.0,
                requested_duration_s=80.0 * 3600.0,
            )
        )

    def test_full_mission_accepts_landing_in_terminal_flare_before_nominal_end(self) -> None:
        self.assertTrue(
            mission_completion_status(
                mission_mode="full",
                flight_profile_completed=True,
                fuel_feasible=True,
                touchdown=True,
                elapsed_s=5.9 * 3600.0,
                requested_duration_s=6.0 * 3600.0,
            )
        )
        self.assertFalse(
            mission_completion_status(
                mission_mode="full",
                flight_profile_completed=True,
                fuel_feasible=True,
                touchdown=False,
                elapsed_s=6.0 * 3600.0,
                requested_duration_s=6.0 * 3600.0,
            )
        )
        self.assertFalse(
            mission_completion_status(
                mission_mode="full",
                flight_profile_completed=True,
                fuel_feasible=False,
                touchdown=True,
                elapsed_s=5.9 * 3600.0,
                requested_duration_s=6.0 * 3600.0,
            )
        )

    def test_flare_is_altitude_gated_and_approach_continues_when_high(self) -> None:
        approach = {"name": "approach_descent", "end_s": 10.0}
        flare = {"name": "flare", "end_s": 20.0}
        selected, active = select_landing_guidance_phase(
            scheduled_phase=flare,
            approach_phase=approach,
            flare_phase=flare,
            altitude_m=1600.0,
            terminal_flare_active=False,
        )
        self.assertEqual(selected["name"], "approach_descent")
        self.assertFalse(active)

        selected, active = select_landing_guidance_phase(
            scheduled_phase=approach,
            approach_phase=approach,
            flare_phase=flare,
            altitude_m=120.0,
            terminal_flare_active=False,
        )
        self.assertEqual(selected["name"], "flare")
        self.assertTrue(active)

    def test_flare_sink_target_never_reverses_into_climb(self) -> None:
        self.assertAlmostEqual(flare_target_vertical_speed_up_mps(300.0), -1.0)
        self.assertAlmostEqual(flare_target_vertical_speed_up_mps(60.0), -0.6)
        self.assertAlmostEqual(flare_target_vertical_speed_up_mps(0.0), -0.25)
        self.assertLess(flare_target_vertical_speed_up_mps(60.0), 0.0)

    def test_candidate_jsbsim_induced_drag_coefficient_uses_cl_squared_once(self) -> None:
        calibration = calibrate_ultra_baseline()
        aero = aero_from_design(BASELINE, calibration)
        model, temporary = PTDL6DoFOptimizerValidator()._write_candidate_model(
            BASELINE,
            aero,
            design_metrics(BASELINE, calibration.reference),
        )
        try:
            root = ET.parse(model).getroot()
            induced_values = root.findall(
                "./aerodynamics/axis[@name='DRAG']/function/product/sum/product/value"
            )
            self.assertTrue(induced_values)
            expected = 1.0 / (
                math.pi
                * BASELINE.aspect_ratio
                * max(aero.oswald_efficiency, 1e-4)
            )
            self.assertAlmostEqual(float(induced_values[0].text), expected, places=10)
        finally:
            temporary.cleanup()

    def test_vspaero_worker_result_is_parsed_amid_solver_console_output(self) -> None:
        payload = LiveOpenVSPAeroProvider._parse_worker_output(
            'VSPAERO solver text {"cl_alpha_per_rad": 6.1, "oswald_efficiency": 0.82} '
            "solver shutdown text\n"
        )

        self.assertEqual(payload["cl_alpha_per_rad"], 6.1)
        self.assertEqual(payload["oswald_efficiency"], 0.82)

    def test_joint_endurance_problem_includes_sixdof_hard_constraints(self) -> None:
        def validator(**_kwargs: object) -> dict[str, dict[str, float]]:
            return {
                "constraints": {"sixdof_test_envelope": -0.25},
                "diagnostics": {"sixdof_test_metric": 1.5},
            }

        problem = PTDLEnduranceJointProblem(sixdof_validator=validator)
        candidate = problem.evaluate(problem.sample(1, np.random.default_rng(19))[0])

        self.assertIn("sixdof_test_envelope", candidate.result.constraints)
        self.assertLessEqual(candidate.result.constraints["sixdof_test_envelope"], 0.0)
        self.assertEqual(candidate.result.diagnostics["sixdof_test_metric"], 1.5)

    def test_full_mission_sixdof_validator_receives_candidate_phase_schedule(self) -> None:
        class FullMissionValidator:
            full_mission = True

            def __init__(self) -> None:
                self.received: dict[str, object] = {}

            def __call__(self, **kwargs: object) -> dict[str, dict[str, float]]:
                self.received = kwargs
                return {
                    "constraints": {"sixdof_full_mission_complete": 0.0},
                    "diagnostics": {"sixdof_full_mission_completed": 1.0},
                }

        validator = FullMissionValidator()
        problem = PTDLEnduranceJointProblem(sixdof_validator=validator)
        decision = problem.sample(1, np.random.default_rng(22))[0]
        candidate = problem.evaluate(decision)

        self.assertIn("engine_caps_kw", validator.received)
        self.assertIn("motor_assist_kw", validator.received)
        self.assertIn("phase_speeds_mps", validator.received)
        self.assertIn("phase_altitudes_m", validator.received)
        self.assertEqual(
            validator.received["engine_caps_kw"]["endurance_loiter"],
            decision[problem.decision_labels.index("engine_cap_endurance_loiter_kw")],
        )
        self.assertIn("sixdof_full_mission_complete", candidate.result.constraints)
        self.assertEqual(candidate.result.diagnostics["sixdof_full_mission_completed"], 1.0)

    def test_powered_descent_disables_motor_optimizer_variable(self) -> None:
        problem = PTDLEnduranceJointProblem()
        motor_index = problem.decision_labels.index("motor_assist_long_descent_kw")

        self.assertEqual(problem.lower_bounds[motor_index], 0.0)
        self.assertEqual(problem.upper_bounds[motor_index], 0.0)

    def test_joint_problem_can_limit_design_to_aerodynamic_doe_bounds(self) -> None:
        problem = PTDLEnduranceJointProblem(
            design_feature_bounds={
                "aspect_ratio": (20.0, 30.0),
                "sweep_deg": (0.2, 7.0),
            }
        )
        aspect_index = problem.decision_labels.index("aspect_ratio")
        sweep_index = problem.decision_labels.index("sweep_deg")
        self.assertEqual(problem.lower_bounds[aspect_index], 20.0)
        self.assertEqual(problem.upper_bounds[aspect_index], 30.0)
        self.assertEqual(problem.lower_bounds[sweep_index], 0.2)
        self.assertEqual(problem.upper_bounds[sweep_index], 7.0)
        samples = problem.sample(12, np.random.default_rng(12))
        self.assertTrue(np.all(samples[:, aspect_index] >= 20.0))
        self.assertTrue(np.all(samples[:, aspect_index] <= 30.0))
        self.assertTrue(np.all(samples[:, sweep_index] >= 0.2))
        self.assertTrue(np.all(samples[:, sweep_index] <= 7.0))

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
            generation_archive = pd.read_csv(
                output_dir / "ptdl_80h_joint_moead_generation_feasible_pareto.csv"
            )
            final_pareto = pd.read_csv(
                output_dir / "ptdl_80h_joint_moead_final_population_feasible_pareto.csv"
            )
            self.assertEqual(set(generation_archive["generation"]), {0, 1})
            self.assertTrue(generation_archive["feasible"].all())
            self.assertTrue(final_pareto["feasible"].all())
            self.assertEqual(
                delivery["run"].history["historical_pareto_count"].tolist(),
                [float((generation_archive["generation"] == generation).sum()) for generation in (0, 1)],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
