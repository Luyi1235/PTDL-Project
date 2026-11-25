from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 指标体系与评价函数 import demo_configuration, evaluate_mission
from 线性动态系统 import (
    ContinuousStateSpace,
    first_order_state_space_step,
    first_order_transfer_step,
)
from 优化算法配置库 import summarize_equal_budget_benchmark
from 飞推控闭环物理模型 import AircraftParameters
from 低阶地面滑跑模型 import simulate_ground_roll


class LinearDynamicsMethodTests(unittest.TestCase):
    def test_first_order_transfer_step_matches_unit_step_response(self) -> None:
        value = first_order_transfer_step(0.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(value, 1.0 - np.exp(-1.0), places=12)

    def test_first_order_state_space_matches_transfer_function_realization(self) -> None:
        transfer = first_order_transfer_step(2.0, 5.0, 0.4, 3.0, gain=1.5)
        state_space = first_order_state_space_step(
            2.0, 5.0, 0.4, 3.0, gain=1.5
        )
        self.assertAlmostEqual(state_space, transfer, places=12)

    def test_state_space_integrates_and_returns_output(self) -> None:
        model = ContinuousStateSpace(
            A=np.array([[-1.0]]),
            B=np.array([[1.0]]),
            C=np.array([[1.0]]),
            D=np.array([[0.0]]),
        )
        state, output = model.step(np.array([0.0]), np.array([1.0]), 0.1)
        self.assertAlmostEqual(state[0], 1.0 - np.exp(-0.1), delta=1e-6)
        self.assertAlmostEqual(output[0], state[0], places=12)


class EnergyManagementMetricTests(unittest.TestCase):
    def test_battery_efficiency_and_losses_are_reported(self) -> None:
        time_s = np.array([0.0, 10.0])
        profile = pd.DataFrame(
            {
                "time_s": time_s,
                "fuel_flow_kgps": [0.0, 0.0],
                "battery_discharge_kw": [2.0, 2.0],
                "battery_charge_kw": [1.0, 1.0],
                "soc": [0.8, 0.79],
                "h_m": [1000.0, 1000.0],
                "v_mps": [30.0, 30.0],
                "p_engine_kw": [10.0, 10.0],
                "p_motor_kw": [1.0, 1.0],
                "e_h_m": [0.0, 0.0],
                "e_v_mps": [0.0, 0.0],
                "e_theta_rad": [0.0, 0.0],
                "q_radps": [0.0, 0.0],
            }
        )
        result = evaluate_mission(profile, demo_configuration())
        self.assertGreater(result.diagnostics["battery_conversion_loss_mj"], 0.0)
        self.assertLess(result.diagnostics["battery_energy_management_efficiency"], 1.0)
        self.assertEqual(result.diagnostics["battery_efficiency_assessment_valid"], 1.0)
        self.assertIn("J_energy_mj", result.objectives)
        self.assertEqual(len(result.objectives), 3)


class GroundRollHandoffTests(unittest.TestCase):
    def test_takeoff_reports_energy_split_fuel_and_soc_handoff(self) -> None:
        aircraft = AircraftParameters(
            wing_area_m2=32.0,
            prop_disk_area_m2=12.0,
            initial_fuel_kg=500.0,
            battery_capacity_mj=120.0,
        )
        result = simulate_ground_roll(
            aircraft,
            "takeoff",
            available_shaft_power_kw=180.0,
            engine_power_fraction=0.8,
            initial_soc=0.9,
            battery_capacity_mj=120.0,
        )
        self.assertTrue(result.feasible)
        self.assertGreater(result.terminal_speed_mps, 0.0)
        self.assertAlmostEqual(
            result.shaft_energy_mj,
            result.engine_shaft_energy_mj + result.motor_shaft_energy_mj,
            places=10,
        )
        self.assertGreater(result.estimated_fuel_used_kg, 0.0)
        self.assertGreater(result.estimated_battery_energy_used_mj, 0.0)
        self.assertLess(result.terminal_soc_estimated, 0.9)


class AlgorithmCatalogTests(unittest.TestCase):
    def test_catalog_keeps_tradeoff_ranks_without_declaring_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats = pd.DataFrame(
                [
                    {
                        "algorithm": "NSGA-II",
                        "seed_count": 3,
                        "final_feasible_ratio_mean": 1.0,
                        "runtime_s_mean": 10.0,
                        "final_shared_hypervolume_mean": 0.6,
                    },
                    {
                        "algorithm": "MOPSO",
                        "seed_count": 3,
                        "final_feasible_ratio_mean": 0.7,
                        "runtime_s_mean": 8.0,
                        "final_shared_hypervolume_mean": 0.9,
                    },
                ]
            )
            stats_path = root / "stats.csv"
            stats.to_csv(stats_path, index=False)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({"generations": 5}), encoding="utf-8")
            catalog = summarize_equal_budget_benchmark(stats_path, metadata_path)
        self.assertEqual(len(catalog["algorithm_profiles"]), 2)
        self.assertIn("No universal winner", catalog["selection_rule"])
        mopso = next(
            item for item in catalog["algorithm_profiles"] if item["algorithm"] == "MOPSO"
        )
        self.assertEqual(mopso["hypervolume_rank"], 1)
        self.assertEqual(mopso["runtime_rank"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
