from dataclasses import replace
from pathlib import Path
import sys
import unittest

CORE = Path(__file__).resolve().parents[1] / "核心模型"
sys.path.insert(0, str(CORE))

from ptdl_hybrid_system_adapter import HybridSystemPowertrain
from 飞推控闭环物理模型 import AircraftParameters, HybridPowertrainRequest
from 电池能量收支 import BatteryEnergyLedger


class BatteryAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.plant = HybridSystemPowertrain()
        self.aircraft = AircraftParameters()
        self.state = self.plant.initial_state(self.aircraft, 0.85)
        self.nominal = self.state.nominal_battery_parameters

    def total_charge(self):
        return (
            self.nominal["BAT_Capacity_Ah"] * self.state.battery_available_fraction
            * self.state.battery.soc + self.state.battery_isolated_charge_ah
        )

    def test_partial_isolation_changes_capacity_and_resistance(self):
        before = self.total_charge()
        self.plant.set_battery_availability(self.state, 0.6)
        self.assertAlmostEqual(self.state.battery.parameters["BAT_Capacity_Ah"], self.nominal["BAT_Capacity_Ah"] * 0.6)
        self.assertAlmostEqual(self.state.battery.parameters["BAT_Rdis_pack_Ohm"][0], self.nominal["BAT_Rdis_pack_Ohm"][0] / 0.6)
        self.assertAlmostEqual(self.total_charge(), before)

    def test_repeated_changes_and_full_isolation_preserve_charge(self):
        before = self.total_charge()
        for fraction in (0.6, 0.2, 0.8, 0.0, 0.4, 1.0, 1.0):
            self.plant.set_battery_availability(self.state, fraction)
            self.assertAlmostEqual(self.total_charge(), before)
        self.assertAlmostEqual(self.state.battery.soc, 0.85)
        self.assertAlmostEqual(self.state.battery_isolated_charge_ah, 0.0)

    def test_discharge_and_recovery_do_not_reset_state(self):
        before = self.total_charge()
        request = HybridPowertrainRequest("test", 0.0, 0.0, 3.0, 0.0, 30.0, 0.6)
        output = self.plant.step(self.state, request, 10.0, self.aircraft)
        self.assertLess(output.soc, 0.85)
        charge_after = self.total_charge()
        self.assertLess(charge_after, before)
        self.plant.set_battery_availability(self.state, 1.0)
        self.assertAlmostEqual(self.total_charge(), charge_after)
        self.assertLess(self.state.battery.soc, 0.85)
        self.assertGreater(self.state.battery.soc, output.soc)

    def test_zero_availability_blocks_battery_power(self):
        output = self.plant.step(
            self.state, HybridPowertrainRequest("test", 0.0, 0.0, 3.0, -30.0, 30.0, 0.0),
            2.0, self.aircraft,
        )
        self.assertAlmostEqual(output.battery_current_a, 0.0)
        self.assertAlmostEqual(output.battery_terminal_power_kw, 0.0)
        self.assertAlmostEqual(output.propulsor_input_power_kw, 0.0)

    def test_unavailable_fraction_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.plant.set_battery_availability(self.state, value)

    def test_unit_availability_preserves_previous_component_result(self):
        request = HybridPowertrainRequest("test", 10.0, 120.0, 15.0, -30.0, 30.0)
        baseline = self.plant.step(self.state, request, 2.0, self.aircraft)
        other = self.plant.initial_state(self.aircraft, 0.85)
        actual = self.plant.step(other, replace(request, battery_available_energy_fraction=1.0), 2.0, self.aircraft)
        self.assertEqual(baseline.soc, actual.soc)
        self.assertEqual(baseline.propulsor_input_power_kw, actual.propulsor_input_power_kw)


class BatteryLedgerTests(unittest.TestCase):
    def test_separate_charge_discharge_efficiency(self):
        ledger = BatteryEnergyLedger()
        ledger.update(9.0, 10.0, 100.0)
        ledger.update(-10.0, -9.0, 100.0)
        result = ledger.summary()
        self.assertAlmostEqual(result["discharge_efficiency"], 0.9)
        self.assertAlmostEqual(result["charge_efficiency"], 0.9)
        self.assertAlmostEqual(result["internal_loss_mj"], 0.2)
        self.assertAlmostEqual(result["net_chemical_withdrawal_mj"], 0.1)
        self.assertAlmostEqual(result["flow_balance_residual_mj"], 0.0)

    def test_no_flow_is_not_perfect_efficiency(self):
        ledger = BatteryEnergyLedger()
        ledger.update(0.0, 0.0, 2.0)
        self.assertIsNone(ledger.summary()["conversion_efficiency"])

    def test_invalid_energy_input_rejected(self):
        for powers in ((10, 9, 1), (1, -1, 1), (0, 0, 0), (float("nan"), 1, 1)):
            with self.assertRaises(ValueError):
                BatteryEnergyLedger().update(*powers)


if __name__ == "__main__":
    unittest.main()
