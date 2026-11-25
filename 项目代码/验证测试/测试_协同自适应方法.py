import sys
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "核心模型"))
from 协同自适应方法 import (
    ResidualHealthEstimator, identify_delayed_response, glide_command,
    EstimatedHealthCoordinator, PhaseScheduledMPC,
)
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator
from 飞推控闭环物理模型 import DispatchObservation


class AdaptiveMethodsTests(unittest.TestCase):
    def test_delay_identification(self):
        rng = np.random.default_rng(7)
        u = rng.uniform(10,80,300)
        y = np.zeros(300)
        for k in range(3,299):
            y[k+1] = np.exp(-1/4)*y[k]+(1-np.exp(-1/4))*u[k-3]
        result = identify_delayed_response(u,y,1,6)
        self.assertEqual(result["delay_steps"],3)
        self.assertAlmostEqual(result["time_constant_s"],4,places=6)
        self.assertLess(result["validation_rmse"],1e-10)

    def test_unexcited_rejected(self):
        with self.assertRaises(ValueError):
            identify_delayed_response(np.ones(40),np.ones(40),1)

    def measurements(self):
        return dict(dt_s=1,soc_before=.8,soc_after=.8-20/(3600*60),
                    current_a=20,nominal_capacity_ah=100,command_kw=20,engine_kw=20,
                    nominal_cap_kw=100,speed_error_mps=0,altitude_error_m=0)

    def test_capacity_estimate_without_fault_truth(self):
        e=ResidualHealthEstimator()
        for _ in range(30):
            e.update(**self.measurements())
        self.assertAlmostEqual(e.battery_fraction,.6,places=5)
        self.assertEqual(e.engine_observations,0)

    def test_soc_clamp_not_capacity_evidence(self):
        e=ResidualHealthEstimator()
        e.update(**(self.measurements()|{"soc_after":.2}))
        self.assertEqual(e.capacity_observations,0)

    def test_engine_excitation_required(self):
        e=ResidualHealthEstimator()
        for _ in range(80):
            e.update(**(self.measurements()|{"command_kw":100,"engine_kw":60}))
        self.assertAlmostEqual(e.engine_fraction,.6,places=5)

    def test_glide_speed_priority(self):
        self.assertLess(glide_command(25,30,500,10000,1000),glide_command(35,30,500,10000,1000))
        self.assertGreater(glide_command(30,30,500,10000,2),-.5)

    def test_invalid_observation(self):
        with self.assertRaises(ValueError):
            ResidualHealthEstimator().update(**(self.measurements()|{"current_a":float("nan")}))

    def test_coulomb_efficiency_is_in_estimate(self):
        e=ResidualHealthEstimator()
        for _ in range(40):
            e.update(**(self.measurements()|{
                "soc_after":.8-.95*20/(3600*60),"coulombic_efficiency":.95}))
        self.assertAlmostEqual(e.battery_fraction,.6,places=6)

    def test_mpc_energy_uses_kw_seconds_in_mj(self):
        from 智能能量管理与容错调度 import fuel_energy_mj
        self.assertAlmostEqual(float(fuel_energy_mj(100,10,.25)),4.0)
        np.testing.assert_allclose(fuel_energy_mj([100,50],10,[.25,.5]),[4,1])

    def test_component_delay_and_lag(self):
        from ptdl_hybrid_system_adapter import HybridSystemPowertrain, HybridSystemAdapterConfig
        from 飞推控闭环物理模型 import AircraftParameters, HybridPowertrainRequest
        aircraft=AircraftParameters()
        plant=HybridSystemPowertrain(HybridSystemAdapterConfig(
            engine_time_constant_s=2,engine_command_delay_s=1))
        state=plant.initial_state(aircraft,.85)
        request=HybridPowertrainRequest("test",20,100,20,-30,30,engine_target_is_dc=True)
        values=[plant.step(state,request,.25,aircraft).engine_power_kw for _ in range(20)]
        np.testing.assert_allclose(values[:4],0)
        self.assertGreater(values[-1],values[4])
        self.assertLess(values[-1],20)

    def test_negative_dynamic_parameter_rejected(self):
        from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig
        with self.assertRaises(ValueError):
            HybridSystemAdapterConfig(engine_time_constant_s=-1)

    def test_short_delay_does_not_expand_to_coarse_update_interval(self):
        from ptdl_hybrid_system_adapter import HybridSystemPowertrain, HybridSystemAdapterConfig
        from 飞推控闭环物理模型 import AircraftParameters, HybridPowertrainRequest
        aircraft=AircraftParameters()
        plant=HybridSystemPowertrain(HybridSystemAdapterConfig(engine_command_delay_s=1))
        state=plant.initial_state(aircraft,.85)
        output=plant.step(state,HybridPowertrainRequest("test",20,100,20,-30,30),300,aircraft)
        self.assertGreater(output.engine_power_kw,19)


if __name__=="__main__":
    unittest.main()
