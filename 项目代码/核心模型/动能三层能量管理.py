"""PTDL dispatch interface for the supplied 1005 three-layer EMS."""

from 动力模型.full_system.init_hybrid_system import init_hybrid_system
from 动力模型.power_electronics_ems.power_electronics_ems import PowerElectronicsEMS
from 飞推控闭环物理模型 import PowerDispatch


class SourceThreeLayerCoordinator:
    PHASES = {
        "ground_test": 1, "taxi": 2, "takeoff": 3, "takeoff_roll": 3,
        "main_climb_to_leveloff": 4, "short_leveloff_to_isr_altitude": 4,
        "climb": 4, "endurance_loiter": 5, "cruise": 5,
        "long_descent": 6, "approach_descent": 6, "descent": 6,
        "glide": 6, "unpowered_glide": 6, "flare": 7,
        "landing": 7, "landing_roll": 7,
    }

    def __init__(self, inverter_efficiency=0.95):
        parameters = dict(init_hybrid_system()["ems"])
        parameters["PE_Eta_inverter"] = float(inverter_efficiency)
        self.ems = PowerElectronicsEMS(parameters)
        self.reset()

    def reset(self):
        self.ems.reset()
        self.generator_ac_w = 0.0
        self.battery_terminal_w = 0.0
        self.motor_ac_w = 0.0

    def update_powertrain_feedback(self, output):
        p = self.ems.parameters
        self.generator_ac_w = output.engine_power_kw * 1000.0 / p["PE_Eta_rectifier"]
        self.battery_terminal_w = output.battery_terminal_power_kw * 1000.0
        self.motor_ac_w = getattr(
            output, "motor_dc_power_kw",
            max(output.engine_power_kw + output.motor_power_kw, 0.0),
        ) * 1000.0 * p["PE_Eta_inverter"]

    def dispatch(self, observation):
        p = self.ems.parameters
        p["PE_Pgen_dc_design_W"] = max(observation.engine_power_cap_kw, 0.0) * 1000.0
        p["PE_SOC_min"] = observation.battery_soc_min
        p["PE_SOC_max"] = min(0.90, observation.battery_soc_max)
        demand_kw = observation.required_motor_dc_kw
        if demand_kw is None:
            demand_kw = observation.required_propulsion_kw / (0.96 * p["PE_Eta_inverter"])
        result = self.ems.outputs(
            self.PHASES.get(observation.phase_name, 5), 3,
            self.motor_ac_w, observation.soc, self.generator_ac_w,
            self.battery_terminal_w, demand_kw * 1000.0,
        )
        engine_kw = min(
            result["P_generator_ac_cmd_W"] * p["PE_Eta_rectifier"] / 1000.0,
            observation.engine_power_cap_kw,
        )
        return PowerDispatch(
            engine_target_kw=max(engine_kw, 0.0),
            soc_target=p["PE_SOC_engine_stop"],
            mode="source_ems_engine_on" if result["Engine_enable_cmd"] else "source_ems_engine_off",
            engine_target_is_dc=True,
        )
