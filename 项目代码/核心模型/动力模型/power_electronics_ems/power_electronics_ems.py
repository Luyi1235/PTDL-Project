"""Average-power converters and the current three-layer EMS state machine."""

from bisect import bisect_right
from math import isfinite

from .init_power_electronics_ems import init_power_electronics_ems


class PowerElectronicsEMS:
    output_names = (
        "P_generator_ac_cmd_W",
        "P_battery_terminal_cmd_W",
        "P_motor_ac_W",
        "P_motor_dc_W",
        "P_generator_dc_W",
        "P_battery_dc_W",
        "P_bus_deficit_W",
        "P_bus_available",
        "P_inverter_loss_W",
        "P_rectifier_loss_W",
        "P_dcdc_loss_W",
        "P_battery_bus_cmd_W",
        "Engine_enable_cmd",
        "Engine_supervisor_state",
    )

    def __init__(self, parameters=None):
        self.parameters = dict(
            init_power_electronics_ems() if parameters is None else parameters
        )
        self.reset()

    def reset(self, engine_state=None):
        state = (
            self.parameters["PE_Engine_state_init"]
            if engine_state is None
            else float(engine_state)
        )
        if state not in (0.0, 1.0):
            raise ValueError("engine_state must be 0 or 1")
        self.engine_state = state

    def outputs(
        self,
        phase_index,
        mode,
        motor_ac_actual_w,
        soc,
        generator_ac_actual_w,
        battery_terminal_actual_w,
        motor_dc_demand_w,
        *,
        update_state=True,
    ):
        values = tuple(
            map(
                float,
                (
                    phase_index,
                    mode,
                    motor_ac_actual_w,
                    soc,
                    generator_ac_actual_w,
                    battery_terminal_actual_w,
                    motor_dc_demand_w,
                ),
            )
        )
        if not all(map(isfinite, values)):
            raise ValueError("EMS inputs must be finite")
        phase, mode, motor_ac, soc, gen_ac, battery_terminal, demand = values
        p = self.parameters

        motor_dc = motor_ac / p["PE_Eta_inverter"]
        generator_dc = gen_ac * p["PE_Eta_rectifier"]
        battery_dc = (
            max(battery_terminal, 0.0) * p["PE_Eta_dcdc"]
            + min(battery_terminal, 0.0) / p["PE_Eta_dcdc"]
        )
        deficit = motor_dc - generator_dc - battery_dc

        critical = int(round(phase)) in (
            p["PE_Critical_phase_takeoff"],
            p["PE_Critical_phase_climb"],
            p["PE_Critical_phase_landing"],
        )
        start_request = (
            critical
            or soc <= p["PE_SOC_engine_start"]
            or deficit > p["PE_Bus_power_tolerance_W"]
        )
        descent = int(round(phase)) == p["PE_Descent_phase"]
        stop_request = (soc >= p["PE_SOC_engine_stop"] and not critical) or descent
        if start_request:
            next_state = 1.0
        elif stop_request:
            next_state = 0.0
        else:
            next_state = self.engine_state

        phase_lookup_index = min(
            max(bisect_right(p["PE_Phase_bp"], phase) - 1, 0),
            len(p["PE_Phase_bp"]) - 1,
        )
        mode2_battery_enabled = p["PE_Mode2_battery_enable"][phase_lookup_index]
        mode1_target = p["PE_Pgen_dc_design_W"]
        mode2_target = (
            mode2_battery_enabled * p["PE_Pgen_dc_design_W"]
            + (1 - mode2_battery_enabled) * demand
        )
        legacy_target = mode2_target if mode >= 1.5 else mode1_target
        mode3_enabled = mode > p["PE_ThreeLayer_mode_threshold"]
        mode3_target = (
            demand if soc >= p["PE_SOC_engine_stop"] else p["PE_Pgen_dc_design_W"]
        ) * next_state
        target = mode3_target if mode3_enabled else legacy_target
        engine_enable = next_state if mode3_enabled else 1.0

        raw_battery = demand - target
        discharge = max(raw_battery, 0.0) * (soc > p["PE_SOC_min"])
        charge = min(raw_battery, 0.0) * (soc < p["PE_SOC_max"])
        battery_bus_command = discharge + charge
        battery_terminal_command = (
            discharge / p["PE_Eta_dcdc"] + charge * p["PE_Eta_dcdc"]
        )
        result = (
            target / p["PE_Eta_rectifier"],
            battery_terminal_command,
            motor_ac,
            motor_dc,
            generator_dc,
            battery_dc,
            deficit,
            deficit <= p["PE_Bus_power_tolerance_W"],
            motor_dc - motor_ac,
            gen_ac - generator_dc,
            battery_terminal - battery_dc,
            battery_bus_command,
            engine_enable,
            next_state,
        )
        if update_state:
            self.engine_state = next_state
        return dict(zip(self.output_names, result))
