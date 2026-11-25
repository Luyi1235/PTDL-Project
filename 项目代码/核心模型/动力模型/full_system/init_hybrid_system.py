"""Assembly parameters from init_hybrid_system.m and execute_hybrid_system.m."""

from ..battery_rint.init_battery_rint import init_battery_rint
from ..generator_gearbox.init_generator_gearbox import init_generator_gearbox
from ..motor_pmsm.init_pmsm import init_pmsm
from ..power_electronics_ems.init_power_electronics_ems import init_power_electronics_ems
from ..propeller.init_propeller import init_propeller


def init_hybrid_system(initial_soc=0.90):
    """Return current 81s3p/dual-EMRAX assembly parameter dictionaries."""
    motor = init_pmsm(dt_s=0.001)
    generator = init_generator_gearbox()
    generator["GEN_N_engine_design_rpm"] = 5000.0
    generator["GEN_GearRatio"] = 5000.0 / generator["GEN_N_design_rpm"]
    ems = init_power_electronics_ems()
    ems["PE_Eta_inverter"] = 0.95
    ems["PE_Eta_rectifier"] = 0.96 / generator["GEN_Eta"]
    ems["PE_Pgen_dc_design_W"] = generator["GEN_P_elec_design_W"] * ems["PE_Eta_rectifier"]
    ems["PE_Eta_dcdc"] = 1.0
    battery = init_battery_rint(initial_soc=initial_soc)
    propeller = init_propeller(dt_s=0.5)
    if battery["BAT_Ns"] != 81 or battery["BAT_Np"] != 3:
        raise ValueError("The active battery must be 81s3p")
    if abs(battery["BAT_Capacity_Ah"] - 55.65) > 1e-12:
        raise ValueError("The active battery capacity must be 55.65 Ah")
    if max(propeller["PROP_Motor_input_table_W"]) > 2 * motor["MOT_P_input_max_W"]:
        raise ValueError("Mission input power exceeds the dual-motor limit")
    return {
        "battery": battery,
        "generator": generator, "motor": motor, "ems": ems,
        "propeller": propeller,
        "number_of_motors": 2, "internal_step_s": 0.001,
        "initial_generator_feedback_W": 0.0,
        "initial_battery_feedback_W": 0.0,
        "initial_available_power_W": ems["PE_Pgen_dc_design_W"],
        "mission_power_peak_target_W": 80000.0,
        "powertrain_status": (
            "Current 81s3p battery, dual EMRAX 228 MV motors, 62.5 kW engine, "
            "60 kW generator and descent-off three-layer EMS parameter set"
        ),
    }
