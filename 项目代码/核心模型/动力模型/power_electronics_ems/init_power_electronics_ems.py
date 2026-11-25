"""Average-power and three-layer EMS parameters from the current MATLAB model."""

from math import isfinite


def ems_parameters(dt_s=1.0):
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite")
    p = {
        "PE_Eta_motor": 0.96,
        "PE_Eta_inverter": 0.95,
        "PE_Pgen_dc_design_W": 4800.0,
        "PE_SOC_min": 0.20,
        "PE_SOC_max": 0.90,
        "PE_Eta_rectifier": 1.0,
        "PE_Eta_generator_rectifier_design_statement": 0.95,
        "PE_Eta_dcdc": 1.0,
        "PE_Phase_bp": (1, 2, 3, 4, 5, 6, 7),
        "PE_Mode2_battery_enable": (1, 1, 1, 1, 0, 0, 1),
        "PE_Bus_power_tolerance_W": 1e-6,
        "PE_dt_s": float(dt_s),
        "PE_ThreeLayer_mode_threshold": 2.5,
        "PE_SOC_engine_stop": 0.90,
        "PE_SOC_engine_hysteresis": 0.50,
        "PE_Engine_state_init": 1.0,
        "PE_Engine_supervisor_dt_s": float(dt_s),
        "PE_Critical_phase_takeoff": 3,
        "PE_Critical_phase_climb": 4,
        "PE_Critical_phase_landing": 7,
        "PE_Descent_phase": 6,
    }
    p["PE_SOC_engine_start"] = (
        p["PE_SOC_engine_stop"] - p["PE_SOC_engine_hysteresis"]
    )
    return p


def init_power_electronics_ems(dt_s=1.0):
    return ems_parameters(dt_s)
