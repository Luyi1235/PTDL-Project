"""Current seven-stage dual-EMRAX mission parameters."""

from itertools import accumulate
from math import isfinite


def propeller_parameters(dt_s=0.5):
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite")
    motor_input = tuple(1000.0 * value for value in (50, 60, 80, 54, 45, 10.97, 75))
    eta_motor = 0.96
    eta_controller = 0.95
    eta_propeller = 0.65
    shaft = tuple(value * eta_motor * eta_controller for value in motor_input)
    aircraft = tuple(value * eta_propeller for value in shaft)
    p = {
        "PROP_Phase_names": (
            "Ground test",
            "Taxi",
            "Takeoff",
            "Climb",
            "Cruise",
            "Descent",
            "Landing",
        ),
        "PROP_Phase_index": (1, 2, 3, 4, 5, 6, 7),
        "PROP_Phase_duration_s": (120, 15, 6, 394, 3600, 666, 10),
        "PROP_Motor_input_table_W": motor_input,
        "PROP_Motor_input_peak_target_W": max(motor_input),
        "PROP_Shaft_power_W": shaft,
        "PROP_Aircraft_power_W": aircraft,
        "PROP_Altitude_boundary_m": (
            0.0,
            0.0,
            0.0,
            0.0,
            5000.0,
            5000.0,
            0.0,
            0.0,
        ),
        "PROP_Eta_design": eta_propeller,
        "PROP_Eta_motor": eta_motor,
        "PROP_Eta_motor_controller": eta_controller,
        "PROP_Diameter_m": 1.4,
        "PROP_n_min_rps": 1e-3,
        "PROP_rho_min_kgpm3": 1e-6,
        "PROP_dt_s": float(dt_s),
    }
    p["PROP_Profile_end_s"] = tuple(accumulate(p["PROP_Phase_duration_s"]))
    p["PROP_Profile_start_s"] = (0,) + p["PROP_Profile_end_s"][:-1]
    p["PROP_Total_time_s"] = sum(p["PROP_Phase_duration_s"])
    return p


def init_propeller(dt_s=0.5):
    return propeller_parameters(dt_s)
