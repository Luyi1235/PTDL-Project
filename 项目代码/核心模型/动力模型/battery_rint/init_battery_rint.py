"""81s3p Rint parameters ported from the current MATLAB initializer.

OCV and resistance arrays are cell-level source data. Pack resistance follows
Rpack=(Ns/Np)*Rcell and pack capacity follows Np*Qcell.
"""

from math import isfinite


def battery_parameters(initial_soc=0.90):
    if not isfinite(initial_soc) or not 0.2 <= initial_soc <= 1.0:
        raise ValueError("initial_soc must be finite and in [0.2, 1.0]")
    p = {
        "BAT_SOC_bp": (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
        "BAT_Cell_Voc_V": (3.00, 3.40, 3.60, 3.70, 3.78, 3.82, 4.10),
        "BAT_Cell_Rdis_Ohm": (
            0.00105103578260870,
            0.000722524086956522,
            0.000554305260869565,
            0.000351218956521739,
            0.000363698956521739,
            0.000363698956521739,
            0.000406422826086957,
        ),
        "BAT_Cell_Rchg_Ohm": (
            0.000684840826086956,
            0.000586878304347826,
            0.000577214130434783,
            0.000512068043478261,
            0.000449946304347826,
            0.000449946304347826,
            0.000489159130434783,
        ),
        "BAT_Ns": 81,
        "BAT_Np": 3,
        "BAT_Cell_V_nom_V": 3.7,
        "BAT_Cell_V_min_V": 2.7,
        "BAT_Cell_V_max_V": 4.2,
        "BAT_Cell_Capacity_Ah": 18.55,
        "BAT_SOC_init": float(initial_soc),
        "BAT_SOC_min": 0.20,
        "BAT_SOC_max": 1.00,
        "BAT_CoulombicEfficiency": 0.95,
        "BAT_Motor_Vmin_pack_V": 240.0,
        "BAT_Discriminant_floor": 0.0,
    }
    p["BAT_Capacity_Ah"] = p["BAT_Np"] * p["BAT_Cell_Capacity_Ah"]
    p["BAT_Rint_series_multiplier"] = p["BAT_Ns"] / p["BAT_Np"]
    p["BAT_Voc_pack_V"] = tuple(
        p["BAT_Ns"] * value for value in p["BAT_Cell_Voc_V"]
    )
    p["BAT_Rdis_pack_Ohm"] = tuple(
        p["BAT_Rint_series_multiplier"] * value
        for value in p["BAT_Cell_Rdis_Ohm"]
    )
    p["BAT_Rchg_pack_Ohm"] = tuple(
        p["BAT_Rint_series_multiplier"] * value
        for value in p["BAT_Cell_Rchg_Ohm"]
    )
    p["BAT_Vmin_pack_V"] = p["BAT_Ns"] * p["BAT_Cell_V_min_V"]
    p["BAT_Vmax_pack_V"] = p["BAT_Ns"] * p["BAT_Cell_V_max_V"]
    p["BAT_Vnom_pack_V"] = p["BAT_Ns"] * p["BAT_Cell_V_nom_V"]
    p["BAT_Energy_nominal_Wh"] = p["BAT_Vnom_pack_V"] * p["BAT_Capacity_Ah"]
    p["BAT_Vmin_required_pack_V"] = max(
        p["BAT_Vmin_pack_V"], p["BAT_Motor_Vmin_pack_V"]
    )
    return p


def init_battery_rint(initial_soc=0.90):
    return battery_parameters(initial_soc)
