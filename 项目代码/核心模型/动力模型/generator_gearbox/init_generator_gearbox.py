"""Component-local parameters translated from the current MATLAB initializer.

No shared mutable parameter dictionary or root parameter module is required.
"""

from math import pi


def generator_parameters(assembly=True):
    """Use the active 62.5 kW/5000 rpm assembly, or the paper design."""
    p = {
        "GEN_Eta": 0.96,
        "GEN_P_engine_design_W": 62500.0 if assembly else 5000.0,
        "GEN_P_elec_design_W": 60000.0 if assembly else 4800.0,
        "GEN_N_design_rpm": 6000.0,
        "GEN_N_engine_design_rpm": 5000.0 if assembly else 50005.0,
        "GEN_GearRatio_paper": 8.33,
        "GEN_GearRatio": (5000.0 / 6000.0) if assembly else 8.33,
        "GEN_GearEta": 1.0,
        "GEN_MinSpeed_rpm": 1.0,
        "GEN_dt_s": 1e-4,
    }
    p["GEN_Omega_design_radps"] = p["GEN_N_design_rpm"] * 2 * pi / 60
    p["GEN_T_generator_design_Nm"] = (
        p["GEN_P_elec_design_W"]
        / (p["GEN_Omega_design_radps"] * p["GEN_Eta"])
    )
    p["GEN_T_engine_design_Nm"] = (
        p["GEN_T_generator_design_Nm"]
        / (p["GEN_GearRatio"] * p["GEN_GearEta"])
    )
    return p


def init_generator_gearbox(assembly=True):
    return generator_parameters(assembly=assembly)
