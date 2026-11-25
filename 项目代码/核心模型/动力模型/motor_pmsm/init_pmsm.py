"""EMRAX 228 MV LC parameters from the current MATLAB initializer."""

from math import hypot, inf, isfinite, pi, sqrt


def init_pmsm(dt_s=1e-5, parameter_set="emrax_228_mv_lc"):
    if str(parameter_set).lower() != "emrax_228_mv_lc":
        raise ValueError("Only the emrax_228_mv_lc parameter set is supported")
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite")
    p = {
        "MOT_ParameterSet": "emrax_228_mv_lc",
        "MOT_ParameterSource": "EMRAX 228 datasheet v1.6; Medium Voltage, Liquid Cooled",
        "MOT_ProductName": "EMRAX 228 MV LC",
        "MOT_PolePairs": 10,
        "MOT_Kt_RMS_NmPerA": 0.61,
        "MOT_Rs_Ohm": 7.06e-3,
        "MOT_Ld_H": 96.5e-6,
        "MOT_Lq_H": 96.5e-6,
        "MOT_J_kgm2": 0.02521,
        "MOT_N_rated_rpm": 2100.0,
        "MOT_N_limit_rpm": 6500.0,
        "MOT_Iq_continuous_A": 180.0 * sqrt(2.0),
        "MOT_Iq_limit_A": 360.0 * sqrt(2.0),
        "MOT_P_input_max_W": 40e3,
        "MOT_Fitted_viscous_loss_fraction": 0.01,
        "MOT_CurrentLoopBandwidth_radps": 2500.0,
        "MOT_SpeedLoopNaturalFrequency_radps": 40.0,
        "MOT_SpeedLoopDampingRatio": 1.0,
        "MOT_Vdc_design_V": 300.0,
        "MOT_Vdc_min_application_V": 240.0,
        "MOT_Udq_limit_V": inf,
        "MOT_Vdq_utilization": 1.0 / sqrt(3.0),
        "MOT_Ba_APerRadps": 0.0,
        "MOT_Ba": 0.0,
        "MOT_Id_init_A": 0.0,
        "MOT_Iq_init_A": 0.0,
        "MOT_Omega_init_radps": 0.0,
        "MOT_dt_s": float(dt_s),
        "MOT_SpeedError_scale": pi / 30.0,
    }
    p["MOT_Kt_NmPerA"] = p["MOT_Kt_RMS_NmPerA"] / sqrt(2.0)
    p["MOT_Psi_f_Wb"] = p["MOT_Kt_NmPerA"] / (1.5 * p["MOT_PolePairs"])
    p["MOT_Omega_rated_radps"] = p["MOT_N_rated_rpm"] * pi / 30.0
    p["MOT_T_continuous_model_Nm"] = (
        p["MOT_Kt_NmPerA"] * p["MOT_Iq_continuous_A"]
    )
    p["MOT_T_peak_model_Nm"] = p["MOT_Kt_NmPerA"] * p["MOT_Iq_limit_A"]
    p["MOT_P_rated_W"] = (
        p["MOT_T_continuous_model_Nm"] * p["MOT_Omega_rated_radps"]
    )
    p["MOT_B_Nms"] = (
        p["MOT_Fitted_viscous_loss_fraction"]
        * p["MOT_P_rated_W"]
        / p["MOT_Omega_rated_radps"] ** 2
    )
    p["MOT_Kp_id"] = p["MOT_Ld_H"] * p["MOT_CurrentLoopBandwidth_radps"]
    p["MOT_Ki_id"] = p["MOT_Rs_Ohm"] * p["MOT_CurrentLoopBandwidth_radps"]
    p["MOT_Kp_iq"] = p["MOT_Lq_H"] * p["MOT_CurrentLoopBandwidth_radps"]
    p["MOT_Ki_iq"] = p["MOT_Ki_id"]
    p["MOT_Kp_speed"] = (
        2.0
        * p["MOT_SpeedLoopDampingRatio"]
        * p["MOT_SpeedLoopNaturalFrequency_radps"]
        * p["MOT_J_kgm2"]
        - p["MOT_B_Nms"]
    ) / p["MOT_Kt_NmPerA"]
    p["MOT_Ki_speed"] = (
        p["MOT_SpeedLoopNaturalFrequency_radps"] ** 2
        * p["MOT_J_kgm2"]
        / p["MOT_Kt_NmPerA"]
    )
    p["MOT_Kaw_speed"] = p["MOT_Ki_speed"] / p["MOT_Kp_speed"]
    p["MOT_Fig420_UseCurveTunedSpeedPI"] = False
    p["MOT_Fig420_J_kgm2"] = p["MOT_J_kgm2"]
    p["MOT_Fig420_Kp_speed"] = p["MOT_Kp_speed"]
    p["MOT_Fig420_Ki_speed"] = p["MOT_Ki_speed"]
    p["MOT_Fig420_Kaw_speed"] = p["MOT_Kaw_speed"]
    p["MOT_T_em_rated_Nm"] = p["MOT_P_rated_W"] / p["MOT_Omega_rated_radps"]
    p["MOT_Iq_rated_A"] = p["MOT_T_em_rated_Nm"] / p["MOT_Kt_NmPerA"]
    p["MOT_Iq_rated_RMS_A"] = p["MOT_Iq_rated_A"] / sqrt(2.0)
    p["MOT_T_load_rated_Nm"] = (
        p["MOT_T_em_rated_Nm"] - p["MOT_B_Nms"] * p["MOT_Omega_rated_radps"]
    )
    p["MOT_T_rated_Nm"] = p["MOT_T_load_rated_Nm"]
    p["MOT_SpeedInt_limit_A"] = p["MOT_Iq_limit_A"]
    p["MOT_Beta_from_Kp"] = (
        p["MOT_Kp_speed"] * p["MOT_PolePairs"] * p["MOT_Psi_f_Wb"]
        / (1.5 * p["MOT_J_kgm2"])
    )
    p["MOT_Beta_from_KiKp"] = p["MOT_Ki_speed"] / p["MOT_Kp_speed"]
    for name in ("Kp", "KiKp"):
        p["MOT_Ba_from_" + name] = (
            1.5 * p["MOT_Beta_from_" + name] * p["MOT_J_kgm2"]
            - p["MOT_B_Nms"]
        ) / (p["MOT_PolePairs"] * p["MOT_Psi_f_Wb"])
    p["MOT_SpeedInt_init"] = 0.0
    p["MOT_IdInt_init_V"] = 0.0
    p["MOT_IqInt_init_V"] = 0.0
    p["MOT_Ud_rated_V"] = (
        -p["MOT_PolePairs"]
        * p["MOT_Omega_rated_radps"]
        * p["MOT_Lq_H"]
        * p["MOT_Iq_rated_A"]
    )
    p["MOT_Uq_rated_V"] = (
        p["MOT_Rs_Ohm"] * p["MOT_Iq_rated_A"]
        + p["MOT_PolePairs"]
        * p["MOT_Omega_rated_radps"]
        * p["MOT_Psi_f_Wb"]
    )
    p["MOT_Udq_rated_V"] = hypot(p["MOT_Ud_rated_V"], p["MOT_Uq_rated_V"])
    p["MOT_Vdc_min_rated_V"] = sqrt(3.0) * p["MOT_Udq_rated_V"]
    p["MOT_Copper_loss_rated_W"] = (
        1.5 * p["MOT_Rs_Ohm"] * p["MOT_Iq_rated_A"] ** 2
    )
    p["MOT_Efficiency_copper_only"] = p["MOT_P_rated_W"] / (
        p["MOT_P_rated_W"] + p["MOT_Copper_loss_rated_W"]
    )
    p["MOT_Task_peak_torque_required_Nm"] = (
        40e3 * 0.96 * 0.95 / p["MOT_Omega_rated_radps"]
    )
    p["MOT_Task_climb_torque_required_Nm"] = (
        27e3 * 0.96 * 0.95 / p["MOT_Omega_rated_radps"]
    )
    p["MOT_Task_cruise_torque_required_Nm"] = (
        22.5e3 * 0.96 * 0.95 / p["MOT_Omega_rated_radps"]
    )
    p["MOT_Task_peak_Iq_required_A"] = (
        p["MOT_Task_peak_torque_required_Nm"]
        + p["MOT_B_Nms"] * p["MOT_Omega_rated_radps"]
    ) / p["MOT_Kt_NmPerA"]
    p["MOT_Task_peak_Ud_V"] = (
        -p["MOT_PolePairs"]
        * p["MOT_Omega_rated_radps"]
        * p["MOT_Lq_H"]
        * p["MOT_Task_peak_Iq_required_A"]
    )
    p["MOT_Task_peak_Uq_V"] = (
        p["MOT_Rs_Ohm"] * p["MOT_Task_peak_Iq_required_A"]
        + p["MOT_PolePairs"]
        * p["MOT_Omega_rated_radps"]
        * p["MOT_Psi_f_Wb"]
    )
    p["MOT_Task_peak_Vdc_min_V"] = sqrt(3.0) * hypot(
        p["MOT_Task_peak_Ud_V"], p["MOT_Task_peak_Uq_V"]
    )
    if p["MOT_T_peak_model_Nm"] < 219.0:
        raise ValueError("EMRAX peak torque conversion is inconsistent")
    if p["MOT_Task_peak_Iq_required_A"] >= p["MOT_Iq_limit_A"]:
        raise ValueError("Mission peak torque exceeds current limit")
    if p["MOT_Task_peak_Vdc_min_V"] >= p["MOT_Vdc_design_V"]:
        raise ValueError("300 V bus has insufficient dq-voltage margin")
    return p
