"""Generator and gearbox algebra ported from Generator_Gearbox.slx."""

from math import isfinite, pi

from .init_generator_gearbox import init_generator_gearbox


class GeneratorGearbox:
    output_names = (
        "N_generator_rpm", "T_engine_load_Nm", "T_generator_req_Nm",
        "P_generator_ac_actual_W", "P_generator_shaft_W",
        "P_engine_mech_used_W", "P_engine_mech_available_W",
        "P_generator_ac_available_W", "P_generator_loss_W", "P_gear_loss_W",
        "P_generator_unserved_W", "T_engine_margin_Nm",
    )

    def __init__(self, parameters=None):
        self.parameters = dict(init_generator_gearbox() if parameters is None else parameters)

    def outputs(self, power_ac_command_w, engine_speed_rpm, engine_torque_available_nm):
        command, speed, torque = map(float, (power_ac_command_w, engine_speed_rpm,
                                             engine_torque_available_nm))
        if not all(map(isfinite, (command, speed, torque))):
            raise ValueError("Inputs must be finite")
        p = self.parameters
        generator_speed = speed / p["GEN_GearRatio"]
        generator_omega = max(generator_speed, 1.0) * 2.0 * pi / 60.0
        engine_omega = speed * 2.0 * pi / 60.0
        mechanical_available = max(torque * engine_omega, 0.0)
        electrical_available = mechanical_available * p["GEN_GearEta"] * p["GEN_Eta"]
        electrical = min(max(command, 0.0), electrical_available)
        shaft = electrical / p["GEN_Eta"]
        generator_torque = shaft / generator_omega
        engine_load_torque = generator_torque / (p["GEN_GearRatio"] * p["GEN_GearEta"])
        engine_used = engine_load_torque * engine_omega
        values = (
            generator_speed, engine_load_torque, generator_torque, electrical, shaft,
            engine_used, mechanical_available, electrical_available, shaft-electrical,
            engine_used-shaft, max(command, 0.0)-electrical, torque-engine_load_torque,
        )
        return dict(zip(self.output_names, values))
