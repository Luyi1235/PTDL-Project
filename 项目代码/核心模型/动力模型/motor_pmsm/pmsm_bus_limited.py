"""Detailed dq PMSM and a mission-scale steady-state evaluator.

The detailed class follows the current PMSM_BusLimited.slx equations and the
M7 corrected controller-integrator initial values. ``step`` integrates all six
continuous states using RK4; callers own the assembly's sampled input delays.
``steady_outputs`` is an explicit equilibrium approximation for mission_fast.
"""

from math import ceil, isfinite, pi, sqrt

from .init_pmsm import init_pmsm


def _clip(value, lower, upper):
    return min(max(value, lower), upper)


class PMSMBusLimited:
    state_names = (
        "id_A", "iq_A", "omega_m_radps", "speed_integrator_A",
        "id_integrator_V", "iq_integrator_V",
    )
    output_names = (
        "N_rpm", "id_A", "iq_A", "Te_Nm", "P_mech_W", "P_elec_W",
        "ud_V", "uq_V", "omega_m_radps", "iq_ref_A", "P_shaft_W",
    )

    def __init__(self, parameters=None, inverter_efficiency=0.95):
        self.parameters = dict(init_pmsm() if parameters is None else parameters)
        self.inverter_efficiency = float(inverter_efficiency)
        if not 0.0 < self.inverter_efficiency <= 1.0:
            raise ValueError("Inverter efficiency must be in (0, 1]")
        self.reset()

    def reset(self):
        """Apply SLX initial-condition expressions to current parameters.

        The SLX uses Rs*I_init and Iq_init+Ba*omega_init expressions directly,
        rather than the initializer's diagnostic MOT_*Int_init variables.
        This matters after execute_hybrid_system overrides physical ICs.
        """
        p = self.parameters
        id_, iq, omega = (p[key] for key in (
            "MOT_Id_init_A", "MOT_Iq_init_A", "MOT_Omega_init_radps"))
        self.state = self._project([
            id_, iq, omega, iq + p["MOT_Ba_APerRadps"]*omega,
            p["MOT_Rs_Ohm"]*id_, p["MOT_Rs_Ohm"]*iq,
        ])
        self.time_s = 0.0

    def initialize_operating_point(self, speed_rpm, load_torque_nm):
        """Set the three mission physical ICs and derive M7 PI states once.

        Equivalent to execute_hybrid_system.m's first-phase initialization;
        no voltage feedforward is added to the current integrator states.
        """
        speed, load = float(speed_rpm), float(load_torque_nm)
        if not isfinite(speed) or speed < 0 or not isfinite(load):
            raise ValueError("Initial speed must be nonnegative and inputs finite")
        p = self.parameters
        omega = speed*pi/30.0
        p["MOT_Id_init_A"] = 0.0
        p["MOT_Iq_init_A"] = (load+p["MOT_B_Nms"]*omega)/p["MOT_Kt_NmPerA"]
        p["MOT_Omega_init_radps"] = omega
        self.reset()

    def steady_outputs(self, speed_rpm, load_torque_nm, vdc_bus_v,
                       dc_power_available_w=float("inf")):
        """Evaluate a requested equilibrium and report whether it is feasible.

        A false ``equilibrium_feasible`` means the imposed speed/load cannot
        be represented by an equilibrium here; use the dynamic solver. No
        independent clamp of reported power is applied, so P=1.5*(ud*id+uq*iq)
        remains true even when this approximate operating point is infeasible.
        """
        p = self.parameters
        speed_rpm, load, vdc = map(float, (speed_rpm, load_torque_nm, vdc_bus_v))
        omega = speed_rpm*pi/30.0
        required_iq = (load + p["MOT_B_Nms"]*omega)/p["MOT_Kt_NmPerA"]
        iq = min(required_iq, p["MOT_Iq_limit_A"])
        id_ = 0.0
        omega_e = p["MOT_PolePairs"]*omega
        ud = -omega_e*p["MOT_Lq_H"]*iq
        uq = p["MOT_Rs_Ohm"]*iq + omega_e*p["MOT_Psi_f_Wb"]
        voltage_limit = vdc*p["MOT_Vdq_utilization"]
        voltage_feasible = ud <= voltage_limit and uq <= voltage_limit
        ud = self._wired_saturation(ud, voltage_limit)
        uq = self._wired_saturation(uq, voltage_limit)
        torque = p["MOT_Kt_NmPerA"]*iq
        shaft = load*omega
        mechanical = torque*omega
        electrical = 1.5*(ud*id_+uq*iq)
        max_ac = max(float(dc_power_available_w), 0.0)*self.inverter_efficiency
        values = (speed_rpm, id_, iq, torque, mechanical, electrical, ud, uq,
                  omega, iq, shaft)
        result = dict(zip(self.output_names, values))
        result["equilibrium_feasible"] = (
            required_iq <= p["MOT_Iq_limit_A"] and voltage_feasible
            and abs(required_iq+p["MOT_Ba_APerRadps"]*omega)
            <= p["MOT_SpeedInt_limit_A"]
            and electrical <= max_ac + 1e-9
        )
        return result

    @staticmethod
    def _wired_saturation(command, positive_limit):
        """Exact generated-C conditional for the actual SLX port wiring.

        Library ports are upper, signal, lower; the model connects command,
        positive limit, negative limit. For nonnegative limits this is min.
        """
        if positive_limit > command:
            return command
        if positive_limit < -positive_limit:
            return -positive_limit
        return positive_limit

    def _project(self, state):
        result = list(state)
        result[2] = max(result[2], 0.0)
        limit = self.parameters["MOT_SpeedInt_limit_A"]
        result[3] = _clip(result[3], -limit, limit)
        return result

    @staticmethod
    def _inputs(reference, load, vdc, available):
        values = tuple(map(float, (reference, load, vdc, available)))
        if not all(map(isfinite, values)):
            raise ValueError("Dynamic PMSM inputs must be finite")
        if values[2] < 0:
            raise ValueError("Vdc must be nonnegative")
        return values

    def _evaluate(self, state, inputs):
        p = self.parameters
        id_, iq, omega, speed_int, id_int, iq_int = self._project(state)
        reference, load, vdc, available = inputs
        omega_e = p["MOT_PolePairs"]*omega
        error = (reference-omega*30.0/pi)*p["MOT_SpeedError_scale"]
        raw = (p["MOT_Kp_speed"]*error+speed_int
               - p["MOT_Ba_APerRadps"]/p["MOT_PolePairs"]*omega_e)
        b = 1.5*p["MOT_Psi_f_Wb"]*abs(omega_e)
        ac_available = max(available, 0.0)*self.inverter_efficiency
        iq_limit = _clip((sqrt(b*b+6*p["MOT_Rs_Ohm"]*ac_available)-b)
                         /(3*p["MOT_Rs_Ohm"]), 0.0, p["MOT_Iq_limit_A"])
        # Preserve current SLX Saturation Dynamic wiring (upper-limit behavior).
        iq_reference = self._wired_saturation(raw, iq_limit)
        error_d, error_q = -id_, iq_reference-iq
        voltage_limit = vdc*p["MOT_Vdq_utilization"]
        ud = self._wired_saturation(
            p["MOT_Kp_id"]*error_d+id_int-omega_e*p["MOT_Lq_H"]*iq,
            voltage_limit)
        uq = self._wired_saturation(
            p["MOT_Kp_iq"]*error_q+iq_int
            + omega_e*(p["MOT_Ld_H"]*id_+p["MOT_Psi_f_Wb"]), voltage_limit)
        torque = p["MOT_Kt_NmPerA"]*iq
        did = (ud-p["MOT_Rs_Ohm"]*id_+omega_e*p["MOT_Lq_H"]*iq)/p["MOT_Ld_H"]
        diq = (uq-p["MOT_Rs_Ohm"]*iq
               - omega_e*(p["MOT_Ld_H"]*id_+p["MOT_Psi_f_Wb"]))/p["MOT_Lq_H"]
        domega = (torque-load-p["MOT_B_Nms"]*omega)/p["MOT_J_kgm2"]
        dspeed = p["MOT_Ki_speed"]*error+p["MOT_Kaw_speed"]*(iq_reference-raw)
        if omega <= 0.0 and domega < 0.0:
            domega = 0.0
        limit = p["MOT_SpeedInt_limit_A"]
        if (speed_int <= -limit and dspeed < 0.0) or (speed_int >= limit and dspeed > 0.0):
            dspeed = 0.0
        derivative = (did, diq, domega, dspeed,
                      p["MOT_Ki_id"]*error_d, p["MOT_Ki_iq"]*error_q)
        outputs = (omega*30/pi, id_, iq, torque, torque*omega,
                   1.5*(ud*id_+uq*iq), ud, uq, omega, iq_reference, load*omega)
        return outputs, derivative

    def outputs(self, speed_reference_rpm, load_torque_nm, vdc_bus_v,
                dc_power_available_w):
        inputs = self._inputs(speed_reference_rpm, load_torque_nm,
                              vdc_bus_v, dc_power_available_w)
        return dict(zip(self.output_names, self._evaluate(self.state, inputs)[0]))

    def derivatives(self, speed_reference_rpm, load_torque_nm, vdc_bus_v,
                    dc_power_available_w, state=None):
        """Return continuous derivatives in ``state_names`` order.

        This method supports integration alongside the battery in a coupled
        RK stage. External delayed inputs are supplied by the assembly.
        """
        inputs = self._inputs(speed_reference_rpm, load_torque_nm,
                              vdc_bus_v, dc_power_available_w)
        return self._evaluate(self.state if state is None else state, inputs)[1]

    def step(self, speed_reference_rpm, load_torque_nm, vdc_bus_v,
             dc_power_available_w, dt_s, max_step_s=None):
        inputs = self._inputs(speed_reference_rpm, load_torque_nm,
                              vdc_bus_v, dc_power_available_w)
        dt = float(dt_s)
        maximum = float(self.parameters["MOT_dt_s"] if max_step_s is None else max_step_s)
        if not isfinite(dt) or dt < 0.0:
            raise ValueError("dt_s must be nonnegative and finite")
        if not isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_step_s must be positive and finite")
        if dt == 0.0:
            return dict(zip(self.output_names, self._evaluate(self.state, inputs)[0]))
        count = max(1, ceil(dt/maximum-1e-10))
        h = dt/count
        for _ in range(count):
            state = self.state
            k1 = self._evaluate(state, inputs)[1]
            k2 = self._evaluate([x+h*k/2 for x, k in zip(state, k1)], inputs)[1]
            k3 = self._evaluate([x+h*k/2 for x, k in zip(state, k2)], inputs)[1]
            k4 = self._evaluate([x+h*k for x, k in zip(state, k3)], inputs)[1]
            self.state = self._project([
                state[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0 for i in range(6)])
            if not all(map(isfinite, self.state)):
                raise ArithmeticError("Nonfinite motor state; reduce the integration step")
        self.time_s += dt
        return dict(zip(self.output_names, self._evaluate(self.state, inputs)[0]))
