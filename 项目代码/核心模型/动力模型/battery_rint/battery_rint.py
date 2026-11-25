"""Rint battery ported from Battery_Rint.slx (positive power is discharge)."""

from bisect import bisect_right
from math import ceil, isfinite, sqrt

from .init_battery_rint import init_battery_rint


def linear_lookup(x, breakpoints, values):
    x = min(max(float(x), breakpoints[0]), breakpoints[-1])
    index = min(max(bisect_right(breakpoints, x) - 1, 0), len(breakpoints) - 2)
    fraction = (x - breakpoints[index]) / (breakpoints[index + 1] - breakpoints[index])
    return values[index] + fraction * (values[index + 1] - values[index])


class BatteryRint:
    output_names = (
        "SOC", "Voc_pack_V", "R_pack_Ohm", "P_bat_W", "I_bat_A",
        "V_bat_V", "Rdis_pack_Ohm", "Rchg_pack_Ohm",
    )

    def __init__(self, parameters=None):
        self.parameters = dict(init_battery_rint() if parameters is None else parameters)
        self.reset()

    def reset(self, soc=None):
        p = self.parameters
        value = p["BAT_SOC_init"] if soc is None else float(soc)
        if not isfinite(value) or not p["BAT_SOC_min"] <= value <= p["BAT_SOC_max"]:
            raise ValueError("SOC is outside configured limits")
        self.soc = value
        self.time_s = 0.0

    def outputs(self, power_request_w, soc=None):
        p = self.parameters
        request = float(power_request_w)
        state = self.soc if soc is None else float(soc)
        if not isfinite(request) or not isfinite(state):
            raise ValueError("Power and SOC must be finite")
        state = min(max(state, p["BAT_SOC_min"]), p["BAT_SOC_max"])
        voc = linear_lookup(state, p["BAT_SOC_bp"], p["BAT_Voc_pack_V"])
        rdis = linear_lookup(state, p["BAT_SOC_bp"], p["BAT_Rdis_pack_Ohm"])
        rchg = linear_lookup(state, p["BAT_SOC_bp"], p["BAT_Rchg_pack_Ohm"])
        resistance = rdis if request >= 0.0 else rchg
        upper = max((voc - p["BAT_Vmin_required_pack_V"]) / resistance
                    * p["BAT_Vmin_required_pack_V"], 0.0)
        lower = min((voc - p["BAT_Vmax_pack_V"]) / resistance
                    * p["BAT_Vmax_pack_V"], 0.0)
        power = min(max(request, lower), upper)
        discriminant = max(voc * voc - 4.0 * power * resistance,
                           p["BAT_Discriminant_floor"])
        current = (voc - sqrt(discriminant)) / (2.0 * resistance)
        voltage = voc - current * resistance
        values = (state, voc, resistance, power, current, voltage, rdis, rchg)
        return dict(zip(self.output_names, values))

    def _soc_rate(self, soc, power_request_w):
        p = self.parameters
        current = self.outputs(power_request_w, soc)["I_bat_A"]
        rate = -p["BAT_CoulombicEfficiency"] * current / (3600.0 * p["BAT_Capacity_Ah"])
        if (soc <= p["BAT_SOC_min"] and rate < 0.0) or (soc >= p["BAT_SOC_max"] and rate > 0.0):
            return 0.0
        return rate

    def step(self, power_request_w, dt_s, max_step_s=0.05):
        dt_s, max_step_s = float(dt_s), float(max_step_s)
        if not isfinite(dt_s) or dt_s <= 0.0 or not isfinite(max_step_s) or max_step_s <= 0.0:
            raise ValueError("Time steps must be positive and finite")
        self.outputs(power_request_w)
        count = max(1, ceil(dt_s / max_step_s - 1e-12))
        h = dt_s / count
        for _ in range(count):
            s = self.soc
            k1 = self._soc_rate(s, power_request_w)
            k2 = self._soc_rate(s + h * k1 / 2.0, power_request_w)
            k3 = self._soc_rate(s + h * k2 / 2.0, power_request_w)
            k4 = self._soc_rate(s + h * k3, power_request_w)
            self.soc = min(max(s + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0,
                               self.parameters["BAT_SOC_min"]),
                           self.parameters["BAT_SOC_max"])
        self.time_s += dt_s
        return self.outputs(power_request_w)
