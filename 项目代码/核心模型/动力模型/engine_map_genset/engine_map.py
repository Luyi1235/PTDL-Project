"""Three-dimensional lookup engine built from the 60-case GT-POWER CSV.

The runtime map coordinates are ISA altitude, engine speed and throttle. The
source CSV does not include a convergence flag, so arithmetic/grid checks are
performed here while physical GT-POWER convergence remains explicitly unknown.
"""

import csv
from bisect import bisect_right
from math import isfinite, pi
from pathlib import Path


DEFAULT_MAP = Path(__file__).with_name("EngineMap_60Case.csv")


def _bracket(value, axis):
    if not isfinite(float(value)):
        raise ValueError("Map coordinates must be finite")
    value = min(max(float(value), axis[0]), axis[-1])
    index = min(max(bisect_right(axis, value) - 1, 0), len(axis) - 2)
    fraction = (value - axis[index]) / (axis[index + 1] - axis[index])
    return value, index, fraction


class EngineMap:
    """Trilinear altitude/speed/throttle map with torque inversion."""

    fields = (
        "BrakeTorque_Nm", "BrakePower_kW", "FuelFlow_kg_s", "BSFC_g_kWh",
    )

    def __init__(self, csv_path=DEFAULT_MAP):
        self.csv_path = Path(csv_path)
        with self.csv_path.open(newline="", encoding="utf-8-sig") as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != 60:
            raise ValueError(f"Expected 60 engine-map rows, found {len(source_rows)}")

        self.altitude_m = tuple(sorted({float(row["Height (m)"])
                                        for row in source_rows}))
        self.rpm = tuple(sorted({float(row["Engine Speed (rpm)"])
                                 for row in source_rows}))
        self.throttle_deg = tuple(sorted({float(row["Throttle (deg)"])
                                          for row in source_rows}))
        if self.altitude_m != (0.0, 3000.0, 5000.0):
            raise ValueError("Engine map altitude axis must be 0, 3000 and 5000 m")
        if self.rpm != (2800.0, 3900.0, 5000.0, 6100.0):
            raise ValueError("Engine map speed axis must be 2800, 3900, 5000 and 6100 rpm")
        if self.throttle_deg != (30.0, 45.0, 60.0, 75.0, 90.0):
            raise ValueError("Engine map throttle axis must be 30, 45, 60, 75 and 90 deg")

        self._rows = {}
        for source in source_rows:
            height = float(source["Height (m)"])
            speed = float(source["Engine Speed (rpm)"])
            throttle = float(source["Throttle (deg)"])
            fuel_kg_h = float(source["Fuel Flow (kg/h)"])
            row = {
                "BrakeTorque_Nm": float(source["Brake Torque (N-m)"]),
                "BrakePower_kW": float(source["Brake Power (kW)"]),
                "FuelFlow_kg_s": fuel_kg_h / 3600.0,
                "BSFC_g_kWh": float(source["BSFC System (g/kWh)"]),
            }
            key = (height, speed, throttle)
            if key in self._rows:
                raise ValueError(f"Duplicate engine-map coordinate: {key}")
            self._rows[key] = row

        expected = {
            (height, speed, throttle)
            for height in self.altitude_m
            for speed in self.rpm
            for throttle in self.throttle_deg
        }
        if set(self._rows) != expected:
            raise ValueError("Engine map must be a complete 3 x 4 x 5 grid")
        if any(not isfinite(row[field])
               for row in self._rows.values() for field in self.fields):
            raise ValueError("Engine map contains non-finite values")

        self.total_points = len(self._rows)
        self.converged_count = None
        self.convergence_status = "not provided by EngineMap_60Case.csv"
        self._validate_source_consistency()

    def _validate_source_consistency(self):
        for (height, speed, _), row in self._rows.items():
            power = row["BrakeTorque_Nm"] * speed * 2.0 * pi / 60.0 / 1000.0
            if abs(power - row["BrakePower_kW"]) / row["BrakePower_kW"] > 1e-4:
                raise ValueError(f"Power/torque inconsistency at {height} m, {speed} rpm")
            bsfc = row["FuelFlow_kg_s"] * 3.6e9 / (row["BrakePower_kW"] * 1000.0)
            if abs(bsfc - row["BSFC_g_kWh"]) / row["BSFC_g_kWh"] > 1e-3:
                raise ValueError(f"Fuel/BSFC inconsistency at {height} m, {speed} rpm")
        for height in self.altitude_m:
            for speed in self.rpm:
                torque = [self._rows[(height, speed, throttle)]["BrakeTorque_Nm"]
                          for throttle in self.throttle_deg]
                if any(right <= left for left, right in zip(torque, torque[1:])):
                    raise ValueError(
                        f"Torque must increase with throttle at {height} m, {speed} rpm"
                    )

    def _value(self, rpm, throttle_deg, altitude_m, field):
        if field not in self.fields:
            raise KeyError(field)
        _, ai, af = _bracket(altitude_m, self.altitude_m)
        _, ri, rf = _bracket(rpm, self.rpm)
        _, ti, tf = _bracket(throttle_deg, self.throttle_deg)
        result = 0.0
        for da, wa in ((0, 1.0 - af), (1, af)):
            for dr, wr in ((0, 1.0 - rf), (1, rf)):
                for dt, wt in ((0, 1.0 - tf), (1, tf)):
                    key = (self.altitude_m[ai + da], self.rpm[ri + dr],
                           self.throttle_deg[ti + dt])
                    result += wa * wr * wt * self._rows[key][field]
        return result

    def lookup(self, rpm, throttle_deg, altitude_m=0.0):
        altitude_m = min(max(float(altitude_m), self.altitude_m[0]),
                         self.altitude_m[-1])
        rpm = min(max(float(rpm), self.rpm[0]), self.rpm[-1])
        throttle_deg = min(max(float(throttle_deg), self.throttle_deg[0]),
                           self.throttle_deg[-1])
        if not all(map(isfinite, (altitude_m, rpm, throttle_deg))):
            raise ValueError("Map coordinates must be finite")
        result = {
            field: self._value(rpm, throttle_deg, altitude_m, field)
            for field in self.fields
        }
        result.update(
            Altitude_m=altitude_m, RPM=rpm, Throttle_deg=throttle_deg,
            FuelFlow_kg_h=result["FuelFlow_kg_s"] * 3600.0,
        )
        return result

    def throttle_for_torque(self, rpm, torque_nm, altitude_m=0.0):
        altitude_m = min(max(float(altitude_m), self.altitude_m[0]),
                         self.altitude_m[-1])
        rpm = min(max(float(rpm), self.rpm[0]), self.rpm[-1])
        requested = float(torque_nm)
        if not all(map(isfinite, (altitude_m, rpm, requested))):
            raise ValueError("Altitude, speed and torque must be finite")
        torque_axis = tuple(
            self._value(rpm, throttle, altitude_m, "BrakeTorque_Nm")
            for throttle in self.throttle_deg
        )
        if any(right <= left for left, right in zip(torque_axis, torque_axis[1:])):
            raise ValueError("Inverse torque lookup requires a strictly increasing torque row")
        clipped = min(max(requested, torque_axis[0]), torque_axis[-1])
        index = min(max(bisect_right(torque_axis, clipped) - 1, 0),
                    len(torque_axis) - 2)
        fraction = ((clipped - torque_axis[index]) /
                    (torque_axis[index + 1] - torque_axis[index]))
        return self.throttle_deg[index] + fraction * (
            self.throttle_deg[index + 1] - self.throttle_deg[index]
        )

    def operating_point_for_power(self, power_w, rpm=5000.0, altitude_m=0.0):
        power_w, rpm, altitude_m = map(float, (power_w, rpm, altitude_m))
        if not all(map(isfinite, (power_w, rpm, altitude_m))) or rpm <= 0.0:
            raise ValueError("Power, speed and altitude must be finite; speed must be positive")
        rpm = min(max(rpm, self.rpm[0]), self.rpm[-1])
        altitude_m = min(max(altitude_m, self.altitude_m[0]), self.altitude_m[-1])
        omega = rpm * 2.0 * pi / 60.0
        torque_request = max(power_w, 0.0) / omega
        throttle = self.throttle_for_torque(rpm, torque_request, altitude_m)
        result = self.lookup(rpm, throttle, altitude_m)
        result["Torque_request_Nm"] = torque_request
        result["Torque_request_clipped"] = (
            abs(result["BrakeTorque_Nm"] - torque_request) > 1e-8
        )
        result["P_engine_W"] = result["BrakeTorque_Nm"] * omega
        result["P_map_column_W"] = result["BrakePower_kW"] * 1000.0
        result["SFC_kg_per_kWh"] = (
            result["FuelFlow_kg_s"] * 3.6e6 / result["P_engine_W"]
            if result["P_engine_W"] > 0.0 else 0.0
        )
        return result
