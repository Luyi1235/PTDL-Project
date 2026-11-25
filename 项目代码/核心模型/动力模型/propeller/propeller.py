"""Seven-stage mission lookup and propeller coefficient equations."""

from bisect import bisect_right
from math import isfinite

from .init_propeller import init_propeller


class Propeller:
    output_names = (
        "Phase_index", "P_aircraft_required_W", "P_propeller_shaft_W",
        "P_motor_input_formula_W", "P_motor_input_table_W", "Advance_ratio",
        "Cp", "Ct", "Eta_prop_calculated", "Eta_profile_table",
        "Altitude_ISA_m",
    )

    def __init__(self, parameters=None):
        self.parameters = dict(init_propeller() if parameters is None else parameters)

    def outputs(self, mission_time_s, flight_speed_mps=60.0, prop_speed_rps=35.0,
                air_density_kgpm3=1.225, thrust_n=0.0):
        t, speed, prop_speed, density, thrust = map(float, (
            mission_time_s, flight_speed_mps, prop_speed_rps, air_density_kgpm3, thrust_n))
        if not all(map(isfinite, (t, speed, prop_speed, density, thrust))):
            raise ValueError("Inputs must be finite")
        p = self.parameters
        index = min(max(bisect_right(p["PROP_Profile_start_s"], t)-1, 0),
                    len(p["PROP_Phase_index"])-1)
        phase_start = p["PROP_Profile_start_s"][index]
        phase_end = p["PROP_Profile_end_s"][index]
        fraction = min(max((t-phase_start)/(phase_end-phase_start), 0.0), 1.0)
        altitude = (p["PROP_Altitude_boundary_m"][index]
                    + fraction*(p["PROP_Altitude_boundary_m"][index+1]
                                - p["PROP_Altitude_boundary_m"][index]))
        aircraft = p["PROP_Aircraft_power_W"][index]
        shaft = p["PROP_Shaft_power_W"][index]
        motor = shaft / (p["PROP_Eta_motor"] * p["PROP_Eta_motor_controller"])
        prop_speed = max(prop_speed, p["PROP_n_min_rps"])
        density = max(density, p["PROP_rho_min_kgpm3"])
        diameter = p["PROP_Diameter_m"]
        advance = speed / (prop_speed * diameter)
        cp = shaft / (density * prop_speed**3 * diameter**5)
        ct = thrust / (density * prop_speed**2 * diameter**4)
        result = (
            p["PROP_Phase_index"][index], aircraft, shaft, motor,
            p["PROP_Motor_input_table_W"][index], advance, cp, ct,
            ct/cp*advance, aircraft/shaft, altitude,
        )
        return dict(zip(self.output_names, result))
