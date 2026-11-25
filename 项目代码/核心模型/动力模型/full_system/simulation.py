"""Top-level coupling for the seven-stage series-hybrid mission."""

import csv
import json
from math import ceil, floor, isfinite, pi
from pathlib import Path
from time import perf_counter

from ..battery_rint.battery_rint import BatteryRint
from ..engine_map_genset.engine_map import EngineMap
from ..generator_gearbox.generator_gearbox import GeneratorGearbox
from ..battery_rint.init_battery_rint import init_battery_rint
from ..generator_gearbox.init_generator_gearbox import init_generator_gearbox
from ..motor_pmsm.init_pmsm import init_pmsm
from ..power_electronics_ems.init_power_electronics_ems import init_power_electronics_ems
from ..propeller.init_propeller import init_propeller
from ..motor_pmsm.pmsm_bus_limited import PMSMBusLimited
from ..power_electronics_ems.power_electronics_ems import PowerElectronicsEMS
from ..propeller.propeller import Propeller
from .init_hybrid_system import init_hybrid_system


RESULT_FIELDS = (
    "Time_s", "Phase", "Phase_name", "Altitude_ISA_m", "P_aircraft_required_W",
    "Pshaft_request_W", "Pmotor_demand_table_W", "Pmotor_output_W",
    "Pmotor_ac_actual_W", "Pmotor_dc_actual_W", "Pengine_demand_W",
    "Pengine_actual_W", "Tengine_Nm", "Nengine_rpm", "Throttle_deg",
    "FuelFlow_kgps", "SFC_kg_per_kWh", "Pgenerator_ac_demand_W",
    "Pgenerator_ac_actual_W", "Pgenerator_dc_W", "Pbattery_command_W",
    "Pbattery_W", "Ibat_A", "Idc_bus_load_A", "Igenerator_bus_A",
    "Vbat_V", "Vdc_bus_V", "SOC", "Pbus_deficit_W",
    "Ppropulsion_unmet_W", "Motor_speed_rpm", "Motor_iq_A",
    "Motor_voltage_dq_V", "Motor_copper_loss_total_W",
    "Motor_id_A", "Motor_ud_V", "Motor_uq_V", "Pbattery_dc_W",
    "Pdc_available_W", "Pdc_for_motors_W", "Engine_enable_cmd",
    "Engine_supervisor_state", "Engine_on", "Solver",
)


class HybridSystemSimulation:
    """Current 81s3p, dual-EMRAX, 60 kW DC EngineMap hybrid assembly."""

    def __init__(self, mode=3, engine_map_path=None, initial_soc=0.90,
                 solver="mission_fast", initialize_motor=True):
        if mode not in (1, 2, 3):
            raise ValueError("mode must be 1, 2 or 3")
        if solver not in ("mission_fast", "dynamic"):
            raise ValueError("solver must be mission_fast or dynamic")
        self.mode, self.solver = int(mode), solver
        self.configuration = init_hybrid_system(initial_soc)
        self.battery = BatteryRint(self.configuration["battery"])
        self.ems = PowerElectronicsEMS(self.configuration["ems"])
        self.engine = EngineMap() if engine_map_path is None else EngineMap(engine_map_path)
        self.generator = GeneratorGearbox(self.configuration["generator"])
        self.motor = PMSMBusLimited(self.configuration["motor"],
                                    inverter_efficiency=self.ems.parameters["PE_Eta_inverter"])
        self.propeller = Propeller(self.configuration["propeller"])
        self.motor_speed_rpm = self.motor.parameters["MOT_N_rated_rpm"]
        self.engine_speed_rpm = self.generator.parameters["GEN_N_engine_design_rpm"]
        self.initialize_motor = bool(initialize_motor)
        self._engine_cache = {}

    def _engine_outputs(self, gen_command, altitude_m, engine_enable):
        enabled = float(engine_enable) >= 0.5
        cache_key = (float(gen_command), float(altitude_m), enabled)
        if cache_key not in self._engine_cache:
            if not enabled:
                engine = {
                    "P_engine_W": 0.0,
                    "BrakeTorque_Nm": 0.0,
                    "Throttle_deg": 0.0,
                    "FuelFlow_kg_s": 0.0,
                    "SFC_kg_per_kWh": 0.0,
                }
                generator = self.generator.outputs(0.0, 0.0, 0.0)
                self._engine_cache[cache_key] = 0.0, engine, generator
                return self._engine_cache[cache_key]
            demand = gen_command / (self.generator.parameters["GEN_Eta"] *
                                    self.generator.parameters["GEN_GearEta"])
            engine = self.engine.operating_point_for_power(
                demand, self.engine_speed_rpm, altitude_m)
            generator = self.generator.outputs(gen_command, self.engine_speed_rpm,
                                               engine["BrakeTorque_Nm"])
            self._engine_cache[cache_key] = demand, engine, generator
        return self._engine_cache[cache_key]

    def _row(self, time_s):
        profile = self.propeller.outputs(time_s)
        phase = int(profile["Phase_index"])
        shaft_total = profile["P_propeller_shaft_W"]
        omega_motor = self.motor_speed_rpm*pi/30.0
        load_each = shaft_total/(2.0*omega_motor)
        gen_prev, bat_prev, soc_prev, vdc_for_motor, available = self._delay
        motor_each = self.motor.steady_outputs(
            self.motor_speed_rpm, load_each, vdc_for_motor,
            # mission_fast is an equilibrium energy approximation; unlike the
            # dynamic solver it does not replay the one-step available-power
            # startup transient.
            dc_power_available_w=self.motor.parameters["MOT_P_input_max_W"])
        if not motor_each["equilibrium_feasible"]:
            raise RuntimeError("Requested motor equilibrium is constrained; use --solver dynamic")
        motor_ac = 2.0*motor_each["P_elec_W"]
        motor_dc = motor_ac/self.ems.parameters["PE_Eta_inverter"]
        ems = self.ems.outputs(
            phase, self.mode, motor_ac, soc_prev, gen_prev, bat_prev,
            profile["P_motor_input_table_W"])
        gen_command = ems["P_generator_ac_cmd_W"]
        engine_demand, engine, generator = self._engine_outputs(
            gen_command, profile["Altitude_ISA_m"], ems["Engine_enable_cmd"])
        battery_command = ems["P_battery_terminal_cmd_W"]
        battery = self.battery.outputs(battery_command)
        pgen_dc = ems["P_generator_dc_W"]
        pbat = battery["P_bat_W"]
        phase_name = self.propeller.parameters["PROP_Phase_names"][phase-1]
        copper_total = (
            3.0*self.motor.parameters["MOT_Rs_Ohm"]
            *(motor_each["id_A"]**2+motor_each["iq_A"]**2)
        )
        row = {
            "Time_s": time_s,
            "Phase": phase,
            "Phase_name": phase_name,
            "Altitude_ISA_m": profile["Altitude_ISA_m"],
            "P_aircraft_required_W": profile["P_aircraft_required_W"],
            "Pshaft_request_W": shaft_total,
            "Pmotor_demand_table_W": profile["P_motor_input_table_W"],
            "Pmotor_output_W": 2.0*motor_each["P_shaft_W"],
            "Pmotor_ac_actual_W": motor_ac,
            "Pmotor_dc_actual_W": motor_dc,
            "Pengine_demand_W": engine_demand,
            "Pengine_actual_W": engine["P_engine_W"],
            "Tengine_Nm": engine["BrakeTorque_Nm"],
            "Nengine_rpm": self.engine_speed_rpm if ems["Engine_enable_cmd"] else 0.0,
            "Throttle_deg": engine["Throttle_deg"],
            "FuelFlow_kgps": engine["FuelFlow_kg_s"],
            "SFC_kg_per_kWh": engine["SFC_kg_per_kWh"],
            "Pgenerator_ac_demand_W": gen_command,
            "Pgenerator_ac_actual_W": generator["P_generator_ac_actual_W"],
            "Pgenerator_dc_W": pgen_dc,
            "Pbattery_command_W": battery_command,
            "Pbattery_W": pbat,
            "Ibat_A": battery["I_bat_A"],
            "Idc_bus_load_A": motor_dc/vdc_for_motor if vdc_for_motor else 0.0,
            "Igenerator_bus_A": pgen_dc/vdc_for_motor if vdc_for_motor else 0.0,
            "Vbat_V": battery["V_bat_V"],
            "Vdc_bus_V": vdc_for_motor,
            "SOC": self.battery.soc,
            "Pbus_deficit_W": ems["P_bus_deficit_W"],
            "Ppropulsion_unmet_W": max(shaft_total-2.0*motor_each["P_shaft_W"], 0.0),
            "Motor_speed_rpm": motor_each["N_rpm"],
            "Motor_iq_A": motor_each["iq_A"],
            "Motor_voltage_dq_V": (motor_each["ud_V"]**2+motor_each["uq_V"]**2)**0.5,
            "Motor_copper_loss_total_W": copper_total,
            "Motor_id_A": motor_each["id_A"],
            "Motor_ud_V": motor_each["ud_V"], "Motor_uq_V": motor_each["uq_V"],
            "Pbattery_dc_W": ems["P_battery_dc_W"],
            "Pdc_available_W": pgen_dc + ems["P_battery_dc_W"],
            "Pdc_for_motors_W": available,
            "Engine_enable_cmd": ems["Engine_enable_cmd"],
            "Engine_supervisor_state": ems["Engine_supervisor_state"],
            "Engine_on": ems["Engine_enable_cmd"],
            "Solver": "mission_fast",
        }
        self._last_vdc = battery["V_bat_V"]
        next_delay = [
            generator["P_generator_ac_actual_W"],
            battery["P_bat_W"],
            self.battery.soc,
            battery["V_bat_V"],
            pgen_dc + ems["P_battery_dc_W"],
        ]
        return row, battery_command, next_delay

    def _reset(self):
        self.battery.reset()
        self.ems.reset()
        for key in ("MOT_Id_init_A", "MOT_Iq_init_A", "MOT_Omega_init_radps"):
            self.motor.parameters[key] = self.configuration["motor"][key]
        self.motor.reset()
        if self.initialize_motor:
            omega = self.motor_speed_rpm*pi/30.0
            load = self.propeller.parameters["PROP_Shaft_power_W"][0] / (2.0*omega)
            self.motor.initialize_operating_point(self.motor_speed_rpm, load)
        self._last_vdc = self.battery.outputs(0.0)["Voc_pack_V"]
        # [generator AC, battery terminal, SOC, motor Vdc, motor available DC]
        self._delay = [0.0, 0.0, self.battery.soc, self._last_vdc,
                       self.configuration["initial_available_power_W"]]

    def _dynamic_sample(self, time_s, record=True):
        # Mission reference is sampled at 0.5 s; integer ticks avoid boundary drift.
        prop_dt = self.propeller.parameters["PROP_dt_s"]
        mission_t = floor((time_s+1e-10)/prop_dt)*prop_dt
        if getattr(self, "_profile_time", None) != mission_t:
            self._profile = self.propeller.outputs(mission_t)
            self._profile_time = mission_t
        profile = self._profile
        phase = int(profile["Phase_index"])
        load = profile["P_propeller_shaft_W"]/(2.0*self.motor_speed_rpm*pi/30.0)
        gen_prev, bat_prev, soc_prev, vdc, available = self._delay
        inputs = (self.motor_speed_rpm, load, vdc, available/2.0)
        motor = self.motor.outputs(*inputs)
        motor_ac = 2.0*motor["P_elec_W"]
        ems = self.ems.outputs(phase, self.mode, motor_ac, soc_prev, gen_prev,
                               bat_prev, profile["P_motor_input_table_W"])
        demand, engine, generator = self._engine_outputs(
            ems["P_generator_ac_cmd_W"], profile["Altitude_ISA_m"],
            ems["Engine_enable_cmd"])
        command = ems["P_battery_terminal_cmd_W"]
        battery = self.battery.outputs(command)
        actual_available = ems["P_generator_dc_W"] + ems["P_battery_dc_W"]
        next_delay = [generator["P_generator_ac_actual_W"], battery["P_bat_W"],
                      self.battery.soc, battery["V_bat_V"], actual_available]
        row = None
        if record:
            row = {
                "Time_s": time_s, "Phase": phase,
                "Phase_name": self.propeller.parameters["PROP_Phase_names"][phase-1],
                "Altitude_ISA_m": profile["Altitude_ISA_m"],
                "P_aircraft_required_W": profile["P_aircraft_required_W"],
                "Pshaft_request_W": profile["P_propeller_shaft_W"],
                "Pmotor_demand_table_W": profile["P_motor_input_table_W"],
                "Pmotor_output_W": 2.0*motor["P_shaft_W"],
                "Pmotor_ac_actual_W": motor_ac, "Pmotor_dc_actual_W": ems["P_motor_dc_W"],
                "Pengine_demand_W": demand, "Pengine_actual_W": engine["P_engine_W"],
                "Tengine_Nm": engine["BrakeTorque_Nm"],
                "Nengine_rpm": self.engine_speed_rpm if ems["Engine_enable_cmd"] else 0.0,
                "Throttle_deg": engine["Throttle_deg"], "FuelFlow_kgps": engine["FuelFlow_kg_s"],
                "SFC_kg_per_kWh": engine["SFC_kg_per_kWh"],
                "Pgenerator_ac_demand_W": ems["P_generator_ac_cmd_W"],
                "Pgenerator_ac_actual_W": generator["P_generator_ac_actual_W"],
                "Pgenerator_dc_W": ems["P_generator_dc_W"],
                "Pbattery_command_W": command, "Pbattery_W": battery["P_bat_W"],
                "Ibat_A": battery["I_bat_A"],
                "Idc_bus_load_A": ems["P_motor_dc_W"]/max(abs(vdc), 1e-6),
                "Igenerator_bus_A": ems["P_generator_dc_W"]/max(abs(vdc), 1e-6),
                "Vbat_V": battery["V_bat_V"], "Vdc_bus_V": vdc, "SOC": self.battery.soc,
                "Pbus_deficit_W": ems["P_bus_deficit_W"],
                "Ppropulsion_unmet_W": max(profile["P_propeller_shaft_W"]-2*motor["P_shaft_W"], 0.0),
                "Motor_speed_rpm": motor["N_rpm"], "Motor_iq_A": motor["iq_A"],
                "Motor_voltage_dq_V": (motor["ud_V"]**2+motor["uq_V"]**2)**0.5,
                "Motor_copper_loss_total_W": 3*self.motor.parameters["MOT_Rs_Ohm"]*(motor["id_A"]**2+motor["iq_A"]**2),
                "Motor_id_A": motor["id_A"], "Motor_ud_V": motor["ud_V"], "Motor_uq_V": motor["uq_V"],
                "Pbattery_dc_W": ems["P_battery_dc_W"], "Pdc_available_W": actual_available,
                "Pdc_for_motors_W": available,
                "Engine_enable_cmd": ems["Engine_enable_cmd"],
                "Engine_supervisor_state": ems["Engine_supervisor_state"],
                "Engine_on": ems["Engine_enable_cmd"],
                "Solver": "dynamic",
            }
        return row, inputs, command, next_delay

    def run(self, output_step_s=0.05, duration_s=None, progress_callback=None):
        """Run the mission; optional callback receives (stage, time_s, duration_s).

        Progress reports use wall time, independently of the numerical steps.
        No callback means no progress output for library callers.
        """
        output_step_s = float(output_step_s)
        total = self.propeller.parameters["PROP_Total_time_s"]
        duration = total if duration_s is None else float(duration_s)
        if not isfinite(output_step_s) or output_step_s <= 0.0:
            raise ValueError("output_step_s must be positive and finite")
        if not isfinite(duration) or not 0.0 <= duration <= total:
            raise ValueError(f"duration_s must be finite and in [0, {total}]")

        last_report_wall = None
        last_report_time = None

        def report_progress(time_s, force=False):
            nonlocal last_report_wall, last_report_time
            if progress_callback is None or time_s == last_report_time:
                return
            now = perf_counter()
            if force or last_report_wall is None or now-last_report_wall >= 2.0:
                progress_callback("simulate", time_s, duration)
                last_report_wall, last_report_time = now, time_s

        self._reset()
        self._profile_time = None
        if self.solver == "dynamic":
            h = self.configuration["internal_step_s"]
            count, stride = round(duration/h), round(output_step_s/h)
            if stride < 1 or abs(stride*h-output_step_s)>1e-9 or abs(count*h-duration)>1e-9:
                raise ValueError("dynamic duration and output step must be multiples of 0.001 s")
            report_progress(0.0, force=True)
            rows = []
            for index in range(count+1):
                record = index % stride == 0 or index == count
                row, inputs, command, next_delay = self._dynamic_sample(index*h, record)
                if record:
                    rows.append(row)
                if index < count:
                    # Both symmetric branches have identical states, so solve one.
                    # UnitDelay outputs remain fixed through all four RK stages.
                    self.motor.step(*inputs, h)
                    self.battery.step(command, h, max_step_s=h)
                    self._delay = next_delay
                    if (index+1) % 100 == 0:
                        report_progress((index+1)*h)
            report_progress(duration, force=True)
            return rows
        # Fast energy model: 0.05 s state/EMS updates with steady motor outputs.
        h = 0.05
        count, stride = round(duration/h), round(output_step_s/h)
        if (stride < 1 or abs(stride*h-output_step_s) > 1e-9
                or abs(count*h-duration) > 1e-9):
            raise ValueError("mission_fast duration and output step must be multiples of 0.05 s")
        rows = []
        report_progress(0.0, force=True)
        for index in range(count+1):
            time_s = index*h
            record = index % stride == 0 or index == count
            row, command, next_delay = self._row(time_s)
            if record:
                rows.append(row)
            if index < count:
                self.battery.step(command, h, max_step_s=h)
                self._delay = next_delay
                if (index+1) % 100 == 0:
                    report_progress((index+1)*h)
        report_progress(duration, force=True)
        return rows


def summarize(rows, mode):
    fuel = 0.0
    battery_energy_wh = 0.0
    engine_off_s = 0.0
    for left, right in zip(rows[:-1], rows[1:]):
        dt = right["Time_s"]-left["Time_s"]
        fuel += 0.5*(left["FuelFlow_kgps"]+right["FuelFlow_kgps"])*dt
        battery_energy_wh += 0.5*(left["Pbattery_W"]+right["Pbattery_W"])*dt/3600.0
        engine_off_s += 0.5*(2.0-left["Engine_enable_cmd"]-right["Engine_enable_cmd"])*dt
    transitions = sum(
        left["Engine_enable_cmd"] != right["Engine_enable_cmd"]
        for left, right in zip(rows[:-1], rows[1:])
    )
    return {
        "mode": int(mode),
        "duration_s": rows[-1]["Time_s"],
        "sample_count": len(rows),
        "fuel_used_kg": fuel,
        "initial_soc": rows[0]["SOC"],
        "final_soc": rows[-1]["SOC"],
        "minimum_soc": min(row["SOC"] for row in rows),
        "maximum_soc": max(row["SOC"] for row in rows),
        "engine_state_transition_count": transitions,
        "engine_off_duration_s": engine_off_s,
        "battery_terminal_energy_Wh_positive_discharge": battery_energy_wh,
        "maximum_bus_deficit_W": max(row["Pbus_deficit_W"] for row in rows),
        "maximum_propulsion_unmet_W": max(row["Ppropulsion_unmet_W"] for row in rows),
        "engine_map_converged_points": None,
        "engine_map_total_points": 60,
        "engine_map_convergence_status": "not provided by EngineMap_60Case.csv",
        "mission_motor_input_peak_target_W": 80000.0,
        "engine_design_power_W": 62500.0,
        "engine_design_speed_rpm": 5000.0,
        "generator_design_speed_rpm": 6000.0,
        "active_gear_ratio_engine_to_generator": 5000.0 / 6000.0,
        "mission_motor_solver": rows[0].get("Solver", "mission_fast"),
        "internal_step_s": 0.001 if rows[0].get("Solver") == "dynamic" else 0.05,
        "symmetrical_motor_branches": 2,
        "capacity_aging_included": False,
        "fuel_integration": "trapezoid over saved output, as MATLAB result.fuel_consumed_kg",
        "validation_note": (
            "Current 81s3p battery, dual EMRAX 228 MV motors, 62.5 kW engine, "
            "60 kW generator and descent-off mode-selectable EMS; capacity aging is omitted."
        ),
    }


def write_results(rows, output_directory, mode):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory/f"hybrid_system_mode{mode}_outputs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, mode)
    summary_path = output_directory/f"hybrid_system_mode{mode}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return csv_path, summary_path, summary


def run_mission(mode, output_directory, output_step_s=0.05, duration_s=None,
                make_plots=True, solver="mission_fast", initialize_motor=True,
                progress_callback=None, initial_soc=0.90):
    simulation = HybridSystemSimulation(
        mode=mode, solver=solver, initialize_motor=initialize_motor,
        initial_soc=initial_soc,
    )
    rows = simulation.run(output_step_s=output_step_s, duration_s=duration_s,
                          progress_callback=progress_callback)
    duration = rows[-1]["Time_s"]
    if progress_callback is not None:
        progress_callback("save", duration, duration)
    csv_path, summary_path, summary = write_results(rows, output_directory, mode)
    plot_paths = []
    if make_plots:
        if progress_callback is not None:
            progress_callback("plot", duration, duration)
        from .plotting import plot_results
        plot_paths = plot_results(rows, output_directory, mode)
    if progress_callback is not None:
        progress_callback("done", duration, duration)
    return {
        "rows": rows, "csv_path": csv_path, "summary_path": summary_path,
        "summary": summary, "plot_paths": plot_paths,
    }
