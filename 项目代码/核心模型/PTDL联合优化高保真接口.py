"""Candidate-level JSBSim and live OpenVSP/VSPAERO adapters for PTDL."""

from __future__ import annotations

import json
import csv
import importlib.util
import math
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from ULTRA基准标定 import UltraCalibration
from 机体参数化设计与气动阻力 import (
    AeroCoefficients,
    AirframeDesign,
    aircraft_from_design,
    design_metrics,
)
from 飞推控闭环物理模型 import isa_density
from 低阶地面滑跑模型 import (
    GroundRollAssumptions,
    GroundRollResult,
    simulate_ground_roll,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PTDL_MODEL_DIR = PACKAGE_ROOT / "高保真参考模型" / "PTDL_EST_6DOF"
FT_PER_M = 3.280839895
MPS_PER_KNOT = 0.514444
LBF_PER_N = 0.224808943
PSF_TO_PA = 47.88025898
SQFT_PER_M2 = 10.76391042
G = 9.80665


class PTDL6DoFOptimizerValidator:
    """Run a short, candidate-specific JSBSim envelope check in each evaluation."""

    def __init__(
        self,
        duration_s: float = 8.0,
        integration_step_s: float | None = None,
        pulse_amplitude: float = 0.015,
        minimum_speed_mps: float = 15.0,
        maximum_bank_deg: float = 35.0,
        maximum_pitch_deg: float = 35.0,
    ) -> None:
        if duration_s < 5.0:
            raise ValueError("duration_s must be at least 5 seconds to exercise all three axes.")
        if not 0.0 < pulse_amplitude <= 0.05:
            raise ValueError("pulse_amplitude must be in (0, 0.05].")
        self.duration_s = float(duration_s)
        self.integration_step_s = integration_step_s
        self.pulse_amplitude = float(pulse_amplitude)
        self.minimum_speed_mps = float(minimum_speed_mps)
        self.maximum_bank_deg = float(maximum_bank_deg)
        self.maximum_pitch_deg = float(maximum_pitch_deg)
        self.records: list[dict[str, float | str]] = []

    def __call__(
        self,
        design: AirframeDesign,
        aero: AeroCoefficients,
        calibration: UltraCalibration,
        cruise_speed_mps: float,
        cruise_altitude_m: float,
        available_power_kw: float,
    ) -> dict[str, Mapping[str, float]]:
        import jsbsim

        metrics = design_metrics(design, calibration.reference)
        if metrics.fuel_mass_kg <= 0.0:
            return self._failed_result("candidate_has_no_fuel")

        model, temporary = self._write_candidate_model(design, aero, metrics)
        try:
            fdm = jsbsim.FGFDMExec(None)
            fdm.set_debug_level(0)
            fdm.set_aircraft_path(str(model.parent.parent))
            fdm.set_engine_path(str(PTDL_MODEL_DIR.parent / "engines"))
            fdm.set_systems_path(str(PTDL_MODEL_DIR.parent / "systems"))
            if not fdm.load_model(model.parent.name):
                return self._failed_result("jsbsim_model_load_failed")
            if self.integration_step_s is not None:
                if not 1.0 / 120.0 <= self.integration_step_s <= 0.25:
                    raise ValueError("integration_step_s must be between 1/120 and 0.25 seconds.")
                fdm.set_dt(self.integration_step_s)

            rho = isa_density(cruise_altitude_m)
            qbar_pa = 0.5 * rho * cruise_speed_mps**2
            flight_mass_kg = metrics.estimated_nonfuel_mass_kg + metrics.fuel_mass_kg
            target_cl = flight_mass_kg * G / max(qbar_pa * design.wing_area_m2, 1.0)
            alpha_rad = float(
                np.clip(
                    (target_cl - 0.25) / max(aero.cl_alpha_per_rad, 1e-6),
                    math.radians(-8.0),
                    math.radians(11.0),
                )
            )
            fdm["ic/h-sl-ft"] = cruise_altitude_m * FT_PER_M
            fdm["ic/vt-kts"] = cruise_speed_mps / MPS_PER_KNOT
            fdm["ic/alpha-deg"] = math.degrees(alpha_rad)
            fdm["ic/theta-deg"] = math.degrees(alpha_rad)
            fdm["ic/psi-true-deg"] = 0.0
            if not fdm.run_ic():
                return self._failed_result("jsbsim_initialization_failed")

            dt = float(fdm.get_delta_t())
            if not math.isfinite(dt) or dt <= 0.0:
                return self._failed_result("invalid_jsbsim_timestep")
            steps = max(1, round(self.duration_s / dt))
            theta_reference = math.radians(float(fdm["attitude/theta-deg"]))
            alpha_reference = float(fdm["aero/alpha-rad"])
            elevator_trim_rad = -(0.0954 - 0.80 * alpha_reference) / -1.10
            elevator_limit_rad = math.radians(20.0)
            aileron_limit_rad = math.radians(18.0)
            rudder_limit_rad = math.radians(22.0)
            thrust_actual_n = 0.0
            max_abs_bank = 0.0
            max_abs_pitch = 0.0
            max_abs_alpha = 0.0
            min_speed = math.inf
            required_speed = max(self.minimum_speed_mps, 1.10 * metrics.stall_speed_mps)
            min_altitude = math.inf
            max_command = 0.0
            saturated_steps = 0
            max_elevator_command = 0.0
            max_aileron_command = 0.0
            max_rudder_command = 0.0
            finite = True

            for step in range(steps + 1):
                time_s = step * dt
                speed_mps = max(float(fdm["velocities/vt-fps"]) / FT_PER_M, 1.0)
                alpha = float(fdm["aero/alpha-rad"])
                beta = float(fdm["aero/beta-rad"])
                bank = float(fdm["attitude/phi-rad"])
                pitch = float(fdm["attitude/theta-rad"])
                heading = float(fdm["attitude/psi-rad"])
                p = float(fdm["velocities/p-rad_sec"])
                q = float(fdm["velocities/q-rad_sec"])
                r = float(fdm["velocities/r-rad_sec"])

                elevator = elevator_trim_rad + 0.85 * (pitch - theta_reference) + 0.28 * q
                aileron = -1.8 * bank - 0.85 * p
                rudder = 0.15 * r
                if 1.0 <= time_s < 1.5:
                    elevator += self.pulse_amplitude * elevator_limit_rad
                if 2.5 <= time_s < 3.0:
                    aileron += self.pulse_amplitude * aileron_limit_rad
                if 4.0 <= time_s < 4.5:
                    rudder += self.pulse_amplitude * rudder_limit_rad
                elevator_norm = float(np.clip(elevator / elevator_limit_rad, -1.0, 1.0))
                aileron_norm = float(np.clip(aileron / aileron_limit_rad, -1.0, 1.0))
                rudder_norm = float(np.clip(rudder / rudder_limit_rad, -1.0, 1.0))
                max_command = max(max_command, abs(elevator_norm), abs(aileron_norm), abs(rudder_norm))
                max_elevator_command = max(max_elevator_command, abs(elevator_norm))
                max_aileron_command = max(max_aileron_command, abs(aileron_norm))
                max_rudder_command = max(max_rudder_command, abs(rudder_norm))
                saturated_steps += int(max(abs(elevator_norm), abs(aileron_norm), abs(rudder_norm)) >= 0.98)
                fdm["fcs/elevator-cmd-norm"] = elevator_norm
                fdm["fcs/aileron-cmd-norm"] = aileron_norm
                fdm["fcs/rudder-cmd-norm"] = rudder_norm

                qbar_psf = float(fdm["aero/qbar-psf"])
                cd = (
                    aero.cd0
                    + (aero.cl_alpha_per_rad * alpha) ** 2
                    / (math.pi * design.aspect_ratio * max(aero.oswald_efficiency, 1e-4))
                    + 0.35 * beta**2
                )
                drag_n = qbar_psf * PSF_TO_PA * design.wing_area_m2 * cd
                power_limited_thrust_n = (
                    max(available_power_kw, 0.0)
                    * 1000.0
                    * aero.propulsive_efficiency
                    / speed_mps
                )
                thrust_command_n = min(max(drag_n, 0.0), power_limited_thrust_n)
                thrust_actual_n += dt / 2.5 * (thrust_command_n - thrust_actual_n)
                fdm["propulsion/ptdl-thrust-lbf"] = thrust_actual_n * LBF_PER_N

                state_values = (
                    time_s,
                    speed_mps,
                    alpha,
                    beta,
                    bank,
                    pitch,
                    heading,
                    p,
                    q,
                    r,
                    float(fdm["position/h-sl-ft"]),
                )
                finite &= all(math.isfinite(value) for value in state_values)
                max_abs_bank = max(max_abs_bank, abs(math.degrees(bank)))
                max_abs_pitch = max(max_abs_pitch, abs(math.degrees(pitch)))
                max_abs_alpha = max(max_abs_alpha, abs(math.degrees(alpha)))
                min_speed = min(min_speed, speed_mps)
                min_altitude = min(min_altitude, float(fdm["position/h-sl-ft"]) / FT_PER_M)
                if step < steps and not fdm.run():
                    finite = False
                    break

            constraints = {
                "sixdof_bank_deg": max_abs_bank - self.maximum_bank_deg,
                "sixdof_pitch_deg": max_abs_pitch - self.maximum_pitch_deg,
                "sixdof_alpha_deg": max_abs_alpha - 14.0,
                "sixdof_min_speed_mps": required_speed - min_speed,
                "sixdof_altitude_m": -min_altitude,
                "sixdof_control_saturation": saturated_steps / max(steps, 1) - 0.10,
                "sixdof_finite_state": 0.0 if finite else 1.0,
            }
            diagnostics = {
                "sixdof_duration_s": self.duration_s,
                "sixdof_integration_step_s": dt,
                "sixdof_integration_steps": steps,
                "sixdof_max_abs_bank_deg": max_abs_bank,
                "sixdof_max_abs_pitch_deg": max_abs_pitch,
                "sixdof_max_abs_alpha_deg": max_abs_alpha,
                "sixdof_min_speed_mps": min_speed,
                "sixdof_required_speed_mps": required_speed,
                "sixdof_min_altitude_m": min_altitude,
                "sixdof_max_control_command_norm": max_command,
                "sixdof_max_elevator_command_norm": max_elevator_command,
                "sixdof_max_aileron_command_norm": max_aileron_command,
                "sixdof_max_rudder_command_norm": max_rudder_command,
                "sixdof_control_saturation_fraction": saturated_steps / max(steps, 1),
                "sixdof_available_power_kw": max(available_power_kw, 0.0),
                "sixdof_finite_state": float(finite),
                "sixdof_status": (
                    "estimated_JSBSim_stepwise_candidate_trajectory"
                    if self.integration_step_s is not None
                    else "estimated_JSBSim_candidate_envelope_check"
                ),
            }
            self.records.append(
                {
                    "wing_area_m2": design.wing_area_m2,
                    "aspect_ratio": design.aspect_ratio,
                    "cruise_speed_mps": cruise_speed_mps,
                    "cruise_altitude_m": cruise_altitude_m,
                    "available_power_kw": available_power_kw,
                    **diagnostics,
                    **{f"g_{name}": value for name, value in constraints.items()},
                }
            )
            return {"constraints": constraints, "diagnostics": diagnostics}
        finally:
            temporary.cleanup()

    def _write_candidate_model(
        self,
        design: AirframeDesign,
        aero: AeroCoefficients,
        metrics: object,
    ) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory(prefix="ptdl_jsbsim_candidate_")
        model_dir = Path(temporary.name) / "PTDL_EST_6DOF"
        model_dir.mkdir(parents=True)
        root = ET.parse(PTDL_MODEL_DIR / "ptdl_est_6dof.xml").getroot()
        metric_values = {
            "wingarea": design.wing_area_m2,
            "wingspan": metrics.wing_span_m,
            "chord": metrics.mean_aerodynamic_chord_m,
            "htailarea": metrics.horizontal_tail_area_m2,
            "htailarm": 0.52 * design.fuselage_length_m,
            "vtailarea": metrics.vertical_tail_area_m2,
            "vtailarm": 0.52 * design.fuselage_length_m,
        }
        for name, value in metric_values.items():
            node = root.find(f"./metrics/{name}")
            if node is not None:
                node.text = f"{float(value):.12g}"

        mass_ratio = max(metrics.estimated_nonfuel_mass_kg + metrics.fuel_mass_kg, 1.0) / 1360.78
        geometry = root.find("./mass_balance")
        if geometry is not None:
            for name, value in {
                "ixx": 42520.0 * mass_ratio * (metrics.wing_span_m / 24.9936) ** 2,
                "iyy": 9710.0 * mass_ratio * (design.fuselage_length_m / 8.38) ** 2,
                "izz": 52230.0 * mass_ratio * (metrics.wing_span_m / 24.9936) ** 2,
                "emptywt": metrics.estimated_nonfuel_mass_kg,
            }.items():
                node = geometry.find(name)
                if node is not None:
                    node.text = f"{float(value):.12g}"
        tank = root.find("./propulsion/tank")
        if tank is not None:
            capacity = tank.find("capacity")
            contents = tank.find("contents")
            if capacity is not None:
                capacity.text = f"{metrics.fuel_mass_kg:.12g}"
            if contents is not None:
                contents.text = f"{metrics.fuel_mass_kg:.12g}"

        drag_values = root.findall("./aerodynamics/axis[@name='DRAG']/function/product/sum/value")
        if drag_values:
            drag_values[0].text = f"{aero.cd0:.12g}"
        induced_values = root.findall(
            "./aerodynamics/axis[@name='DRAG']/function/product/sum/product/value"
        )
        if induced_values:
            induced_values[0].text = (
                f"{1.0 / (math.pi * design.aspect_ratio * max(aero.oswald_efficiency, 1e-4)):.12g}"
            )
        lift_axis = root.find("./aerodynamics/axis[@name='LIFT']")
        if lift_axis is not None:
            for product in lift_axis.iter("product"):
                children = list(product)
                for index, child in enumerate(children[:-1]):
                    if child.tag == "property" and (child.text or "").strip() == "aero/alpha-rad":
                        if children[index + 1].tag == "value":
                            children[index + 1].text = f"{aero.cl_alpha_per_rad:.12g}"
                        break

        aircraft_xml = model_dir / "ptdl_est_6dof.xml"
        ET.ElementTree(root).write(aircraft_xml, encoding="utf-8", xml_declaration=True)
        reset_root = ET.parse(PTDL_MODEL_DIR / "reset00.xml").getroot()
        for name, value in {
            "ubody": 36.011,
            "wbody": -4.30,
            "altitude": 3000.0,
        }.items():
            node = reset_root.find(name)
            if node is not None:
                node.text = f"{value:.12g}"
        ET.ElementTree(reset_root).write(
            model_dir / "reset00.xml",
            encoding="utf-8",
            xml_declaration=True,
        )
        return aircraft_xml, temporary

    def _failed_result(self, reason: str) -> dict[str, Mapping[str, float]]:
        constraints = {
            "sixdof_bank_deg": 1e6,
            "sixdof_pitch_deg": 1e6,
            "sixdof_alpha_deg": 1e6,
            "sixdof_min_speed_mps": 1e6,
            "sixdof_altitude_m": 1e6,
            "sixdof_control_saturation": 1e6,
            "sixdof_finite_state": 1.0,
        }
        diagnostics: dict[str, float | str] = {
            "sixdof_finite_state": 0.0,
            "sixdof_status": reason,
        }
        self.records.append({"sixdof_status": reason, **{f"g_{key}": value for key, value in constraints.items()}})
        return {"constraints": constraints, "diagnostics": diagnostics}


class PTDL6DoFFullMissionOptimizerValidator:
    """Run the complete JSBSim mission once for every optimizer candidate."""

    full_mission = True
    accepts_powertrain_sizing = True

    def __init__(
        self,
        output_dir: str | Path,
        mission_hours: float = 80.0,
        mission_duration_basis: str = "total-including-landing",
        integration_step_s: float = 0.05,
        control_period_s: float = 2.0,
        terminal_control_period_s: float = 0.5,
        sample_period_s: float = 600.0,
        pitch_kp: float = 4.0,
        pitch_kd: float = 4.0,
        terminal_pitch_kp: float | None = 2.5,
        terminal_pitch_kd: float | None = 4.0,
        terminal_pitch_ki: float = 0.15,
        battery_energy_multiplier: float = 1.20,
        engine_efficiency_values: tuple[float, ...] | None = None,
        dispatch_policy_factory: object | None = None,
        powertrain_config: Mapping[str, object] | None = None,
        minimum_soc: float = 0.20,
        fuel_reserve_fraction: float = 0.05,
        ground_roll_assumptions: GroundRollAssumptions = GroundRollAssumptions(),
        runway_length_m: float | None = None,
    ) -> None:
        if mission_hours < 3.0:
            raise ValueError("Full-mission JSBSim evaluation requires at least 3 mission hours.")
        if not 1.0 / 120.0 <= integration_step_s <= 0.25:
            raise ValueError("integration_step_s must be between 1/120 and 0.25 seconds.")
        if control_period_s < integration_step_s or sample_period_s <= 0.0:
            raise ValueError("Control and sample periods must be positive and consistent.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mission_hours = float(mission_hours)
        if mission_duration_basis not in {
            "total-including-landing",
            "endurance-before-landing",
        }:
            raise ValueError(
                "mission_duration_basis must be 'total-including-landing' "
                "or 'endurance-before-landing'."
            )
        self.mission_duration_basis = mission_duration_basis
        self.integration_step_s = float(integration_step_s)
        self.control_period_s = float(control_period_s)
        self.terminal_control_period_s = float(terminal_control_period_s)
        self.sample_period_s = float(sample_period_s)
        self.pitch_kp = float(pitch_kp)
        self.pitch_kd = float(pitch_kd)
        self.terminal_pitch_kp = (
            None if terminal_pitch_kp is None else float(terminal_pitch_kp)
        )
        self.terminal_pitch_kd = (
            None if terminal_pitch_kd is None else float(terminal_pitch_kd)
        )
        self.terminal_pitch_ki = float(terminal_pitch_ki)
        self.battery_energy_multiplier = float(battery_energy_multiplier)
        self.engine_efficiency_values = engine_efficiency_values
        self.dispatch_policy_factory = dispatch_policy_factory
        self.powertrain_config = dict(powertrain_config or {})
        self.minimum_soc = float(minimum_soc)
        self.fuel_reserve_fraction = float(fuel_reserve_fraction)
        if runway_length_m is not None and runway_length_m <= 0.0:
            raise ValueError("runway_length_m must be positive when configured.")
        self.ground_roll_assumptions = ground_roll_assumptions
        self.runway_length_m = runway_length_m
        self.records: list[dict[str, float | str]] = []
        self._model_writer = PTDL6DoFOptimizerValidator()

    def __call__(
        self,
        design: AirframeDesign,
        aero: AeroCoefficients,
        calibration: UltraCalibration,
        cruise_speed_mps: float,
        cruise_altitude_m: float,
        available_power_kw: float,
        *,
        engine_caps_kw: Mapping[str, float] | None = None,
        motor_assist_kw: Mapping[str, float] | None = None,
        phase_speeds_mps: Mapping[str, float] | None = None,
        phase_altitudes_m: Mapping[str, float] | None = None,
        initial_soc: float = 0.95,
        powertrain_sizing: Mapping[str, float] | None = None,
    ) -> dict[str, Mapping[str, float]]:
        metrics = design_metrics(design, calibration.reference)
        if metrics.fuel_mass_kg <= 0.0:
            return self._failed_result("candidate_has_no_fuel")

        candidate_index = len(self.records) + 1
        candidate_output = self.output_dir / f"candidate_{candidate_index:04d}"
        phase_names = (
            "main_climb_to_leveloff",
            "short_leveloff_to_isr_altitude",
            "endurance_loiter",
            "long_descent",
            "approach_descent",
            "flare",
        )
        overrides: dict[str, dict[str, float]] = {}
        engine_caps_kw = engine_caps_kw or {}
        motor_assist_kw = motor_assist_kw or {}
        phase_speeds_mps = phase_speeds_mps or {}
        phase_altitudes_m = phase_altitudes_m or {}
        for name in phase_names:
            phase_override: dict[str, float] = {}
            if name in engine_caps_kw:
                phase_override["engine_cap_kw"] = max(0.0, float(engine_caps_kw[name]))
            if name in motor_assist_kw:
                phase_override["motor_cap_kw"] = max(0.0, float(motor_assist_kw[name]))
            if name in phase_speeds_mps:
                phase_override["speed_mps"] = float(phase_speeds_mps[name])
            if name in phase_altitudes_m:
                phase_override["altitude_m"] = float(phase_altitudes_m[name])
            overrides[name] = phase_override
        required_landing_speed_mps = max(15.0, 1.10 * float(metrics.stall_speed_mps))
        approach_speed = max(
            float(overrides["approach_descent"].get("speed_mps", 34.0)),
            required_landing_speed_mps + 1.0,
        )
        flare_speed = max(
            float(overrides["flare"].get("speed_mps", 30.0)),
            required_landing_speed_mps + 1.5,
        )
        overrides["approach_descent"]["speed_mps"] = approach_speed
        overrides["flare"]["speed_mps"] = flare_speed

        from 机体参数化设计与气动阻力 import design_metrics as compute_metrics

        span_metrics = compute_metrics(design, calibration.reference)
        takeoff_engine_kw = max(
            0.0, float(engine_caps_kw.get("main_climb_to_leveloff", 120.0))
        )
        takeoff_motor_kw = max(
            0.0, float(motor_assist_kw.get("main_climb_to_leveloff", 30.0))
        )
        takeoff_bus_power_kw = takeoff_engine_kw + takeoff_motor_kw
        engine_fraction = (
            takeoff_engine_kw / takeoff_bus_power_kw
            if takeoff_bus_power_kw > 0.0
            else 1.0
        )
        battery_capacity_mj = (
            max(0.0, float(span_metrics.battery_capacity_mj))
            * self.battery_energy_multiplier
        )
        sizing = dict(powertrain_sizing or {})
        battery_capacity_mj = float(sizing.get("battery_capacity_mj", battery_capacity_mj))
        coupled_aircraft = aircraft_from_design(design, calibration, aero)
        from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
        ground_powertrain = HybridSystemPowertrain(
            HybridSystemAdapterConfig(**self.powertrain_config)
        )
        coupled_aircraft = replace(
            coupled_aircraft,
            engine_electrical_power_max_kw=float(sizing.get(
                "engine_electrical_power_max_kw",
                max(float(p.get("engine_cap_kw", 120.0)) for p in overrides.values()),
            )),
            motor_power_max_kw=float(sizing.get(
                "motor_power_max_kw",
                max(float(p.get("motor_cap_kw", 30.0)) for p in overrides.values()),
            )),
            battery_capacity_mj=battery_capacity_mj,
        )
        takeoff_roll = simulate_ground_roll(
            coupled_aircraft,
            "takeoff",
            takeoff_bus_power_kw * aero.propulsive_efficiency,
            assumptions=self.ground_roll_assumptions,
            mass_kg=coupled_aircraft.initial_mass_kg,
            engine_power_fraction=engine_fraction,
            initial_soc=initial_soc,
            battery_capacity_mj=battery_capacity_mj,
            powertrain=ground_powertrain,
        )
        if not takeoff_roll.feasible:
            return self._failed_ground_takeoff(
                candidate_index,
                design,
                takeoff_roll,
                candidate_output,
            )

        model_xml, temporary = self._model_writer._write_candidate_model(
            design, aero, metrics
        )
        try:
            runner_path = (
                PACKAGE_ROOT
                / "运行脚本"
                / "运行_PTDL估计六自由度全任务闭环.py"
            )
            spec = importlib.util.spec_from_file_location(
                f"_ptdl_full_mission_runner_{candidate_index}", runner_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load the PTDL full-mission JSBSim runner.")
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            summary = runner.run(
                output_dir=candidate_output,
                integration_step_s=self.integration_step_s,
                control_period_s=self.control_period_s,
                terminal_control_period_s=self.terminal_control_period_s,
                sample_period_s=self.sample_period_s,
                pitch_kp=self.pitch_kp,
                pitch_kd=self.pitch_kd,
                terminal_pitch_kp=self.terminal_pitch_kp,
                terminal_pitch_kd=self.terminal_pitch_kd,
                terminal_pitch_ki=self.terminal_pitch_ki,
                mission_hours=self.mission_hours,
                loiter_speed_mps=float(cruise_speed_mps),
                loiter_altitude_m=float(cruise_altitude_m),
                mission_mode=(
                    "endurance_then_land"
                    if self.mission_duration_basis == "endurance-before-landing"
                    else "full"
                ),
                model_dir=model_xml.parent,
                phase_overrides=overrides,
                aircraft_overrides={
                    "reference_wing_area_m2": float(design.wing_area_m2),
                    "wingspan_m": float(span_metrics.wing_span_m),
                    "cd0": float(aero.cd0),
                    "oswald_efficiency": float(aero.oswald_efficiency),
                    "cl_alpha_per_rad": float(aero.cl_alpha_per_rad),
                    "propulsive_efficiency": float(aero.propulsive_efficiency),
                },
                initial_soc=takeoff_roll.terminal_soc_estimated,
                battery_capacity_mj=battery_capacity_mj,
                installed_engine_rating_kw=coupled_aircraft.engine_electrical_power_max_kw,
                installed_motor_assist_kw=coupled_aircraft.motor_power_max_kw,
                fuel_capacity_kg=max(
                    0.0, float(metrics.fuel_mass_kg) - takeoff_roll.estimated_fuel_used_kg
                ),
                engine_efficiency_values=self.engine_efficiency_values,
                dispatch_policy=(
                    self.dispatch_policy_factory()
                    if self.dispatch_policy_factory is not None
                    else None
                ),
                powertrain_config=self.powertrain_config or None,
                initial_altitude_m=0.0,
                initial_speed_mps=takeoff_roll.terminal_speed_mps,
            )
        except Exception as exc:
            return self._failed_result(
                f"full_mission_exception:{type(exc).__name__}:{exc}"
            )
        finally:
            temporary.cleanup()

        phase_metrics = summary.get("phase_metrics_sampled", {})
        flare = phase_metrics.get("flare", {})
        minimum_speed = float(summary.get("minimum_speed_mps_sampled", 0.0))
        required_speed = required_landing_speed_mps
        fuel_margin = float(summary.get("fuel_margin_kg_estimated", -1e6))
        touchdown_speed = float(summary.get("terminal_speed_mps", 0.0))
        landing_fuel_remaining = max(
            0.0,
            float(metrics.fuel_mass_kg)
            - takeoff_roll.estimated_fuel_used_kg
            - float(summary.get("fuel_used_kg_estimated", 0.0)),
        )
        landing_roll = (
            simulate_ground_roll(
                coupled_aircraft,
                "landing",
                0.0,
                assumptions=self.ground_roll_assumptions,
                initial_speed_mps=touchdown_speed,
                mass_kg=float(metrics.estimated_nonfuel_mass_kg)
                + landing_fuel_remaining,
                initial_soc=float(summary["terminal_soc"]),
                battery_capacity_mj=battery_capacity_mj,
                powertrain=ground_powertrain,
            )
            if bool(summary.get("touchdown_detected"))
            else None
        )
        constraints = {
            "sixdof_full_mission_complete": (
                0.0 if bool(summary.get("mission_completed")) else 1.0
            ),
            "sixdof_touchdown": (
                0.0 if bool(summary.get("touchdown_detected")) else 1.0
            ),
            "sixdof_min_speed_mps": required_speed - minimum_speed,
            "sixdof_fuel_reserve_kg": (
                self.fuel_reserve_fraction * float(metrics.fuel_mass_kg) - fuel_margin
            ),
            "sixdof_pitch_deg": (
                float(summary.get("maximum_abs_pitch_deg_sampled", 1e6)) - 25.0
            ),
            "ground_takeoff_liftoff": 0.0 if takeoff_roll.feasible else 1.0,
            "ground_landing_stop": (
                0.0 if landing_roll is not None and landing_roll.feasible else 1.0
            ),
        }
        if self.runway_length_m is not None:
            constraints["ground_takeoff_runway_length_m"] = (
                takeoff_roll.distance_m - self.runway_length_m
            )
            if landing_roll is not None:
                constraints["ground_landing_runway_length_m"] = (
                    landing_roll.distance_m - self.runway_length_m
                )
        diagnostics = {
            "sixdof_full_mission_duration_h": float(
                summary.get("simulation_duration_s", 0.0)
            )
            / 3600.0,
            "sixdof_minimum_endurance_before_landing_h": float(
                summary.get("minimum_flight_hours_before_landing") or 0.0
            ),
            "sixdof_full_mission_integration_steps": float(
                summary.get("integration_steps", 0.0)
            ),
            "sixdof_full_mission_touchdown": float(
                bool(summary.get("touchdown_detected"))
            ),
            "sixdof_full_mission_completed": float(
                bool(summary.get("mission_completed"))
            ),
            "sixdof_full_mission_min_speed_mps": minimum_speed,
            "sixdof_full_mission_required_speed_mps": required_speed,
            "sixdof_full_mission_approach_target_speed_mps": approach_speed,
            "sixdof_full_mission_flare_target_speed_mps": flare_speed,
            "sixdof_full_mission_fuel_used_kg": float(
                summary.get("fuel_used_kg_estimated", 0.0)
            ),
            "sixdof_full_mission_fuel_margin_kg": fuel_margin,
            "sixdof_full_mission_minimum_soc": float(
                summary.get("minimum_soc", 0.0)
            ),
            "sixdof_full_mission_max_pitch_deg": float(
                summary.get("maximum_abs_pitch_deg_sampled", 0.0)
            ),
            "sixdof_full_mission_flare_altitude_error_abs_max_m": float(
                flare.get("altitude_error_abs_max_m", 1e6)
            ),
            "sixdof_full_mission_summary_path": str(
                candidate_output / "PTDL_JSBSim全任务闭环摘要.json"
            ),
            "sixdof_full_mission_status": str(summary.get("status", "unknown")),
            "ground_takeoff_distance_m": takeoff_roll.distance_m,
            "ground_takeoff_duration_s": takeoff_roll.duration_s,
            "ground_takeoff_handoff_speed_mps": takeoff_roll.terminal_speed_mps,
            "ground_takeoff_fuel_used_kg_estimated": takeoff_roll.estimated_fuel_used_kg,
            "ground_takeoff_battery_energy_used_mj_estimated": (
                takeoff_roll.estimated_battery_energy_used_mj
            ),
            "ground_takeoff_terminal_soc_estimated": takeoff_roll.terminal_soc_estimated,
            "ground_landing_distance_m": (
                landing_roll.distance_m if landing_roll is not None else float("nan")
            ),
            "ground_landing_duration_s": (
                landing_roll.duration_s if landing_roll is not None else float("nan")
            ),
            "ground_landing_touchdown_speed_mps": touchdown_speed,
            "ground_runway_length_limit_m": (
                self.runway_length_m if self.runway_length_m is not None else float("nan")
            ),
        }
        candidate_output.mkdir(parents=True, exist_ok=True)
        (candidate_output / "地面起降滑跑耦合摘要.json").write_text(
            json.dumps(
                {
                    "status": (
                        "estimated_ground_roll_screen_linked_to_JSBSim_candidate"
                    ),
                    "handoff": {
                        "initial_altitude_m": 0.0,
                        "takeoff_speed_mps": takeoff_roll.terminal_speed_mps,
                        "fuel_used_before_airborne_kg": (
                            takeoff_roll.estimated_fuel_used_kg
                        ),
                        "soc_at_airborne_initialization": (
                            takeoff_roll.terminal_soc_estimated
                        ),
                        "touchdown_speed_mps": touchdown_speed,
                        "landing_mass_kg_estimated": (
                            float(metrics.estimated_nonfuel_mass_kg)
                            + landing_fuel_remaining
                        ),
                    },
                    "takeoff": takeoff_roll.__dict__,
                    "landing": (
                        landing_roll.__dict__ if landing_roll is not None else None
                    ),
                    "runway_length_limit_m": self.runway_length_m,
                    "constraints": constraints,
                    "limitations": [
                        "Ground phases use a separate longitudinal point-mass model.",
                        "Ground roll duration is excluded from airborne mission endurance.",
                        "No runway length constraint is applied unless configured.",
                        "Energy handoff uses estimated component efficiency and ground-roll power split.",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.records.append(
            {
                "candidate_index": candidate_index,
                "wing_area_m2": design.wing_area_m2,
                "aspect_ratio": design.aspect_ratio,
                "mission_completed": diagnostics["sixdof_full_mission_completed"],
                "touchdown": diagnostics["sixdof_full_mission_touchdown"],
                **diagnostics,
                **{f"g_{name}": value for name, value in constraints.items()},
            }
        )
        return {"constraints": constraints, "diagnostics": diagnostics}

    def _failed_ground_takeoff(
        self,
        candidate_index: int,
        design: AirframeDesign,
        takeoff_roll: GroundRollResult,
        candidate_output: Path,
    ) -> dict[str, Mapping[str, float]]:
        constraints = {
            "sixdof_full_mission_complete": 1.0,
            "sixdof_touchdown": 1.0,
            "sixdof_min_speed_mps": 1e6,
            "sixdof_fuel_reserve_kg": 1e6,
            "sixdof_pitch_deg": 1e6,
            "ground_takeoff_liftoff": 1.0,
            "ground_landing_stop": 1.0,
        }
        if self.runway_length_m is not None:
            constraints["ground_takeoff_runway_length_m"] = (
                takeoff_roll.distance_m - self.runway_length_m
            )
        diagnostics = {
            "sixdof_full_mission_completed": 0.0,
            "sixdof_full_mission_status": "ground_takeoff_screen_failed",
            "ground_takeoff_distance_m": takeoff_roll.distance_m,
            "ground_takeoff_duration_s": takeoff_roll.duration_s,
            "ground_takeoff_handoff_speed_mps": takeoff_roll.terminal_speed_mps,
            "ground_takeoff_fuel_used_kg_estimated": takeoff_roll.estimated_fuel_used_kg,
            "ground_takeoff_battery_energy_used_mj_estimated": (
                takeoff_roll.estimated_battery_energy_used_mj
            ),
            "ground_takeoff_terminal_soc_estimated": takeoff_roll.terminal_soc_estimated,
            "ground_takeoff_termination_reason": takeoff_roll.termination_reason,
        }
        self.records.append(
            {
                "candidate_index": candidate_index,
                "wing_area_m2": design.wing_area_m2,
                "aspect_ratio": design.aspect_ratio,
                **diagnostics,
                **{f"g_{name}": value for name, value in constraints.items()},
            }
        )
        candidate_output.mkdir(parents=True, exist_ok=True)
        (candidate_output / "地面起飞筛查.json").write_text(
            json.dumps(
                {
                    "status": "takeoff_screen_failed_before_JSBSim",
                    "ground_roll": {
                        **takeoff_roll.__dict__,
                    },
                    "constraints": constraints,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"constraints": constraints, "diagnostics": diagnostics}

    def _failed_result(self, reason: str) -> dict[str, Mapping[str, float]]:
        constraints = {
            "sixdof_full_mission_complete": 1.0,
            "sixdof_touchdown": 1.0,
            "sixdof_min_speed_mps": 1e6,
            "sixdof_fuel_reserve_kg": 1e6,
            "sixdof_pitch_deg": 1e6,
            "ground_takeoff_liftoff": 1.0,
            "ground_landing_stop": 1.0,
        }
        diagnostics = {
            "sixdof_full_mission_completed": 0.0,
            "sixdof_full_mission_status": reason,
        }
        self.records.append(
            {"sixdof_full_mission_status": reason}
            | {f"g_{name}": value for name, value in constraints.items()}
        )
        return {"constraints": constraints, "diagnostics": diagnostics}


class StepwiseOpenVSPAeroCandidateValidator:
    """Run one live VSPAERO solve at every step of a short point-mass trajectory."""

    def __init__(
        self,
        python_executable: str | Path,
        output_dir: str | Path,
        duration_s: float = 8.0,
        integration_step_s: float = 0.25,
        couple_with_jsbsim: bool = False,
        jsbsim_duration_s: float | None = None,
        jsbsim_integration_step_s: float | None = None,
        mesh_level: str = "legacy",
    ) -> None:
        if not 0.1 <= duration_s <= 60.0:
            raise ValueError("duration_s must be between 0.1 and 60 seconds.")
        if not 1.0 / 120.0 <= integration_step_s <= 0.25:
            raise ValueError("integration_step_s must be between 1/120 and 0.25 seconds.")
        if couple_with_jsbsim:
            if jsbsim_duration_s is not None and not 5.0 <= jsbsim_duration_s <= 60.0:
                raise ValueError("jsbsim_duration_s must be between 5 and 60 seconds.")
            if (
                jsbsim_integration_step_s is not None
                and not 1.0 / 120.0 <= jsbsim_integration_step_s <= 0.25
            ):
                raise ValueError(
                    "jsbsim_integration_step_s must be between 1/120 and 0.25 seconds."
                )
        self.python_executable = Path(python_executable)
        self.mesh_level = mesh_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duration_s = float(duration_s)
        self.integration_step_s = float(integration_step_s)
        self.couple_with_jsbsim = bool(couple_with_jsbsim)
        self.jsbsim_duration_s = (
            float(jsbsim_duration_s)
            if jsbsim_duration_s is not None
            else self.duration_s
        )
        self.jsbsim_integration_step_s = (
            float(jsbsim_integration_step_s)
            if jsbsim_integration_step_s is not None
            else self.integration_step_s
        )
        self.records: list[dict[str, float | str]] = []

    def evaluate_candidate(
        self,
        design: AirframeDesign,
        calibration: UltraCalibration,
        *,
        speed_mps: float,
        altitude_m: float,
    ) -> dict[str, object]:
        from 机体参数化设计与气动阻力 import aero_from_design

        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_index = len(self.records) + 1
        candidate_dir = self.output_dir / f"candidate_{candidate_index:05d}"
        candidate_dir.mkdir(parents=True, exist_ok=False)
        low_order = aero_from_design(
            design,
            calibration,
            speed_mps=speed_mps,
            altitude_m=altitude_m,
        )
        metrics = design_metrics(design, calibration.reference)
        if self.couple_with_jsbsim:
            return self._evaluate_coupled_candidate(
                design,
                calibration,
                low_order,
                metrics,
                candidate_index,
                candidate_dir,
                speed_mps,
                altitude_m,
            )
        mass_kg = metrics.estimated_nonfuel_mass_kg + metrics.fuel_mass_kg
        steps = max(1, round(self.duration_s / self.integration_step_s))
        dt = self.duration_s / steps
        speed = max(float(speed_mps), 1.0)
        altitude = max(float(altitude_m), 0.0)
        target_speed = speed
        target_altitude = altitude
        gamma = 0.0
        qbar = max(0.5 * isa_density(altitude) * speed**2, 1.0)
        target_cl = mass_kg * G / max(qbar * design.wing_area_m2, 1.0)
        alpha = float(
            np.clip(
                (target_cl - 0.25) / max(low_order.cl_alpha_per_rad, 1e-6),
                math.radians(-8.0),
                math.radians(11.0),
            )
        )
        rows: list[dict[str, float]] = []
        process = None
        started = time.perf_counter()
        try:
            process_type = _stepwise_vspaero_process_type()
            process = process_type(self.python_executable, design, candidate_dir,
                                   mesh_level=self.mesh_level)
            for step in range(steps):
                rho = isa_density(altitude)
                qbar = max(0.5 * rho * speed**2, 1.0)
                remaining_s = max(self.duration_s - step * dt, dt)
                hdot_command = float(
                    np.clip(
                        1.2 * (target_altitude - altitude) / remaining_s,
                        -speed * math.sin(math.radians(10.0)),
                        speed * math.sin(math.radians(10.0)),
                    )
                )
                gamma_command = math.asin(
                    float(np.clip(hdot_command / max(speed, 1.0), -0.98, 0.98))
                )
                gamma_rate_command = float(
                    np.clip(
                        (1.0 - math.exp(-0.045 * dt))
                        * (gamma_command - gamma)
                        / max(dt, 1e-9),
                        -math.radians(0.20),
                        math.radians(0.20),
                    )
                )
                lift_command = mass_kg * (G * math.cos(gamma) + speed * gamma_rate_command)
                cl_command = lift_command / max(qbar * design.wing_area_m2, 1.0)
                alpha_command = float(
                    np.clip(
                        (cl_command - 0.25) / max(low_order.cl_alpha_per_rad, 1e-6),
                        math.radians(-8.0),
                        math.radians(11.0),
                    )
                )
                alpha += float(
                    np.clip(
                        alpha_command - alpha,
                        -math.radians(4.0) * dt,
                        math.radians(4.0) * dt,
                    )
                )
                result = process.solve(
                    {
                        "step": step,
                        "speed_mps": speed,
                        "altitude_m": altitude,
                        "alpha_deg": math.degrees(alpha),
                        "beta_deg": 0.0,
                    }
                )
                cl = float(result["cl"])
                cdi = max(0.0, float(result["cd"]))
                cd = low_order.cd0 + cdi
                lift = qbar * design.wing_area_m2 * cl
                drag = qbar * design.wing_area_m2 * cd
                accel_command = float(
                    np.clip(1.2 * (target_speed - speed) / 30.0, -1.2, 1.2)
                )
                thrust_request = max(
                    0.0,
                    drag + mass_kg * accel_command + mass_kg * G * math.sin(gamma),
                )
                thrust = min(thrust_request, 180_000.0 / max(speed, 1.0))
                speed_dot = float(
                    np.clip(
                        (thrust - drag - mass_kg * G * math.sin(gamma)) / mass_kg,
                        -1.2,
                        1.2,
                    )
                )
                gamma_dot = float(
                    np.clip(
                        (lift - mass_kg * G * math.cos(gamma))
                        / max(mass_kg * speed, 1.0),
                        -math.radians(2.0),
                        math.radians(2.0),
                    )
                )
                rows.append(
                    {
                        "time_s": step * dt,
                        "speed_mps": speed,
                        "altitude_m": altitude,
                        "gamma_deg": math.degrees(gamma),
                        "alpha_deg": math.degrees(alpha),
                        "cl_vspaero": cl,
                        "cd_vspaero_induced": cdi,
                        "e_wake": float(result["oswald_efficiency"]),
                        "cd0_component_build_up": low_order.cd0,
                        "lift_n": lift,
                        "drag_n": drag,
                        "thrust_n": thrust,
                        "solver_elapsed_s": float(result["solver_elapsed_s"]),
                    }
                )
                speed = max(1.0, speed + speed_dot * dt)
                gamma = float(
                    np.clip(
                        gamma + gamma_dot * dt,
                        -math.radians(25.0),
                        math.radians(25.0),
                    )
                )
                altitude = max(0.0, altitude + speed * math.sin(gamma) * dt)

            alpha_rad = np.radians([row["alpha_deg"] for row in rows])
            cls = np.asarray([row["cl_vspaero"] for row in rows], dtype=float)
            cl_alpha = (
                float(np.polyfit(alpha_rad, cls, 1)[0])
                if np.ptp(alpha_rad) >= math.radians(0.05)
                else low_order.cl_alpha_per_rad
            )
            oswald_values = [
                row["e_wake"]
                for row in rows
                if math.isfinite(row["e_wake"]) and row["e_wake"] > 0.0
            ]
            aero = AeroCoefficients(
                cd0=low_order.cd0,
                oswald_efficiency=float(
                    np.clip(
                        np.median(oswald_values) if oswald_values else low_order.oswald_efficiency,
                        0.45,
                        0.98,
                    )
                ),
                cl_alpha_per_rad=float(np.clip(cl_alpha, 3.0, 9.0)),
                cl_max=low_order.cl_max,
                propulsive_efficiency=low_order.propulsive_efficiency,
                reference_reynolds=low_order.reference_reynolds,
            )
            minimum_speed = min(row["speed_mps"] for row in rows)
            max_alpha = max(abs(row["alpha_deg"]) for row in rows)
            altitude_deviation = max(abs(row["altitude_m"] - target_altitude) for row in rows)
            constraints = {
                "vsp_stepwise_min_speed_mps": max(15.0, 1.10 * metrics.stall_speed_mps) - minimum_speed,
                "vsp_stepwise_alpha_deg": max_alpha - 14.0,
                "vsp_stepwise_altitude_deviation_m": altitude_deviation - 250.0,
                "vsp_stepwise_finite_state": 0.0
                if all(math.isfinite(value) for row in rows for value in row.values())
                else 1.0,
            }
            diagnostics: dict[str, float | str] = {
                "vsp_stepwise_duration_s": len(rows) * dt,
                "vsp_stepwise_integration_step_s": dt,
                "vsp_stepwise_integration_steps": len(rows),
                "vsp_stepwise_vspaero_calls": len(rows),
                "vsp_stepwise_wall_time_s": time.perf_counter() - started,
                "vsp_stepwise_mean_cl": float(np.mean(cls)),
                "vsp_stepwise_mean_cd_induced": float(
                    np.mean([row["cd_vspaero_induced"] for row in rows])
                ),
                "vsp_stepwise_oswald_definition": (
                    "median VSPAERO Ew at integration-step operating points"
                ),
                "vsp_stepwise_cl_alpha_per_rad": aero.cl_alpha_per_rad,
                "vsp_stepwise_oswald_efficiency": aero.oswald_efficiency,
                "vsp_stepwise_status": "candidate_point_mass_each_step_live_vspaero",
            }
            with (candidate_dir / "VSPAERO逐积分步时序.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            record: dict[str, float | str] = {
                "candidate_index": candidate_index,
                "wing_area_m2": design.wing_area_m2,
                "aspect_ratio": design.aspect_ratio,
                "speed_mps": speed_mps,
                "altitude_m": altitude_m,
                **diagnostics,
                **{f"g_{name}": value for name, value in constraints.items()},
            }
            self.records.append(record)
            return {"aero": aero, "constraints": constraints, "diagnostics": diagnostics}
        except Exception as error:
            constraints = {
                "vsp_stepwise_min_speed_mps": 1e6,
                "vsp_stepwise_alpha_deg": 1e6,
                "vsp_stepwise_altitude_deviation_m": 1e6,
                "vsp_stepwise_finite_state": 1.0,
            }
            diagnostics: dict[str, float | str] = {
                "vsp_stepwise_status": f"failed: {error}",
                "vsp_stepwise_integration_step_s": dt,
                "vsp_stepwise_integration_steps": 0.0,
            }
            self.records.append(
                {
                    "candidate_index": candidate_index,
                    "status": f"failed: {error}",
                    **{f"g_{name}": value for name, value in constraints.items()},
                }
            )
            return {"aero": low_order, "constraints": constraints, "diagnostics": diagnostics}
        finally:
            if process is not None:
                process.close(candidate_dir)

    def _evaluate_coupled_candidate(
        self,
        design: AirframeDesign,
        calibration: UltraCalibration,
        low_order: AeroCoefficients,
        metrics,
        candidate_index: int,
        candidate_dir: Path,
        speed_mps: float,
        altitude_m: float,
    ) -> dict[str, object]:
        started = time.perf_counter()
        dt = min(self.integration_step_s, self.jsbsim_integration_step_s)
        duration_s = max(self.duration_s, self.jsbsim_duration_s)
        rho = isa_density(max(float(altitude_m), 0.0))
        qbar = max(0.5 * rho * max(float(speed_mps), 1.0) ** 2, 1.0)
        total_mass = metrics.estimated_nonfuel_mass_kg + metrics.fuel_mass_kg
        target_cl = total_mass * G / max(qbar * design.wing_area_m2, 1.0)
        initial_alpha_deg = math.degrees(
            float(
                np.clip(
                    (target_cl - 0.25) / max(low_order.cl_alpha_per_rad, 1e-6),
                    math.radians(-8.0),
                    math.radians(11.0),
                )
            )
        )
        try:
            runner = _stepwise_coupled_runner_module()
            summary = runner.run(
                duration_s=duration_s,
                output_dir=candidate_dir,
                vsp_python=self.python_executable,
                design=design,
                initial_speed_mps=float(speed_mps),
                initial_altitude_m=float(altitude_m),
                initial_alpha_deg=initial_alpha_deg,
                integration_step_s=dt,
                mesh_level=self.mesh_level,
            )
            aero = AeroCoefficients(
                cd0=low_order.cd0,
                oswald_efficiency=float(
                    np.clip(summary["oswald_efficiency"], 0.45, 0.98)
                ),
                cl_alpha_per_rad=float(
                    np.clip(summary["cl_alpha_per_rad"], 3.0, 9.0)
                ),
                cl_max=low_order.cl_max,
                propulsive_efficiency=low_order.propulsive_efficiency,
                reference_reynolds=low_order.reference_reynolds,
            )
            constraints = {
                **summary["aero_constraints"],
                **summary["sixdof_constraints"],
            }
            diagnostics: dict[str, float | str] = {
                "vsp_stepwise_duration_s": float(summary["duration_s"]),
                "vsp_stepwise_integration_step_s": float(
                    summary["integration_step_s"]
                ),
                "vsp_stepwise_integration_steps": float(
                    summary["integration_steps"]
                ),
                "vsp_stepwise_vspaero_calls": float(summary["vspaero_calls"]),
                "vsp_stepwise_wall_time_s": float(summary["wall_time_s"]),
                "vsp_stepwise_mean_cl": float(summary["mean_cl"]),
                "vsp_stepwise_mean_cd_induced": float(
                    summary["mean_cd_induced"]
                ),
                "vsp_stepwise_cl_alpha_per_rad": aero.cl_alpha_per_rad,
                "vsp_stepwise_oswald_efficiency": aero.oswald_efficiency,
                "vsp_stepwise_status": "coupled_vspaero_jsbsim_each_integration_step",
                "sixdof_stepwise_duration_s": float(summary["duration_s"]),
                "sixdof_stepwise_integration_step_s": float(
                    summary["integration_step_s"]
                ),
                "sixdof_stepwise_integration_steps": float(
                    summary["integration_steps"]
                ),
                "sixdof_stepwise_max_abs_bank_deg": float(
                    summary["max_abs_bank_deg"]
                ),
                "sixdof_stepwise_max_abs_pitch_deg": float(
                    summary["max_abs_pitch_deg"]
                ),
                "sixdof_stepwise_max_abs_alpha_deg": float(
                    summary["max_abs_alpha_deg"]
                ),
                "sixdof_stepwise_minimum_speed_mps": float(
                    summary["minimum_speed_mps"]
                ),
                "sixdof_stepwise_status": str(summary["status"]),
            }
            record: dict[str, float | str] = {
                "candidate_index": candidate_index,
                "wing_area_m2": design.wing_area_m2,
                "aspect_ratio": design.aspect_ratio,
                "speed_mps": speed_mps,
                "altitude_m": altitude_m,
                **diagnostics,
                **{f"g_{name}": float(value) for name, value in constraints.items()},
            }
            self.records.append(record)
            return {
                "aero": aero,
                "constraints": constraints,
                "diagnostics": diagnostics,
            }
        except Exception as error:
            constraints = {
                "vsp_stepwise_min_speed_mps": 1e6,
                "vsp_stepwise_alpha_deg": 1e6,
                "vsp_stepwise_altitude_deviation_m": 1e6,
                "vsp_stepwise_finite_state": 1.0,
                "sixdof_bank_deg": 1e6,
                "sixdof_pitch_deg": 1e6,
                "sixdof_alpha_deg": 1e6,
                "sixdof_min_speed_mps": 1e6,
                "sixdof_altitude_m": 1e6,
                "sixdof_control_saturation": 1e6,
                "sixdof_finite_state": 1.0,
            }
            diagnostics = {
                "vsp_stepwise_status": f"coupled_failed: {error}",
                "sixdof_stepwise_status": f"coupled_failed: {error}",
                "vsp_stepwise_integration_step_s": dt,
                "vsp_stepwise_integration_steps": 0.0,
                "vsp_stepwise_wall_time_s": time.perf_counter() - started,
            }
            self.records.append(
                {
                    "candidate_index": candidate_index,
                    "status": f"coupled_failed: {error}",
                    **{f"g_{name}": value for name, value in constraints.items()},
                }
            )
            return {
                "aero": low_order,
                "constraints": constraints,
                "diagnostics": diagnostics,
            }


_STEPWISE_PROCESS_TYPE = None


def _stepwise_coupled_runner_module():
    path = PACKAGE_ROOT / "运行脚本" / "运行_PTDL逐积分步VSPAERO_6DOF耦合.py"
    module_name = "ptdl_stepwise_vspaero_runner"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load VSPAERO/JSBSim stepwise runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stepwise_vspaero_process_type():
    global _STEPWISE_PROCESS_TYPE
    if _STEPWISE_PROCESS_TYPE is not None:
        return _STEPWISE_PROCESS_TYPE
    module = _stepwise_coupled_runner_module()
    _STEPWISE_PROCESS_TYPE = module.VSPAEROProcess
    return _STEPWISE_PROCESS_TYPE


class LiveOpenVSPAeroProvider:
    """Use the installed CPython 3.11 OpenVSP binding and cache geometry results."""

    def __init__(
        self,
        python_executable: str | Path,
        worker_script: str | Path,
        output_dir: str | Path,
        calibration: UltraCalibration,
        timeout_s: float = 180.0,
        mesh_level: str = "legacy",
    ) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = Path(worker_script)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration = calibration
        self.timeout_s = float(timeout_s)
        if mesh_level not in {"legacy", "medium", "fine", "verification"}:
            raise ValueError("Unknown VSPAERO mesh level")
        self.mesh_level = mesh_level
        self._cache: dict[tuple[float, ...], AeroCoefficients] = {}
        self._failures: dict[tuple[float, ...], str] = {}
        self.last_failure: str | None = None
        self.records: list[dict[str, float | str]] = []
        self._baselines: dict[tuple[float, float], tuple[float, float]] = {}
        self.accepts_operating_point = True
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"OpenVSP Python runtime not found: {self.python_executable}")

    def __call__(
        self,
        design: AirframeDesign,
        calibration: UltraCalibration,
        *,
        speed_mps: float | None = None,
        altitude_m: float | None = None,
    ) -> AeroCoefficients:
        from 机体参数化设计与气动阻力 import aero_from_design

        speed = float(speed_mps or calibration.reference.cruise_speed_mps)
        altitude = float(
            calibration.reference.calibration_altitude_m
            if altitude_m is None
            else altitude_m
        )
        condition = (round(speed, 5), round(altitude, 2))
        geometry_key = tuple(
            round(float(getattr(design, name)), 7)
            for name in (
                "wing_area_m2",
                "aspect_ratio",
                "taper_ratio",
                "sweep_deg",
                "tip_twist_deg",
                "fuselage_length_m",
                "fuselage_diameter_m",
            )
        )
        key = geometry_key + condition
        if key in self._cache:
            self.last_failure = self._failures.get(key)
            return self._cache[key]
        self.last_failure = None
        if condition not in self._baselines:
            self._baselines[condition] = self._run_polar(
                AirframeDesign(),
                f"baseline_{condition[0]:.3f}mps_{condition[1]:.0f}m",
                speed,
                altitude,
            )
            baseline = self._baselines[condition]
            self.records.append(
                {
                    "geometry_id": f"baseline_{condition[0]:.3f}mps_{condition[1]:.0f}m",
                    "speed_mps": speed,
                    "altitude_m": altitude,
                    "vspaero_cl_alpha_per_rad": baseline[0],
                    "vspaero_oswald_efficiency": baseline[1],
                    "status": "live_OpenVSP_VSPAERO_reference",
                }
            )
        geometry_id = f"candidate_{len(self.records):04d}"
        baseline_cl, baseline_e = self._baselines[condition]
        low_order = aero_from_design(
            design,
            calibration,
            speed_mps=speed,
            altitude_m=altitude,
        )
        try:
            cl_alpha, oswald = self._run_polar(design, geometry_id, speed, altitude)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            self.last_failure = str(error)
            self._failures[key] = self.last_failure
            self._cache[key] = low_order
            self.records.append(
                {
                    **dict(zip(
                        (
                            "wing_area_m2",
                            "aspect_ratio",
                            "taper_ratio",
                            "sweep_deg",
                            "tip_twist_deg",
                            "fuselage_length_m",
                            "fuselage_diameter_m",
                        ),
                        key,
                    )),
                    "geometry_id": geometry_id,
                    "speed_mps": speed,
                    "altitude_m": altitude,
                    "polar_csv": str(self.output_dir / geometry_id / "vspaero_polar.csv"),
                    "status": "vspaero_failed_low_order_fallback_infeasible",
                    "failure_reason": self.last_failure,
                }
            )
            return low_order
        result = AeroCoefficients(
            cd0=low_order.cd0,
            oswald_efficiency=float(np.clip(low_order.oswald_efficiency * oswald / baseline_e, 0.45, 0.98)),
            cl_alpha_per_rad=float(np.clip(low_order.cl_alpha_per_rad * cl_alpha / baseline_cl, 3.0, 9.0)),
            cl_max=low_order.cl_max,
            propulsive_efficiency=low_order.propulsive_efficiency,
            reference_reynolds=low_order.reference_reynolds,
        )
        self._cache[key] = result
        self.records.append(
            {
                **dict(zip(
                    (
                        "wing_area_m2",
                        "aspect_ratio",
                        "taper_ratio",
                        "sweep_deg",
                        "tip_twist_deg",
                        "fuselage_length_m",
                        "fuselage_diameter_m",
                    ),
                    key,
                )),
                "geometry_id": geometry_id,
                "speed_mps": speed,
                "altitude_m": altitude,
                "vspaero_cl_alpha_per_rad": cl_alpha,
                "vspaero_oswald_efficiency": oswald,
                "corrected_cl_alpha_per_rad": result.cl_alpha_per_rad,
                "corrected_oswald_efficiency": result.oswald_efficiency,
                "cd0_component_build_up": result.cd0,
                "polar_csv": str(self.output_dir / geometry_id / "vspaero_polar.csv"),
                "status": "live_OpenVSP_VSPAERO",
            }
        )
        return result

    def _run_polar(
        self,
        design: AirframeDesign,
        label: str,
        speed_mps: float,
        altitude_m: float,
    ) -> tuple[float, float]:
        destination = self.output_dir / label
        request = {
            "design": {
                name: float(getattr(design, name))
                for name in (
                    "wing_area_m2",
                    "aspect_ratio",
                    "taper_ratio",
                    "sweep_deg",
                    "tip_twist_deg",
                    "fuselage_length_m",
                    "fuselage_diameter_m",
                    "propeller_diameter_m",
                    "payload_mass_kg",
                    "hybrid_system_mass_kg",
                )
            },
            "output_dir": str(destination),
            "speed_mps": float(speed_mps),
            "altitude_m": float(altitude_m),
            "mesh_level": self.mesh_level,
        }
        process = subprocess.run(
            [str(self.python_executable), str(self.worker_script)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
            cwd=str(PACKAGE_ROOT),
        )
        if process.returncode != 0:
            raise RuntimeError(
                "OpenVSP/VSPAERO candidate analysis failed "
                f"({process.returncode}):\n{process.stdout[-2500:]}\n{process.stderr[-2500:]}"
            )
        payload = self._parse_worker_output(process.stdout)
        cl_alpha = float(payload["cl_alpha_per_rad"])
        oswald = float(payload["oswald_efficiency"])
        if not np.isfinite([cl_alpha, oswald]).all() or cl_alpha <= 0.0 or oswald <= 0.0:
            raise RuntimeError("OpenVSP/VSPAERO returned invalid aerodynamic coefficients.")
        return cl_alpha, oswald

    @staticmethod
    def _parse_worker_output(output: str) -> dict[str, object]:
        for marker in reversed(list(re.finditer(r'"cl_alpha_per_rad"\s*:', output))):
            start = output.rfind("{", 0, marker.start())
            end = output.find("}", marker.end())
            if start < 0 or end < 0:
                continue
            try:
                payload = json.loads(output[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and {
                "cl_alpha_per_rad",
                "oswald_efficiency",
            }.issubset(payload):
                return payload
        raise RuntimeError(f"Could not parse OpenVSP/VSPAERO result: {output[-2500:]}")
