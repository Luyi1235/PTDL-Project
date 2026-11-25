"""Run a quasi-steady VSPAERO solve before every JSBSim 6-DoF integration step."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

import jsbsim


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "高保真参考模型" / "PTDL_EST_6DOF"
CORE = ROOT / "核心模型"
VSP_WORKER = ROOT / "运行脚本" / "VSPAERO逐步求解工作进程.py"
FT_PER_M = 3.280839895
MPS_PER_KNOT = 0.514444
LBF_PER_N = 0.224808943
PSF_TO_PA = 47.88025898

if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PTDL六自由度参考模型 import PTDL6DoFParameters
from ptdl_标定缓存 import load_dimension_calibration
from 机体参数化设计与气动阻力 import AirframeDesign, aero_from_design, design_metrics


def _set_text(parent: ET.Element, path: str, value: float) -> None:
    element = parent.find(path)
    if element is None:
        raise RuntimeError(f"Aircraft XML is missing {path}.")
    element.text = f"{value:.12g}"


def _set_coefficient_axis(root: ET.Element, axis_name: str, property_name: str) -> None:
    axis = root.find(f"./aerodynamics/axis[@name='{axis_name}']")
    if axis is None:
        raise RuntimeError(f"Aircraft XML is missing {axis_name} axis.")
    function = axis.find("function")
    if function is None:
        function = ET.SubElement(axis, "function")
    else:
        function.clear()
    function.set("name", f"aero/force/{axis_name.lower()}")
    product = ET.SubElement(function, "product")
    for value in ("aero/qbar-psf", "metrics/Sw-sqft", property_name):
        ET.SubElement(product, "property").text = value


def prepare_aircraft(temp_root: Path, design: AirframeDesign) -> tuple[Path, float]:
    target_dir = temp_root / "aircraft" / "PTDL_EST_6DOF"
    shutil.copytree(MODEL_DIR, target_dir)
    model_path = target_dir / "ptdl_est_6dof.xml"
    tree = ET.parse(model_path)
    root = tree.getroot()
    metrics = design_metrics(design)
    aero = aero_from_design(design, load_dimension_calibration())
    reactions = root.find("./external_reactions")
    if reactions is None:
        raise RuntimeError("Aircraft XML is missing external_reactions.")
    for property_name in ("aero/ptdl-cl", "aero/ptdl-cd"):
        ET.SubElement(reactions, "property").text = property_name
    _set_text(root, "./metrics/wingarea", design.wing_area_m2)
    _set_text(root, "./metrics/wingspan", metrics.wing_span_m)
    _set_text(root, "./metrics/chord", metrics.mean_aerodynamic_chord_m)
    _set_text(root, "./mass_balance/emptywt", metrics.estimated_nonfuel_mass_kg)
    tank = root.find("./propulsion/tank")
    if tank is None:
        raise RuntimeError("Aircraft XML is missing its fuel tank.")
    _set_text(tank, "capacity", metrics.fuel_mass_kg)
    _set_text(tank, "contents", metrics.fuel_mass_kg)
    _set_coefficient_axis(root, "DRAG", "aero/ptdl-cd")
    _set_coefficient_axis(root, "LIFT", "aero/ptdl-cl")
    tree.write(model_path, encoding="utf-8", xml_declaration=True)
    return target_dir, aero.cd0


class VSPAEROProcess:
    def __init__(self, python: Path, design: AirframeDesign, output: Path,
                 mesh_level: str = "legacy"):
        self.temporary = tempfile.TemporaryDirectory(prefix="ptdl_vspaero_steps_")
        self.workdir = Path(self.temporary.name)
        (self.workdir / "init.json").write_text(
            json.dumps({"design": design.__dict__, "mesh_level": mesh_level}), encoding="utf-8"
        )
        self.process = subprocess.Popen(
            [str(python), str(VSP_WORKER), str(self.workdir)],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            error_file = self.workdir / "worker_error.json"
            if error_file.is_file():
                error = json.loads(error_file.read_text(encoding="utf-8"))
                raise RuntimeError(error.get("error", "VSPAERO worker failed to initialize."))
            ready_file = self.workdir / "ready.json"
            if ready_file.is_file():
                ready = json.loads(ready_file.read_text(encoding="utf-8"))
                break
            if self.process.poll() is not None:
                raise RuntimeError(f"VSPAERO worker exited with code {self.process.returncode}.")
            time.sleep(0.05)
        else:
            raise TimeoutError("VSPAERO worker geometry initialization exceeded 120 seconds.")
        if not ready.get("ready"):
            raise RuntimeError(
                ready.get("error", "VSPAERO worker did not initialize.")
                + "\n"
                + ready.get("traceback", "")
            )

    def solve(self, state: dict) -> dict:
        request_id = uuid.uuid4().hex
        request = {**state, "request_id": request_id}
        temporary_request = self.workdir / "request.tmp"
        request_path = self.workdir / "request.json"
        response_path = self.workdir / "response.json"
        response_path.unlink(missing_ok=True)
        temporary_request.write_text(json.dumps(request), encoding="utf-8")
        temporary_request.replace(request_path)
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            error_file = self.workdir / "worker_error.json"
            if error_file.is_file():
                error = json.loads(error_file.read_text(encoding="utf-8"))
                raise RuntimeError(error.get("error", "VSPAERO worker crashed."))
            if response_path.is_file():
                try:
                    result = json.loads(response_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.02)
                    continue
                if result.get("request_id") == request_id:
                    response_path.unlink(missing_ok=True)
                    break
            if self.process.poll() is not None:
                raise RuntimeError(f"VSPAERO worker exited with code {self.process.returncode}.")
            time.sleep(0.02)
        else:
            raise TimeoutError(f"VSPAERO solve for step {state['step']} exceeded 180 seconds.")
        if not result.get("ok"):
            raise RuntimeError(
                result.get("error", "VSPAERO step failed.")
                + "\n"
                + result.get("traceback", "")
            )
        return result

    def close(self, output: Path) -> None:
        if self.process.poll() is None:
            shutdown = self.workdir / "request.json"
            shutdown.write_text(json.dumps({"shutdown": True}), encoding="utf-8")
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        artifact_dir = output / "VSPAERO逐步联算"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for item in self.workdir.iterdir():
            if item.is_file():
                shutil.copy2(item, artifact_dir / item.name)
        self.temporary.cleanup()


def run(
    duration_s: float,
    output_dir: Path,
    vsp_python: Path,
    design: AirframeDesign | None = None,
    progress_file: Path | None = None,
    initial_speed_mps: float = 36.011,
    initial_altitude_m: float = 3000.0,
    initial_alpha_deg: float = 6.84,
    integration_step_s: float | None = None,
    mesh_level: str = "legacy",
) -> dict:
    if not 1.0 / 120.0 <= duration_s <= 10.0:
        raise ValueError("联算时长必须在 0.008333–10 秒内；逐步 VSPAERO 求解计算量很大。")
    if not vsp_python.is_file():
        raise FileNotFoundError(f"OpenVSP Python was not found: {vsp_python}")
    design = design or AirframeDesign()
    output_dir.mkdir(parents=True, exist_ok=True)
    fdm_params = PTDL6DoFParameters()
    vsp_process = VSPAEROProcess(vsp_python, design, output_dir, mesh_level=mesh_level)
    dt = 0.0
    calls = 0
    rows: list[dict] = []
    start_wall = time.perf_counter()

    try:
        with tempfile.TemporaryDirectory(prefix="ptdl_live_6dof_") as temporary:
            aircraft_path, cd0 = prepare_aircraft(Path(temporary), design)
            fdm = jsbsim.FGFDMExec(None)
            fdm.set_debug_level(0)
            fdm.set_aircraft_path(str(aircraft_path.parent))
            if not fdm.load_model("PTDL_EST_6DOF"):
                raise RuntimeError("JSBSim could not load the VSPAERO-coupled aircraft.")
            if integration_step_s is not None:
                if not 1.0 / 120.0 <= integration_step_s <= 0.25:
                    raise ValueError("积分步长必须在 0.008333–0.25 秒内。")
                fdm.set_dt(integration_step_s)
            fdm["aero/ptdl-cl"] = 0.5
            fdm["aero/ptdl-cd"] = cd0 + 0.02
            fdm["ic/h-sl-ft"] = initial_altitude_m * FT_PER_M
            fdm["ic/vt-kts"] = initial_speed_mps / MPS_PER_KNOT
            fdm["ic/alpha-deg"] = initial_alpha_deg
            fdm["ic/theta-deg"] = initial_alpha_deg
            fdm["ic/psi-true-deg"] = 0.0
            if not fdm.run_ic():
                raise RuntimeError("JSBSim initial-condition setup failed.")
            dt = float(fdm.get_delta_t())
            steps = max(1, round(duration_s / dt))
            actual_duration = steps * dt
            speed0 = max(float(fdm["velocities/vt-fps"]) / FT_PER_M, 1.0)
            alpha0 = math.radians(float(fdm["aero/alpha-deg"]))
            q0_pa = float(fdm["aero/qbar-psf"]) * PSF_TO_PA
            initial_drag = q0_pa * design.wing_area_m2 * (
                cd0 + max(0.0, 0.004 + 0.002 * (float(fdm["aero/alpha-deg"]) / 5.0) ** 2)
            )
            shaft_power_kw = min(fdm_params.hybrid_shaft_power_max_kw, initial_drag * speed0 / 0.86 / 1000.0)
            thrust_n = shaft_power_kw * 1000.0 * 0.86 / speed0

            for step in range(steps):
                speed = max(float(fdm["velocities/vt-fps"]) / FT_PER_M, 1.0)
                altitude = max(0.0, float(fdm["position/h-sl-ft"]) / FT_PER_M)
                alpha_deg = float(fdm["aero/alpha-deg"])
                beta_deg = float(fdm["aero/beta-deg"])
                result = vsp_process.solve({
                    "step": step,
                    "speed_mps": speed,
                    "altitude_m": altitude,
                    "alpha_deg": alpha_deg,
                    "beta_deg": beta_deg,
                })
                calls += 1
                # VSPAERO supplies the inviscid lift and induced drag; the
                # geometry build-up CD0 remains the package's profile-drag model.
                cd = cd0 + max(0.0, result["cd"])
                fdm["aero/ptdl-cl"] = result["cl"]
                fdm["aero/ptdl-cd"] = cd
                qbar_pa = float(fdm["aero/qbar-psf"]) * PSF_TO_PA
                drag_n = qbar_pa * design.wing_area_m2 * cd
                lift_n = qbar_pa * design.wing_area_m2 * result["cl"]
                fdm["propulsion/ptdl-thrust-lbf"] = min(
                    thrust_n, fdm_params.hybrid_shaft_power_max_kw * 1000.0 * 0.86 / speed
                ) * LBF_PER_N
                fdm["fcs/elevator-cmd-norm"] = 0.0
                fdm["fcs/aileron-cmd-norm"] = 0.0
                fdm["fcs/rudder-cmd-norm"] = 0.0
                state_before = {
                    "time_s": step * dt,
                    "altitude_m": altitude,
                    "speed_mps": speed,
                    "alpha_deg": alpha_deg,
                    "beta_deg": beta_deg,
                    "cl_vspaero": result["cl"],
                    "cd_vspaero_induced": result["cd"],
                    "oswald_efficiency_wake": result["oswald_efficiency"],
                    "cd0_geometry": cd0,
                    "cd_total": cd,
                    "lift_n": lift_n,
                    "drag_n": drag_n,
                    "solver_elapsed_s": result["solver_elapsed_s"],
                    "vsp_call_index": calls,
                }
                if not fdm.run():
                    raise RuntimeError(f"JSBSim stopped at integration step {step}.")
                state_before.update({
                    "altitude_next_m": float(fdm["position/h-sl-ft"]) / FT_PER_M,
                    "speed_next_mps": float(fdm["velocities/vt-fps"]) / FT_PER_M,
                    "roll_deg": float(fdm["attitude/phi-deg"]),
                    "pitch_deg": float(fdm["attitude/theta-deg"]),
                    "yaw_deg": float(fdm["attitude/psi-deg"]),
                })
                rows.append(state_before)
                if progress_file:
                    progress_file.write_text(json.dumps({
                        "percent": round(100 * (step + 1) / steps),
                        "stage": "逐步气动联算",
                        "detail": f"VSPAERO 求解并推进 6-DoF：{step + 1}/{steps} 步",
                        "completed_steps": step + 1,
                        "total_steps": steps,
                        "vsp_calls": calls,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        vsp_process.close(output_dir)

    if calls != len(rows) or calls != round(actual_duration / dt):
        raise RuntimeError(f"Coupling count mismatch: VSPAERO={calls}, flight steps={len(rows)}.")
    if not all(math.isfinite(value) for row in rows for value in row.values()):
        raise RuntimeError("Coupled time history contains non-finite values.")
    alpha_values = [math.radians(row["alpha_deg"]) for row in rows]
    cl_values = [row["cl_vspaero"] for row in rows]
    alpha_mean = sum(alpha_values) / len(alpha_values)
    cl_mean = sum(cl_values) / len(cl_values)
    alpha_variance = sum((value - alpha_mean) ** 2 for value in alpha_values)
    cl_alpha = (
        sum((alpha - alpha_mean) * (cl - cl_mean) for alpha, cl in zip(alpha_values, cl_values))
        / alpha_variance
        if alpha_variance > 1e-10
        else 5.2
    )
    oswald_values = [
        row["oswald_efficiency_wake"]
        for row in rows
        if row["oswald_efficiency_wake"] > 0.0
    ]
    oswald = (
        sorted(oswald_values)[len(oswald_values) // 2]
        if oswald_values
        else 0.84
    )
    design_mass = design_metrics(design, load_dimension_calibration().reference)
    max_bank = max(abs(row["roll_deg"]) for row in rows)
    max_pitch = max(abs(row["pitch_deg"]) for row in rows)
    max_alpha = max(abs(row["alpha_deg"]) for row in rows)
    min_speed = min(row["speed_mps"] for row in rows)
    min_altitude = min(row["altitude_m"] for row in rows)
    sixdof_constraints = {
        "sixdof_bank_deg": max_bank - 35.0,
        "sixdof_pitch_deg": max_pitch - 35.0,
        "sixdof_alpha_deg": max_alpha - 14.0,
        "sixdof_min_speed_mps": max(15.0, 1.10 * design_mass.stall_speed_mps) - min_speed,
        "sixdof_altitude_m": -min_altitude,
        "sixdof_finite_state": 0.0,
    }
    aero_constraints = {
        "vsp_stepwise_min_speed_mps": max(15.0, 1.10 * design_mass.stall_speed_mps) - min_speed,
        "vsp_stepwise_alpha_deg": max_alpha - 14.0,
        "vsp_stepwise_altitude_deviation_m": max(
            abs(row["altitude_m"] - rows[0]["altitude_m"]) for row in rows
        ) - 250.0,
        "vsp_stepwise_finite_state": 0.0,
    }
    csv_path = output_dir / "逐积分步VSPAERO_6DOF时序.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model_id": "PTDL-VSPAERO-JSBSIM-STEPWISE",
        "status": "quasi_steady_vlm_each_jsbsim_integration_step",
        "model_boundary": "每个 JSBSim 积分步调用一次稳态 VSPAERO 单工况求解；非非定常尾涡/气弹模型。VSPAERO 更新升力与诱导阻力，型阻使用几何估算；侧向力及三轴力矩保留六自由度参考模型估算项。",
        "duration_s": len(rows) * dt,
        "integration_step_s": dt,
        "requested_integration_step_s": integration_step_s or dt,
        "integration_steps": len(rows),
        "vspaero_calls": calls,
        "calls_per_integration_step": calls / len(rows),
        "mean_cl": cl_mean,
        "mean_cd_induced": sum(row["cd_vspaero_induced"] for row in rows) / len(rows),
        "cl_alpha_per_rad": float(max(3.0, min(9.0, cl_alpha))),
        "oswald_efficiency": float(max(0.45, min(0.98, oswald))),
        "oswald_efficiency_definition": (
            "median VSPAERO Ew at integration-step operating points"
        ),
        "sixdof_constraints": sixdof_constraints,
        "aero_constraints": aero_constraints,
        "max_abs_bank_deg": max_bank,
        "max_abs_pitch_deg": max_pitch,
        "max_abs_alpha_deg": max_alpha,
        "minimum_speed_mps": min_speed,
        "openvsp_geometry": str(output_dir / "VSPAERO逐步联算" / "airframe.vsp3"),
        "wall_time_s": time.perf_counter() - start_wall,
        "initial_state": {key: rows[0][key] for key in ("altitude_m", "speed_mps", "alpha_deg", "beta_deg")},
        "final_state": {key: rows[-1][key] for key in ("altitude_next_m", "speed_next_mps", "roll_deg", "pitch_deg", "yaw_deg")},
        "max_single_vspaero_s": max(row["solver_elapsed_s"] for row in rows),
        "outputs": [str(csv_path)],
        "evidence": "vspaero_calls equals integration_steps; every row includes current state, VSPAERO coefficients, loads, and the subsequent 6-DoF state.",
    }
    (output_dir / "逐积分步VSPAERO_6DOF摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vsp-python", type=Path, required=True)
    parser.add_argument("--design-json", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--initial-speed-mps", type=float, default=36.011)
    parser.add_argument("--initial-altitude-m", type=float, default=3000.0)
    parser.add_argument("--initial-alpha-deg", type=float, default=6.84)
    parser.add_argument("--integration-step-s", type=float, default=1.0 / 120.0)
    args = parser.parse_args()
    design = AirframeDesign()
    if args.design_json:
        payload = json.loads(args.design_json.read_text(encoding="utf-8"))
        design = AirframeDesign(**payload.get("design", payload))
    summary = run(
        args.duration_s,
        args.output_dir,
        args.vsp_python,
        design,
        args.progress_file,
        args.initial_speed_mps,
        args.initial_altitude_m,
        args.initial_alpha_deg,
        args.integration_step_s,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
