"""Run PTDL's estimated JSBSim 6-DoF model with three-axis control pulses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import jsbsim

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
MODEL_DIR = ROOT / "高保真参考模型" / "PTDL_EST_6DOF"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from PTDL六自由度参考模型 import PTDL6DoFParameters, write_parameter_manifest


FT_PER_M = 3.280839895
MPS_PER_KNOT = 0.514444
LBF_PER_N = 0.224808943
PSF_TO_PA = 47.88025898
SQFT_PER_M2 = 10.76391042


def build_fdm() -> jsbsim.FGFDMExec:
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.set_aircraft_path(str(MODEL_DIR.parent))
    fdm.set_engine_path(str(MODEL_DIR.parent / "engines"))
    fdm.set_systems_path(str(MODEL_DIR.parent / "systems"))
    if not fdm.load_model("PTDL_EST_6DOF"):
        raise RuntimeError("JSBSim 无法加载 PTDL_EST_6DOF 估计机型。")
    fdm["ic/h-sl-ft"] = 3000.0 * FT_PER_M
    fdm["ic/vt-kts"] = 36.011 / MPS_PER_KNOT
    fdm["ic/alpha-deg"] = 6.84
    fdm["ic/theta-deg"] = 6.84
    fdm["ic/psi-true-deg"] = 0.0
    if not fdm.run_ic():
        raise RuntimeError("PTDL 六自由度初始状态初始化失败。")
    return fdm


def run(duration_s: float, sample_period_s: float, output_dir: Path) -> dict[str, object]:
    if duration_s <= 0.0 or sample_period_s <= 0.0:
        raise ValueError("仿真时长和采样周期必须为正数。")

    params = PTDL6DoFParameters()
    fdm = build_fdm()
    dt = float(fdm.get_delta_t())
    total_steps = round(duration_s / dt)
    sample_stride = max(1, round(sample_period_s / dt))

    # The existing hybrid estimate provides shaft power; this adapter applies
    # propulsive efficiency and converts power to thrust at the current speed.
    qbar_pa = float(fdm["aero/qbar-psf"]) * PSF_TO_PA
    alpha_rad = math.radians(float(fdm["aero/alpha-deg"]))
    beta_rad = math.radians(float(fdm["aero/beta-deg"]))
    ar = params.wingspan_m**2 / params.reference_wing_area_m2
    cd = params.cd0 + params.cl_alpha_per_rad**2 * alpha_rad**2 / (
        math.pi * ar * params.oswald_efficiency
    ) + 0.35 * beta_rad**2
    drag_n = qbar_pa * params.reference_wing_area_m2 * cd
    speed_mps = max(float(fdm["velocities/vt-fps"]) / FT_PER_M, 10.0)
    shaft_power_kw = min(
        params.hybrid_shaft_power_max_kw,
        drag_n * speed_mps / params.propulsive_efficiency / 1000.0,
    )
    thrust_n = shaft_power_kw * 1000.0 * params.propulsive_efficiency / speed_mps
    fdm["propulsion/ptdl-thrust-lbf"] = thrust_n * LBF_PER_N
    thrust_actual_n = thrust_n
    # Estimated elevator trim balances the assumed Cm0/Cma at the reference
    # angle of attack; the following pulse is an increment around this trim.
    elevator_trim_deg = (
        params.cm0 + params.cm_alpha_per_rad * alpha_rad
    ) / -params.cm_elevator_per_rad * 180.0 / math.pi
    elevator_trim_norm = elevator_trim_deg / params.elevator_limit_deg

    props = {
        "time_s": "simulation/sim-time-sec",
        "altitude_m": "position/h-sl-ft",
        "speed_mps": "velocities/vt-fps",
        "alpha_deg": "aero/alpha-deg",
        "beta_deg": "aero/beta-deg",
        "roll_deg": "attitude/phi-deg",
        "pitch_deg": "attitude/theta-deg",
        "yaw_deg": "attitude/psi-deg",
        "p_rad_s": "velocities/p-rad_sec",
        "q_rad_s": "velocities/q-rad_sec",
        "r_rad_s": "velocities/r-rad_sec",
        "elevator_rad": "fcs/elevator-pos-rad",
        "aileron_rad": "fcs/left-aileron-pos-rad",
        "rudder_rad": "fcs/rudder-pos-rad",
        "thrust_lbf": "propulsion/ptdl-thrust-lbf",
    }
    records: list[dict[str, float]] = []
    max_abs_attitude_deg = 0.0
    minimum_altitude_m = math.inf
    minimum_speed_mps = math.inf
    test_input = 0.02
    for step in range(total_steps + 1):
        time_s = step * dt
        current_speed_mps = max(float(fdm["velocities/vt-fps"]) / FT_PER_M, 10.0)
        thrust_command_n = shaft_power_kw * 1000.0 * params.propulsive_efficiency / current_speed_mps
        thrust_actual_n += dt / params.propulsion_time_constant_s * (thrust_command_n - thrust_actual_n)
        fdm["propulsion/ptdl-thrust-lbf"] = thrust_actual_n * LBF_PER_N
        # Independent small command pulses check each modeled control axis.
        fdm["fcs/elevator-cmd-norm"] = (
            elevator_trim_norm + test_input if 10.0 <= time_s < 12.0 else elevator_trim_norm
        )
        fdm["fcs/aileron-cmd-norm"] = test_input if 20.0 <= time_s < 22.0 else 0.0
        fdm["fcs/rudder-cmd-norm"] = test_input if 30.0 <= time_s < 32.0 else 0.0
        if step % sample_stride == 0 or step == total_steps:
            record = {name: float(fdm[prop]) for name, prop in props.items()}
            record["altitude_m"] /= FT_PER_M
            record["speed_mps"] /= FT_PER_M
            records.append(record)
            max_abs_attitude_deg = max(
                max_abs_attitude_deg,
                abs(record["roll_deg"]),
                abs(record["pitch_deg"]),
            )
            minimum_altitude_m = min(minimum_altitude_m, record["altitude_m"])
            minimum_speed_mps = min(minimum_speed_mps, record["speed_mps"])
        if step < total_steps and not fdm.run():
            raise RuntimeError(f"PTDL JSBSim 仿真在第 {step} 步停止。")

    values = [value for row in records for value in row.values()]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("PTDL 六自由度时序包含非有限值。")
    if max_abs_attitude_deg > 35.0 or minimum_altitude_m < 0.0 or minimum_speed_mps < 15.0:
        raise RuntimeError(
            "参考工况超出软件冒烟验收包线："
            f"姿态峰值={max_abs_attitude_deg:.2f} deg，"
            f"最低高度={minimum_altitude_m:.1f} m，"
            f"最低空速={minimum_speed_mps:.1f} m/s。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "PTDL_六自由度三轴操纵响应.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(props))
        writer.writeheader()
        writer.writerows(records)
    manifest_path = write_parameter_manifest(output_dir / "PTDL_六自由度参数清单.json")
    summary = {
        "model_id": "PTDL-EST-6DOF-ULTRA-SCALE",
        "model_status": "estimate_only_not_calibrated_not_flight_validated",
        "jsbsim_version": jsbsim.__version__,
        "duration_s": duration_s,
        "integration_step_s": dt,
        "sample_count": len(records),
        "initial_mass_closure_error_kg": params.mass_closure_error_kg,
        "initial_alpha_deg": records[0]["alpha_deg"],
        "initial_speed_mps": records[0]["speed_mps"],
        "initial_altitude_m": records[0]["altitude_m"],
        "estimated_cruise_shaft_power_kw": shaft_power_kw,
        "estimated_cruise_thrust_n": thrust_n,
        "estimated_elevator_trim_deg": elevator_trim_deg,
        "maximum_absolute_roll_or_pitch_deg": max_abs_attitude_deg,
        "minimum_altitude_m": minimum_altitude_m,
        "minimum_speed_mps": minimum_speed_mps,
        "finite_state_history": True,
        "software_smoke_envelope_passed": True,
        "three_axis_command_pulses": {
            "elevator": [10.0, 12.0, test_input],
            "aileron": [20.0, 22.0, test_input],
            "rudder": [30.0, 32.0, test_input],
        },
        "outputs": [str(csv_path), str(manifest_path)],
        "boundary": "Software-level 6-DoF model execution and control response only; not HIL, flight, or engineering validation.",
    }
    (output_dir / "PTDL_六自由度验证摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--sample-period-s", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "PTDL六自由度估计验证",
    )
    args = parser.parse_args()
    summary = run(args.duration_s, args.sample_period_s, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
