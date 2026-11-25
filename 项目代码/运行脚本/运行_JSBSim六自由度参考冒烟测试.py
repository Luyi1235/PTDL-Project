"""Exercise the installed JSBSim 6-DoF runtime on its bundled C172 reference model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jsbsim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "运行结果" / "JSBSim六自由度参考冒烟验证" / "c172p_reference.csv",
    )
    args = parser.parse_args()
    if args.duration_s <= 0:
        raise SystemExit("duration-s 必须为正数。")

    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    if not fdm.load_model("c172p"):
        raise RuntimeError("JSBSim bundled reference aircraft c172p failed to load.")
    fdm["ic/h-sl-ft"] = 3000.0
    fdm["ic/vt-kts"] = 100.0
    if not fdm.run_ic():
        raise RuntimeError("JSBSim initial-condition setup failed.")

    sample_period = 0.1
    total_steps = round(args.duration_s / fdm.get_delta_t())
    sample_stride = max(1, round(sample_period / fdm.get_delta_t()))
    properties = {
        "time_s": "simulation/sim-time-sec",
        "altitude_ft": "position/h-sl-ft",
        "latitude_rad": "position/lat-geod-rad",
        "longitude_rad": "position/long-gc-rad",
        "roll_rad": "attitude/phi-rad",
        "pitch_rad": "attitude/theta-rad",
        "yaw_rad": "attitude/psi-rad",
        "u_fps": "velocities/u-fps",
        "v_fps": "velocities/v-fps",
        "w_fps": "velocities/w-fps",
        "p_radps": "velocities/p-rad_sec",
        "q_radps": "velocities/q-rad_sec",
        "r_radps": "velocities/r-rad_sec",
    }
    records = []
    for step in range(total_steps + 1):
        if step % sample_stride == 0:
            records.append({name: float(fdm[prop]) for name, prop in properties.items()})
        if step < total_steps and not fdm.run():
            raise RuntimeError(f"JSBSim stopped at integration step {step}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=properties)
        writer.writeheader()
        writer.writerows(records)
    print(f"JSBSim {jsbsim.__version__}: {len(records)} samples written to {args.output}")
    print("STATUS=reference_aircraft_smoke_test; NOT a PTDL six-degree-of-freedom validation.")


if __name__ == "__main__":
    main()
