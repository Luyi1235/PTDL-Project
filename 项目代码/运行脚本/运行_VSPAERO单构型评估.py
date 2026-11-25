"""Evaluate one PTDL design with the installed OpenVSP/VSPAERO Python API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from OpenVSP_VSPAERO接口 import (
    OSWALD_EFFICIENCY_DEFINITION,
    run_vspaero_polar,
)
from 机体参数化设计与气动阻力 import AirframeDesign
from 八十小时续航任务模型 import calibrate_endurance_task_baseline


def main() -> None:
    request = json.loads(sys.stdin.read())
    design = AirframeDesign(**request["design"])
    calibration = calibrate_endurance_task_baseline().dimension_calibration
    polar = run_vspaero_polar(
        design,
        request["output_dir"],
        speed_mps=float(request.get("speed_mps", calibration.reference.cruise_speed_mps)),
        altitude_m=float(request.get("altitude_m", calibration.reference.calibration_altitude_m)),
        alpha_deg=tuple(request.get("alpha_deg", (-3.0, -1.0, 1.0, 3.0, 5.0))),
        wake_iterations=int(request.get("wake_iterations", 30)),
        mesh_level=request.get("mesh_level", "legacy"),
    )
    print(
        json.dumps(
            {
                "cl_alpha_per_rad": polar.cl_alpha_per_rad,
                "oswald_efficiency": polar.oswald_efficiency,
                "oswald_efficiency_definition": OSWALD_EFFICIENCY_DEFINITION,
                "polar_csv": str(Path(request["output_dir"]) / "vspaero_polar.csv"),
            }
        )
    )


if __name__ == "__main__":
    main()
