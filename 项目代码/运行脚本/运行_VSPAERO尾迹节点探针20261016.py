"""Isolated native worker for a recorded streamwise wake discretization."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def main():
    request = json.loads(sys.stdin.read())
    nodes = int(os.environ["PTDL_DIAGNOSTIC_WAKE_NODES"])
    if nodes not in (8, 16, 32, 64, 128):
        raise ValueError("Unsupported diagnostic wake node count")
    with native_solver_lock():
        import openvsp as vsp
        import OpenVSP_VSPAERO接口 as adapter
        from 机体参数化设计与气动阻力 import AirframeDesign
        original = vsp.ExecAnalysis

        def execute(name):
            if name == "VSPAEROSweep":
                if "NumWakeNodes" not in vsp.GetAnalysisInputNames(name):
                    raise RuntimeError("Installed solver has no NumWakeNodes input")
                vsp.SetIntAnalysisInput(name, "NumWakeNodes", [nodes])
            return original(name)

        vsp.ExecAnalysis = execute
        try:
            polar = adapter.run_vspaero_polar(
                AirframeDesign(**request["design"]), request["output_dir"],
                speed_mps=request["speed_mps"], altitude_m=request["altitude_m"],
                wake_iterations=request["wake_iterations"], mesh_level=request["mesh_level"])
            print(json.dumps(dict(
                cl_alpha_per_rad=polar.cl_alpha_per_rad, oswald_efficiency=polar.oswald_efficiency,
                oswald_efficiency_definition=adapter.OSWALD_EFFICIENCY_DEFINITION,
                polar_csv=str(Path(request["output_dir"]) / "vspaero_polar.csv"),
                wake_nodes=nodes)), flush=True)
        finally:
            vsp.ExecAnalysis = original


if __name__ == "__main__":
    main()
