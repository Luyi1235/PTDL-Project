"""Isolated solver worker with a tighter forward GMRES reduction factor."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def main():
    request = json.loads(sys.stdin.read())
    with native_solver_lock():
        import openvsp as vsp
        import OpenVSP_VSPAERO接口 as adapter
        from 机体参数化设计与气动阻力 import AirframeDesign
        original = vsp.ExecAnalysis

        def execute(name):
            if name == "VSPAEROSweep":
                inputs = list(vsp.GetAnalysisInputNames(name))
                parameter = "ForwardGMRESConvergenceFactor"
                if parameter not in inputs:
                    raise RuntimeError(f"{parameter} absent; installed inputs: {inputs}")
                vsp.SetDoubleAnalysisInput(name, parameter, [.001])
            return original(name)

        vsp.ExecAnalysis = execute
        try:
            polar = adapter.run_vspaero_polar(
                AirframeDesign(**request["design"]), request["output_dir"],
                speed_mps=request["speed_mps"], altitude_m=request["altitude_m"],
                wake_iterations=request["wake_iterations"], mesh_level=request["mesh_level"])
            print(json.dumps(dict(
                cl_alpha_per_rad=polar.cl_alpha_per_rad,
                oswald_efficiency=polar.oswald_efficiency,
                oswald_efficiency_definition=adapter.OSWALD_EFFICIENCY_DEFINITION,
                polar_csv=str(Path(request["output_dir"]) / "vspaero_polar.csv"),
                forward_gmres_convergence_factor=.001,
            )), flush=True)
        finally:
            vsp.ExecAnalysis = original


if __name__ == "__main__":
    main()
