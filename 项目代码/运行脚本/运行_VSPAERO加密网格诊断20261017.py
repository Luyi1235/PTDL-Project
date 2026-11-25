"""Keep free-wake settings fixed while varying only geometric panel density."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def main():
    request = json.loads(sys.stdin.read())
    mesh = request["mesh_level"]
    if mesh not in ("medium", "fine") or request["wake_iterations"] != 30:
        raise ValueError("Unexpected mesh diagnostic protocol")
    with native_solver_lock():
        import openvsp as vsp
        import OpenVSP_VSPAERO接口 as adapter
        from 机体参数化设计与气动阻力 import AirframeDesign
        original = vsp.ExecAnalysis
        applied = {}

        def execute(name):
            if name == "VSPAEROSweep":
                for key, value in dict(FixedWakeFlag=0, NumWakeNodes=8, NCPU=8,
                                       FarDistToggle=0, WakeNumIter=30).items():
                    vsp.SetIntAnalysisInput(name, key, [value])
                    applied[key] = int(vsp.GetIntAnalysisInput(name, key)[0])
            return original(name)

        vsp.ExecAnalysis = execute
        try:
            polar = adapter.run_vspaero_polar(
                AirframeDesign(**request["design"]), request["output_dir"],
                speed_mps=request["speed_mps"], altitude_m=request["altitude_m"],
                wake_iterations=30, mesh_level=mesh)
            result = dict(cl_alpha_per_rad=polar.cl_alpha_per_rad,
                          oswald_efficiency=polar.oswald_efficiency,
                          oswald_efficiency_definition=adapter.OSWALD_EFFICIENCY_DEFINITION,
                          polar_csv=str(Path(request["output_dir"]) / "vspaero_polar.csv"),
                          wake_protocol="free-wake-grid-diagnostic-v1", mesh_level=mesh,
                          wake_nodes=8, native_cores_requested=8, applied=applied)
            (Path(request["output_dir"]) / "冻结求解设置.json").write_text(
                json.dumps(result, indent=2), encoding="utf8")
            print(json.dumps(result), flush=True)
        finally:
            vsp.ExecAnalysis = original


if __name__ == "__main__":
    main()
