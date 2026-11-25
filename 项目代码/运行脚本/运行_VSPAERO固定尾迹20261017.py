"""Separate fixed-wake aerodynamic protocol, retaining the original Ew definition."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def main():
    request = json.loads(sys.stdin.read())
    nodes = int(os.environ.get("PTDL_FIXED_WAKE_NODES", "64"))
    if nodes not in (32, 64, 128):
        raise ValueError("Unsupported fixed-wake resolution")
    with native_solver_lock():
        import openvsp as vsp
        import OpenVSP_VSPAERO接口 as adapter
        from 机体参数化设计与气动阻力 import AirframeDesign
        far = 20.*(request["design"]["wing_area_m2"]*request["design"]["aspect_ratio"])**.5
        original = vsp.ExecAnalysis
        applied = {}

        def execute(name):
            if name == "VSPAEROSweep":
                names = set(vsp.GetAnalysisInputNames(name))
                if not {"FixedWakeFlag", "NumWakeNodes", "FarDistToggle", "FarDist"}.issubset(names):
                    raise RuntimeError("Installed solver lacks explicit fixed-wake controls")
                for key, value in dict(FixedWakeFlag=1, NumWakeNodes=nodes,
                                       NCPU=8, FarDistToggle=1).items():
                    vsp.SetIntAnalysisInput(name, key, [value])
                    applied[key] = int(vsp.GetIntAnalysisInput(name, key)[0])
                vsp.SetDoubleAnalysisInput(name, "FarDist", [far])
                applied["FarDist"] = float(vsp.GetDoubleAnalysisInput(name, "FarDist")[0])
            return original(name)

        vsp.ExecAnalysis = execute
        try:
            polar = adapter.run_vspaero_polar(
                AirframeDesign(**request["design"]), request["output_dir"],
                speed_mps=request["speed_mps"], altitude_m=request["altitude_m"],
                wake_iterations=request["wake_iterations"], mesh_level=request["mesh_level"])
            result = dict(
                cl_alpha_per_rad=polar.cl_alpha_per_rad, oswald_efficiency=polar.oswald_efficiency,
                oswald_efficiency_definition=adapter.OSWALD_EFFICIENCY_DEFINITION,
                polar_csv=str(Path(request["output_dir"]) / "vspaero_polar.csv"),
                wake_protocol="fixed-wake-v1", wake_nodes=nodes, native_cores_requested=8,
                far_distance_spans=20., mesh_level=request["mesh_level"], applied=applied)
            (Path(request["output_dir"]) / "自适应求解设置.json").write_text(
                json.dumps(result, indent=2), encoding="utf8")
            print(json.dumps(result), flush=True)
        finally:
            vsp.ExecAnalysis = original


if __name__ == "__main__":
    main()
