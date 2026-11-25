"""Native wake/grid worker with explicit solver threads and optional finer mesh."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def main():
    request = json.loads(sys.stdin.read())
    nodes = int(os.environ["PTDL_ADAPTIVE_WAKE_NODES"])
    cores = int(os.environ.get("PTDL_ADAPTIVE_NATIVE_CORES", "8"))
    if nodes not in (64, 128, 256) or not 1 <= cores <= 16:
        raise ValueError("Invalid adaptive native budget")
    with native_solver_lock():
        import openvsp as vsp
        import OpenVSP_VSPAERO接口 as adapter
        import 机体参数化设计与气动阻力 as geometry
        original_analysis, original_mesh = vsp.ExecAnalysis, geometry.configure_geometry_mesh

        def configure_mesh(api, level):
            if level != "ultra":
                return original_mesh(api, level)
            for geom in api.FindGeoms():
                name = str(api.GetGeomName(geom))
                api.SetParmVal(geom, "Tess_W", "Shape", 49 if name == "Fuselage" else 81)
                if name == "Fuselage":
                    for section in range(5):
                        api.SetParmVal(geom, "SectTess_U", f"XSec_{section}", 14)
                else:
                    api.SetParmVal(geom, "SectTess_U", "XSec_1", 49 if name == "Main_Wing" else 25)

        def execute(name):
            if name == "VSPAEROSweep":
                vsp.SetIntAnalysisInput(name, "NumWakeNodes", [nodes])
                vsp.SetIntAnalysisInput(name, "NCPU", [cores])
            return original_analysis(name)

        geometry.configure_geometry_mesh, vsp.ExecAnalysis = configure_mesh, execute
        try:
            polar = adapter.run_vspaero_polar(
                geometry.AirframeDesign(**request["design"]), request["output_dir"],
                speed_mps=request["speed_mps"], altitude_m=request["altitude_m"],
                wake_iterations=request["wake_iterations"], mesh_level=request["mesh_level"])
            result = dict(
                cl_alpha_per_rad=polar.cl_alpha_per_rad, oswald_efficiency=polar.oswald_efficiency,
                oswald_efficiency_definition=adapter.OSWALD_EFFICIENCY_DEFINITION,
                polar_csv=str(Path(request["output_dir"]) / "vspaero_polar.csv"),
                wake_nodes=nodes, native_cores_requested=cores, mesh_level=request["mesh_level"])
            (Path(request["output_dir"]) / "自适应求解设置.json").write_text(
                json.dumps(result, indent=2), encoding="utf8")
            print(json.dumps(result), flush=True)
        finally:
            geometry.configure_geometry_mesh, vsp.ExecAnalysis = original_mesh, original_analysis


if __name__ == "__main__":
    main()
