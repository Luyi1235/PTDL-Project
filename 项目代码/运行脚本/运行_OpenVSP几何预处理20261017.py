"""Compute only geometry topology; never run a VSPAERO force or wake sweep."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def process(design, output):
    import openvsp as vsp
    from OpenVSP_VSPAERO接口 import configure_vspaero_geometry_sets
    from 机体参数化设计与气动阻力 import AirframeDesign, write_openvsp_geometry
    started = time.monotonic()
    cwd = Path.cwd()
    with native_solver_lock(), tempfile.TemporaryDirectory(prefix="ptdl_mesh_") as folder:
        work = Path(folder)
        try:
            os.chdir(work)
            geometry = work / "airframe.vsp3"
            write_openvsp_geometry(AirframeDesign(**design), geometry, mesh_level="legacy")
            vsp.ClearVSPModel()
            vsp.ReadVSPFile(str(geometry))
            wing = vsp.FindGeomsWithName("Main_Wing")[0]
            thick, thin = configure_vspaero_geometry_sets(vsp)
            vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
            vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [thick])
            vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "ThinGeomSet", [thin])
            vsp.SetVSPAERORefWingID(wing)
            vsp.ExecAnalysis("VSPAEROComputeGeometry")
            lines = (work / "airframe.vspgeom").read_text(encoding="utf8").splitlines()
            if not lines[0].startswith("# vspgeom") or int(lines[1]) != 1:
                raise ValueError("Unsupported geometry format")
            count = [int(v) for v in lines[2].split()]
            record = dict(node_count=count[0], loop_count=count[1], surface_count=count[2],
                          wall_s=time.monotonic()-started, analysis="VSPAEROComputeGeometry",
                          force_solver_called=False, mesh_level="legacy", design=design)
            Path(output).write_text(json.dumps(record, indent=2), encoding="utf8")
            return record
        finally:
            vsp.ClearVSPModel()
            os.chdir(cwd)


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    print(json.dumps(process(request["design"], request["output"])), flush=True)
