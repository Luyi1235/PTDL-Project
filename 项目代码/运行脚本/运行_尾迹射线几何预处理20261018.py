"""Save geometry meshes for prediction before any aerodynamic test labels exist."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def main(request):
    import openvsp as vsp
    from OpenVSP_VSPAERO接口 import configure_vspaero_geometry_sets
    from 机体参数化设计与气动阻力 import AirframeDesign, write_openvsp_geometry
    output = Path(request["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    started, cwd = time.monotonic(), Path.cwd()
    try:
        for index, job in enumerate(request["jobs"], 1):
            folder = output / job["sample"]
            folder.mkdir(exist_ok=True)
            path, record_path = folder / "airframe.vspgeom", folder / "geometry.json"
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf8"))
                if (record["design"] != job["design"]
                        or record["mesh_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
                    raise RuntimeError("Precomputed geometry changed")
            else:
                with native_solver_lock(), tempfile.TemporaryDirectory(prefix="ptdl_ray_") as directory:
                    work = Path(directory)
                    try:
                        os.chdir(work)
                        write_openvsp_geometry(AirframeDesign(**job["design"]), work / "airframe.vsp3",
                                               mesh_level="legacy")
                        vsp.ClearVSPModel()
                        vsp.ReadVSPFile(str(work / "airframe.vsp3"))
                        wing = vsp.FindGeomsWithName("Main_Wing")[0]
                        thick, thin = configure_vspaero_geometry_sets(vsp)
                        vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
                        vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [thick])
                        vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "ThinGeomSet", [thin])
                        vsp.SetVSPAERORefWingID(wing)
                        vsp.ExecAnalysis("VSPAEROComputeGeometry")
                        shutil.copyfile(work / "airframe.vspgeom", path)
                        shutil.copyfile(work / "airframe.vsp3", folder / "airframe.vsp3")
                        record = dict(sample=job["sample"], design=job["design"],
                            force_solver_called=False, analysis="VSPAEROComputeGeometry",
                            mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        record_path.write_text(json.dumps(record, indent=2), encoding="utf8")
                    finally:
                        vsp.ClearVSPModel()
                        os.chdir(cwd)
            (output.parent / "预处理进度.json").write_text(json.dumps(dict(
                stage="geometry_preprocessing", completed=index, requested=len(request["jobs"]),
                force_solver_called=False, elapsed_s=time.monotonic()-started,
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))), encoding="utf8")
        print(json.dumps(dict(completed=len(request["jobs"]), force_solver_called=False)), flush=True)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main(json.loads(sys.stdin.read()))
