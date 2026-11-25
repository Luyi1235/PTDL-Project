"""Isolated, explicitly configured numerical-convergence experiment."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO原生进程锁 import native_solver_lock


def main():
    request = json.loads(sys.stdin.read())
    ints = request["integer_settings"]
    doubles = request["double_settings"]
    with native_solver_lock():
        import openvsp as vsp
        import OpenVSP_VSPAERO接口 as adapter
        from 机体参数化设计与气动阻力 import AirframeDesign
        execute_original = vsp.ExecAnalysis
        applied = {}

        def execute(name):
            if name == "VSPAEROSweep":
                for key, value in ints.items():
                    vsp.SetIntAnalysisInput(name, key, [value])
                    actual = int(vsp.GetIntAnalysisInput(name, key)[0])
                    if actual != value:
                        raise RuntimeError(f"Unapplied integer setting {key}")
                    applied[key] = actual
                for key, value in doubles.items():
                    vsp.SetDoubleAnalysisInput(name, key, [value])
                    actual = float(vsp.GetDoubleAnalysisInput(name, key)[0])
                    if abs(actual-value) > 1e-12:
                        raise RuntimeError(f"Unapplied real setting {key}")
                    applied[key] = actual
            return execute_original(name)

        vsp.ExecAnalysis = execute
        try:
            result = adapter.run_vspaero_polar(
                AirframeDesign(**request["design"]), request["output_dir"],
                speed_mps=35.99708, altitude_m=3000.,
                wake_iterations=ints["WakeNumIter"], mesh_level=request["mesh"])
            payload = dict(cl_alpha_per_rad=result.cl_alpha_per_rad,
                           oswald_efficiency=result.oswald_efficiency, applied=applied,
                           mesh=request["mesh"], protocol=request["protocol"],
                           oswald_efficiency_definition=adapter.OSWALD_EFFICIENCY_DEFINITION,
                           polar_csv=str(Path(request["output_dir"]) / "vspaero_polar.csv"))
            (Path(request["output_dir"]) / "诊断求解设置.json").write_text(
                json.dumps(payload, indent=2), encoding="utf8")
            print(json.dumps(payload), flush=True)
        finally:
            vsp.ExecAnalysis = execute_original


if __name__ == "__main__":
    main()
