"""Persistent OpenVSP/VSPAERO worker for one flight-condition solve per request."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))

from OpenVSP_VSPAERO接口 import (
    configure_vspaero_geometry_sets,
)
from 机体参数化设计与气动阻力 import AirframeDesign, design_metrics, write_openvsp_geometry
from 飞推控闭环物理模型 import isa_density


def run() -> None:
    import openvsp as vsp

    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request = json.loads((output / "init.json").read_text(encoding="utf-8"))
    design = AirframeDesign(**request["design"])
    metrics = design_metrics(design)
    geometry = output / "airframe.vsp3"
    write_openvsp_geometry(design, geometry, mesh_level=request.get("mesh_level", "legacy"))
    os.chdir(output)
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(str(geometry))
    wing_ids = list(vsp.FindGeomsWithName("Main_Wing"))
    if not wing_ids:
        raise RuntimeError("OpenVSP geometry is missing Main_Wing.")
    wing_id = wing_ids[0]
    thick_set, thin_set = configure_vspaero_geometry_sets(vsp)

    geometry_analysis = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(geometry_analysis)
    vsp.SetIntAnalysisInput(geometry_analysis, "GeomSet", [thick_set])
    vsp.SetIntAnalysisInput(geometry_analysis, "ThinGeomSet", [thin_set])
    vsp.SetVSPAERORefWingID(wing_id)
    vsp.ExecAnalysis(geometry_analysis)
    (output / "ready.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    request_path = output / "request.json"
    response_path = output / "response.json"
    while True:
        if not request_path.exists():
            time.sleep(0.01)
            continue
        try:
            case = json.loads(request_path.read_text(encoding="utf-8"))
            request_path.unlink(missing_ok=True)
            request_id = case.pop("request_id", None)
            if case.get("shutdown"):
                break
            speed = max(float(case["speed_mps"]), 1.0)
            altitude = max(float(case["altitude_m"]), 0.0)
            alpha = max(-12.0, min(14.0, float(case["alpha_deg"])))
            beta = max(-15.0, min(15.0, float(case["beta_deg"])))
            rho = isa_density(altitude)
            reynolds = rho * speed * metrics.mean_aerodynamic_chord_m / 1.789e-5

            sweep = "VSPAEROSweep"
            vsp.SetAnalysisInputDefaults(sweep)
            vsp.SetIntAnalysisInput(sweep, "GeomSet", [thick_set])
            vsp.SetIntAnalysisInput(sweep, "ThinGeomSet", [thin_set])
            vsp.SetIntAnalysisInput(sweep, "RefFlag", [1])
            vsp.SetStringAnalysisInput(sweep, "WingID", [wing_id], 0)
            vsp.SetDoubleAnalysisInput(sweep, "MachStart", [speed / 340.0])
            vsp.SetDoubleAnalysisInput(sweep, "MachEnd", [speed / 340.0])
            vsp.SetIntAnalysisInput(sweep, "MachNpts", [1])
            vsp.SetDoubleAnalysisInput(sweep, "Vinf", [speed])
            vsp.SetDoubleAnalysisInput(sweep, "Rho", [rho])
            vsp.SetDoubleAnalysisInput(sweep, "ReCref", [reynolds])
            vsp.SetDoubleAnalysisInput(sweep, "ReCrefEnd", [reynolds])
            vsp.SetIntAnalysisInput(sweep, "ReCrefNpts", [1])
            vsp.SetDoubleAnalysisInput(sweep, "AlphaStart", [alpha])
            vsp.SetDoubleAnalysisInput(sweep, "AlphaEnd", [alpha])
            vsp.SetIntAnalysisInput(sweep, "AlphaNpts", [1])
            vsp.SetDoubleAnalysisInput(sweep, "BetaStart", [beta])
            vsp.SetDoubleAnalysisInput(sweep, "BetaEnd", [beta])
            vsp.SetIntAnalysisInput(sweep, "BetaNpts", [1])
            vsp.SetIntAnalysisInput(sweep, "NCPU", [1])
            vsp.SetIntAnalysisInput(sweep, "WakeNumIter", [10])
            started = time.perf_counter()
            for stale_polar in output.glob("*.polar"):
                stale_polar.unlink()
            vsp.ExecAnalysis(sweep)
            polar_files = sorted(output.glob("*.polar"), key=lambda item: item.stat().st_mtime)
            if not polar_files:
                raise RuntimeError("VSPAERO did not produce a .polar file.")
            polar_path = polar_files[-1]
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    lines = polar_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    header = next((i for i, row in enumerate(lines) if "AoA" in row.split() and "CLtot" in row.split()), None)
                    if header is not None and len(lines) > header + 1:
                        columns = lines[header].split()
                        values = lines[header + 1].split()
                        if len(values) >= len(columns):
                            data = {key: float(value) for key, value in zip(columns, values)}
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError("Could not parse the current VSPAERO single-point polar row.")

            required = ("CLtot", "CDi", "Ew")
            if any(name not in data or not math.isfinite(data[name]) for name in required):
                raise RuntimeError(f"VSPAERO row lacks finite coefficients: {required}")
            response = {
                "ok": True,
                "request_id": request_id,
                "step": int(case["step"]),
                "alpha_deg": alpha,
                "beta_deg": beta,
                "speed_mps": speed,
                "altitude_m": altitude,
                "cl": data["CLtot"],
                "cd": data["CDi"],
                "oswald_efficiency": data["Ew"],
                "oswald_efficiency_definition": "VSPAERO Ew at this integration-step operating point",
                "cs": data.get("CStot", 0.0),
                "cm_x": data.get("CMxtot", 0.0),
                "cm_y": data.get("CMytot", 0.0),
                "cm_z": data.get("CMztot", 0.0),
                "solver_elapsed_s": time.perf_counter() - started,
            }
            temporary_response = output / "response.tmp"
            temporary_response.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")
            temporary_response.replace(response_path)
        except Exception as error:
            response = {
                "ok": False,
                "request_id": locals().get("request_id"),
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            temporary_response = output / "response.tmp"
            temporary_response.write_text(json.dumps(response), encoding="utf-8")
            temporary_response.replace(response_path)

    vsp.ClearVSPModel()


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        error_path = Path(sys.argv[1]) / "worker_error.json"
        error_path.write_text(json.dumps({"error": str(error), "traceback": traceback.format_exc()}), encoding="utf-8")
