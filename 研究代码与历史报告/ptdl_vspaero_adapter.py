"""Clean OpenVSP/VSPAERO adapter for PTDL geometry-level aerodynamic checks."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ptdl_airframe_design import AirframeDesign, design_metrics, write_openvsp_geometry
from ptdl_flight_propulsion_control import isa_density


AIR_DYNAMIC_VISCOSITY = 1.789e-5


@dataclass(frozen=True)
class VSPAeroPolar:
    data: pd.DataFrame
    cl_alpha_per_rad: float
    oswald_efficiency: float
    geometry_path: Path


def run_vspaero_polar(
    design: AirframeDesign,
    output_dir: str | Path,
    speed_mps: float = 36.01,
    altitude_m: float = 3000.0,
    alpha_deg: Sequence[float] = (-3.0, -1.0, 1.0, 3.0, 5.0),
) -> VSPAeroPolar:
    """Generate a VLM polar for one geometry and export it as a CSV.

    VSPAERO supplies lift slope and induced-drag behavior.  Its inviscid VLM
    result is deliberately not used as CD0; the geometry-level parasite drag
    model in ``ptdl_airframe_design`` supplies that component.
    """
    import openvsp as vsp

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    geometry_path = output_path / "airframe.vsp3"
    write_openvsp_geometry(design, geometry_path)
    metrics = design_metrics(design)
    alpha = np.asarray(alpha_deg, dtype=float)
    if alpha.size < 3 or not np.all(np.diff(alpha) > 0.0):
        raise ValueError("alpha_deg must contain at least three increasing angles.")
    original_cwd = Path.cwd()

    try:
        os.chdir(output_path)
        for stale_file in output_path.glob("*.polar"):
            stale_file.unlink()
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(str(geometry_path))
        wing_id = _find_geometry(vsp, "Main_Wing")
        rho = isa_density(altitude_m)
        reynolds = rho * speed_mps * metrics.mean_aerodynamic_chord_m / AIR_DYNAMIC_VISCOSITY

        geometry_analysis = "VSPAEROComputeGeometry"
        vsp.SetAnalysisInputDefaults(geometry_analysis)
        vsp.SetIntAnalysisInput(geometry_analysis, "AnalysisMethod", [vsp.VORTEX_LATTICE])
        vsp.SetVSPAERORefWingID(wing_id)
        vsp.ExecAnalysis(geometry_analysis)

        sweep_analysis = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(sweep_analysis)
        vsp.SetIntAnalysisInput(sweep_analysis, "AnalysisMethod", [vsp.VORTEX_LATTICE])
        vsp.SetIntAnalysisInput(sweep_analysis, "GeomSet", [vsp.SET_ALL])
        vsp.SetIntAnalysisInput(sweep_analysis, "RefFlag", [1])
        vsp.SetStringAnalysisInput(sweep_analysis, "WingID", [wing_id], 0)
        vsp.SetDoubleAnalysisInput(sweep_analysis, "MachStart", [speed_mps / 340.0])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "MachEnd", [speed_mps / 340.0])
        vsp.SetIntAnalysisInput(sweep_analysis, "MachNpts", [1])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "Vinf", [speed_mps])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "Rho", [rho])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "ReCref", [reynolds])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "ReCrefEnd", [reynolds])
        vsp.SetIntAnalysisInput(sweep_analysis, "ReCrefNpts", [1])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "AlphaStart", [float(alpha[0])])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "AlphaEnd", [float(alpha[-1])])
        vsp.SetIntAnalysisInput(sweep_analysis, "AlphaNpts", [len(alpha)])
        vsp.SetIntAnalysisInput(sweep_analysis, "NCPU", [1])
        vsp.SetIntAnalysisInput(sweep_analysis, "WakeNumIter", [5])
        vsp.SetVSPAERORefWingID(wing_id)
        vsp.ExecAnalysis(sweep_analysis)
        # The Python API returns an analysis-summary object in OpenVSP 3.36,
        # while VSPAERO writes the actual polar deterministically to disk.
        polar_files = _wait_for_polar_file(output_path)
        if not polar_files:
            raise RuntimeError("VSPAERO completed without writing a .polar file.")
        return load_vspaero_polar(polar_files[-1], design, geometry_path, output_path / "vspaero_polar.csv")
    except Exception as error:
        (output_path / "vspaero_adapter_error.txt").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
        raise
    finally:
        vsp.ClearVSPModel()
        os.chdir(original_cwd)


def _wait_for_polar_file(output_path: Path, timeout_s: float = 15.0) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    previous_size = -1
    stable_count = 0
    while time.monotonic() < deadline:
        polar_files = sorted(output_path.glob("*.polar"), key=lambda item: item.stat().st_mtime)
        if polar_files:
            size = polar_files[-1].stat().st_size
            if size == previous_size and size > 0:
                stable_count += 1
                if stable_count >= 2:
                    return polar_files
            else:
                previous_size = size
                stable_count = 0
        time.sleep(0.2)
    raise RuntimeError("VSPAERO did not finish writing its .polar file within the timeout.")


def load_vspaero_polar(
    polar_path: str | Path,
    design: AirframeDesign,
    geometry_path: str | Path,
    output_csv: str | Path | None = None,
) -> VSPAeroPolar:
    """Parse a VSPAERO ``.polar`` file independently of the OpenVSP API state."""
    # VSPAERO writes aligned fixed-width headers and data.  A regex whitespace
    # parser treats the padded header as one field in this OpenVSP version.
    raw_polar = pd.read_fwf(polar_path)
    polar = raw_polar.loc[:, ["AoA", "CL", "CDi"]].rename(
        columns={"AoA": "alpha_deg", "CL": "cl", "CDi": "cd_induced"}
    ).dropna()
    if len(polar) < 3:
        raise RuntimeError("VSPAERO returned fewer than three valid polar points.")
    cl_slope = _least_squares_slope(np.deg2rad(polar["alpha_deg"].to_numpy(dtype=float)), polar["cl"].to_numpy(dtype=float))
    cd_slope = _least_squares_slope(polar["cl"].to_numpy(dtype=float) ** 2, polar["cd_induced"].to_numpy(dtype=float))
    oswald = float(1.0 / (np.pi * design.aspect_ratio * cd_slope)) if cd_slope > 1e-8 else float("nan")
    if output_csv is not None:
        polar.to_csv(output_csv, index=False)
    return VSPAeroPolar(polar, cl_slope, oswald, Path(geometry_path))


def _least_squares_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the scalar least-squares slope without invoking LAPACK."""
    dx = x - float(np.mean(x))
    denominator = float(np.sum(dx * dx))
    if denominator <= 1e-15:
        raise RuntimeError("VSPAERO polar has insufficient variation for slope estimation.")
    return float(np.sum(dx * (y - float(np.mean(y)))) / denominator)


def _find_geometry(vsp: object, name: str) -> str:
    geometry_ids = list(vsp.FindGeomsWithName(name))
    if geometry_ids:
        return geometry_ids[0]
    raise RuntimeError(f"Geometry named {name!r} was not found in the VSP model.")


if __name__ == "__main__":
    result = run_vspaero_polar(AirframeDesign(), "ptdl_airframe_results/vspaero_baseline")
    print(result.data.to_string(index=False))
    print(f"CL_alpha={result.cl_alpha_per_rad:.4f}/rad, e={result.oswald_efficiency:.4f}")
