"""Clean OpenVSP/VSPAERO adapter for PTDL geometry-level aerodynamic checks."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from 机体参数化设计与气动阻力 import AirframeDesign, design_metrics, write_openvsp_geometry
from 飞推控闭环物理模型 import isa_density
from VSPAERO原生进程锁 import serialized_native_solver


AIR_DYNAMIC_VISCOSITY = 1.789e-5
VSPAERO_WAKE_ITERATIONS = 30
OSWALD_EFFICIENCY_DEFINITION = (
    "median VSPAERO Ew for polar points with abs(CLiw) >= 0.15"
)


@dataclass(frozen=True)
class VSPAeroPolar:
    data: pd.DataFrame
    cl_alpha_per_rad: float
    oswald_efficiency: float
    geometry_path: Path


@serialized_native_solver
def run_vspaero_polar(
    design: AirframeDesign,
    output_dir: str | Path,
    speed_mps: float = 36.01,
    altitude_m: float = 3000.0,
    alpha_deg: Sequence[float] = (-3.0, -1.0, 1.0, 3.0, 5.0),
    wake_iterations: int = VSPAERO_WAKE_ITERATIONS,
    mesh_level: str = "legacy",
) -> VSPAeroPolar:
    """Generate a VLM polar for one geometry and export it as a CSV.

    VSPAERO supplies lift slope and induced-drag behavior.  Its inviscid VLM
    result is deliberately not used as CD0; the geometry-level parasite drag
    model in ``ptdl_airframe_design`` supplies that component.
    """
    import openvsp as vsp

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    temp_work = tempfile.TemporaryDirectory(prefix="ptdl_vspaero_") if not str(output_path).isascii() else None
    solver_path = Path(temp_work.name) if temp_work is not None else output_path
    geometry_path = solver_path / "airframe.vsp3"
    write_openvsp_geometry(design, geometry_path, mesh_level=mesh_level)
    metrics = design_metrics(design)
    alpha = np.asarray(alpha_deg, dtype=float)
    if alpha.size < 3 or not np.all(np.diff(alpha) > 0.0):
        raise ValueError("alpha_deg must contain at least three increasing angles.")
    if wake_iterations < 1:
        raise ValueError("wake_iterations must be a positive integer.")
    original_cwd = Path.cwd()

    try:
        os.chdir(solver_path)
        for stale_file in solver_path.glob("*.polar"):
            stale_file.unlink()
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(str(geometry_path))
        wing_id = _find_geometry(vsp, "Main_Wing")
        thick_set, thin_set = configure_vspaero_geometry_sets(vsp)
        rho = isa_density(altitude_m)
        reynolds = rho * speed_mps * metrics.mean_aerodynamic_chord_m / AIR_DYNAMIC_VISCOSITY

        geometry_analysis = "VSPAEROComputeGeometry"
        vsp.SetAnalysisInputDefaults(geometry_analysis)
        # OpenVSP 3.53 selects VLM through ThinGeomSet; older releases exposed
        # an AnalysisMethod enum that is no longer present in the Python API.
        vsp.SetIntAnalysisInput(geometry_analysis, "GeomSet", [thick_set])
        vsp.SetIntAnalysisInput(geometry_analysis, "ThinGeomSet", [thin_set])
        vsp.SetVSPAERORefWingID(wing_id)
        vsp.ExecAnalysis(geometry_analysis)

        sweep_analysis = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(sweep_analysis)
        vsp.SetIntAnalysisInput(sweep_analysis, "GeomSet", [thick_set])
        vsp.SetIntAnalysisInput(sweep_analysis, "ThinGeomSet", [thin_set])
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
        vsp.SetIntAnalysisInput(sweep_analysis, "WakeNumIter", [int(wake_iterations)])
        vsp.SetVSPAERORefWingID(wing_id)
        vsp.ExecAnalysis(sweep_analysis)
        # The Python API returns an analysis-summary object in OpenVSP 3.36,
        # while VSPAERO writes the actual polar deterministically to disk.
        polar_files = _wait_for_polar_file(solver_path)
        if not polar_files:
            raise RuntimeError("VSPAERO completed without writing a .polar file.")
        result = load_vspaero_polar(
            polar_files[-1], design, geometry_path, solver_path / "vspaero_polar.csv"
        )
        import json
        (solver_path / "solver_settings.json").write_text(json.dumps({
            "mesh_level": mesh_level, "wake_iterations": wake_iterations,
            "speed_mps": speed_mps, "altitude_m": altitude_m,
            "alpha_deg": alpha.tolist(), "efficiency_definition": OSWALD_EFFICIENCY_DEFINITION,
        }, indent=2), encoding="utf-8")
        if temp_work is not None:
            _copy_solver_artifacts(solver_path, output_path)
            result = VSPAeroPolar(
                result.data,
                result.cl_alpha_per_rad,
                result.oswald_efficiency,
                output_path / geometry_path.name,
            )
        return result
    except Exception as error:
        (output_path / "vspaero_adapter_error.txt").write_text(
            "".join(traceback.format_exception(error)), encoding="utf-8"
        )
        if temp_work is not None:
            _copy_solver_artifacts(solver_path, output_path)
        raise
    finally:
        vsp.ClearVSPModel()
        os.chdir(original_cwd)
        if temp_work is not None:
            temp_work.cleanup()


def _copy_solver_artifacts(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, destination / item.name)


def configure_vspaero_geometry_sets(vsp: object) -> tuple[int, int]:
    """Separate lifting surfaces from the fuselage for mixed VLM/panel analysis."""
    thin_set = int(vsp.SET_FIRST_USER)
    thick_set = thin_set + 1
    vsp.SetSetName(thin_set, "PTDL_VSPAERO_LiftingSurfaces")
    vsp.SetSetName(thick_set, "PTDL_VSPAERO_Bodies")

    thin_names = {"Main_Wing", "Horizontal_Tail", "Vertical_Tail"}
    required_names = thin_names | {"Fuselage"}
    found_names: set[str] = set()
    for geometry_id in vsp.FindGeoms():
        name = str(vsp.GetGeomName(geometry_id))
        found_names.add(name)
        vsp.SetSetFlag(geometry_id, thin_set, name in thin_names)
        vsp.SetSetFlag(geometry_id, thick_set, name == "Fuselage")
    missing_names = required_names - found_names
    if missing_names:
        raise RuntimeError(
            "OpenVSP geometry is missing VSPAERO components: "
            + ", ".join(sorted(missing_names))
        )
    return thick_set, thin_set


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
    # Header width and lift-column names differ between VSPAERO releases.
    lines = Path(polar_path).read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "AoA" in line.split() and ("CLtot" in line.split() or "CL" in line.split())
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError("Could not identify AoA and lift columns in the VSPAERO polar header.")
    columns = lines[header_index].split()
    lift_column = "CLtot" if "CLtot" in columns else "CL"
    required_columns = ("CDi", "E", "CLiw", "CDiw", "Ew")
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise RuntimeError(
            "VSPAERO polar is missing required force/wake columns: "
            + ", ".join(missing_columns)
        )
    raw_polar = pd.read_csv(
        polar_path,
        sep=r"\s+",
        skiprows=header_index + 1,
        names=columns,
        header=None,
        engine="python",
    )
    polar = raw_polar.loc[
        :, ["AoA", lift_column, "CDi", "E", "CLiw", "CDiw", "Ew"]
    ].rename(
        columns={
            "AoA": "alpha_deg",
            lift_column: "cl",
            "CDi": "cd_induced",
            "E": "e_surface",
            "CLiw": "cl_wake",
            "CDiw": "cd_induced_wake",
            "Ew": "e_wake",
        }
    ).dropna()
    polar = polar.apply(pd.to_numeric, errors="coerce").dropna()
    polar = polar.loc[np.isfinite(polar.to_numpy(dtype=float)).all(axis=1)]
    if len(polar) < 3:
        raise RuntimeError(
            "VSPAERO polar has fewer than three finite points "
            f"({len(polar)} valid of {len(raw_polar)} data rows); "
            f"inspect solver convergence and geometry in {Path(polar_path)}."
        )
    cl_slope = _least_squares_slope(np.deg2rad(polar["alpha_deg"].to_numpy(dtype=float)), polar["cl"].to_numpy(dtype=float))
    valid_wake = polar.loc[
        (polar["cl_wake"].abs() >= 0.15) & (polar["e_wake"] > 0.0),
        "e_wake",
    ]
    if valid_wake.empty:
        raise RuntimeError(
            "VSPAERO polar has no valid wake-efficiency points with |CLiw| >= 0.15."
        )
    oswald = float(valid_wake.median())
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
