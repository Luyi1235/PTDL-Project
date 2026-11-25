"""Recompute the original 23 VSPAERO designs with wake Ew, extend to 40/100, and refit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (  # noqa: E402
    FEATURE_LABELS,
    OSWALD_EFFICIENCY_DEFINITION,
    fit_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import (  # noqa: E402
    AirframeDesign,
    geometry_constraints,
)


DATA_DIR = ROOT / "气动代理模型数据"
SEED_DESIGN = ROOT / "验证测试" / "预估额定值基准候选.json"
WORKER = ROOT / "运行脚本" / "运行_VSPAERO单构型评估.py"
DEFAULT_PYTHON = ROOT.parent.parent / "tools" / "Python311" / "python.exe"
DEFAULT_OUTPUT = ROOT.parent / "VSPAERO_DOE_Ew_physical_20261006"
VERSION = "Ew_物理选点_网格修复_20261006"
FILES = {
    "doe40": f"VSPAERO_DOE基准样本_40点_{VERSION}.csv",
    "doe100": f"VSPAERO_DOE扩充样本_100点_{VERSION}.csv",
    "model40": f"气动代理模型_基准DOE40点_{VERSION}.json",
    "model100": f"气动代理模型_扩充DOE100点_{VERSION}.json",
    "metadata": f"VSPAERO_DOE40_100实算元数据_{VERSION}.json",
    "validation": f"VSPAERO_DOE代理模型留一验证_{VERSION}.json",
}
ROW_COLUMNS = (
    "sample",
    *FEATURE_LABELS,
    "propeller_diameter_m",
    "payload_mass_kg",
    "hybrid_system_mass_kg",
    "vspaero_cl_alpha_per_rad",
    "vspaero_oswald_efficiency",
    "oswald_efficiency_definition",
    "polar_points",
    "geometry_file",
    "selection_role",
    "selection_basis",
)
TARGET_COLUMNS = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")


def _extract_worker_result(stdout: str) -> dict:
    for match in reversed(list(re.finditer(r'"cl_alpha_per_rad"\s*:', stdout))):
        start = stdout.rfind("{", 0, match.start())
        if start < 0:
            continue
        try:
            result, _ = json.JSONDecoder().raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and {
            "cl_alpha_per_rad",
            "oswald_efficiency",
            "oswald_efficiency_definition",
            "polar_csv",
        }.issubset(result):
            return result
    raise RuntimeError(f"VSPAERO worker returned no result JSON: {stdout[-1500:]}")


def _record_from_result(
    sample: str,
    design: dict,
    result: dict,
    case_dir: Path,
    worker_log: str,
    selection_role: str,
    selection_basis: str,
    oswald_label_bounds: tuple[float, float] = (0.45, 1.4),
) -> dict:
    if result["oswald_efficiency_definition"] != OSWALD_EFFICIENCY_DEFINITION:
        raise RuntimeError("VSPAERO worker returned an unexpected Oswald-efficiency definition.")
    if "PGFace Invalid" in worker_log or "Faces have errors" in worker_log:
        raise RuntimeError("Geometry mesh reported PGFace/face errors; sample rejected.")
    polar_path = Path(result["polar_csv"])
    polar = pd.read_csv(polar_path)
    required = {"e_surface", "cl_wake", "cd_induced_wake", "e_wake"}
    if not required.issubset(polar.columns) or len(polar) < 3:
        raise RuntimeError("VSPAERO polar is incomplete or lacks separate E/Ew columns.")
    eligible = polar.loc[
        (polar["cl_wake"].abs() >= 0.15) & (polar["e_wake"] > 0.0),
        "e_wake",
    ]
    if eligible.empty:
        raise RuntimeError("VSPAERO polar has no valid wake-efficiency points.")
    ew = float(eligible.median())
    if abs(ew - float(result["oswald_efficiency"])) > 1e-8:
        raise RuntimeError("Worker Ew scalar does not match the saved polar's eligible Ew median.")
    targets = np.asarray(
        [result["cl_alpha_per_rad"], ew],
        dtype=float,
    )
    if not np.isfinite(targets).all() or not (3.5 <= targets[0] <= 8.5):
        raise RuntimeError(f"Invalid aerodynamic labels: {targets.tolist()}")
    if not (oswald_label_bounds[0] <= targets[1] <= oswald_label_bounds[1]):
        raise RuntimeError(f"Wake-based Ew outside permitted sample envelope: {targets[1]:.6f}")
    return {
        "sample": sample,
        **design,
        "vspaero_cl_alpha_per_rad": float(targets[0]),
        "vspaero_oswald_efficiency": ew,
        "oswald_efficiency_definition": OSWALD_EFFICIENCY_DEFINITION,
        "polar_points": int(len(polar)),
        "geometry_file": str(case_dir / "airframe.vsp3"),
        "selection_role": selection_role,
        "selection_basis": selection_basis,
    }


def _run_case(
    python_exe: Path,
    design: dict,
    sample: str,
    case_dir: Path,
    timeout_s: float,
    selection_role: str,
    selection_basis: str,
    oswald_label_bounds: tuple[float, float] = (0.45, 1.4),
    wake_iterations: int = 30,
    mesh_level: str = "legacy",
) -> dict:
    case_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "design": design,
        "output_dir": str(case_dir),
        "speed_mps": 35.99708,
        "altitude_m": 3000.0,
        "wake_iterations": wake_iterations,
        "mesh_level": mesh_level,
    }
    started = time.perf_counter()
    try:
        # OpenVSP may create scratch files during import, before solver chdir.
        with tempfile.TemporaryDirectory(prefix="ptdl_vsp_worker_") as startup_dir:
            process = subprocess.run(
                [str(python_exe), str(WORKER)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                cwd=startup_dir,
                check=False,
            )
        log = process.stdout + "\n--- STDERR ---\n" + process.stderr
        (case_dir / "worker.log").write_text(log, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(
                f"worker exit {process.returncode}: {process.stderr[-1200:]}"
            )
        result = _extract_worker_result(process.stdout)
        row = _record_from_result(
            sample,
            design,
            result,
            case_dir,
            log,
            selection_role,
            selection_basis,
            oswald_label_bounds,
        )
        row["solver_wall_time_s"] = time.perf_counter() - started
        return row
    except Exception as error:
        (case_dir / "case_error.txt").write_text(str(error), encoding="utf-8")
        raise


def _load_saved_case(
    sample: str,
    design: dict,
    case_dir: Path,
    selection_role: str,
    selection_basis: str,
    oswald_label_bounds: tuple[float, float] = (0.45, 1.4),
) -> dict:
    polar_path = case_dir / "vspaero_polar.csv"
    log_path = case_dir / "worker.log"
    if not polar_path.is_file() or not log_path.is_file():
        raise RuntimeError("Existing case has no complete polar and solver log.")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "PGFace Invalid" in log or "Faces have errors" in log:
        raise RuntimeError("Existing case has geometry-mesh warnings.")
    polar = pd.read_csv(polar_path)
    required = {"alpha_deg", "cl", "cl_wake", "e_wake"}
    if not required.issubset(polar.columns) or len(polar) < 3:
        raise RuntimeError("Existing case polar is incomplete.")
    alpha = np.deg2rad(polar["alpha_deg"].to_numpy(float))
    cl = polar["cl"].to_numpy(float)
    cl_alpha = float(np.polyfit(alpha, cl, 1)[0])
    eligible = polar.loc[
        (polar["cl_wake"].abs() >= 0.15) & (polar["e_wake"] > 0.0),
        "e_wake",
    ]
    if eligible.empty:
        raise RuntimeError("Existing case has no valid Ew operating points.")
    ew = float(eligible.median())
    if not np.isfinite([cl_alpha, ew]).all() or not (3.5 <= cl_alpha <= 8.5):
        raise RuntimeError("Existing case has invalid aerodynamic labels.")
    if not (oswald_label_bounds[0] <= ew <= oswald_label_bounds[1]):
        raise RuntimeError(f"Existing case Ew is outside the accepted range: {ew:.6f}.")
    return {
        "sample": sample,
        **design,
        "vspaero_cl_alpha_per_rad": cl_alpha,
        "vspaero_oswald_efficiency": ew,
        "oswald_efficiency_definition": OSWALD_EFFICIENCY_DEFINITION,
        "polar_points": int(len(polar)),
        "geometry_file": str(case_dir / "airframe.vsp3"),
        "selection_role": selection_role,
        "selection_basis": selection_basis,
        "solver_wall_time_s": float("nan"),
    }


def _feasible_pool(
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    rng: np.random.Generator,
    non_geometry: dict[str, float],
) -> np.ndarray:
    values: list[np.ndarray] = []
    while len(values) < count:
        batch = rng.uniform(lower, upper, size=(max(5000, count), len(FEATURE_LABELS)))
        valid: list[np.ndarray] = []
        for row in batch:
            design = AirframeDesign(
                **non_geometry,
                **dict(zip(FEATURE_LABELS, row.tolist())),
            )
            if all(value <= 1e-9 for value in geometry_constraints(design).values()):
                valid.append(row)
        valid_array = np.asarray(valid, dtype=float).reshape((-1, len(FEATURE_LABELS)))
        values.extend(valid_array[: count - len(values)])
    return np.asarray(values, dtype=float)


def _fit_and_save(frame: pd.DataFrame, model_path: Path) -> None:
    designs = [
        AirframeDesign(**{label: float(row[label]) for label in FEATURE_LABELS})
        for _, row in frame.iterrows()
    ]
    model = fit_vsp_surrogate(
        designs,
        frame[TARGET_COLUMNS[0]].to_numpy(float),
        frame[TARGET_COLUMNS[1]].to_numpy(float),
    )
    model.to_json(model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload["oswald_efficiency_definition"] = OSWALD_EFFICIENCY_DEFINITION
    payload["dataset_version"] = VERSION
    payload["mesh_policy"] = (
        "Main_Wing/Horizontal_Tail/Vertical_Tail thin set; Fuselage thick set; "
        "samples with PGFace Invalid or Faces have errors rejected"
    )
    model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _loo_metrics(frame: pd.DataFrame) -> dict:
    actual = frame.loc[:, TARGET_COLUMNS].to_numpy(float)
    predictions = np.empty_like(actual)
    for index in range(len(frame)):
        training = frame.drop(frame.index[index])
        designs = [
            AirframeDesign(**{label: float(row[label]) for label in FEATURE_LABELS})
            for _, row in training.iterrows()
        ]
        model = fit_vsp_surrogate(
            designs,
            training[TARGET_COLUMNS[0]].to_numpy(float),
            training[TARGET_COLUMNS[1]].to_numpy(float),
        )
        design = AirframeDesign(
            **{label: float(frame.iloc[index][label]) for label in FEATURE_LABELS}
        )
        predictions[index] = model.predict(design)
    metrics: dict[str, dict[str, float]] = {}
    for index, label in enumerate(TARGET_COLUMNS):
        residual = predictions[:, index] - actual[:, index]
        denominator = float(np.sum((actual[:, index] - actual[:, index].mean()) ** 2))
        metrics[label] = {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - np.sum(residual**2) / denominator)
            if denominator > 1e-15
            else float("nan"),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsp-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-size", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this versioned run from its saved per-case polar files.",
    )
    args = parser.parse_args()
    if not args.vsp_python.is_file():
        raise FileNotFoundError(f"OpenVSP Python runtime not found: {args.vsp_python}")

    output_dir = args.output_dir.resolve()
    data_paths = {key: DATA_DIR / value for key, value in FILES.items()}
    protected = [output_dir, *data_paths.values()]
    existing = [path for path in data_paths.values() if path.exists()]
    if output_dir.exists() and not args.resume:
        existing.append(output_dir)
    if args.resume and not output_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume; run directory does not exist: {output_dir}")
    if args.resume:
        allowed_partial = {data_paths["doe40"], data_paths["model40"]}
        existing = [path for path in existing if path not in allowed_partial]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing DOE/model/evidence outputs: "
            + ", ".join(str(path) for path in existing)
        )
    seed_payload = json.loads(SEED_DESIGN.read_text(encoding="utf-8"))
    nominal_design = {
        name: float(seed_payload["design"][name])
        for name in (
            *FEATURE_LABELS,
            "propeller_diameter_m",
            "payload_mass_kg",
            "hybrid_system_mass_kg",
        )
    }
    ultra_reference = {
        **nominal_design,
        "wing_area_m2": 26.0,
        "aspect_ratio": 24.02,
        "taper_ratio": 0.45,
        "sweep_deg": 1.0,
        "tip_twist_deg": 0.0,
        "fuselage_length_m": 8.38,
        "fuselage_diameter_m": 0.75,
    }
    bounds = np.asarray(
        [
            [22.0, 40.0],
            [18.0, 31.0],
            [0.30, 0.70],
            [0.0, 8.0],
            [-3.0, 1.5],
            [7.5, 10.0],
            [0.65, 0.95],
        ],
        dtype=float,
    )
    lower, upper = bounds[:, 0], bounds[:, 1]
    non_geometry = {
        name: nominal_design[name]
        for name in (
            "propeller_diameter_m",
            "payload_mass_kg",
            "hybrid_system_mass_kg",
        )
    }
    anchors: list[tuple[str, dict, str, str]] = [
        (
            "baseline_ultra_reference",
            ultra_reference,
            "reference",
            "ULTRA量级参考：26 m²翼面积、AR 24.02对应约24.99 m翼展；质量/载荷采用本项目H=6混动基准。",
        ),
        (
            "h6_nominal_candidate",
            nominal_design,
            "nominal_candidate",
            "采用既有可行H=6联合优化候选作为PTDL混动构型中心。",
        ),
    ]
    nominal_values = np.asarray(
        [nominal_design[name] for name in FEATURE_LABELS], dtype=float
    )
    perturbations = (
        ("wing_area_low", 0, -1.0),
        ("wing_area_high", 0, 1.0),
        ("aspect_ratio_low", 1, -1.0),
        ("aspect_ratio_high", 1, 1.0),
        ("taper_low", 2, -1.0),
        ("taper_high", 2, 1.0),
        ("sweep_low", 3, -1.0),
        ("sweep_high", 3, 1.0),
        ("twist_low", 4, -1.0),
        ("twist_high", 4, 1.0),
        ("fuselage_length_low", 5, -1.0),
        ("fuselage_length_high", 5, 1.0),
        ("fuselage_diameter_low", 6, -1.0),
        ("fuselage_diameter_high", 6, 1.0),
    )
    local_scales = np.asarray([1.2, 1.5, 0.06, 2.0, 0.75, 0.35, 0.035])
    for name, index, direction in perturbations:
        values = nominal_values.copy()
        values[index] = np.clip(
            values[index] + direction * local_scales[index],
            lower[index],
            upper[index],
        )
        candidate = {
            **nominal_design,
            **dict(zip(FEATURE_LABELS, values.tolist())),
        }
        if not all(
            value <= 1e-9 for value in geometry_constraints(AirframeDesign(**candidate)).values()
        ):
            continue
        anchors.append(
            (
                f"local_{name}",
                candidate,
                "local_sensitivity",
                f"围绕H=6名义构型仅改变{name}对应的一个几何变量，检查局部气动响应。",
            )
        )

    output_dir.mkdir(parents=True, exist_ok=args.resume)
    status_path = output_dir / "rebuild_progress.json"
    log_path = output_dir / "rebuild_attempts.csv"
    successful_rows: list[dict] = []
    failed_rows: list[dict] = []
    if args.resume and log_path.is_file() and log_path.stat().st_size > 4:
        try:
            failed_rows = pd.read_csv(log_path).to_dict(orient="records")
        except pd.errors.EmptyDataError:
            failed_rows = []
    started = time.perf_counter()

    def save_status(current: str) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "status": current,
                    "reference_and_local_anchors_computed": len(
                        [row for row in successful_rows if row["selection_role"] != "space_filling"]
                    ),
                    "space_filling_designs_computed": len(
                        [row for row in successful_rows if row["selection_role"] == "space_filling"]
                    ),
                    "total_valid_designs": len(successful_rows),
                    "target_designs": 100,
                    "failed_attempts": len(failed_rows),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_s": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(failed_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    save_status("computing_reference_and_local_anchors")
    for index, (sample, design, role, basis) in enumerate(anchors, start=1):
        case_dir = output_dir / sample
        try:
            if case_dir.exists() and args.resume:
                computed = _load_saved_case(sample, design, case_dir, role, basis)
            else:
                computed = _run_case(
                    args.vsp_python,
                    design,
                    sample,
                    case_dir,
                    args.timeout_s,
                    role,
                    basis,
                )
        except Exception as error:
            failed_rows.append({"sample": sample, "error": str(error)})
            save_status("failed_computing_required_anchor")
            raise RuntimeError(f"Required representative configuration {sample} failed.") from error
        successful_rows.append(computed)
        save_status(f"anchor_{index}_of_{len(anchors)}_complete")
        print(
            f"[anchor {index:02d}/{len(anchors)}] {sample}: "
            f"CLalpha={computed[TARGET_COLUMNS[0]]:.5f}, "
            f"Ew={computed[TARGET_COLUMNS[1]]:.5f}",
            flush=True,
        )

    rng = np.random.default_rng(args.seed)
    pool = _feasible_pool(lower, upper, args.pool_size, rng, non_geometry)
    scale = np.maximum(upper - lower, 1e-12)
    normalized_pool = (pool - lower) / scale
    anchor_features = np.asarray(
        [
            [float(row[label]) for label in FEATURE_LABELS]
            for row in successful_rows
        ],
        dtype=float,
    )
    normalized_known = (anchor_features - lower) / scale
    nearest = np.linalg.norm(
        normalized_pool[:, None, :] - normalized_known[None, :, :], axis=2
    ).min(axis=1)

    minimum_40 = 40
    total_target = 100
    attempts = 0
    replaying_legacy_names = True
    while len(successful_rows) < total_target:
        selected = int(np.argmax(nearest))
        feature_values = pool[selected].copy()
        normalized = normalized_pool[selected].copy()
        nearest[selected] = -1.0
        design = dict(non_geometry)
        design.update(dict(zip(FEATURE_LABELS, feature_values.tolist())))
        sample = (
            f"spacefill_{len(successful_rows) + 1:03d}"
            if replaying_legacy_names
            else f"spacefill_attempt_{attempts:03d}"
        )
        case_dir = output_dir / sample
        attempts += 1
        try:
            basis = (
                "在优化器设计变量边界内，经翼展、失速、燃油质量/容积和机身长细比约束筛选后进行归一化maximin空间填充。"
            )
            if case_dir.exists() and args.resume:
                computed = _load_saved_case(
                    sample, design, case_dir, "space_filling", basis
                )
            else:
                computed = _run_case(
                    args.vsp_python,
                    design,
                    sample,
                    case_dir,
                    args.timeout_s,
                    "space_filling",
                    basis,
                )
            successful_rows.append(computed)
            role_index = len(successful_rows)
            print(
                f"[DOE {role_index:03d}/{total_target}] {sample}: "
                f"CLalpha={computed[TARGET_COLUMNS[0]]:.5f}, "
                f"Ew={computed[TARGET_COLUMNS[1]]:.5f}",
                flush=True,
            )
        except Exception as error:
            replaying_legacy_names = False
            failed_rows.append(
                {
                    "attempt": attempts,
                    "sample": sample,
                    "status": "rejected",
                    "error": str(error),
                }
            )
            safe_error = str(error).encode("ascii", "backslashreplace").decode("ascii")
            print(f"[reject {attempts}] {sample}: {safe_error}", flush=True)
        if not failed_rows or failed_rows[-1].get("sample") != sample:
            normalized_known = np.vstack((normalized_known, normalized))
        distances = np.linalg.norm(normalized_pool - normalized[None, :], axis=1)
        nearest = np.minimum(nearest, distances)
        if (
            len(successful_rows) == minimum_40
            and not data_paths["model40"].exists()
            and not data_paths["doe40"].exists()
        ):
            base40 = pd.DataFrame(successful_rows).loc[:, ROW_COLUMNS]
            _fit_and_save(base40, data_paths["model40"])
            base40.to_csv(data_paths["doe40"], index=False, encoding="utf-8-sig")
        save_status(f"computing_doe_{len(successful_rows)}_of_{total_target}")
        if attempts >= args.pool_size:
            raise RuntimeError(
                f"Stopped at {len(successful_rows)}/{total_target} valid labels after {attempts} attempts."
            )

    full = pd.DataFrame(successful_rows).loc[:, ROW_COLUMNS]
    base40 = full.iloc[:minimum_40].copy()
    if len(base40) != 40 or len(full) != 100:
        raise RuntimeError("Nested DOE sizes are inconsistent.")
    if not data_paths["model40"].is_file() or not data_paths["doe40"].is_file():
        raise RuntimeError("The 40-point model and dataset checkpoint are missing.")
    _fit_and_save(full, data_paths["model100"])
    full.to_csv(data_paths["doe100"], index=False, encoding="utf-8-sig")
    validation = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_definition": OSWALD_EFFICIENCY_DEFINITION,
        "model_family": "quadratic-basis ridge surrogate used by the PTDL optimizer",
        "datasets": {
            "40": {"rows": 40, "loo": _loo_metrics(base40)},
            "100": {"rows": 100, "loo": _loo_metrics(full)},
        },
        "mesh_warning_policy": "Reject any sample with PGFace Invalid or Faces have errors.",
        "caveat": "VSPAERO VLM computational labels; not wind-tunnel or flight validation.",
    }
    data_paths["validation"].write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_version": VERSION,
        "representative_anchor_count": len(anchors),
        "additional_valid_vspaero_solves": total_target - len(anchors),
        "failed_or_rejected_attempts": len(failed_rows),
        "attempts_for_new_designs": attempts,
        "target_definition": OSWALD_EFFICIENCY_DEFINITION,
        "geometry_mesh_policy": (
            "Main_Wing/Horizontal_Tail/Vertical_Tail thin set; "
            "Fuselage thick set; reject mesh face warnings"
        ),
        "wake_iterations": 30,
        "runtime_s": time.perf_counter() - started,
        "files": {key: str(path) for key, path in data_paths.items()},
        "design_selection_policy": (
            "ULTRA-span-matched geometric reference plus feasible H=6 nominal hybrid design, "
            "one-factor local sensitivity points, and feasible maximin space-filling points."
        ),
        "design_bounds": {
            label: [float(low), float(high)]
            for label, low, high in zip(FEATURE_LABELS, lower, upper)
        },
        "fixed_powertrain_context": non_geometry,
    }
    data_paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_status("complete")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
