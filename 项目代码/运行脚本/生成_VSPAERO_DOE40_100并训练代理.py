"""Extend real VSPAERO samples to nested 40/100-point DOE sets and refit models."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (
    FEATURE_LABELS,
    OSWALD_EFFICIENCY_DEFINITION,
    fit_vsp_surrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign


SOURCE_DATA = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_23点.csv"
DATA_DIR = ROOT / "气动代理模型数据"
SEED_DESIGN = ROOT / "验证测试" / "预估额定值基准候选.json"
WORKER = ROOT / "运行脚本" / "运行_VSPAERO单构型评估.py"
DEFAULT_VSP_PYTHON = ROOT.parent.parent / "tools" / "Python311" / "python.exe"
DEFAULT_OUTPUT = ROOT / "运行结果" / "VSPAERO_DOE扩展实算_Ew_20261006"
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
)


def worker_payload(stdout: str) -> dict:
    for match in reversed(list(re.finditer(r'"cl_alpha_per_rad"\s*:', stdout))):
        start = stdout.rfind("{", 0, match.start())
        if start < 0:
            continue
        try:
            payload, _ = json.JSONDecoder().raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and {
            "cl_alpha_per_rad",
            "oswald_efficiency",
        }.issubset(payload):
            return payload
    raise RuntimeError(f"Worker did not return a polar result: {stdout[-1200:]}")


def feasible_pool(
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool: list[np.ndarray] = []
    while len(pool) < count:
        batch = rng.uniform(lower, upper, size=(max(5000, count), len(FEATURE_LABELS)))
        span = np.sqrt(batch[:, 0] * batch[:, 1])
        fineness = batch[:, 5] / batch[:, 6]
        valid = batch[(span <= 32.0) & (fineness >= 8.0)]
        pool.extend(valid[: count - len(pool)])
    return np.asarray(pool, dtype=float)


def save_models(frame: pd.DataFrame) -> None:
    designs = [
        AirframeDesign(**{label: float(row[label]) for label in FEATURE_LABELS})
        for _, row in frame.iterrows()
    ]
    model = fit_vsp_surrogate(
        designs,
        frame["vspaero_cl_alpha_per_rad"].to_numpy(dtype=float),
        frame["vspaero_oswald_efficiency"].to_numpy(dtype=float),
    )
    point_count = len(frame)
    if point_count == 40:
        model_path = DATA_DIR / "气动代理模型_基准DOE40点_Ew.json"
    elif point_count == 100:
        model_path = DATA_DIR / "气动代理模型_扩充DOE100点_Ew.json"
    else:
        raise ValueError(f"Unexpected DOE size {point_count}.")
    model.to_json(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsp-python", type=Path, default=DEFAULT_VSP_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    if not args.vsp_python.is_file():
        raise FileNotFoundError(f"OpenVSP Python runtime not found: {args.vsp_python}")
    if args.pool_size < 1000:
        raise ValueError("pool-size must be at least 1000.")

    output_dir = args.output_dir.resolve()
    protected_outputs = (
        output_dir,
        DATA_DIR / "VSPAERO_DOE基准样本_40点_Ew.csv",
        DATA_DIR / "VSPAERO_DOE扩充样本_100点_Ew.csv",
        DATA_DIR / "气动代理模型_基准DOE40点_Ew.json",
        DATA_DIR / "气动代理模型_扩充DOE100点_Ew.json",
        DATA_DIR / "VSPAERO_DOE40_100实算元数据_Ew.json",
    )
    existing_outputs = [path for path in protected_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing Ew outputs: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    source = pd.read_csv(SOURCE_DATA)
    if "oswald_efficiency_definition" not in source.columns:
        raise ValueError(
            "The 23-point source DOE predates the wake-based Ew target. "
            "Recompute those labels before combining them with new Ew samples."
        )
    if set(source["oswald_efficiency_definition"].dropna()) != {
        OSWALD_EFFICIENCY_DEFINITION
    }:
        raise ValueError("The source DOE does not use the selected VSPAERO Ew definition.")
    source = source.loc[:, [column for column in ROW_COLUMNS if column in source.columns]]
    if len(source) != 23 or source.loc[:, FEATURE_LABELS].isna().any().any():
        raise ValueError("The source DOE must contain 23 complete VSPAERO rows.")
    if not np.isfinite(
        source[["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]].to_numpy(float)
    ).all():
        raise ValueError("The source DOE contains invalid VSPAERO target values.")

    seed_payload = json.loads(SEED_DESIGN.read_text(encoding="utf-8"))
    base_design = dict(seed_payload["design"])
    known = source.loc[:, FEATURE_LABELS].to_numpy(dtype=float)
    lower = known.min(axis=0)
    upper = known.max(axis=0)
    for index, name in enumerate(FEATURE_LABELS):
        value = float(base_design[name])
        lower[index] = min(lower[index], value)
        upper[index] = max(upper[index], value)
    feature_scale = np.maximum(upper - lower, 1e-12)
    rng = np.random.default_rng(args.seed)
    pool = feasible_pool(lower, upper, args.pool_size, rng)
    normalized_pool = (pool - lower) / feature_scale
    normalized_known = (known - lower) / feature_scale
    seed_vector = np.asarray([base_design[name] for name in FEATURE_LABELS], dtype=float)
    normalized_seed = (seed_vector - lower) / feature_scale
    if (
        np.sqrt(seed_vector[0] * seed_vector[1]) <= 32.0
        and seed_vector[5] / seed_vector[6] >= 8.0
    ):
        normalized_known = np.vstack((normalized_known, normalized_seed))

    nearest = np.linalg.norm(
        normalized_pool[:, None, :] - normalized_known[None, :, :], axis=2
    ).min(axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "DOE实算进度.json"
    log_path = output_dir / "DOE实算尝试记录.csv"
    success_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    required_new_points = 77
    successful = 0
    attempts = 0

    def persist_progress() -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "source_points": 23,
                    "successful_new_points": successful,
                    "required_new_points": required_new_points,
                    "attempts": attempts,
                    "target_points": 100,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(success_rows + failed_rows).to_csv(log_path, index=False)
        if successful == 17:
            base40 = pd.concat(
                [source, pd.DataFrame(success_rows[:17])], ignore_index=True
            ).loc[:, ROW_COLUMNS]
            base40.to_csv(DATA_DIR / "VSPAERO_DOE基准样本_40点_Ew.csv", index=False)
            save_models(base40)
        if successful == required_new_points:
            full100 = pd.concat(
                [source, pd.DataFrame(success_rows)], ignore_index=True
            ).loc[:, ROW_COLUMNS]
            full100.to_csv(DATA_DIR / "VSPAERO_DOE扩充样本_100点_Ew.csv", index=False)
            save_models(full100)

    while successful < required_new_points:
        index = int(np.argmax(nearest))
        values = pool[index].copy()
        normalized = normalized_pool[index].copy()
        nearest[index] = -1.0
        design = dict(base_design)
        design.update(dict(zip(FEATURE_LABELS, values.tolist())))
        sample_number = 23 + successful
        sample_name = f"doe_{sample_number:02d}"
        case_dir = output_dir / sample_name
        case_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "design": design,
            "output_dir": str(case_dir),
            "speed_mps": 35.99708,
            "altitude_m": 3000.0,
        }
        attempts += 1
        process = None
        error = ""
        try:
            command = [str(args.vsp_python), str(WORKER)]
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8"
            ) as stderr_file:
                worker = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    cwd=str(ROOT),
                )
                try:
                    worker.communicate(
                        input=json.dumps(request),
                        timeout=args.timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    if sys.platform == "win32":
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(worker.pid)],
                                capture_output=True,
                                text=True,
                                timeout=15,
                                check=False,
                            )
                        except (OSError, subprocess.SubprocessError):
                            pass
                    if worker.poll() is None:
                        worker.kill()
                    try:
                        worker.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                    stdout_file.seek(0)
                    stderr_file.seek(0)
                    process = subprocess.CompletedProcess(
                        command,
                        worker.returncode,
                        stdout_file.read(),
                        stderr_file.read(),
                    )
                    raise TimeoutError(
                        f"VSPAERO worker process tree exceeded {args.timeout_s:.0f} seconds."
                    ) from exc
                stdout_file.seek(0)
                stderr_file.seek(0)
                process = subprocess.CompletedProcess(
                    command,
                    worker.returncode,
                    stdout_file.read(),
                    stderr_file.read(),
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"worker exit {process.returncode}: "
                    f"{process.stdout[-1000:]} {process.stderr[-1000:]}"
                )
            result = worker_payload(process.stdout)
            if result.get("oswald_efficiency_definition") != OSWALD_EFFICIENCY_DEFINITION:
                raise RuntimeError("The VSPAERO worker returned a different Oswald definition.")
            targets = np.asarray(
                [result["cl_alpha_per_rad"], result["oswald_efficiency"]],
                dtype=float,
            )
            if not np.isfinite(targets).all() or np.any(targets <= 0):
                raise RuntimeError(f"invalid VSPAERO targets {targets.tolist()}")
            if not (3.5 <= targets[0] <= 8.5 and 0.55 <= targets[1] <= 1.4):
                raise RuntimeError(
                    "VSPAERO converged but its fitted polar response is outside "
                    f"the usable range: CL_alpha={targets[0]:.5f}, "
                    f"Oswald={targets[1]:.5f}."
                )
            row = {
                "sample": sample_name,
                **design,
                "vspaero_cl_alpha_per_rad": float(targets[0]),
                "vspaero_oswald_efficiency": float(targets[1]),
                "oswald_efficiency_definition": OSWALD_EFFICIENCY_DEFINITION,
                "polar_points": 5,
                "geometry_file": str(case_dir / "airframe.vsp3"),
            }
            success_rows.append(row)
            successful += 1
            print(
                f"[{successful:02d}/{required_new_points}] {sample_name} "
                f"CLalpha={targets[0]:.5f} e={targets[1]:.5f}",
                flush=True,
            )
        except Exception as exc:
            error = str(exc)
            failed_rows.append(
                {
                    "attempt": attempts,
                    "sample": sample_name,
                    **dict(zip(FEATURE_LABELS, values.tolist())),
                    "status": "failed",
                    "error": error,
                }
            )
            (case_dir / "worker.log").write_text(
                (process.stdout if process else "")
                + "\n--- STDERR ---\n"
                + (process.stderr if process else "")
                + "\n"
                + error,
                encoding="utf-8",
            )
            print(f"[skip {attempts}] VSPAERO did not converge: {error[:180]}", flush=True)

        normalized_known = np.vstack((normalized_known, normalized))
        distance = np.linalg.norm(normalized_pool - normalized[None, :], axis=1)
        nearest = np.minimum(nearest, distance)
        persist_progress()
        if attempts >= args.pool_size:
            raise RuntimeError(
                f"Stopped after {attempts} attempts with {successful}/{required_new_points} valid new points."
            )

    full_frame = pd.read_csv(DATA_DIR / "VSPAERO_DOE扩充样本_100点_Ew.csv")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "nested maximin DOE; all new responses computed using the OpenVSP Python runtime and VSPAERO",
        "source_file": str(SOURCE_DATA),
        "source_point_count": 23,
        "baseline_point_count": 40,
        "expanded_point_count": 100,
        "new_successful_vspaero_solves": successful,
        "failed_attempts_excluded_from_training": len(failed_rows),
        "attempt_count": attempts,
        "seed": args.seed,
        "feature_bounds": {
            label: [
                float(full_frame[label].min()),
                float(full_frame[label].max()),
            ]
            for label in FEATURE_LABELS
        },
        "quality_scope": "VSPAERO VLM labels CL_alpha and Oswald efficiency; parasite drag remains in the separate low-order buildup.",
    }
    (DATA_DIR / "VSPAERO_DOE40_100实算元数据_Ew.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created real VSPAERO DOE datasets and surrogate models: {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
