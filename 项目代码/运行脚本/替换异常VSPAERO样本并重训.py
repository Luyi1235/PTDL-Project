"""Replace successful but unusable VSPAERO polar responses in a 100-point DOE."""

from __future__ import annotations

import argparse
import json
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

from VSPAERO气动代理模型 import FEATURE_LABELS, fit_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign
from 生成_VSPAERO_DOE40_100并训练代理 import feasible_pool, worker_payload


DATA_DIR = ROOT / "气动代理模型数据"
DOE100 = DATA_DIR / "VSPAERO_DOE扩充样本_100点.csv"
WORKER = ROOT / "运行脚本" / "运行_VSPAERO单构型评估.py"
DEFAULT_VSP_PYTHON = ROOT.parent.parent / "tools" / "Python311" / "python.exe"
DEFAULT_OUTPUT = ROOT / "运行结果" / "VSPAERO_DOE100异常点替换_20261006"


def usable_targets(cl_alpha: float, oswald: float) -> bool:
    return (
        np.isfinite([cl_alpha, oswald]).all()
        and 3.5 <= cl_alpha <= 8.5
        and 0.55 <= oswald <= 1.4
    )


def terminate_worker_tree(worker: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(worker.pid)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    else:
        worker.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsp-python", type=Path, default=DEFAULT_VSP_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    if not args.vsp_python.is_file():
        raise FileNotFoundError(f"OpenVSP Python runtime not found: {args.vsp_python}")

    frame = pd.read_csv(DOE100)
    required = {
        *FEATURE_LABELS,
        "vspaero_cl_alpha_per_rad",
        "vspaero_oswald_efficiency",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or len(frame) != 100:
        raise ValueError(f"Expected a complete 100-row DOE; missing={missing}, rows={len(frame)}")

    bad_mask = ~frame.apply(
        lambda row: usable_targets(
            float(row["vspaero_cl_alpha_per_rad"]),
            float(row["vspaero_oswald_efficiency"]),
        ),
        axis=1,
    )
    bad_indices = frame.index[bad_mask].tolist()
    if not bad_indices:
        print("No unusable VSPAERO response rows found.")
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = DATA_DIR / "VSPAERO_DOE扩充样本_100点_异常筛查前.csv"
    if not backup.exists():
        frame.to_csv(backup, index=False)
    all_features = frame.loc[:, FEATURE_LABELS].to_numpy(dtype=float)
    lower = all_features.min(axis=0)
    upper = all_features.max(axis=0)
    pool = feasible_pool(lower, upper, args.pool_size, np.random.default_rng(args.seed))
    scales = np.maximum(upper - lower, 1e-12)
    normalized_pool = (pool - lower) / scales
    normalized_known = (all_features - lower) / scales
    nearest = np.linalg.norm(
        normalized_pool[:, None, :] - normalized_known[None, :, :], axis=2
    ).min(axis=1)

    attempts: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    attempted = 0
    for row_index in bad_indices:
        old_sample = str(frame.at[row_index, "sample"])
        while True:
            candidate_index = int(np.argmax(nearest))
            if nearest[candidate_index] < 0:
                raise RuntimeError("Exhausted the replacement DOE candidate pool.")
            values = pool[candidate_index].copy()
            normalized = normalized_pool[candidate_index].copy()
            nearest[candidate_index] = -1.0
            design = dict(zip(FEATURE_LABELS, values.tolist()))
            attempted += 1
            case_name = f"replacement_{attempted:03d}"
            case_dir = output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            request = {
                "design": {
                    **{
                        key: float(frame.iloc[row_index][key])
                        for key in (
                            "propeller_diameter_m",
                            "payload_mass_kg",
                            "hybrid_system_mass_kg",
                        )
                        if key in frame.columns
                    },
                    **design,
                },
                "output_dir": str(case_dir),
                "speed_mps": 35.99708,
                "altitude_m": 3000.0,
            }
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
                    stdout_file.seek(0)
                    stderr_file.seek(0)
                    stdout = stdout_file.read()
                    stderr = stderr_file.read()
                    if worker.returncode != 0:
                        raise RuntimeError(
                            f"worker exit {worker.returncode}: "
                            f"{stdout[-900:]} {stderr[-900:]}"
                        )
                    result = worker_payload(stdout)
                    cl_alpha = float(result["cl_alpha_per_rad"])
                    oswald = float(result["oswald_efficiency"])
                    if not usable_targets(cl_alpha, oswald):
                        raise RuntimeError(
                            f"polar response outside usable limits: "
                            f"CL_alpha={cl_alpha:.5f}, Oswald={oswald:.5f}"
                        )
                    replacement = {
                        **design,
                        "sample": f"{old_sample}_replacement",
                        "vspaero_cl_alpha_per_rad": cl_alpha,
                        "vspaero_oswald_efficiency": oswald,
                        "polar_points": int(result.get("polar_points", 5)),
                        "geometry_file": str(case_dir / "airframe.vsp3"),
                        "replaced_sample": old_sample,
                    }
                    for key in replacement:
                        if key not in frame.columns:
                            frame[key] = (
                                pd.Series(index=frame.index, dtype="object")
                                if key == "replaced_sample"
                                else np.nan
                            )
                    for key, value in replacement.items():
                        frame.at[row_index, key] = value
                    accepted.append(
                        {
                            "replaced_sample": old_sample,
                            "replacement_sample": replacement["sample"],
                            "CL_alpha_per_rad": cl_alpha,
                            "oswald_efficiency": oswald,
                            **design,
                        }
                    )
                    print(
                        f"[{len(accepted)}/{len(bad_indices)}] {old_sample} -> "
                        f"{replacement['sample']}: CLalpha={cl_alpha:.5f}, e={oswald:.5f}",
                        flush=True,
                    )
                    break
                except subprocess.TimeoutExpired:
                    terminate_worker_tree(worker)
                    if worker.poll() is None:
                        worker.kill()
                    worker.wait(timeout=15)
                    error = f"worker process tree timed out after {args.timeout_s:.0f}s"
                except Exception as exc:
                    error = str(exc)
                    if worker.poll() is None:
                        terminate_worker_tree(worker)
                        if worker.poll() is None:
                            worker.kill()
                        worker.wait(timeout=15)

            attempts.append(
                {
                    "attempt": attempted,
                    "case": case_name,
                    **design,
                    "status": "rejected",
                    "error": error,
                }
            )
            normalized_known = np.vstack((normalized_known, normalized))
            nearest = np.minimum(
                nearest,
                np.linalg.norm(normalized_pool - normalized[None, :], axis=1),
            )
            pd.DataFrame(attempts).to_csv(output_dir / "替换尝试记录.csv", index=False)
            if attempted >= args.pool_size:
                raise RuntimeError("Replacement search exhausted its attempt budget.")

        normalized_known = np.vstack((normalized_known, normalized))
        nearest = np.minimum(
            nearest,
            np.linalg.norm(normalized_pool - normalized[None, :], axis=1),
        )

    frame.to_csv(DOE100, index=False)
    designs = [
        AirframeDesign(**{key: float(row[key]) for key in FEATURE_LABELS})
        for _, row in frame.iterrows()
    ]
    model = fit_vsp_surrogate(
        designs,
        frame["vspaero_cl_alpha_per_rad"].to_numpy(dtype=float),
        frame["vspaero_oswald_efficiency"].to_numpy(dtype=float),
    )
    model.to_json(DATA_DIR / "气动代理模型_扩充DOE100点.json")
    metadata_path = DATA_DIR / "VSPAERO_DOE40_100实算元数据.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "usable_polar_response_limits": {
                "CL_alpha_per_rad": [3.5, 8.5],
                "oswald_efficiency": [0.55, 1.4],
            },
            "replaced_unusable_rows": [str(frame.at[index, "replaced_sample"]) for index in bad_indices],
            "replacement_solves_accepted": len(accepted),
            "replacement_attempts": attempted,
            "failed_solver_attempts_during_replacement": len(attempts),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(accepted).to_csv(output_dir / "已接受替换点.csv", index=False)
    pd.DataFrame(attempts).to_csv(output_dir / "替换尝试记录.csv", index=False)
    print(f"Updated 100-point DOE and surrogate. Backup: {backup}", flush=True)


if __name__ == "__main__":
    main()
