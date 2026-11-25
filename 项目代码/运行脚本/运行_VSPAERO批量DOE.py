"""Run VSPAERO for pending adaptive DOE points and save only solver-produced labels."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        "当前 OpenVSP 3.53.1 Python 扩展为 CPython 3.11 ABI，请使用 D:\\LDTP\\tools\\Python311\\python.exe。"
    )
OPENVSP_HOME = Path(
    os.environ.get(
        "PTDL_OPENVSP_HOME",
        ROOT.parents[1] / "tools" / "OpenVSP-3.53.1-win64",
    )
)
if OPENVSP_HOME.is_dir():
    os.environ["PATH"] = str(OPENVSP_HOME) + os.pathsep + os.environ.get("PATH", "")
    _OPENVSP_DLL_DIRECTORY = os.add_dll_directory(str(OPENVSP_HOME))

import openvsp as vsp

from OpenVSP_VSPAERO接口 import run_vspaero_polar
from VSPAERO气动代理模型 import FEATURE_LABELS
from 机体参数化设计与气动阻力 import AirframeDesign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "运行结果" / "VSPAERO自适应DOE建议_20261004" / "VSPAERO自适应DOE待计算点.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "运行结果" / "VSPAERO新增DOE实算")
    parser.add_argument("--speed-mps", type=float, default=36.01)
    parser.add_argument("--altitude-m", type=float, default=3000.0)
    args = parser.parse_args()
    points = pd.read_csv(args.input)
    missing = set(FEATURE_LABELS).difference(points.columns)
    if missing:
        raise SystemExit(f"DOE 输入缺少构型字段: {sorted(missing)}")

    records: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in points.iterrows():
        label = str(row.get("suggestion", f"sample_{len(records) + 1:02d}"))
        result_row: dict[str, object] = {
            "sample": label,
            "source": str(args.input),
            "openvsp_version": vsp.GetVSPVersion(),
            "speed_mps": args.speed_mps,
            "altitude_m": args.altitude_m,
            "aero_method": "VLM",
            "status": "failed",
        }
        try:
            design = AirframeDesign(**{key: float(row[key]) for key in FEATURE_LABELS})
            result = run_vspaero_polar(
                design,
                args.output_dir / label,
                speed_mps=args.speed_mps,
                altitude_m=args.altitude_m,
            )
            result_row.update(
                {
                    **{key: float(row[key]) for key in FEATURE_LABELS},
                    "cl_alpha_per_rad": result.cl_alpha_per_rad,
                    "oswald_efficiency": result.oswald_efficiency,
                    "polar_points": len(result.data),
                    "geometry_file": str(result.geometry_path),
                    "status": "solver_completed",
                }
            )
        except Exception as error:
            result_row["error"] = f"{type(error).__name__}: {error}"
        records.append(result_row)
        pd.DataFrame(records).to_csv(args.output_dir / "VSPAERO新增样本运行状态.csv", index=False)

    results = pd.DataFrame(records)
    successful = results.loc[results["status"] == "solver_completed"]
    successful.to_csv(args.output_dir / "VSPAERO新增DOE实算标签.csv", index=False)
    print(results[["sample", "status"]].to_string(index=False))
    print(f"成功 {len(successful)}/{len(results)}；标签文件：{args.output_dir / 'VSPAERO新增DOE实算标签.csv'}")
    return 0 if len(successful) == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
