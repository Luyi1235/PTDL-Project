"""PTDL calibration caches bundled for repeatable standalone runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


CORE = Path(__file__).resolve().parent
DIMENSION_CACHE = CORE / "ptdl_dimension_calibration_cache.json"
TASK_CACHE = CORE / "ptdl_endurance_task_calibration_cache.json"
DIMENSION_SCHEMA = "dimension_calibration_dynamic_cd0_v1"
TASK_SCHEMA = "full_task_endurance_calibration_dynamic_cd0_v3"


def load_dimension_calibration():
    if DIMENSION_CACHE.exists():
        payload = json.loads(DIMENSION_CACHE.read_text(encoding="utf-8"))
        if payload.get("schema") == DIMENSION_SCHEMA:
            from ULTRA基准标定 import UltraCalibration, UltraOpenData

            values = payload["calibration"]
            return UltraCalibration(
                cd0=float(values["cd0"]),
                baseline_endurance_h=float(values["baseline_endurance_h"]),
                baseline_fuel_used_kg=float(values["baseline_fuel_used_kg"]),
                cruise_power_kw=float(values["cruise_power_kw"]),
                reference=UltraOpenData(**values["reference"]),
            )

    from 固定起飞质量尺寸标定 import calibrate_dimension_baseline as calibrate

    calibration = calibrate()
    DIMENSION_CACHE.write_text(
        json.dumps(
            {
                "schema": DIMENSION_SCHEMA,
                "purpose": "Reusable inverse calibration for the dimension-driven reference model.",
                "calibration": {**asdict(calibration), "reference": asdict(calibration.reference)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return calibration


def load_task_calibration_values(assumptions=None) -> tuple[float, float] | None:
    if not TASK_CACHE.exists():
        return None
    payload = json.loads(TASK_CACHE.read_text(encoding="utf-8"))
    if payload.get("schema") != TASK_SCHEMA:
        return None
    if assumptions is not None:
        expected = json.loads(json.dumps(asdict(assumptions)))
        if payload.get("propulsion_assumptions") != expected:
            return None
    return float(payload["efficiency_scale"]), float(payload["baseline_duration_h"])


def save_task_calibration_values(scale: float, duration_h: float, assumptions=None) -> None:
    TASK_CACHE.write_text(
        json.dumps(
            {
                "schema": TASK_SCHEMA,
                "purpose": "Reusable full-task 72-hour inverse calibration for the reference model.",
                "efficiency_scale": float(scale),
                "baseline_duration_h": float(duration_h),
                "propulsion_assumptions": asdict(assumptions) if assumptions is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
