"""Persistent cache for the expensive dimension-layer ULTRA calibration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ptdl_fixed_mtow_sizing import calibrate_dimension_baseline
from ptdl_ultra_benchmark import UltraCalibration, UltraOpenData


ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "ptdl_dimension_calibration_cache.json"
CACHE_SCHEMA = "dimension_calibration_dynamic_cd0_v1"


def load_dimension_calibration() -> UltraCalibration:
    """Load a traceable calibration cache or calculate it once when absent."""
    if CACHE_PATH.exists():
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if payload.get("schema") == CACHE_SCHEMA:
            values = payload["calibration"]
            return UltraCalibration(
                cd0=float(values["cd0"]),
                baseline_endurance_h=float(values["baseline_endurance_h"]),
                baseline_fuel_used_kg=float(values["baseline_fuel_used_kg"]),
                cruise_power_kw=float(values["cruise_power_kw"]),
                reference=UltraOpenData(**values["reference"]),
            )
    calibration = calibrate_dimension_baseline()
    CACHE_PATH.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "purpose": "Reusable inverse calibration for the current dimension-driven, dynamic-CD0 reference model.",
                "calibration": {**asdict(calibration), "reference": asdict(calibration.reference)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return calibration
