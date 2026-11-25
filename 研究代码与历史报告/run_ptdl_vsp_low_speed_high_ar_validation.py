"""VSPAERO validation samples for low-speed and high-aspect-ratio PTDL candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from ptdl_airframe_design import AirframeDesign, design_metrics
from ptdl_vsp_aero_surrogate import VSPAeroSurrogate
from ptdl_vspaero_adapter import load_vspaero_polar, run_vspaero_polar


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ptdl_vsp_low_speed_high_ar_validation_results"
PARETO = ROOT / "ptdl_moead_h6_robust_vsp_results" / "稳健H6_MOEAD可行Pareto候选.csv"
SURROGATE = ROOT / "ptdl_vsp_targeted_candidate_results" / "VSPAERO扩充代理模型.json"
ALTITUDE_M = 3000.0


def design_from_row(row: pd.Series) -> AirframeDesign:
    fields = (
        "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_deg", "tip_twist_deg",
        "fuselage_length_m", "fuselage_diameter_m", "propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg",
    )
    return AirframeDesign(**{field: float(row[field]) for field in fields})


def samples() -> list[tuple[str, str, AirframeDesign, float]]:
    candidates = pd.read_csv(PARETO).sort_values("candidate_index").reset_index(drop=True)
    a, b = design_from_row(candidates.iloc[0]), design_from_row(candidates.iloc[1])
    return [
        ("candidate6_low_speed", "候选6，低速 ISR", a, 30.0),
        ("candidate6_high_speed", "候选6，较高巡航速度", a, 42.0),
        ("candidate23_low_speed", "候选23，低速 ISR", b, 30.0),
        ("high_ar31_area_minus_low_speed", "高展弦比31，翼面积-5%，低速", replace(b, wing_area_m2=b.wing_area_m2 * 0.95), 30.0),
        ("high_ar31_area_plus_low_speed", "高展弦比31，翼面积+5%，低速", replace(b, wing_area_m2=b.wing_area_m2 * 1.05), 30.0),
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    surrogate = VSPAeroSurrogate.from_json(SURROGATE)
    rows: list[dict[str, object]] = []
    for index, (label, description, design, speed) in enumerate(samples(), start=1):
        folder = OUTPUT / "polars" / label
        polar_csv = folder / "vspaero_polar.csv"
        if polar_csv.exists() and (folder / "airframe.vsp3").exists():
            polar_file = next(folder.glob("*.polar"), None)
            if polar_file is None:
                raise RuntimeError(f"Cached VSPAERO sample {label} has no .polar file.")
            polar = load_vspaero_polar(polar_file, design, folder / "airframe.vsp3", polar_csv)
            print(f"[{index}/5] Reusing {label}", flush=True)
        else:
            print(f"[{index}/5] Running VSPAERO {label}", flush=True)
            polar = run_vspaero_polar(design, folder, speed_mps=speed, altitude_m=ALTITUDE_M)
        predicted_cl, predicted_e = surrogate.predict(design)
        rows.append({
            "sample": label, "description": description, "altitude_m": ALTITUDE_M, "speed_mps": speed,
            **asdict(design), "wing_span_m": design_metrics(design).wing_span_m,
            "vspaero_cl_alpha_per_rad": polar.cl_alpha_per_rad,
            "vspaero_oswald_efficiency": polar.oswald_efficiency,
            "surrogate_cl_alpha_per_rad": predicted_cl,
            "surrogate_oswald_efficiency": predicted_e,
            "cl_alpha_error_per_rad": predicted_cl - polar.cl_alpha_per_rad,
            "oswald_error": predicted_e - polar.oswald_efficiency,
            "polar_points": len(polar.data),
            "geometry_file": str(polar.geometry_path.relative_to(ROOT)),
        })
    pd.DataFrame(rows).to_csv(OUTPUT / "低速高展弦比VSPAERO验证结果.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "运行说明.json").write_text(json.dumps({
        "scope": "low-speed and high-aspect-ratio VSPAERO validation around robust candidates",
        "samples": len(rows), "altitude_m": ALTITUDE_M,
        "model_boundary": "VSPAERO potential-flow coefficients are expected to show little direct speed dependence. Parasite-drag Reynolds effects remain in the calibrated dynamic CD0 model.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
