"""Replay a previously feasible 80 h PTDL candidate with sized component models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from 八十小时续航任务模型 import (
    build_endurance_mission,
    calibrate_endurance_task_baseline,
    evaluate_endurance_task,
    load_endurance_profile,
)
from 全任务剖面定义 import load_preliminary_propulsion_config, preliminary_hybrid_aircraft
from 机体参数化设计与气动阻力 import AirframeDesign
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain
from VSPAERO气动代理模型 import VSPAeroSurrogate


DEFAULT_CANDIDATE = ROOT / "验证测试" / "预估额定值基准候选.json"
DOE_VERSION = "Ew_物理选点_网格修复_20261006"
MODEL_VERSION = f"{DOE_VERSION}_边界适配"
DEFAULT_SURROGATE = (
    ROOT
    / "气动代理模型数据"
    / f"气动代理模型_扩充DOE100点_{MODEL_VERSION}.json"
)
DEFAULT_OUTPUT = ROOT / "运行结果" / "预估额定值闭环验证"


def _font(size: int) -> ImageFont.ImageFont:
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_series(
    draw: ImageDraw.ImageDraw,
    frame: tuple[int, int, int, int],
    x_values: list[float],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    left, top, right, bottom = frame
    ink = (33, 46, 58)
    muted = (97, 110, 120)
    grid = (220, 226, 230)
    title_font = _font(23)
    label_font = _font(17)
    small_font = _font(14)
    draw.text((left, top - 68), title, fill=ink, font=title_font)
    draw.text((left, top - 29), y_label, fill=muted, font=small_font)
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill=grid, width=1)
        tick_label = f"{value:.2f}" if y_max - y_min <= 2.0 else f"{value:.0f}"
        draw.text((left - 58, y - 10), tick_label, fill=muted, font=small_font)
    draw.rectangle((left, top, right, bottom), outline=(165, 176, 184), width=1)
    draw.text((left, bottom + 12), f"{min(x_values):.0f}", fill=muted, font=small_font)
    draw.text((right - 30, bottom + 12), f"{max(x_values):.0f}", fill=muted, font=small_font)
    draw.text(((left + right) // 2 - 42, bottom + 33), x_label, fill=muted, font=label_font)

    for index, (name, values, color) in enumerate(series):
        points = []
        span_x = max(x_values[-1] - x_values[0], 1e-9)
        span_y = max(y_max - y_min, 1e-9)
        stride = max(1, len(x_values) // 2500)
        for i in range(0, min(len(x_values), len(values)), stride):
            px = left + (right - left) * (x_values[i] - x_values[0]) / span_x
            py = bottom - (bottom - top) * (values[i] - y_min) / span_y
            points.append((int(px), int(py)))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        legend_x = left + 135 + index * 205
        legend_y = top - 25
        draw.line((legend_x, legend_y, legend_x + 26, legend_y), fill=color, width=4)
        draw.text((legend_x + 32, legend_y - 10), name, fill=ink, font=small_font)


def _write_visual(profile: pd.DataFrame, summary: dict, path: Path) -> None:
    width, height = 1720, 1050
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    ink = (33, 46, 58)
    muted = (97, 110, 120)
    title_font = _font(30)
    label_font = _font(18)
    small_font = _font(15)
    draw.text((70, 35), "PTDL 80 h Hybrid-System Replay", fill=ink, font=title_font)
    achieved = summary["achieved_total_duration_h"]
    status = "PASS" if summary["feasible"] else "CHECK"
    draw.text(
        (70, 83),
        f"{status} | {achieved:.2f} h / 80 h | terminal SOC {summary['terminal_soc']:.3f} | {summary['termination_reason']}",
        fill=(38, 132, 91) if summary["feasible"] else (191, 83, 60),
        font=label_font,
    )

    time_h = (profile["time_s"] / 3600.0).tolist()
    power_keys = (
        ("Engine bus", "p_engine_kw", (38, 116, 168)),
        ("Battery bus", "p_motor_kw", (216, 126, 48)),
        ("Motor shaft", "p_propulsor_input_kw", (42, 145, 103)),
        ("Shaft request", "p_propulsion_request_kw", (181, 76, 72)),
    )
    values = [(label, profile[key].astype(float).tolist(), color) for label, key, color in power_keys]
    power_min = min(float(profile[key].min()) for _, key, _ in power_keys)
    power_max = max(float(profile[key].max()) for _, key, _ in power_keys)
    pad = max((power_max - power_min) * 0.08, 1.0)
    _draw_series(
        draw,
        (115, 175, 1060, 515),
        time_h,
        values,
        power_min - pad,
        power_max + pad,
        "Mission time (h)",
        "Power (kW)",
        "Power flow",
    )
    _draw_series(
        draw,
        (115, 660, 1060, 925),
        time_h,
        [
            ("SOC", profile["soc"].astype(float).tolist(), (121, 82, 160)),
        ],
        0.0,
        1.0,
        "Mission time (h)",
        "SOC (-)",
        "Battery state of charge",
    )

    x0, x1 = 1190, 1650
    draw.text((x0, 170), "Sizing basis", fill=ink, font=_font(24))
    metrics = [
        ("Engine-generator", summary["native_generator_ac_kw"], summary["estimated_engine_limit_kw"], "kW"),
        ("Motor shaft total", summary["native_motor_total_kw"], summary["estimated_motor_limit_kw"], "kW"),
        ("Battery energy", summary["native_battery_kwh"], summary["estimated_battery_kwh"], "kWh"),
    ]
    bar_max = 285
    for index, (name, native, estimate, unit) in enumerate(metrics):
        y = 245 + index * 190
        draw.text((x0, y), name, fill=ink, font=label_font)
        native_w = max(6, bar_max * min(native / max(estimate, 1e-9), 1.0))
        draw.rectangle((x0, y + 38, x0 + native_w, y + 60), fill=(111, 128, 139))
        draw.rectangle((x0, y + 82, x0 + bar_max, y + 104), fill=(38, 116, 168))
        draw.text((x0 + bar_max + 12, y + 39), f"{native:.1f} {unit}", fill=muted, font=small_font)
        draw.text((x0 + bar_max + 12, y + 83), f"{estimate:.1f} {unit}", fill=ink, font=small_font)
        draw.text((x0, y + 119), "source module", fill=muted, font=small_font)
        draw.text((x0, y + 146), "PTDL sizing estimate", fill=ink, font=small_font)

    note_y = 865
    draw.line((x0, note_y - 25, x1, note_y - 25), fill=(220, 226, 230), width=1)
    notes = (
        f"Hybrid-system mass: {summary['hybrid_system_mass_kg']:.1f} kg",
        f"Estimated battery parallel equivalent: {summary['battery_parallel_equivalent']:.2f}",
        f"Equivalent PMSM branches: {summary['motor_branches']}",
        (
            "Rating / energy multipliers: "
            f"{summary['rating_energy_multipliers']['engine']:.2f} / "
            f"{summary['rating_energy_multipliers']['motor']:.2f} / "
            f"{summary['rating_energy_multipliers']['battery_energy']:.2f}"
        ),
        "Equivalent sizing model; not hardware validation.",
    )
    for index, text in enumerate(notes):
        draw.text((x0, note_y + index * 32), text, fill=muted, font=small_font)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a prior feasible PTDL candidate with estimated component ratings.")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--surrogate", type=Path, default=DEFAULT_SURROGATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rating-factor",
        type=float,
        default=None,
        help="Override engine, motor, and battery multipliers together for a controlled comparison.",
    )
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    design = AirframeDesign(**candidate["design"])
    task_calibration = calibrate_endurance_task_baseline()
    assumptions = task_calibration.assumptions
    if args.rating_factor is not None:
        if not 0.25 <= args.rating_factor <= 3.0:
            raise SystemExit("--rating-factor must be between 0.25 and 3.0.")
        assumptions = replace(
            assumptions,
            engine_rating_multiplier=args.rating_factor,
            motor_rating_multiplier=args.rating_factor,
            battery_energy_multiplier=args.rating_factor,
        )
    surrogate = VSPAeroSurrogate.from_json(args.surrogate)
    aero = surrogate.corrected_aero(design, task_calibration.dimension_calibration)
    profile = load_endurance_profile()
    mission = build_endurance_mission(profile)
    speeds = [phase.speed_target_mps for phase in mission.phases]
    speeds[2] = float(candidate["isr_cruise_speed_mps"])
    powertrain = HybridSystemPowertrain(config=HybridSystemAdapterConfig())
    aircraft = preliminary_hybrid_aircraft(
        design,
        task_calibration.dimension_calibration,
        assumptions,
        aero_override=aero,
    )
    initial_state = powertrain.initial_state(aircraft, aircraft.initial_soc)
    _, sizing_config = load_preliminary_propulsion_config()
    evaluation = evaluate_endurance_task(
        design,
        engine_caps_kw=candidate["engine_caps_kw"],
        motor_assist_kw=candidate["motor_assist_kw"],
        speeds_mps=speeds,
        calibration=task_calibration.dimension_calibration,
        assumptions=assumptions,
        aero_override=aero,
        powertrain=powertrain,
    )
    frame = evaluation.simulation.profile
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "ptdl_hybrid_sized_time_history.csv", index=False)
    summary = {
        "candidate_provenance": candidate["provenance"],
        "model_status": "PTDL preliminary equivalent component sizing; not validated hardware performance",
        "feasible": bool(evaluation.result.feasible),
        "achieved_total_duration_h": evaluation.achieved_duration_h,
        "requested_total_duration_h": 80.0,
        "termination_reason": evaluation.simulation.termination_reason,
        "total_violation": evaluation.result.total_violation,
        "fuel_used_kg": aircraft.initial_fuel_kg - evaluation.simulation.terminal_fuel_kg,
        "terminal_fuel_kg": evaluation.simulation.terminal_fuel_kg,
        "terminal_soc": float(frame["soc"].iloc[-1]),
        "minimum_soc": float(frame["soc"].min()),
        "hybrid_system_mass_kg": design.hybrid_system_mass_kg,
        "ultra_reference_basis": sizing_config["reference_basis"],
        "propulsion_sizing_note": sizing_config["sizing_note"],
        "preliminary_propulsion_assumptions": asdict(assumptions),
        "rating_energy_multipliers": {
            "engine": assumptions.engine_rating_multiplier,
            "motor": assumptions.motor_rating_multiplier,
            "battery_energy": assumptions.battery_energy_multiplier,
        },
        "rating_factor_comparison_override": args.rating_factor,
        "estimated_engine_limit_kw": aircraft.engine_electrical_power_max_kw,
        "engine_rating_scale_over_native_generator": initial_state.engine_rating_scale,
        "engine_fuel_flow_override": powertrain.config.engine_fuel_flow_scale,
        "engine_efficiency_curve": {
            "load_fractions": list(aircraft.engine_efficiency_power_fractions),
            "net_efficiencies": list(aircraft.engine_efficiency_values),
        },
        "estimated_motor_limit_kw": aircraft.motor_power_max_kw,
        "motor_rated_kw_per_branch": initial_state.motor.parameters["MOT_P_rated_W"] / 1000.0,
        "motor_branches": initial_state.motor_branches,
        "estimated_battery_capacity_mj": aircraft.battery_capacity_mj,
        "estimated_battery_kwh": aircraft.battery_capacity_mj / 3.6,
        "engine_equivalent_expansion_over_component_generator": initial_state.engine_rating_scale,
        "motor_equivalent_branch_count_from_native_motor": initial_state.motor_branches,
        "battery_equivalent_parallel_count_from_native_pack": initial_state.battery_parallel_equivalent,
        "native_generator_ac_kw": powertrain.generator_rated_ac_kw,
        "native_motor_total_kw": powertrain.motor_rated_mechanical_kw * powertrain.config.motor_branches,
        "native_battery_kwh": powertrain.battery_parameters["BAT_Energy_nominal_Wh"] / 1000.0,
        "battery_parallel_equivalent": initial_state.battery_parallel_equivalent,
        "component_map_converged_points": powertrain.engine_map.converged_count,
        "component_map_total_points": len(powertrain.engine_map._rows),
        "constraints": evaluation.result.constraints,
    }
    (args.output_dir / "ptdl_hybrid_sized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_visual(frame, summary, args.output_dir / "ptdl_hybrid_sized_validation.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
