"""Plot the H=6 dynamic-CD0 MOEA/D Pareto set without modifying run outputs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "ptdl_moead_h6_vsp_dynamic_cd0_results"
PARETO = RESULTS / "h6_dynamic_cd0_feasible_pareto_with_aero.csv"


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _normalize(values: pd.Series) -> list[float]:
    lower, upper = float(values.min()), float(values.max())
    if upper - lower < 1.0e-12:
        return [0.5] * len(values)
    return [float((value - lower) / (upper - lower)) for value in values]


def _color(value: float) -> tuple[int, int, int]:
    return int(41 + 190 * value), int(117 - 65 * value), int(181 - 105 * value)


def _draw_3d(frame: pd.DataFrame) -> None:
    image = Image.new("RGB", (1500, 950), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, note_font = _font(31, True), _font(21), _font(17)
    draw.text((84, 55), "H=6 MOEA/D Pareto Set: VSPAERO Correction + Dynamic CD0", fill="#152238", font=title_font)
    origin = (260, 750)
    x_end, y_end, z_end = (1070, 855), (660, 335), (260, 205)
    for endpoint in (x_end, y_end, z_end):
        draw.line((origin, endpoint), fill="#64748B", width=3)
    draw.text((1090, 842), "J_energy (MJ)", fill="#334155", font=label_font)
    draw.text((665, 300), "J_stability", fill="#334155", font=label_font)
    draw.text((274, 172), "J_response (s)", fill="#334155", font=label_font)
    x_values, y_values, z_values = _normalize(frame["J_energy_mj"]), _normalize(frame["J_stability"]), _normalize(frame["J_response_s"])
    cd_values = _normalize(frame["cd0_loiter_dynamic"])
    for x, y, z, cd0 in sorted(zip(x_values, y_values, z_values, cd_values), key=lambda item: item[2]):
        px = origin[0] + 810 * x + 400 * y
        py = origin[1] + 105 * x - 415 * y - 545 * z
        radius = 10
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=_color(cd0), outline="white", width=2)
    draw.text((88, 866), "Each point is a feasible non-dominated MOEA/D record. Color: dynamic loiter CD0 (blue low -> red high).", fill="#475569", font=note_font)
    image.save(RESULTS / "h6_dynamic_cd0_pareto_3d.png")


def _draw_2d(frame: pd.DataFrame) -> None:
    image = Image.new("RGB", (1200, 820), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, note_font = _font(29, True), _font(20), _font(16)
    draw.text((76, 42), "Dynamic-CD0 Feasible Pareto Projection", fill="#152238", font=title_font)
    left, top, right, bottom = 150, 140, 1080, 680
    draw.rectangle((left, top, right, bottom), outline="#94A3B8", width=2)
    for fraction in (0.25, 0.50, 0.75):
        draw.line((left + (right - left) * fraction, top, left + (right - left) * fraction, bottom), fill="#E2E8F0", width=1)
        draw.line((left, top + (bottom - top) * fraction, right, top + (bottom - top) * fraction), fill="#E2E8F0", width=1)
    x_values, y_values, z_values = _normalize(frame["J_energy_mj"]), _normalize(frame["J_stability"]), _normalize(frame["J_response_s"])
    for x, y, z in zip(x_values, y_values, z_values):
        px = left + (right - left) * x
        py = bottom - (bottom - top) * y
        radius = 10
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=_color(z), outline="white", width=2)
    draw.text((530, 718), "J_energy (MJ)", fill="#334155", font=label_font)
    draw.text((55, 385), "J_stability", fill="#334155", font=label_font)
    draw.text((150, 770), "Color: J_response (blue low -> red high)", fill="#475569", font=note_font)
    image.save(RESULTS / "h6_dynamic_cd0_pareto_energy_stability.png")


def main() -> None:
    frame = pd.read_csv(PARETO)
    _draw_3d(frame)
    _draw_2d(frame)


if __name__ == "__main__":
    main()
