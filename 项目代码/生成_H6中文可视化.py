"""Generate Chinese-labeled static figures for an H=6 integrated PTDL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900
PLOT = (150, 145, 1460, 750)
COLORS = {
    "NSGA-II": "#1976A8",
    "MOEA/D": "#E07A24",
    "发动机": "#1976A8",
    "电机/电池": "#E07A24",
    "推进需求": "#424B54",
    "高度": "#2A8C72",
    "速度": "#7B61A8",
    "SOC": "#2A8C72",
    "SOC下限": "#C44536",
    "SOC目标": "#E07A24",
}
FONT_PATHS = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = list(FONT_PATHS)
    if bold:
        candidates.insert(0, Path("C:/Windows/Fonts/msyhbd.ttc"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    draw.text((78, 38), title, font=_font(38, True), fill="#172B3A")
    if subtitle:
        draw.text((82, 92), subtitle, font=_font(21), fill="#52616B")
    return image, draw


def _axes(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x_label: str,
    y_label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    x_format: str = "{:.0f}",
    y_format: str = "{:.2f}",
) -> tuple[object, object]:
    left, top, right, bottom = PLOT
    draw.rectangle(PLOT, outline="#687782", width=2)
    for tick in np.linspace(0.0, 1.0, 6):
        x = left + tick * (right - left)
        y = bottom - tick * (bottom - top)
        xv = xlim[0] + tick * (xlim[1] - xlim[0])
        yv = ylim[0] + tick * (ylim[1] - ylim[0])
        draw.line((x, top, x, bottom), fill="#E5EAED", width=1)
        draw.line((left, y, right, y), fill="#E5EAED", width=1)
        draw.text((x - 24, bottom + 14), x_format.format(xv), font=_font(17), fill="#52616B")
        draw.text((left - 112, y - 12), y_format.format(yv), font=_font(17), fill="#52616B")
    draw.text(((left + right) / 2 - 80, bottom + 58), x_label, font=_font(21), fill="#263943")
    label_layer = Image.new("RGBA", (520, 42), (255, 255, 255, 0))
    ImageDraw.Draw(label_layer).text((4, 2), y_label, font=_font(20), fill="#263943")
    label_layer = label_layer.rotate(90, expand=True)
    image.paste(label_layer, (20, int((top + bottom - label_layer.height) / 2)), label_layer)

    def tx(value: float) -> float:
        return left + (value - xlim[0]) / max(xlim[1] - xlim[0], 1e-12) * (right - left)

    def ty(value: float) -> float:
        return bottom - (value - ylim[0]) / max(ylim[1] - ylim[0], 1e-12) * (bottom - top)

    return tx, ty


def _legend(draw: ImageDraw.ImageDraw, entries: list[tuple[str, str]]) -> None:
    x, y = PLOT[0] + 12, PLOT[1] + 12
    for label, color in entries:
        draw.line((x, y + 11, x + 34, y + 11), fill=color, width=4)
        draw.ellipse((x + 13, y + 5, x + 25, y + 17), fill=color)
        draw.text((x + 43, y), label, font=_font(18), fill="#263943")
        x += 70 + int(draw.textlength(label, font=_font(18)))


def _line_chart(
    path: Path,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    x_format: str = "{:.0f}",
    y_format: str = "{:.2f}",
    horizontal_lines: list[tuple[float, str, str]] | None = None,
) -> None:
    finite_x = [x[np.isfinite(x)] for _, x, y, _ in series if len(x)]
    finite_y = [y[np.isfinite(y)] for _, x, y, _ in series if len(y)]
    if not finite_x or not finite_y:
        return
    all_x = np.concatenate(finite_x)
    all_y = np.concatenate(finite_y)
    xlim = xlim or (float(np.min(all_x)), float(np.max(all_x)))
    if xlim[0] == xlim[1]:
        xlim = (xlim[0] - 1, xlim[1] + 1)
    if ylim is None:
        low, high = float(np.min(all_y)), float(np.max(all_y))
        padding = max((high - low) * 0.10, abs(high) * 0.02, 1e-6)
        ylim = (low - padding, high + padding)
    image, draw = _canvas(title, subtitle)
    tx, ty = _axes(image, draw, x_label, y_label, xlim, ylim, x_format=x_format, y_format=y_format)
    if horizontal_lines:
        for value, label, color in horizontal_lines:
            if ylim[0] <= value <= ylim[1]:
                yy = ty(value)
                draw.line((PLOT[0], yy, PLOT[2], yy), fill=color, width=3)
                draw.text((PLOT[2] - 190, yy - 28), label, font=_font(16), fill=color)
    for label, xs, ys, color in series:
        points = [
            (tx(float(x)), ty(float(y)))
            for x, y in zip(xs, ys)
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        stride = max(1, len(points) // 35)
        for x, y in points[::stride]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    _legend(draw, [(label, color) for label, _, _, color in series])
    image.save(path)


def _scatter_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    groups: list[tuple[str, pd.DataFrame, str]],
    x_column: str,
    y_column: str,
) -> None:
    frames = [frame for _, frame, _ in groups if not frame.empty]
    if not frames:
        return
    all_x = np.concatenate([frame[x_column].to_numpy(float) for frame in frames])
    all_y = np.concatenate([frame[y_column].to_numpy(float) for frame in frames])
    pad_x = max(float(np.ptp(all_x)) * 0.08, 1.0)
    pad_y = max(float(np.ptp(all_y)) * 0.12, 0.1)
    image, draw = _canvas(title, "点表示满足全部硬约束的非支配候选；越靠左下通常越优")
    tx, ty = _axes(
        image,
        draw,
        x_label,
        y_label,
        (float(np.min(all_x) - pad_x), float(np.max(all_x) + pad_x)),
        (float(np.min(all_y) - pad_y), float(np.max(all_y) + pad_y)),
        x_format="{:.0f}",
        y_format="{:.2f}" if "MJ" not in y_label else "{:.0f}",
    )
    for label, frame, color in groups:
        for x, y in zip(frame[x_column], frame[y_column]):
            px, py = tx(float(x)), ty(float(y))
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline="#FFFFFF", width=2)
    _legend(draw, [(label, color) for label, _, color in groups])
    image.save(path)


def _bar_chart(path: Path, comparison: pd.DataFrame) -> None:
    image, draw = _canvas(
        "算法搜索结果对比",
        "主比较口径：历代累计可行 Pareto 外部档案；同时列出末代种群统计",
    )
    labels = ["末代可行个体", "末代种群 Pareto", "历代档案 Pareto", "运行时间（秒）"]
    columns = [
        "final_population_feasible_count",
        "final_population_pareto_count",
        "historical_feasible_pareto_count",
        "runtime_s",
    ]
    values = comparison.set_index("algorithm")
    left, top, right, bottom = PLOT
    plot_w = right - left
    slot_h = (bottom - top) / len(labels)
    for row, (label, column) in enumerate(zip(labels, columns)):
        y_center = top + (row + 0.5) * slot_h
        draw.text((left - 142, y_center - 14), label, font=_font(19), fill="#263943")
        max_value = max(float(values[column].max()), 1e-9)
        bar_y = y_center - 27
        for index, algorithm in enumerate(("NSGA-II", "MOEA/D")):
            value = float(values.loc[algorithm, column])
            width = (plot_w - 180) * value / max_value
            y = bar_y + index * 34
            color = COLORS[algorithm]
            draw.rounded_rectangle((left + 8, y, left + 8 + width, y + 24), radius=4, fill=color)
            draw.text((left + 18 + width, y + 1), f"{value:.1f}" if column == "runtime_s" else f"{value:.0f}",
                      font=_font(17), fill="#263943")
            draw.text((left + 12, y + 2), algorithm, font=_font(15), fill="#FFFFFF")
        draw.line((left, top + (row + 1) * slot_h, right, top + (row + 1) * slot_h), fill="#E5EAED", width=1)
    image.save(path)


def _phase_chart(path: Path, profile: pd.DataFrame) -> None:
    phase_order = list(dict.fromkeys(profile["phase"].astype(str)))
    durations: list[float] = []
    fuel: list[float] = []
    for index, phase in enumerate(phase_order):
        frame = profile.loc[profile["phase"].astype(str) == phase]
        if len(frame) > 1:
            dt = np.diff(frame["time_s"].to_numpy(float))
            durations.append(float(np.sum(dt)) / 3600.0)
            flows = frame["fuel_flow_kgps"].to_numpy(float)
            fuel.append(float(np.sum(0.5 * (flows[:-1] + flows[1:]) * dt)))
        else:
            durations.append(0.0)
            fuel.append(0.0)
    image, draw = _canvas("任务阶段统计", "按闭环时序积分估算各阶段持续时间与燃油消耗")
    left, top, right, bottom = PLOT
    chart_h = bottom - top
    row_h = chart_h / max(len(phase_order), 1)
    max_fuel = max(fuel) if fuel else 1.0
    max_duration = max(durations) if durations else 1.0
    for index, phase in enumerate(phase_order):
        y = top + index * row_h
        draw.text((left - 135, y + 9), phase.replace("_", " "), font=_font(16), fill="#263943")
        duration_width = (right - left) * durations[index] / max_duration * 0.42
        fuel_width = (right - left) * fuel[index] / max_fuel * 0.42
        draw.rounded_rectangle((left + 6, y + 7, left + 6 + duration_width, y + 27), radius=3, fill="#2A8C72")
        draw.text((left + duration_width + 18, y + 6), f"{durations[index]:.2f} 小时", font=_font(15), fill="#263943")
        draw.rounded_rectangle((left + 570, y + 7, left + 570 + fuel_width, y + 27), radius=3, fill="#E07A24")
        draw.text((left + 580 + fuel_width, y + 6), f"{fuel[index]:.1f} kg", font=_font(15), fill="#263943")
        draw.line((left, y + row_h, right, y + row_h), fill="#E5EAED", width=1)
    draw.rectangle((left + 6, top - 26, left + 28, top - 10), fill="#2A8C72")
    draw.text((left + 36, top - 31), "持续时间", font=_font(16), fill="#263943")
    draw.rectangle((left + 185, top - 26, left + 207, top - 10), fill="#E07A24")
    draw.text((left + 215, top - 31), "燃油消耗", font=_font(16), fill="#263943")
    image.save(path)


def _contact_sheet(paths: list[Path], destination: Path) -> None:
    thumb_w, thumb_h = 640, 360
    columns = 2
    rows = int(np.ceil(len(paths) / columns))
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 42)), "#EEF2F4")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + 42)
        sheet.paste(image, (x, y))
        label = _CHART_LABELS.get(path.stem, path.stem)
        draw.text((x + 12, y + thumb_h + 8), label, font=_font(19, True), fill="#263943")
    sheet.save(destination)


_CHART_LABELS = {
    "收敛_feasible_ratio": "可行候选比例收敛",
    "收敛_best_J_energy_mj": "任务能量收敛",
    "收敛_best_J_stability": "稳定性指标收敛",
    "收敛_best_J_response_s": "响应时间指标收敛",
    "Pareto_能量-稳定性": "Pareto：能量与稳定性",
    "Pareto_能量-响应": "Pareto：能量与响应",
    "Pareto_稳定性-响应": "Pareto：稳定性与响应",
    "算法结果对比": "优化算法结果对比",
    "功率分配时序": "80小时功率分配",
    "电池SOC时序": "电池SOC变化",
    "速度高度时序": "飞行速度变化",
    "高度时序": "飞行高度变化",
    "任务阶段统计": "各阶段时长与燃油",
    "总体结果摘要": "H=6优化结果摘要",
    "扰动恢复响应": "等效阻力扰动响应",
    "统一归一化权重扫描": "目标权重与裕度筛选",
}


def _summary_chart(
    path: Path,
    comparison: pd.DataFrame,
    candidate_path: Path,
) -> None:
    import json

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    replay = candidate["independent_replay"]
    image, draw = _canvas(
        "H=6 优化结果摘要",
        "NSGA-II 与 MOEA/D 同模型比较；PPT三项目标与约束保持不变",
    )
    x0, y0 = 160, 160
    rows = [
        ("任务时长", f"{replay['achieved_total_duration_h']:.1f} / {replay['requested_total_duration_h']:.0f} 小时"),
        ("独立回放", "可行" if replay["feasible"] else "未通过"),
        ("燃油消耗", f"{replay['fuel_used_kg']:.1f} kg"),
        ("终端电池SOC", f"{replay['terminal_soc']:.1%}（附加筛选下限 25%）"),
        ("剩余燃油", f"{replay['terminal_fuel_kg']:.1f} kg（5%参考值未达标）"),
        ("等效任务能量", f"{replay['objectives']['J_energy_mj']:.1f} MJ"),
        ("稳定性指标", f"{replay['objectives']['J_stability']:.3f}"),
        ("响应时间指标", f"{replay['objectives']['J_response_s']:.0f} s"),
        ("扰动恢复", (
            f"{replay['response_metrics']['recovery_time_s']:.0f} s"
            if replay["response_metrics"].get("recovery_success", 0.0) >= 1.0
            else f"未在 {replay['response_metrics']['recovery_time_s']:.0f} s 窗口内恢复"
        )),
        ("候选算法", str(candidate["optimizer"])),
    ]
    for index, (label, value) in enumerate(rows):
        y = y0 + index * 52
        draw.text((x0, y), label, font=_font(22, True), fill="#263943")
        draw.text((x0 + 310, y), value, font=_font(23), fill="#1976A8" if index < 2 else "#33454F")
        draw.line((x0, y + 38, 1370, y + 38), fill="#E5EAED", width=2)
    nsga = comparison.loc[comparison["algorithm"] == "NSGA-II"].iloc[0]
    moead = comparison.loc[comparison["algorithm"] == "MOEA/D"].iloc[0]
    note = (
        f"NSGA-II：历代可行 Pareto 档案 {int(nsga['historical_feasible_pareto_count'])} 个，"
        f"耗时 {nsga['runtime_s']:.1f} 秒    |    "
        f"MOEA/D：历代可行 Pareto 档案 {int(moead['historical_feasible_pareto_count'])} 个，"
        f"耗时 {moead['runtime_s']:.1f} 秒"
    )
    draw.text((x0, 710), note, font=_font(20), fill="#52616B")
    draw.text((x0, 755), "SOC≥25%为附加筛选；5%燃油储备参考值未达，申请书硬约束未变；动力参数仍为预估值。", font=_font(19), fill="#A13D32")
    image.save(path)


def _weight_scan_chart(path: Path, scan: pd.DataFrame) -> None:
    selected = scan.loc[scan["selected"].astype(bool)].copy()
    if selected.empty:
        return
    image, draw = _canvas("统一归一化后的权重扫描", "在 SOC≥25%、剩余燃油为正的历史非支配候选中比较四组偏好")
    x_positions = [95, 490, 880, 1110, 1320]
    headers = ["偏好权重", "算法/候选", "归一化加权分", "终端 SOC", "燃油裕度 kg"]
    for x, label in zip(x_positions, headers):
        draw.text((x, 170), label, font=_font(19, True), fill="#263943")
    draw.line((90, 205, 1490, 205), fill="#687782", width=2)
    for index, (_, row) in enumerate(selected.iterrows()):
        y = 230 + index * 105
        values = [
            {
                "energy_priority": "能耗优先",
                "stability_priority": "稳定性优先",
                "response_priority": "响应优先",
                "balanced": "均衡偏好",
            }.get(str(row["preference"]), str(row["preference"])),
            f"{row['algorithm']} #{int(row['archive_candidate_index'])}",
            f"{row['weighted_score']:.4f}",
            f"{row['terminal_soc']:.1%}",
            f"{row['fuel_margin_kg']:.1f}",
        ]
        for x, value in zip(x_positions, values):
            draw.text((x, y), value, font=_font(18), fill="#1976A8")
        draw.line((90, y + 55, 1490, y + 55), fill="#E5EAED", width=1)
    image.save(path)


def generate_visualizations(output_dir: Path) -> list[str]:
    output_dir = output_dir.resolve()
    image_dir = output_dir / "中文可视化"
    image_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "NSGA-II": "ptdl_80h_joint_nsga_ii",
        "MOEA/D": "ptdl_80h_joint_moead",
    }
    convergence = {
        algorithm: pd.read_csv(output_dir / directory / f"{prefix}_convergence.csv")
        for (algorithm, directory), prefix in zip(
            (("NSGA-II", "NSGA-II"), ("MOEA/D", "MOEA-D")), names.values()
        )
    }
    archives = {
        algorithm: pd.read_csv(output_dir / directory / f"{prefix}_feasible_pareto.csv")
        for (algorithm, directory), prefix in zip(
            (("NSGA-II", "NSGA-II"), ("MOEA/D", "MOEA-D")), names.values()
        )
    }
    comparison = pd.read_csv(output_dir / "NSGA-II与MOEA-D同模型算法比较.csv")
    profile = pd.read_csv(output_dir / "NSGA-II与MOEA-D择优候选_MPC时序.csv")
    outputs: list[Path] = []

    for metric, title, label, formatter in (
        ("feasible_ratio", "优化收敛：可行解比例", "可行解比例", "{:.0%}"),
        ("best_J_energy_mj", "优化收敛：最低等效任务能量", "等效任务能量（MJ）", "{:.0f}"),
        ("best_J_stability", "优化收敛：最低稳定性指标", "稳定性指标（归一化）", "{:.2f}"),
        ("best_J_response_s", "优化收敛：最低响应时间指标", "响应时间指标（秒）", "{:.0f}"),
    ):
        series = [
            (
                algorithm,
                frame["generation"].to_numpy(float),
                frame[metric].to_numpy(float),
                COLORS[algorithm],
            )
            for algorithm, frame in convergence.items()
        ]
        path = image_dir / f"收敛_{metric}.png"
        _line_chart(
            path, title, "实线为跨代历史最好值；可行率为当代种群统计",
            "迭代代数", label, series,
            xlim=(0, max(float(frame["generation"].max()) for frame in convergence.values())),
            ylim=(0.0, 1.0) if metric == "feasible_ratio" else None,
            y_format=formatter,
        )
        outputs.append(path)

    pareto_specs = (
        ("J_energy_mj", "J_stability", "Pareto_能量-稳定性.png",
         "Pareto候选：任务能量与稳定性", "等效任务能量（MJ）", "稳定性指标"),
        ("J_energy_mj", "J_response_s", "Pareto_能量-响应.png",
         "Pareto候选：任务能量与响应时间", "等效任务能量（MJ）", "响应时间指标（秒）"),
        ("J_stability", "J_response_s", "Pareto_稳定性-响应.png",
         "Pareto候选：稳定性与响应时间", "稳定性指标", "响应时间指标（秒）"),
    )
    for x_col, y_col, filename, title, x_label, y_label in pareto_specs:
        path = image_dir / filename
        _scatter_chart(
            path, title, x_label, y_label,
            [
                (algorithm, archives[algorithm], COLORS[algorithm])
                for algorithm in ("NSGA-II", "MOEA/D")
            ],
            x_col, y_col,
        )
        outputs.append(path)

    path = image_dir / "算法结果对比.png"
    _bar_chart(path, comparison)
    outputs.append(path)

    time_h = profile["time_s"].to_numpy(float) / 3600.0
    signed_motor = profile["p_motor_kw"].to_numpy(float)
    path = image_dir / "功率分配时序.png"
    _line_chart(
        path,
        "80小时飞推控功率分配",
        "正电机功率表示电池放电助推；负值表示发动机富余功率充电",
        "任务时间（小时）", "功率（kW）",
        [
            ("发动机/发电机实际功率", time_h, profile["p_engine_kw"].to_numpy(float), COLORS["发动机"]),
            ("电机/电池功率（正助推、负充电）", time_h, signed_motor, COLORS["电机/电池"]),
            ("推进功率需求", time_h, profile["p_propulsion_request_kw"].to_numpy(float), COLORS["推进需求"]),
        ],
        xlim=(0, 80),
        y_format="{:.0f}",
    )
    outputs.append(path)

    path = image_dir / "电池SOC时序.png"
    soc_series = [
        ("SOC", time_h, profile["soc"].to_numpy(float), COLORS["SOC"]),
    ]
    _line_chart(
        path,
        "电池荷电状态（SOC）",
        "图中包含安全下限与控制目标；SOC为0到1之间的比例",
        "任务时间（小时）", "SOC",
        soc_series,
        xlim=(0, 80),
        ylim=(0.0, 1.0),
        y_format="{:.0%}",
        horizontal_lines=[
            (0.20, "电池模型SOC下限 20%", COLORS["SOC下限"]),
            (0.55, "巡航阶段参考目标 55%", COLORS["SOC目标"]),
        ],
    )
    outputs.append(path)

    path = image_dir / "速度高度时序.png"
    _line_chart(
        path,
        "飞行状态时序：速度",
        "纵向低阶飞行动力学闭环输出",
        "任务时间（小时）", "速度（m/s）",
        [("飞行速度", time_h, profile["v_mps"].to_numpy(float), COLORS["速度"])],
        xlim=(0, 80),
        y_format="{:.1f}",
    )
    outputs.append(path)

    path = image_dir / "高度时序.png"
    _line_chart(
        path,
        "飞行状态时序：高度",
        "纵向低阶飞行动力学闭环输出",
        "任务时间（小时）", "高度（m）",
        [("飞行高度", time_h, profile["h_m"].to_numpy(float), COLORS["高度"])],
        xlim=(0, 80),
        y_format="{:.0f}",
    )
    outputs.append(path)

    path = image_dir / "任务阶段统计.png"
    _phase_chart(path, profile)
    outputs.append(path)

    scan_path = output_dir / "多目标权重扫描与安全裕度筛选.csv"
    if scan_path.exists():
        path = image_dir / "统一归一化权重扫描.png"
        _weight_scan_chart(path, pd.read_csv(scan_path))
        outputs.append(path)

    disturbed = profile.loc[profile["disturbance_drag_n"] > 0.0, "time_s"]
    if not disturbed.empty:
        event_start = float(disturbed.min())
        event_end = float(disturbed.max())
        response_window = profile.loc[
            (profile["time_s"] >= event_start - 120.0)
            & (profile["time_s"] <= event_end + 900.0)
        ]
        response_path = image_dir / "扰动恢复响应.png"
        speed_offset = response_window["disturbance_delta_v_mps"].to_numpy(float)
        peak_offset = float(np.nanmax(np.abs(speed_offset)))
        recovery_metrics = json.loads(
            (output_dir / "NSGA-II与MOEA-D择优可行候选.json").read_text(encoding="utf-8")
        )["independent_replay"]["response_metrics"]
        response_ylim = max(0.05, peak_offset * 1.25)
        _line_chart(
            response_path,
            "巡航等效阻力扰动后的速度响应",
            (
                f"等效附加阻力 100 N、持续 10 分钟；峰值偏移 {peak_offset:.3f} m/s，"
                f"回归 {recovery_metrics['recovery_time_s']:.0f} s；判据为相对扰动前状态 ±0.8 m/s"
            ),
            "扰动开始后的时间（分钟）",
            "相对扰动前速度变化（m/s）",
            [(
                "速度偏移",
                (response_window["time_s"].to_numpy(float) - event_start) / 60.0,
                response_window["disturbance_delta_v_mps"].to_numpy(float),
                COLORS["速度"],
            )],
            xlim=(-2.0, max(15.0, (event_end + 900.0 - event_start) / 60.0)),
            ylim=(-response_ylim, response_ylim),
            y_format="{:.2f}",
        )
        outputs.append(response_path)

    path = image_dir / "总体结果摘要.png"
    _summary_chart(path, comparison, output_dir / "NSGA-II与MOEA-D择优可行候选.json")
    outputs.append(path)

    contact = image_dir / "H6中文图表总览.png"
    _contact_sheet(outputs, contact)
    outputs.append(contact)
    return [str(path) for path in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description="为 H=6 PTDL 集成优化结果生成中文可视化图表。")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in generate_visualizations(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
