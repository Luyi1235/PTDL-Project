from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "PTDL飞推控耦合模型与MOEA_D优化阶段进展报告.docx"
PLOT = ROOT / "ptdl_report_86h_endurance_history.png"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_text(cell, text: str, bold=False, size=8.5, color=None, align=None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    if align is not None:
        p.alignment = align
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_cell_margins(cell)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_table_widths(table, widths_cm) -> None:
    for row in table.rows:
        for idx, width in enumerate(widths_cm):
            row.cells[idx].width = Cm(width)


def style_table(table, header_fill="1F4E79") -> None:
    table.style = "Table Grid"
    for cell in table.rows[0].cells:
        set_cell_shading(cell, header_fill)
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)
    set_repeat_table_header(table.rows[0])
    for row_index, row in enumerate(table.rows[1:], start=1):
        if row_index % 2 == 0:
            for cell in row.cells:
                set_cell_shading(cell, "F3F7FA")


def add_table(doc, headers, rows, widths_cm, font_size=8.5) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, size=font_size, color=(255, 255, 255), align=WD_ALIGN_PARAGRAPH.CENTER)
    for values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(values):
            align = WD_ALIGN_PARAGRAPH.CENTER if i == 0 and len(str(value)) < 18 else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cells[i], value, size=font_size, align=align)
        prevent_row_split(table.rows[-1])
    set_table_widths(table, widths_cm)
    style_table(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_paragraph(doc, text: str, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.35
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.name = "宋体"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        text = text[len(bold_lead):]
    r = p.add_run(text)
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(10.5)
    return p


def add_bullet(doc, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.25
    r = p.add_run(text)
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(10)


def add_heading(doc, text: str, level: int) -> None:
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(5)
    for r in p.runs:
        r.font.name = "黑体" if level <= 2 else "宋体"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体" if level <= 2 else "宋体")
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.font.size = Pt(14 if level == 1 else 12 if level == 2 else 11)
    return p


def create_plot() -> None:
    data = pd.read_csv(ROOT / "ptdl_endurance_extension_results" / "ptdl_endurance_extension_time_history.csv")
    time_h = data["time_s"].to_numpy() / 3600.0
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(time_h, data["h_m"], color="#1F4E79", lw=1.4, label="Altitude")
    axes[0].set_ylabel("Altitude\n(m)")
    axes[0].grid(alpha=0.25)
    ax0b = axes[0].twinx()
    ax0b.plot(time_h, data["v_mps"], color="#C55A11", lw=1.1, label="Speed")
    ax0b.set_ylabel("Speed (m/s)", color="#C55A11")
    axes[1].plot(time_h, data["mass_kg"], color="#548235", lw=1.4)
    axes[1].set_ylabel("Total mass\n(kg)")
    axes[1].grid(alpha=0.25)
    ax1b = axes[1].twinx()
    ax1b.plot(time_h, data["soc"], color="#7030A0", lw=1.1)
    ax1b.set_ylim(0, 1)
    ax1b.set_ylabel("SOC", color="#7030A0")
    axes[2].plot(time_h, data["p_engine_kw"], color="#7F6000", lw=1.2, label="Engine")
    axes[2].plot(time_h, data["p_motor_kw"], color="#5B9BD5", lw=1.1, label="Motor")
    axes[2].set_ylabel("Power\n(kW)")
    axes[2].set_xlabel("Mission time (h)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right", frameon=False, ncol=2)
    fig.savefig(PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def configure_doc(doc: Document) -> None:
    sec = doc.sections[0]
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.25)
    sec.right_margin = Cm(2.25)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    for name in ("Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.color.rgb = RGBColor(0, 0, 0)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = header.add_run("PTDL 项目阶段进展报告")
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(8.5)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("PTDL 飞推控耦合模型与 MOEA/D 优化阶段成果")
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(8)


def main() -> None:
    create_plot()
    doc = Document()
    configure_doc(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("PTDL 飞推控耦合模型与 MOEA/D 优化阶段进展报告")
    r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor(0, 0, 0)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(14)
    r = subtitle.add_run("面向混合动力超长航时无人机全任务剖面优化的阶段汇报")
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(11)

    add_heading(doc, "一 阶段结论", 1)
    add_paragraph(doc, "本阶段已按导师讨论结论，将多目标进化算法的主线由 NSGA-II 调整为 MOEA/D，并打通了“任务剖面 - 飞行动力学 - 推进与能量管理 - 三目标评价 - 强约束判定 - 联合设计优化 - 结果导出”的可运行计算链。该链条的用途是作为 PTDL 2.2 后续算法开发与验证的可复现实验算例，而非替代真实飞行器的最终高保真数字孪生。")
    add_paragraph(doc, "对标 ULTRA 72 h 公开续航锚点后，已构建带起飞爬升、进入留空、长时间留空、下降和着陆收尾的 PTDL 耐久任务剖面，并将基线在该完整剖面上标定为 72 h。随后以“续航时间不低于 80 h”为硬约束，采用 23 个几何、质量分配和分阶段控制变量进行 MOEA/D 联合优化，获得 80 h 可行候选解；在相同模型和候选解上将任务延拓至 86 h，仍保持全部已实现约束可行。")
    add_paragraph(doc, "结论的工程含义需限定为：在当前低阶物理模型、公开数据锚点和暂定推进参数下，80 h 指标已经具备可行性证据；其数值不等同于已完成 ULTRA 实物复现，也不能直接作为工程定型参数。后续必须以真实发动机耗油图、电池参数、结构质量、气动极线和任务要求完成高保真校核。", bold_lead="结论的工程含义需限定为：")

    add_heading(doc, "二 研究问题的落实与边界", 1)
    add_paragraph(doc, "本阶段将原始表述细化为一个可计算的联合设计与控制问题。优化的直接对象不是 CD0 或奥斯瓦尔德效率 e 等气动结果参数，而是翼面积、展弦比、翼型平面形状参数、机身尺度、螺旋桨直径、载荷质量、混动系统质量、巡航速度以及各飞行阶段发动机功率上限和电机助推功率。气动阻力、诱导效率、燃油容量和起飞重量闭合均由这些设计变量和模型关系派生得到。")
    add_paragraph(doc, "早期算法原型采用三个核心目标：总能量消耗 J_energy、稳定性与跟踪品质 J_stability、任务响应代价 J_response。80 h 不是与三目标并列的第四个优化目标，而是强制可行性条件 T_endurance >= 80 h；同时保留终端与最低 SOC、速度与高度包线、发动机和电机功率及变化率、迎角、载荷因子、燃油、翼展、失速速度、燃油容积、最小载荷、最大起飞重量和机身细长比等约束。")
    doc.add_page_break()
    add_table(doc,
        ["对象", "本阶段确定的内容", "说明"],
        [
            ["算法主线", "MOEA/D", "按权重向量把三目标问题分解为多个标量子问题，适合后续按任务偏好选择解。"],
            ["优化目标", "能量、稳定性、响应", "沿用 2.2.1 评价函数接口，数值均由任务全历程计算。"],
            ["核心硬约束", "T_endurance >= 80 h", "候选解必须完整完成任务，不以不可行解的目标函数优劣替代续航达标。"],
            ["设计变量", "23 个联合变量", "10 个几何与质量变量，1 个 ISR 巡航速度，6 个发动机功率上限，6 个电机助推量。"],
            ["模型定位", "可运行低阶物理算例", "用于算法验证、敏感性研究和变量筛选；尚非结构、CFD 和推进试验数据闭环的数字孪生。"],
        ], [2.5, 5.1, 8.0], 8.8)

    add_heading(doc, "三 已完成的建模与算例构建", 1)
    add_heading(doc, "3.1 飞推控耦合模型", 2)
    add_paragraph(doc, "飞行部分采用固定翼纵向点质量模型，以航程、几何高度、真空速、航迹角、剩余燃油和电池 SOC 为状态。模型逐时间步由高度和速度跟踪误差形成推力需求，再由机翼面积、展弦比、阻力系数和空气密度计算升阻力，最终积分得到状态历程。该做法保留了航迹、气动载荷、速度高度和燃油质量变化对推进需求的耦合。")
    add_paragraph(doc, "推进与能量部分采用螺旋桨推进效率、发动机功率上限、发动机效率图和电机助推功率共同决定可用推进功率。发动机燃油流量由轴功率和效率图计算；电池仅在电机助推或充电时改变 SOC，并受功率、SOC、温度和电流接口约束。控制部分以分阶段功率上限和电机助推为控制量，通过跟踪误差、功率变化率和飞行状态边界对控制策略施加评价与约束。")
    add_table(doc,
        ["耦合环节", "计算关系", "在优化中的作用"],
        [
            ["飞行", "升阻力、重力与推力平衡，积分更新高度、速度、航程", "不同翼面与阻力水平改变所需功率和燃油消耗。"],
            ["推进", "发动机功率经效率图换算燃油流量，螺旋桨效率换算推力", "决定留空阶段的单位时间燃油消耗。"],
            ["能量管理", "电机功率引起电池能量与 SOC 变化", "在爬升或瞬态阶段支撑推进需求，并受 SOC 硬约束。"],
            ["控制与任务", "分阶段功率上限和速度目标驱动状态跟踪", "将任务剖面转化为可优化的控制量序列。"],
            ["总体设计", "尺寸和质量分配派生翼展、燃油容量、阻力和 MTOW 闭合", "避免直接把结果参数当作无约束优化变量。"],
        ], [2.1, 6.7, 6.8], 8.6)

    add_heading(doc, "3.2 原始任务文件审查与耐久任务重构", 2)
    add_paragraph(doc, "已解析 flight_profile_ultra_70h.json。该文件名虽包含 70 h，但按其中的距离、70 m/s 巡航速度和时间字段直接积分，初始任务时长约为 21.21 h，且核心巡航高度约 7600 m、速度为 70 m/s。因此不能将其原样视作 70 h 或 80 h 留空任务，更不能要求任何优化算法在固定 MTOW 下直接完成 80 h。对原始文件进行严格控制审计时，模型出现燃油耗尽，最长完成时间约 18.36 h，这一结果用于识别任务定义不一致，而不是评价 MOEA/D 的算法能力。")
    add_paragraph(doc, "在保留“爬升 - 进入任务高度 - 长时间留空 - 下降 - 进近 - 收尾”结构的前提下，新增 ptdl_80h_endurance_profile.json。新剖面将留空段设在 3000 m 和约 36.01 m/s，优化时允许 ISR 巡航速度在 30 至 42 m/s 的范围内变化；全程目标总时长为 80 h。这个修改使任务定义与“超长航时留空”研究目标一致，并为 80 h 硬约束提供了明确的积分口径。")

    add_heading(doc, "四 基线校准与优化方法", 1)
    add_heading(doc, "4.1 72 h 基线校准", 2)
    add_paragraph(doc, "首先使用 ULTRA 等效留空基准模型核对公开锚点，再在完整 PTDL 任务剖面上进行任务层级标定。为使基线在完整剖面中达到公开的 72 h 续航锚点，对暂定发动机效率图统一施加 1.1398 的校正倍率；该倍率用于消除低阶模型与公开性能锚点之间的系统差异，并在所有后续候选解上保持不变。它不是实测发动机效率，也不意味着已经掌握真实发动机燃油消耗图。")
    add_table(doc,
        ["检查项", "结果", "解释"],
        [
            ["完整剖面基线续航", "72.00 h", "以 ULTRA 公开续航锚点进行任务层级标定。"],
            ["基线燃油质量", "约 490.59 kg", "由固定 MTOW 质量闭合关系得到。"],
            ["基线燃油消耗", "约 491.28 kg", "72 h 时燃油耗尽，因而未满足 80 h 硬约束。"],
            ["全程效率校正倍率", "1.1398", "暂定效率图的统一校正，仅用于基线一致性。"],
            ["基线任务航程", "约 9380.58 km", "包含任务剖面中的各阶段，不等同于纯留空航程。"],
        ], [4.0, 3.1, 8.5], 8.6)
    add_paragraph(doc, "此前出现的约 54.08 h 基线并不是 ULTRA 性能复现失败，而是将纯留空模型的 72 h 标定与未经完整任务校准的爬升、下降、混动质量和效率假设混合使用的结果。现已改为在完整任务剖面上统一校准，基线口径已经固定为 72 h。")

    add_heading(doc, "4.2 MOEA/D 实现与约束处理", 2)
    add_paragraph(doc, "MOEA/D 将三目标问题按权重单纯形上的多个权重向量分解为若干子问题。当前原型使用 H = 2，得到 6 个权重子问题；每个子问题以相同的飞推控任务仿真返回三目标值和全部约束值。比较候选解时采用可行性优先规则：可行解优先于不可行解；两者均可行时比较相应标量化目标；两者均不可行时比较总违反度。这样可以确保 80 h 硬约束不会被目标函数中的局部改进掩盖。")
    add_paragraph(doc, "H = 2、6 代的本次运行是用于证明端到端计算链能够产生可行解的快速原型，而不是最终的 Pareto 前沿收敛试验。末代 6 个子问题均已达到可行，且候选解高度相近，说明需要在下一阶段增加权重分辨率、代数、重复随机种子和局部搜索，才能获得分布充分、可用于决策的 Pareto 解集。")

    doc.add_page_break()
    add_heading(doc, "五 当前优化结果", 1)
    add_heading(doc, "5.1 80 h 可行候选解", 2)
    add_paragraph(doc, "在固定 MTOW、允许翼展增大、允许燃油与载荷及混动质量再分配、暂定最小载荷 100 kg 和最大翼展 32 m 的边界下，联合 MOEA/D 获得了一个 80 h 可行候选解。该解满足已实现的 80 h、终端燃油、SOC、翼展、失速速度、燃油容积、MTOW 闭合等约束。由于载荷和混动质量均贴近其下限，结果反映了当前约束和目标下的优化倾向，不能直接解释为推荐构型。")
    add_table(doc,
        ["类别", "候选解参数", "结果或裕度"],
        [
            ["机翼", "S = 31.99 m2，AR = 28.92，展弦长度约 30.42 m", "距 32 m 翼展上限约 1.58 m。"],
            ["平面形状", "锥度 0.323，后掠 3.03 deg，翼尖扭转 0.22 deg", "由尺寸变量派生低阻巡航构型。"],
            ["机身与螺旋桨", "机长 8.24 m，机身直径 0.707 m，桨径 2.153 m", "满足当前细长比和布置边界。"],
            ["质量分配", "载荷 100 kg，混动系统 0 kg，燃油约 534.84 kg", "当前输入边界下燃油质量保有裕度约 234.84 kg。"],
            ["关键巡航控制", "ISR 巡航 30.0 m/s，留空发动机上限 100 kW", "电机在留空段为 0；长下降段约 0.30 kW。"],
            ["三目标", "J_energy = 21051.65 MJ，J_stability = 62.43，J_response = 3550 s", "为当前评价函数下的三目标值，不能跨模型直接比较。"],
            ["续航与终端状态", "任务完成 80.00 h，终端燃油 45.53 kg，终端 SOC 0.50", "T_endurance >= 80 h 满足，全部已实现约束可行。"],
        ], [3.0, 6.4, 6.2], 8.4)

    add_heading(doc, "5.2 86 h 延拓验证", 2)
    add_paragraph(doc, "将上述 80 h 可行候选解直接代入同一模型，仅把耐久留空段延长至 86 h，计算得到任务完整完成、终端燃油约 12.66 kg、终端 SOC 0.50、燃油消耗约 522.44 kg，所有已实现约束仍为可行。该算例说明当前优化结果对 80 h 门槛并非恰好卡边，但它仍只是在当前假设集合中的延拓验证，而非新一轮以 86 h 为目标的最优设计。")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(PLOT), width=Cm(15.0))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(7)
    r = cap.add_run("图 1  86 h 延拓验证的高度 速度 总质量 SOC 与功率历程")
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(9)

    add_heading(doc, "六 代码文件与已实现功能", 1)
    add_paragraph(doc, "以下程序均位于项目目录根目录，数据结果分别写入对应的 results 文件夹。评价接口与优化接口保持分离，因此后续替换气动、发动机或电池子模型时，不需要改写 MOEA/D 的主流程。")
    add_table(doc,
        ["文件", "已实现功能", "当前用途"],
        [
            ["ptdl_2_2_1_evaluator.py", "三目标评价、硬约束、固定参考归一化、权重生成与偏好后处理", "承接 2.2.1 的评价合同。"],
            ["ptdl_2_2_2_optimizer.py", "MOEA/D、约束优先比较、23 变量联合问题、结果导出", "主优化入口与 80 h 联合设计优化。"],
            ["ptdl_flight_propulsion_control.py", "固定翼纵向飞行、气动阻力、推进效率、发动机效率图、SOC 更新", "飞推控耦合仿真内核。"],
            ["ptdl_airframe_design.py", "尺寸驱动几何、质量闭合、燃油容积、翼展与失速约束", "避免直接优化 CD0 和 e 等结果量。"],
            ["ptdl_vspaero_adapter.py", "解析已有 VSPAERO 极线并提取 CL_alpha 和诱导效率", "低阶模型的 VLM 交叉检查接口。"],
            ["ptdl_ultra_benchmark.py", "ULTRA 等效留空公开锚点与 72 h 基准计算", "基线一致性检查。"],
            ["ptdl_full_task_profile.py", "原始 JSON 解析、源任务审计与严格可行性测试", "识别原始 70 h 文件的时长与工况不一致。"],
            ["ptdl_80h_endurance_profile.json", "重构后的 80 h 任务阶段定义", "优化与延拓验证的任务输入。"],
            ["ptdl_80h_endurance_profile.py", "完整任务 72 h 校准、耐久评估、基线与延拓结果导出", "全任务剖面的统一执行接口。"],
            ["ptdl_fixed_mtow_sizing.py", "固定 MTOW 的初步几何质量筛选", "联合优化前的可行域摸底。"],
            ["test_ptdl_*.py", "评价器、优化器、飞推控、任务和基准的单元与回归测试", "当前核心测试已通过。"],
        ], [4.3, 7.4, 3.9], 8.1)

    add_heading(doc, "七 已产出文件与可复核证据", 1)
    add_table(doc,
        ["目录", "主要文件", "可复核内容"],
        [
            ["ptdl_80h_endurance_baseline_results", "baseline_evaluation.csv，time_history.csv，metadata.json", "72 h 全任务基线及校准后的状态历程。"],
            ["ptdl_80h_joint_moead_results", "convergence.csv，feasible_pareto.csv，final_population.csv，least_violating.csv，time_history.csv，metadata.json", "MOEA/D 收敛、末代解集、80 h 可行候选解与任务历程。"],
            ["ptdl_endurance_extension_results", "extension_evaluation.csv，time_history.csv，metadata.json", "80 h 候选解的 86 h 延拓可行性验证。"],
            ["ptdl_fixed_mtow_sizing_results", "fixed_mtow_sizing_screen.csv", "初步尺寸与质量分配筛选结果。"],
            ["ptdl_source_profile_moead_results", "原始任务审计结果", "原始 JSON 未按 70 h 定义的可复核证据。"],
        ], [4.4, 7.2, 4.0], 8.4)

    add_heading(doc, "八 当前不足与下一步工作", 1)
    add_paragraph(doc, "当前模型的优势是变量、状态、约束和优化接口已经闭合，可以立即开展 MOEA/D 算法参数、权重和可行性研究；其限制是气动阻力、发动机效率图、混动系统比功率比能量和质量模型仍含暂定值，VSPAERO 仅用于已有极线的交叉检查，尚未形成 OpenVSP/VSPAERO、RCAIDE、CFD 或试验数据驱动的高保真闭环。因而本阶段的 80 h 与 86 h 应作为“模型内可行性”而非“实物性能承诺”。")
    add_bullet(doc, "补齐真实或可追溯的发动机燃油消耗图、螺旋桨效率图、电池容量和倍率热边界、混动系统质量功率数据，并替换当前暂定参数。")
    add_bullet(doc, "围绕候选几何构建 OpenVSP 参数化模型，批量获取 VSPAERO 极线；对关键构型采用更高保真气动或推进分析进行校核。")
    add_bullet(doc, "将 MOEA/D 从 H = 2、6 代的流程验证扩展到足够的权重分辨率、迭代代数、随机重复与敏感性分析，形成有离散度的 Pareto 解集。")
    add_bullet(doc, "明确实际最低任务载荷、最大允许翼展、任务高度速度窗口、发动机额定功率与 MTOW 后，重新进行边界收敛和鲁棒性分析。")
    add_bullet(doc, "在控制层引入直接配点或伪谱离散，将目前的分阶段功率控制扩展为连续控制量优化，并与低阶 MOEA/D 外层设计变量协同。")

    add_heading(doc, "九 建议向导师汇报的决策点", 1)
    add_paragraph(doc, "建议本次汇报确认三项决策：第一，认可 MOEA/D 作为 2.2.2 的主线算法，NSGA-II 保留为对照而非主实现；第二，认可当前低阶耦合模型作为算法算例和可行性筛选平台，并允许在取得部件数据后替换子模型；第三，尽快明确最低有效载荷、翼展上限和推进系统参数范围。前两项决定后续代码架构，第三项直接决定 80 h 可行域是否仍然存在。")

    add_heading(doc, "附 注", 1)
    add_paragraph(doc, "本报告的引用基线来自项目已有的 2.2.1 评价函数交付件、2.2.2 算法文献调研与适配报告、Tilt-Wing eVTOL Takeoff Trajectory Optimization 参考论文、学长代码与论文，以及项目目录内的任务 JSON 和本阶段运行结果。ULTRA 的 72 h 指标在此作为公开性能锚点使用。")

    doc.core_properties.title = "PTDL 飞推控耦合模型与 MOEA D 优化阶段进展报告"
    doc.core_properties.subject = "PTDL 2.2 阶段进展"
    doc.core_properties.author = "PTDL 项目组"
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
