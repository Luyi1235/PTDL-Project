from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


OUTPUT = Path(r"D:\LDTP\申请书1.2与2.2完成情况对照表.docx")

ROWS = [
    (
        "1.2（1）构建混合动力系统与飞行控制策略协同模型",
        "已建立发动机、电池、电机模型，并与飞行任务、控制策略及功率分配接入闭环仿真，参数采用当前估计值。",
        "部分完成：姿态、轨迹等飞行控制环节的覆盖仍需完善。",
    ),
    (
        "1.2（2）发展协同控制策略与智能调度算法",
        "已实现 MPC、规则控制、模糊控制及能量管理策略，并开展仿真比较。",
        "部分完成：神经网络等方法及不同飞行状态下的系统性比较尚未全部完成。",
    ),
    (
        "1.2（2）验证起降、低速高升力、巡航及滑翔状态",
        "已开展多个飞行阶段的任务仿真，并分析响应、稳定性和能耗表现。",
        "部分完成：典型状态已有仿真，完整覆盖和统一比较尚未全部完成。",
    ),
    (
        "1.2（3）多工况综合验证与优化迭代",
        "已开展多工况仿真、参数调整、扰动分析和多目标优化迭代。",
        "部分完成：仍需扩大工况并继续迭代。",
    ),
    (
        "2.2（1）建立多目标性能指标体系与数学模型",
        "已建立能量、稳定性、响应时间三个正式优化目标及约束，并补充燃油、末端 SOC 等诊断指标。",
        "基本完成：指标间权衡分析可继续完善。",
    ),
    (
        "2.2（2）发展多目标优化算法及代理模型",
        "已实现并比较 NSGA-II、MOEA/D、MOPSO、混合算法；已比较 Kriging、高斯过程等代理模型，并增加 VSPAERO DOE 样本。",
        "部分完成：算法表现和代理精度仍需改进，最新代理模型尚未统一接入优化器。",
    ),
    (
        "2.2（3）建立多目标优化仿真分析平台并开展对比分析",
        "已建立参数配置、优化运行、进度显示和三目标 Pareto 可视化平台，并开展算法对比、参数敏感性分析和优化迭代。",
        "部分完成：完整任务剖面的算法比较及最终算法库仍需完善。",
    ),
]


def set_cell_shading(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=95, start=110, bottom=95, end=110):
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def remove_paragraph_borders(paragraph_properties):
    if paragraph_properties is None:
        return
    borders = paragraph_properties.find(qn("w:pBdr"))
    if borders is not None:
        paragraph_properties.remove(borders)


def set_table_borders(table, color="D9D9D9"):
    properties = table._tbl.tblPr
    borders = properties.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "5")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_run_font(run, size=9.5, bold=False, color="202A33"):
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def main():
    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7)
    section.page_height = Cm(21.0)
    section.top_margin = Cm(1.25)
    section.bottom_margin = Cm(1.25)
    section.left_margin = Cm(1.35)
    section.right_margin = Cm(1.35)

    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.08

    title = document.add_paragraph(style="Title")
    title_style = document.styles["Title"]
    remove_paragraph_borders(title_style._element.pPr)
    remove_paragraph_borders(title._p.pPr)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    title_run = title.add_run("申请书 1.2 与 2.2 完成情况对照表")
    set_run_font(title_run, size=16, bold=True, color="172B3A")

    table = document.add_table(rows=1, cols=3)
    table.autofit = False
    table.style = "Table Grid"
    set_table_borders(table)
    widths = (Cm(6.0), Cm(12.8), Cm(7.0))
    headers = ("申请书条目", "已完成部分", "完成情况")

    for index, (cell, text) in enumerate(zip(table.rows[0].cells, headers)):
        cell.width = widths[index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, "244B63")
        set_cell_margins(cell, top=105, bottom=105)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        set_run_font(run, size=10, bold=True, color="FFFFFF")
    set_repeat_table_header(table.rows[0])

    for row_index, values in enumerate(ROWS):
        cells = table.add_row().cells
        for column_index, (cell, text) in enumerate(zip(cells, values)):
            cell.width = widths[column_index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index % 2:
                set_cell_shading(cell, "F2F6F8")
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.keep_together = True
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.05
            if column_index == 2:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = paragraph.add_run(text)
            set_run_font(run, size=9.2, bold=(column_index == 0), color="202A33")

    document.core_properties.title = "申请书 1.2 与 2.2 完成情况对照表"
    document.core_properties.subject = "推进系统与控制策略协同作用及多目标优化控制算法研究"
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
