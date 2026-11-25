"""Build the Word stage report from the reproducible MOEA/D validation matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ptdl_moead_stage_validation_results"
OUT = ROOT / "PTDL MOEA D 阶段性进展试验报告.docx"


def font(run, name: str = "宋体", size: float = 10.5, bold: bool = False, color=None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_margins(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", 80), ("start", 95), ("bottom", 80), ("end", 95)):
        tag = qn(f"w:{edge}")
        node = tc_mar.find(tag)
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    element = OxmlElement("w:tblHeader")
    element.set(qn("w:val"), "true")
    tr_pr.append(element)


def remove_title_border(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "nil")
    borders.append(bottom)
    p_pr.append(borders)


def add_heading(doc: Document, text: str, level: int) -> None:
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    paragraph.paragraph_format.space_after = Pt(4)
    for run in paragraph.runs:
        font(run, "黑体" if level <= 2 else "宋体", 14 if level == 1 else 12 if level == 2 else 11, True)


def add_body(doc: Document, text: str, lead: str | None = None) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.line_spacing = 1.35
    paragraph.paragraph_format.space_after = Pt(5)
    if lead and text.startswith(lead):
        run = paragraph.add_run(lead)
        font(run, size=10.5, bold=True)
        text = text[len(lead):]
    run = paragraph.add_run(text)
    font(run)


def add_bullet(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.2
    font(paragraph.add_run(text), size=10)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float], size: float = 8.5) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    for index, label in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(label)
        font(run, size=size, bold=True, color=(255, 255, 255))
        shade(cell, "1F4E79")
        set_margins(cell)
        cell.width = Cm(widths[index])
    set_repeat_header(table.rows[0])
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cell = cells[index]
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if index < 3 else WD_ALIGN_PARAGRAPH.LEFT
            font(paragraph.add_run(str(value)), size=size)
            if row_index % 2:
                shade(cell, "F3F7FA")
            set_margins(cell)
            cell.width = Cm(widths[index])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_figure(doc: Document, path: Path, caption: str, width: float = 15.5) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Cm(width))
    caption_paragraph = doc.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(6)
    font(caption_paragraph.add_run(caption), size=9)


def configure(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    font(header.add_run("PTDL 项目阶段性进展试验报告"), size=8.5)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(footer.add_run("PTDL 飞推控耦合多目标优化"), size=8)


def objective_text(summary: pd.DataFrame, resolution: int, column: str) -> float:
    return float(summary.loc[summary["resolution_H"] == resolution, column].iloc[0])


def main() -> None:
    summary = pd.read_csv(DATA / "validation_resolution_summary.csv")
    run_summary = pd.read_csv(DATA / "validation_run_summary.csv")
    history = pd.read_csv(DATA / "validation_convergence_history.csv")
    final = history.loc[history["generation"] == 60]
    first_full = history.loc[history["feasible_ratio"] >= 1.0].groupby(["resolution", "seed"])["generation"].min()
    first_stats = first_full.groupby(level=0).agg(["mean", "min", "max"])

    doc = Document()
    configure(doc)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    remove_title_border(title)
    font(title.add_run("PTDL MOEA D 阶段性进展试验报告"), "黑体", 20, True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(12)
    font(subtitle.add_run("飞推控耦合三目标优化算法验证矩阵 60 代 3 次重复"), size=11)

    add_heading(doc, "一 阶段结论", 1)
    add_body(doc, "本阶段在统一的 PTDL 80 h 飞推控联合优化问题上完成了 MOEA/D 参数分辨率验证。试验以 H=6、H=8 和 H=12 三种权重分辨率为对象，每种设置采用 3 个独立随机种子并运行 60 代，共完成 9 次独立优化和 492 个最终样本评价。全部最终样本在当前评价模型下满足 T_endurance >= 80 h 的硬约束。")
    add_body(doc, "主要结论：H=12 在相同迭代代数下提供了最密集的 Pareto 覆盖，每次运行平均得到 70.0 个非支配解，跨种子标准差仅为 0.82；H=6 的可行性建立最快，平均第 2.33 代达到全体可行；H=8 在本轮中获得最小响应指标端点 672.28 s。由此建议将 H=12 作为后续 Pareto 分析与候选解筛选的主配置，将 H=6 保留为模型调试和快速可行性检查配置。", lead="主要结论：")
    add_body(doc, "结论边界：上述结果证明了当前低阶飞推控耦合算例中 MOEA/D 优化流程的可运行性、约束处理能力和重复性，不能直接等同于 PTDL 实机的工程定型性能。发动机燃油消耗图、电池效率与热模型、结构质量分布及高保真气动极线仍为暂定或待校核输入。", lead="结论边界：")

    add_heading(doc, "二 试验目的与统一问题定义", 1)
    add_body(doc, "试验目的不是重复证明存在单个 80 h 解，而是在完全一致的任务、目标、约束和迭代预算下，检验 MOEA/D 权重分割数 H 对可行性收敛速度、三目标折中解覆盖和随机重复性的影响，为后续算法配置选择提供可复现实证。")
    add_table(doc, ["项目", "统一设置", "说明"], [
        ["任务与约束", "PTDL 全任务剖面，T_endurance >= 80 h", "80 h 为硬约束，不作为第四个目标"],
        ["决策变量", "23 个", "几何、质量分配、留空速度及分阶段发动机/电机功率控制量"],
        ["优化目标", "J_energy、J_stability、J_response", "分别表征总能耗、跟踪稳定性和任务响应代价"],
        ["算法", "MOEA/D 邻域分解", "同一交叉、变异、替换和约束处理逻辑"],
        ["重复性", "每个 H 使用 3 个随机种子", "种子为 20260950、20260951、20260952"],
        ["迭代预算", "每次 60 代", "仅改变 H，避免与迭代代数混淆"],
    ], [3.0, 5.2, 8.0])
    add_body(doc, "H 对应权重单纯形的离散程度。三目标问题中，权重子问题数为 (H+1)(H+2)/2，因此 H=6、8、12 分别对应 28、45、91 个子问题。H 越大，Pareto 前沿上的取样理论上越密，但单次计算成本和达到全体可行所需代数通常也会增加。")

    add_heading(doc, "三 试验矩阵与数据完整性", 1)
    matrix_rows = []
    for _, row in summary.iterrows():
        h = int(row["resolution_H"])
        first = first_stats.loc[h]
        matrix_rows.append([
            f"H={h}", str(int(row["weights_per_run"])), str(int(row["runs"])),
            str(int(row["final_samples"])), f"{row['final_feasible_ratio']:.0%}",
            f"{first['mean']:.2f} ({int(first['min'])}-{int(first['max'])})",
            f"{row['mean_run_pareto_count']:.1f} +/- {row['std_run_pareto_count']:.2f}",
        ])
    add_table(doc, ["配置", "权重数", "独立运行", "最终样本", "最终可行率", "全体可行代数 平均 最小 最大", "单次非支配解 平均 标准差"], matrix_rows, [1.25, 1.15, 1.15, 1.25, 1.25, 4.0, 4.2], 8.1)
    add_body(doc, "9 次运行均顺利结束，未发生优化器中断、空可行集或约束违反回升。随着 H 从 6 增加到 12，单次运行的权重子问题由 28 增至 91，最终样本总数相应增加；H=12 的非支配解数量最多且种子间波动最小。")
    add_figure(doc, DATA / "validation_feasibility_convergence.png", "图 1 不同 H 的可行率收敛曲线 实线为三次重复均值 阴影为最小最大包络", 15.4)

    doc.add_page_break()
    add_heading(doc, "四 收敛性与随机重复性结果", 1)
    add_body(doc, "可行性曲线表明三种配置均能迅速进入可行域。H 较小时每代处理的子问题较少，因此 H=6 首先达到全体可行；H=12 的平均全体可行代数为 6.00 代，仍明显早于 60 代终止条件。由此可将后续 60 代的大部分计算预算用于目标改进与 Pareto 覆盖，而不是寻找可行解。")
    add_figure(doc, DATA / "validation_objective_convergence.png", "图 2 三个目标的收敛历史 实线为三次重复均值 阴影为种子包络", 16.0)
    add_body(doc, "终代最优端点的跨种子标准差显示：H=12 的最小能耗端点均值为 20004.28 MJ，标准差为 143.36 MJ；H=8 的最小响应端点均值为 685.43 s，但标准差为 17.01 s；H=6 的响应端点离散度最低，但解集覆盖较少。这里的“最优端点”分别来自各自最有利的候选解，不能将三个端点拼接为同一架飞机或同一控制策略。")

    add_heading(doc, "五 Pareto 覆盖与 H 的适用性比较", 1)
    endpoint_rows = []
    for _, row in summary.iterrows():
        endpoint_rows.append([
            f"H={int(row['resolution_H'])}",
            f"{row['pooled_pareto_count']:.0f}",
            f"{row['J_energy_mj_pareto_min']:.2f}",
            f"{row['J_stability_pareto_min']:.4f}",
            f"{row['J_response_s_pareto_min']:.2f}",
        ])
    add_table(doc, ["配置", "该 H 汇合非支配解", "最小 J_energy MJ", "最小 J_stability", "最小 J_response s"], endpoint_rows, [2.0, 3.3, 3.5, 3.5, 3.5], 8.5)
    add_body(doc, "H=12 的汇合非支配解数量为 142，明显多于 H=8 的 72 和 H=6 的 49，说明其对折中区域的覆盖更充分。与此同时，H=12 并未在三个目标端点上全部占优：本轮最小能耗和最小稳定性端点分别来自 H=6，最小响应端点来自 H=8。这是多目标问题的正常现象，不能以单个端点宣称某一 H 在所有维度上更优。")
    add_body(doc, "推荐策略：后续正式 Pareto 搜索使用 H=12，以获得更密集、重复性更好的候选集；需要快速检查模型修改是否仍可行时使用 H=6；若后续任务更偏重响应指标，可将 H=8 与 H=12 同时运行并对响应端的候选解进行局部加密。", lead="推荐策略：")
    add_figure(doc, DATA / "validation_pooled_pareto_3d.png", "图 3 九次独立运行汇合后的全局非支配解点云 不使用三角插值", 12.5)

    add_heading(doc, "六 阶段成果与可交付数据", 1)
    add_bullet(doc, "已形成统一的 H=6、H=8、H=12、60 代、3 随机种子的 MOEA/D 验证矩阵，共 9 次独立运行。")
    add_bullet(doc, "已导出全体最终样本、每代收敛历史、单次运行摘要、H 分辨率摘要和汇合非支配解集，可复算图 1 至图 3。")
    add_bullet(doc, "已验证 80 h 硬约束在当前算例与决策边界内可稳定满足；492 个最终样本的最终可行率为 100%。")
    add_bullet(doc, "已给出 H=12 作为后续 Pareto 主配置的依据：候选覆盖最多，单次非支配解数量的跨种子离散度最小。")

    add_heading(doc, "七 当前边界与下一阶段工作", 1)
    add_body(doc, "当前模型为面向算法验证的低阶联合尺寸质量控制模型。其价值在于已经形成“任务剖面、飞行与推进能量计算、三目标评价、约束处理、MOEA/D 搜索和结果审计”的完整可运行闭环；其限制在于发动机燃油消耗图、混动系统比功率和比能量、电池热约束、结构质量、真实气动极线及控制响应模型仍需用项目数据和高保真分析替换或校核。")
    add_bullet(doc, "以 H=12 为主配置，将迭代代数扩展至 100 代以上，并增加至不少于 5 个随机种子，评估 Pareto 解集是否继续扩展。")
    add_bullet(doc, "对 H=12 汇合 Pareto 解开展聚类和偏好筛选，提取能耗优先、响应优先和均衡型代表方案，并输出对应的几何、质量和分阶段功率控制序列。")
    add_bullet(doc, "接入发动机耗油图、电池效率和热边界，以及 OpenVSP VSPAERO 或更高保真气动结果，重新进行 80 h 约束校核。")
    add_bullet(doc, "将当前响应代理指标替换或标定为可验证的闭环控制动态响应指标，再对 Pareto 关系进行复验。")

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
