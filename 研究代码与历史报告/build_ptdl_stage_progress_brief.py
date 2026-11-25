"""Build a concise Word brief for the recent PTDL substantive work."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PTDL阶段推进简报_飞推控协同优化与鲁棒性验证_更新版.docx"
FIGURES = {
    "joint": ROOT / "ptdl_moead_h6_robust_fuzzy_repeatability_results" / "图16_H6联合优化三随机种子收敛与可行性统计.png",
    "pareto": ROOT / "ptdl_moead_h6_robust_fuzzy_repeatability_results" / "图17_H6联合优化三随机种子Pareto解集.png",
    "aero": ROOT / "ptdl_vsp_low_speed_high_ar_validation_results" / "图14_低速高展弦比候选VSPAERO气动极线.png",
    "robust": ROOT / "ptdl_1_2_3_robustness_validation_results" / "图12_1_2_3任务完成率与80小时鲁棒性.png",
    "adaptive": ROOT / "ptdl_1_2_3_adaptive_recovery_results" / "图20_风扰补偿与健康自适应调度局部恢复对比.png",
    "repair": ROOT / "ptdl_combined_degradation_refinement_results" / "图22_组合退化续航与失速约束定向修复筛选.png",
    "combined": ROOT / "ptdl_moead_h6_combined_degradation_seeded_results" / "图18_组合退化约束MOEAD收敛与80小时复核.png",
}


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(9)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(9)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(8 if level == 1 else 4)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(14 if level == 1 else 11)
    run.font.color.rgb = RGBColor(0, 0, 0)


def add_body(doc: Document, text: str, bold_lead: str | None = None) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.35
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        lead.bold = True
        lead.font.name = "Microsoft YaHei"
        lead._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        tail = paragraph.add_run(text[len(bold_lead):])
        tail.font.name = "Microsoft YaHei"
        tail._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    else:
        tail = paragraph.add_run(text)
        tail.font.name = "Microsoft YaHei"
        tail._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for run in paragraph.runs:
        run.font.size = Pt(10.5)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(2.05)
    section.right_margin = Cm(2.05)
    style = doc.styles["Normal"]
    style.font.name = "Microsoft YaHei"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    style.font.size = Pt(10.5)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("PTDL 项目阶段推进简报")
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(100, 100, 100)


def add_key_table(doc: Document) -> None:
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    headers = ("推进方向", "本阶段实质性新增", "可供审阅的结果")
    for cell, text in zip(table.rows[0].cells, headers):
        set_cell_shading(cell, "1F4E79")
        set_cell_text(cell, text, bold=True, color="FFFFFF")
    rows = (
        ("1.2.1 闭环模型", "沿用并完善飞行控制、混合动力、螺旋桨、纵向动力学与状态反馈构成的内层物理模型。", "为调度优化与扰动验证提供统一飞推控算例。"),
        ("1.2.2 联合优化", "新增风扰前馈、发动机降额识别和 SOC 储备自适应；5 个模糊参数纳入稳健 MOEA/D。", "局部恢复对比、H=6 收敛带与候选参数记录。"),
        ("1.2.3 鲁棒验证", "组合退化场景直接进入闭环仿真与外层硬约束，开展翼面积-混动质量定向筛选。", "28 组局部工况、组合退化可行域与 80 h 复核。"),
        ("2.2 优化支撑", "扩充 VSPAERO 气动代理模型，并由 2.2.1 评价约束和 2.2.2 MOEA/D 承担外层寻优。", "气动极线、代理误差和组合退化 Pareto 解集。"),
    )
    for index, values in enumerate(rows):
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            if index % 2 == 1:
                set_cell_shading(cell, "EEF4F8")
            set_cell_text(cell, value)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
    widths = (Cm(3.0), Cm(7.1), Cm(5.5))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width


def main() -> None:
    if not all(path.exists() for path in FIGURES.values()):
        missing = [str(path) for path in FIGURES.values() if not path.exists()]
        raise FileNotFoundError("Missing report figures: " + "; ".join(missing))
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    title_run = title.add_run("PTDL 飞推控协同优化阶段推进简报")
    title_run.font.name = "Microsoft YaHei"
    title_run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    title_run.font.size = Pt(20)
    title_run.font.color.rgb = RGBColor(0, 0, 0)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(13)
    run = subtitle.add_run("从模型可运行到联合优化、气动校核与鲁棒性验证")
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(11)

    add_body(doc, "本阶段工作围绕“形成可用于导师审阅的实质性推进成果”展开，不以既有结果重绘为目标，而是补充了调度参数对实际推进能力的作用链路、候选构型的真实 VSPAERO 校核、扰动下的闭环飞推控响应，以及优化结果的多随机种子重复验证。")
    add_heading(doc, "阶段结论")
    add_body(doc, "当前已形成“多目标指标与约束 - 飞推控闭环模型 - 模糊能量调度 - VSPAERO 气动校正 - 稳健 MOEA/D - 多工况验证”的可运行链路。其中，1.2.1 提供内层飞推控物理模型，1.2.2 在其上完成调度器的联合寻优与风扰/健康自适应改进，1.2.3 完成扰动鲁棒验证并将组合退化引入 80 h 硬约束；2.2.1 和 2.2.2 分别提供评价约束和外层 MOEA/D 寻优支撑。修复失速约束后，H=6 组合退化优化获得 7 个可行 Pareto 候选，但该结论仍受当前低阶部件模型和载荷下限假设约束。")
    add_heading(doc, "本阶段工作概览")
    add_key_table(doc)

    doc.add_page_break()
    add_heading(doc, "一 1.2.2 协同调度与联合优化")
    add_body(doc, "在原有飞推控闭环模型中，飞行控制器给出推进需求，模糊调度器原本只改变发动机与电机的功率分配。为使调度变量真正进入优化问题，本阶段引入“低误差时保留部分电机能力、高误差时释放储备”的机制；因此在发动机动态、功率边界或扰动作用下，SOC 目标、充放电偏置、助力偏置、储备功率和发动机目标功率变化率会影响实际推力与状态误差。")
    add_body(doc, "在此基础上，构型变量、六个任务阶段功率变量和五个模糊调度参数共同纳入 MOEA/D，并保持名义工况和 CD0 增加 10% 保守工况下的 80 h 硬约束。三次 H=6、12 代重复试验均在第 5 代前达到全体可行，最终可行率均为 100%；不同随机种子得到 1、4、1 个唯一 Pareto 候选，说明在当前边界与模型口径下可行性较稳定。")
    doc.add_picture(str(FIGURES["joint"]), width=Cm(16.4))
    add_caption(doc, "图 1 H=6 稳健模糊联合优化的三随机种子收敛带与 Pareto 候选数量")
    doc.add_page_break()
    add_heading(doc, "二 1.2.1 飞行子模型气动校正")
    add_body(doc, "1.2.1 的飞行子模型以气动代理参数驱动低阶实时升力、阻力计算。为降低 80 h 候选的气动趋势不确定性，先在两个稳健候选附近补充 8 个 VSPAERO 样本，将原 15 点几何代理扩充为 23 点模型。全局留一验证中，升力线斜率 CLα 的 RMSE 由 0.2466 降至 0.2316 rad^-1，奥斯瓦尔德效率 e 的 RMSE 由 0.0497 降至 0.0376。")
    add_body(doc, "随后在 30 m/s 低速 ISR、42 m/s 巡航以及展弦比 31 的翼面积扰动区域补充 5 组独立 VSPAERO 极线。候选 6 的低速 CLα 误差约为 0.175 rad^-1，候选 23 误差接近零，表明候选区气动不确定性存在差异，应在后续优化中对候选 6 保持更保守的气动裕度。VSPAERO 为势流级计算，黏性阻力及 Reynolds 修正仍由动态 CD0 模型承担。")
    doc.add_picture(str(FIGURES["aero"]), width=Cm(16.4))
    add_caption(doc, "图 3 低速与高展弦比稳健候选的 VSPAERO 升力和诱导阻力极线")

    doc.add_page_break()
    add_heading(doc, "三 1.2.3 多工况鲁棒性验证")
    add_body(doc, "以稳健候选 6 为对象，完成起飞后初始爬升、低速高升力、稳定巡航和无动力滑翔四类工况，在上升风、下沉风、CD0 偏差、发动机降额、电池容量衰减及保守组合下共开展 28 组闭环仿真。垂直风扰通过高度状态积分进入模型，阻力、发动机和电池退化分别通过气动、可用功率和 SOC 边界进入模型。")
    add_body(doc, "结果显示，名义、CD0 增加 10%、发动机可用功率 85% 和电池容量 80% 时，四类局部工况均满足设定终端跟踪要求；强垂直风扰会使低速高升力、稳定巡航和无动力滑翔的高度保持超限。对同一候选进行 7 组 86 h 延伸任务复核时，6 组仍满足 80 h，保守组合仅达到 77.55 h。该结果为后续风扰补偿、参数鲁棒优化和动能学院部件模型替换提供了明确优先级。")
    doc.add_picture(str(FIGURES["robust"]), width=Cm(16.4))
    add_caption(doc, "图 4 四类标准工况的扰动跟踪判定与完整任务 80 h 鲁棒性")
    add_heading(doc, "后续工作重点", level=2)
    add_body(doc, "下一阶段应优先针对强垂直风扰设计飞行控制补偿或鲁棒调度机制，并将保守组合 77.55 h 作为联合优化的约束场景；获得发动机 BSFC 图、电池热衰减和螺旋桨安装干扰数据后，可直接替换当前接口中的预估模型，复用本阶段建立的优化与验证流程。")

    doc.add_page_break()
    add_heading(doc, "四 1.2.2 风扰补偿与健康自适应调度")
    add_body(doc, "为使协同控制策略能够针对扰动主动调整，飞行控制器增加垂直风前馈：将估计的垂直气团速度转换为所需的空速航迹角补偿；健康管理层则通过有效发动机功率上限进行一阶降额识别，并据此提高 SOC 储备目标。该结构保留了发动机、电池模型的可替换接口，后续可将当前参数化可用功率输入替换为动能学院提供的实测或高保真状态量。")
    add_body(doc, "在下沉风、CD0 增加 10%、发动机可用功率 85%、电池能量 80% 和效率下降的组合局部工况下，初始爬升终端高度误差由 20.36 m 降至 4.13 m，低速高升力工况由 279.44 m 降至 2.70 m。稳定巡航的高度偏差虽由约 1723 m 降至约 254 m，但仍不满足终端要求，原因是该阶段预设最大航迹角仅为 1 deg，持续 -1.5 m/s 下沉风超出其可补偿包线；无动力滑翔也不应由能量调度器虚构推进能力。")
    doc.add_picture(str(FIGURES["adaptive"]), width=Cm(16.4))
    add_caption(doc, "图 5 组合扰动下风扰补偿与健康自适应调度的局部恢复对比")

    doc.add_page_break()
    add_heading(doc, "五 组合退化 80 h 硬约束与可行区域筛选")
    add_body(doc, "将 CD0 增加 10%、发动机可用功率 85%、电池可用能量 80% 和发动机效率 95% 同时施加于完整任务，并要求所有能源、功率、飞行包线约束和 T_endurance >= 80 h 同时满足。首次完整 H=6 搜索发现续航可达 80 h 的候选存在约 0.46 m/s 的失速速度超限，说明组合退化下的主要矛盾由“低阻力小翼面”与失速包线共同决定，而非单纯的燃油耗尽。")
    add_body(doc, "以最小违反度候选为基准，固定任务功率和模糊调度参数，对翼面积和混动系统质量开展定向筛选，并将载荷保持在当前研究下限 100 kg。结果表明，混动质量 60 至 70 kg、翼面积不小于约 24.58 m2 的区域可同时满足续航和失速约束。该区域是当前参数化模型中的可行方向，真实最低载荷和部件质量参数确定后必须复核。")
    doc.add_picture(str(FIGURES["repair"]), width=Cm(16.4))
    add_caption(doc, "图 6 组合退化下翼面积与混动质量的定向可行域筛选")

    doc.add_page_break()
    add_heading(doc, "六 H=6 组合退化约束 MOEA/D 结果")
    add_body(doc, "将定向筛选得到的可行点作为 MOEA/D 的初始化锚点，在构型、六阶段发动机功率上限、六阶段电机辅助功率和五个模糊调度参数的联合设计空间内开展 H=6、8 代搜索。初始可行率为 3.6%，第 2 代达到 100%，最终得到 7 个唯一可行 Pareto 候选；每个候选均通过名义、CD0 增加 10% 和组合退化三种完整任务评价，并满足 80 h 硬约束。")
    add_body(doc, "在当前目标定义下，候选的综合能耗约为 19192 至 19441 MJ，飞行稳定性指标约为 61.052 至 61.083，任务响应指标约为 8242 至 8468 s。候选构型集中在翼面积约 24.58 m2、展弦比约 30.98、载荷 100 kg、混动系统质量约 70 kg 和 ISR 巡航速度约 30.0 至 30.5 m/s，表明组合退化可行区较窄，后续应优先使用真实部件数据复核这一结论。")
    doc.add_picture(str(FIGURES["combined"]), width=Cm(16.4))
    add_caption(doc, "图 7 H=6 组合退化约束 MOEA/D 的可行率收敛与 80 h 候选复核")

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
