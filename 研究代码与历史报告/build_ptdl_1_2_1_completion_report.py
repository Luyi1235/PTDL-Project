"""Build the PTDL 1.2.1 completion-status report as a Word document."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PTDL 1.2.1 混合动力与飞行控制协同模型完成情况说明.docx"
FIGURE = ROOT / "ptdl_1_2_1_coordination_results" / "coordination_soc_power_comparison.png"


def set_run_font(run, font: str, size: float, bold: bool = False, color: str = "000000") -> None:
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_paragraph_spacing(paragraph, before: float = 0, after: float = 6, line: float = 1.45) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line


def add_text(document: Document, text: str, *, bold_lead: str | None = None) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    set_paragraph_spacing(paragraph, after=6)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, "宋体", 11, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest, "宋体", 11)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, "宋体", 11)


def add_bullet(document: Document, lead: str, text: str) -> None:
    paragraph = document.add_paragraph(style="PTDL Bullet")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_paragraph_spacing(paragraph, after=4, line=1.35)
    a = paragraph.add_run(lead)
    set_run_font(a, "宋体", 10.8, bold=True)
    b = paragraph.add_run(text)
    set_run_font(b, "宋体", 10.8)


def add_heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.page_break_before = False
    run = paragraph.add_run(text)
    set_run_font(run, "黑体", 15 if level == 1 else 12, bold=True)
    set_paragraph_spacing(paragraph, before=14 if level == 1 else 9, after=7, line=1.2)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_run_font(run, "宋体", 9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)
    run = paragraph.add_run(" 页")
    set_run_font(run, "宋体", 9)


def configure(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.35)
    section.right_margin = Cm(2.15)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)

    normal = document.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)

    for level in (1, 2, 3):
        style = document.styles[f"Heading {level}"]
        style.font.name = "黑体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.color.rgb = RGBColor(0, 0, 0)

    bullet = document.styles.add_style("PTDL Bullet", WD_STYLE_TYPE.PARAGRAPH)
    bullet.base_style = document.styles["Normal"]
    bullet.paragraph_format.left_indent = Cm(0.55)
    bullet.paragraph_format.first_line_indent = Cm(-0.4)
    bullet.paragraph_format.space_after = Pt(4)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_run = header.add_run("PTDL 项目研究内容一 1.2.1 阶段说明")
    set_run_font(header_run, "宋体", 9, color="555555")
    footer = section.footer.paragraphs[0]
    add_page_number(footer)


def add_title(document: Document) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(54)
    paragraph.paragraph_format.space_after = Pt(16)
    run = paragraph.add_run("PTDL 1.2.1 混合动力与飞行控制协同模型完成情况说明")
    set_run_font(run, "黑体", 17, bold=True)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(28)
    run = subtitle.add_run("现有基础  新增工作  动能学院替换边界  后续补充项")
    set_run_font(run, "宋体", 11, color="555555")

    info = document.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run("适用对象：项目导师、总体与控制方向、动力与能源方向协同评审")
    set_run_font(run, "宋体", 10, color="555555")
    document.add_page_break()


def main() -> None:
    document = Document()
    configure(document)
    add_title(document)

    add_heading(document, "一 结论与报告目的", 1)
    add_text(
        document,
        "本报告用于说明 PTDL 项目研究内容一中 1.2.1 混合动力与飞行控制策略协同模型构建工作的实际完成边界，并将其与研究内容二中已完成的任务剖面、评价体系和 MOEA/D 优化工作明确区分。当前已形成可运行的飞推控协同仿真原型，可完成全任务剖面下的功率需求计算、发动机电机功率分配、电池 SOC 演化和阶段切换仿真；但发动机、电机、电池及螺旋桨仍采用可替换的预估低阶模型，尚不构成经部件试验或高保真动力学验证的最终模型。",
    )
    add_text(
        document,
        "核心判断：现阶段工作已经完成了协同模型的总体架构、统一接口与初步控制逻辑，可作为后续算法开发和联合建模的集成平台；正式开展动力系统性能结论、动态响应结论和鲁棒性结论前，必须由动能学院接入真实或经验证的发动机、电机、电池及推进系统模型。",
        bold_lead="核心判断：",
    )

    add_heading(document, "二 申请书对 1.2.1 的要求与当前对应关系", 1)
    add_text(
        document,
        "申请书要求针对航空活塞发动机与电池混合动力系统，分别建立发动机、电池和电动机的动力学与能量转换模型，并结合无人机姿态、轨迹和航迹规划建立整体系统模型；同时采用传递函数、状态空间方法和非线性微分方程描述实时响应、功率分配、能量转换、模式切换与延迟效应。",
    )
    add_bullet(document, "已满足的部分：", "已形成任务剖面驱动的非线性纵向飞行与推进功率平衡模型，包含飞行状态、燃油、SOC 和功率流的连续时程计算，并建立了阶段 SOC 目标、功率分配和功率变化率限制。")
    add_bullet(document, "部分满足的部分：", "发动机效率、燃油流量、电池充放电和电机功率边界已进入模型，但目前以静态或准静态关系表达，尚未体现真实部件的内部状态和完整动态过程。")
    add_bullet(document, "尚未满足的部分：", "尚未建立发动机转速与扭矩、电机电流与转速、电池极化与热状态、螺旋桨转速与桨距、执行机构和完整姿态航迹控制的状态空间或传递函数模型；模式切换延迟仍为规则和速率限制，尚未由部件动态模型给出。")

    add_heading(document, "三 来自 2.1 和 2.2 的已有基础", 1)
    add_heading(document, "3.1 来自 2.1 的任务与飞行仿真基础", 2)
    add_text(
        document,
        "研究内容二的 2.1 已建立 80 h 目标下的全任务剖面，将飞行过程划分为主爬升、转入 ISR 高度、长时间留空、长下降、进近下降和拉平六个空中阶段；每一阶段具有高度目标、速度目标、持续时间、飞行航迹角限制、发动机功率范围、电机辅助功率范围和时间步长。该任务剖面是 1.2.1 的工况输入，而不是 1.2.1 新增的动力学模型。",
    )
    add_text(
        document,
        "2.1 同时提供了纵向点质量飞行模型的基础：模型以航程、高度、速度、航迹角、燃油消耗和 SOC 为主要状态，采用非线性关系计算升力、阻力、推力和所需推进功率。它能够反映高度与速度指令对推进功率需求的影响，但尚不是包含横滚、偏航、舵面和执行机构的完整姿态控制模型。",
    )
    add_heading(document, "3.2 来自 2.2 的评价与优化基础", 2)
    add_text(
        document,
        "研究内容二的 2.2 已建立三目标评价体系，即综合能耗、飞行稳定性和任务响应性能，并建立续航时间不少于 80 h、终端与最低 SOC、发动机和电机功率及其变化率、飞行包线、电池电流、燃油和构型约束。1.2.1 的仿真时程直接输出燃油流量、SOC、发动机功率、电机功率、高度、速度和跟踪误差，因此可无缝接入该评价体系。",
    )
    add_text(
        document,
        "2.2 已完成 MOEA/D 主算法选型和初步优化平台搭建，并以 VSPAERO 极线数据校正候选构型的升力线斜率和奥斯瓦尔德效率。MOEA/D 属于外层离线优化工具，负责搜索构型参数与控制参数的 Pareto 最优组合；它不是 1.2.1 中实时功率分配控制器的替代品。",
    )

    add_heading(document, "四 1.2.1 新增完成的协同模型内容", 1)
    add_heading(document, "4.1 协同模型总体结构", 2)
    add_text(
        document,
        "1.2.1 在既有任务和飞行模型基础上新增了混动能量管理层，形成“任务阶段与飞行参考 - 飞行状态跟踪与推进功率需求 - 发动机电机功率分配与电池状态更新”的三级协同结构。飞行层计算满足高度、速度和航迹要求所需的推进功率；协同层根据 SOC、阶段目标和跟踪误差确定发动机目标功率；电机在其功率边界内补足或吸收功率差额，电池 SOC 随充放电功率更新。",
    )
    add_text(document, "当前实现的状态与接口可概括为：飞行状态包括航程、高度、速度和航迹角，能源状态包括燃油消耗与 SOC；控制输入包括阶段高度和速度目标、发动机功率上限及协同调度器参数；输出包括发动机功率、电机功率、燃油流量、SOC、功率模式和飞行跟踪误差。")
    add_heading(document, "4.2 新增的 SOC 反馈协同调度器", 2)
    add_text(
        document,
        "新增规则型协同调度器为各阶段设置 SOC 目标：主爬升 0.72、转入 ISR 高度 0.66、长时间留空 0.55、长下降 0.42、进近下降 0.34、拉平 0.30。调度器根据 SOC 偏差产生充电或放电修正量，并叠加速度和高度跟踪误差产生的功率储备需求，生成发动机目标功率；电机功率则由瞬时推进功率平衡确定。",
    )
    add_text(
        document,
        "调度器目前能够识别电机助力、发动机跟随和电池充电三类工作状态，并通过发动机目标功率变化率限制避免相邻计算步之间出现无约束的功率跃变。这一策略的作用是建立可解释的协同控制基线，后续可作为 1.2.2 智能调度和 2.2 外层优化的初始策略。",
    )
    add_heading(document, "4.3 初步算例与可复现结果", 2)
    add_text(
        document,
        "以 H=6、VSPAERO 修正后的 MOEA/D 可行 Pareto 集中能耗最优构型为算例，在相同构型、任务剖面、发动机功率上限和气动修正条件下，对比原阶段定值调度与 SOC 反馈协同调度。两种策略均完成 80 h 任务；SOC 反馈协同调度的综合能耗目标由 20411.62 MJ 降至 20399.40 MJ，降低约 0.06%，终端 SOC 为 0.318，最低 SOC 为 0.250。该结果用于证明协同控制链路能够运行并影响能量分配，不作为真实动力系统节油幅度结论。",
    )
    if FIGURE.exists():
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(FIGURE), width=Cm(15.5))
        caption = document.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_paragraph_spacing(caption, after=8, line=1.2)
        run = caption.add_run("图 1  阶段定值调度与 SOC 反馈协同调度的 SOC 和功率时程对比")
        set_run_font(run, "宋体", 9.5, color="555555")

    add_heading(document, "五 需由动能学院替换或校核的模块", 1)
    add_text(document, "以下模块已保留在总体框架中的明确接口，但当前数值均为预估参数或低阶关系。动能学院接入模型时，应优先替换这些模块，而保持任务剖面、统一评价函数和 MOEA/D 外层接口不变。")
    add_heading(document, "5.1 航空活塞发动机与发电机模型", 2)
    add_text(document, "当前采用功率分数到效率的静态插值，并以电功率除以效率和燃油低热值计算燃油流量；尚未描述转速、扭矩、节气门、进气和燃烧过程、发电机效率、最小稳定功率及动态滞后。动能学院应提供发动机和发电机的功率转速扭矩边界、燃油消耗率图或效率图、功率爬升率、启动与停机约束、动态响应时间常数及实际电气输出关系。")
    add_heading(document, "5.2 电池模型", 2)
    add_text(document, "当前电池采用 SOC 积分、固定充放电效率和功率边界，未包含开路电压、内阻、极化、电流倍率、容量衰减和热状态。建议以等效电路模型替换，至少包含 SOC、极化电压和温度状态，并提供 OCV SOC 曲线、内阻参数、容量、最大充放电电流、热参数与安全边界。")
    add_heading(document, "5.3 电动机与逆变器模型", 2)
    add_text(document, "当前电机以功率边界直接响应需求，未包含转速、转矩、电流、逆变器损耗及转矩建立时间。动能学院应提供电机和逆变器效率图、最大连续与峰值功率、最大转矩、电流限制、转速范围、热限制和控制响应时间，用于建立从功率指令到轴功率的动态转换。")
    add_heading(document, "5.4 螺旋桨与机械传动模型", 2)
    add_text(document, "当前推力计算采用动量理论和固定推进效率，未描述螺旋桨转速、桨距、前进比与效率图，也未明确串联、并联或混联构型中的实际机械耦合路径。动能学院需明确系统拓扑，并提供螺旋桨性能图、传动效率、转动惯量和必要的轴系动态参数。")

    add_heading(document, "六 尚需补充的总体与控制能力", 1)
    add_bullet(document, "完整飞行控制：", "现有模型只描述纵向高度和速度跟踪；后续需补充姿态内环、舵面与执行机构动态、横滚和偏航控制，以及航迹规划层。")
    add_bullet(document, "混杂模式切换：", "当前模式由 SOC 与功率平衡规则识别，且仅通过功率变化率实现平滑；后续需建立纯油、纯电、混动助力和充电等离散状态机，明确进入退出判据、滞环、最小驻留时间和切换时延。")
    add_bullet(document, "联合校核：", "需要以动能学院模型重新校核功率流、燃油、SOC、温度和动态响应，并与任务剖面中的爬升、留空、下降和进近典型工况逐段比对。")
    add_bullet(document, "鲁棒性输入：", "为支撑 1.2.3 和申请书中的鲁棒控制研究，后续需要加入风扰、温度变化、载荷波动、发动机性能退化和电池衰减等扰动模型。")

    add_heading(document, "七 推荐的后续衔接路径", 1)
    add_text(document, "第一步，由动能学院提供或确认发动机发电机、电机逆变器、电池和螺旋桨模型及参数边界，并替换当前低阶功率流模块；替换后保持现有输入输出字段不变，以确保与 2.1 任务剖面和 2.2.1 评价函数继续兼容。")
    add_text(document, "第二步，在真实部件动态基础上，将 SOC 目标、SOC 反馈增益、功率储备、发动机功率变化率、模式切换阈值和切换平滑参数定义为可调控制器参数，形成 1.2.2 的协同控制与智能调度算法对象。")
    add_text(document, "第三步，将 1.2.2 的协同控制器嵌入 2.2 的 MOEA/D 外层，联合优化构型参数、质量分配、阶段速度和协同控制参数；随后使用 2.1 的典型任务工况及扰动场景开展 1.2.3 多工况验证与迭代。")

    add_heading(document, "八 当前工作边界的汇总表述", 1)
    add_text(document, "当前完成的是面向全任务剖面的混合动力飞推控协同仿真与规则型能量管理原型：它继承 2.1 的任务剖面和飞行状态模型，调用 2.2 的评价函数和 MOEA/D 优化接口，并新增 SOC 反馈、功率分配和模式识别逻辑。待动能学院接入经验证的发动机、电机、电池和螺旋桨动态模型后，现有原型将升级为申请书所要求的、能够描述能量转换、功率分配、模式切换与延迟效应的混合动力系统与飞行控制策略协同作用模型。")

    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
