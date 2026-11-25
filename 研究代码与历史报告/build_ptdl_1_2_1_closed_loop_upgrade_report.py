"""Build the PTDL 1.2.1 closed-loop upgrade report as a Word document."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PTDL 1.2.1 混合动力系统与飞行控制协同作用模型闭环升级说明 CD0更新版.docx"
FIGURE = ROOT / "ptdl_1_2_2_intelligent_energy_management_results" / "fuzzy_adaptive_soc_power_comparison.png"


def set_run_font(run, font: str, size: float, bold: bool = False, color: str = "000000") -> None:
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_spacing(paragraph, before: float = 0, after: float = 6, line: float = 1.45) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line


def add_body(document: Document, text: str, lead: str | None = None) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    set_spacing(paragraph)
    if lead and text.startswith(lead):
        first = paragraph.add_run(lead)
        set_run_font(first, "宋体", 11, bold=True)
        rest = paragraph.add_run(text[len(lead):])
        set_run_font(rest, "宋体", 11)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, "宋体", 11)


def add_bullet(document: Document, lead: str, text: str) -> None:
    paragraph = document.add_paragraph(style="PTDL Bullet")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_spacing(paragraph, after=4, line=1.35)
    run = paragraph.add_run(lead)
    set_run_font(run, "宋体", 10.8, bold=True)
    run = paragraph.add_run(text)
    set_run_font(run, "宋体", 10.8)


def add_heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    set_spacing(paragraph, before=14 if level == 1 else 9, after=7, line=1.2)
    run = paragraph.add_run(text)
    set_run_font(run, "黑体", 15 if level == 1 else 12, bold=True)


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

    title = document.styles["Title"]
    title.font.name = "黑体"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.font.size = Pt(15.5)
    title_ppr = title._element.get_or_add_pPr()
    title_borders = title_ppr.find(qn("w:pBdr"))
    if title_borders is not None:
        title_ppr.remove(title_borders)

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
    run = header.add_run("PTDL 项目研究内容一 1.2.1 闭环升级说明")
    set_run_font(run, "宋体", 9, color="555555")
    add_page_number(section.footer.paragraphs[0])


def add_title(document: Document) -> None:
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(54)
    title.paragraph_format.space_after = Pt(16)
    run = title.add_run("PTDL 1.2.1 混合动力系统与飞行控制协同作用模型闭环升级说明")
    set_run_font(run, "黑体", 15.5, bold=True)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(26)
    run = subtitle.add_run("实际功率到推力到飞行状态闭环  动能模型替换接口  与 2.2 协同关系")
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

    add_heading(document, "一 升级结论与报告目的", 1)
    add_body(document, "本报告说明 PTDL 项目研究内容一中 1.2.1 混合动力系统与飞行控制协同作用模型的当前闭环升级状态，并重点澄清其与研究内容二中 2.2.1 多目标性能指标体系、2.2.2 MOEA/D 多目标优化算法之间的分工关系。当前模型已从“根据目标状态计算所需功率，再按指令推进飞行状态”的准稳态流程，升级为“实际发动机和电池功率决定实际推力，实际推力决定速度、航迹角和高度，再反馈给控制器”的纵向飞推控闭环模型。")
    add_body(document, "核心结论：2.2.1 解决评价问题，2.2.2 解决搜索问题，1.2.1 解决每一个候选方案在物理系统中如何实际运行的问题。1.2.1 不是第二套多目标算法，也不重复建立评价函数；它是 2.2.2 在评估每一个候选解时必须调用的内层飞推控仿真对象。", "核心结论：")
    add_body(document, "目前已完成可替换发动机发电机接口、可整体替换混动总成接口、实际功率到螺旋桨推力模型、VSPAERO 气动趋势校正、随构型与雷诺数变化的部件阻力分解模型、二维纵向动力学积分、任务阶段控制器、SOC 与燃油状态反馈、多速率积分策略和回归测试。默认部件参数仍为低阶参考参数，待动能学院模型接入后可替换为经验证的发动机、电机、电池和传动模型。")

    add_heading(document, "二 1.2.1 与 2.2.1 和 2.2.2 的边界", 1)
    add_heading(document, "2.1 2.2.1 的作用是定义评价标准", 2)
    add_body(document, "2.2.1 已建立面向混合动力超长航时无人机的三目标性能指标体系，包括综合能耗、飞行稳定性和任务响应性能，并明确了 80 h 续航硬约束、SOC 终端和最低值约束、燃油约束、发动机和电机功率与变化率约束、飞行包线约束等。它回答的是“仿真完成后，如何量化判断某个方案是否可行、各项性能有多好”。")
    add_body(document, "2.2.1 的评价函数本身不生成飞行状态，也不决定发动机实际能否达到给定功率，更不计算功率不足时飞机会怎样变化。它接收一段完整时程，读取燃油流量、SOC、功率、高度、速度、误差等数据，再输出目标函数、约束违反度和诊断量。")
    add_heading(document, "2.2.2 的作用是搜索 Pareto 最优方案", 2)
    add_body(document, "2.2.2 已确定采用 MOEA/D 作为外层多目标优化算法。MOEA/D 将多目标问题分解为一组权重子问题，反复生成候选构型与控制参数，调用仿真和评价模块获得三目标及约束，再更新邻域解和 Pareto 解集。它回答的是“在大量可选构型、速度、高度、发动机功率上限、电机辅助能力和控制器参数中，哪些组合能够在约束下取得更优权衡”。")
    add_body(document, "MOEA/D 并不直接理解发动机转速、燃油流量图、电机功率建立过程或飞行方程。若没有内层模型，它只能优化代数化或经验化的结果；因此，算法本身不保证候选解在实际推力不足、部件滞后或能量边界收紧时仍能完成任务。")
    add_heading(document, "2.3 1.2.1 的作用是提供被优化的物理闭环对象", 2)
    add_body(document, "1.2.1 位于 2.2.2 的每一个候选解与 2.2.1 的评价结果之间。对某一组构型与控制参数，1.2.1 根据任务剖面和真实状态进行逐步仿真，输出实际飞行与能量时程；2.2.1 再以此计算目标和约束；MOEA/D 依据评价结果决定该候选解是否保留和如何更新。")
    add_body(document, "因此，三者的关系可概括为：MOEA/D 提出“试什么方案”，1.2.1 计算“该方案实际上怎样飞、怎样供能、能否跟踪任务”，2.2.1 判断“该结果是否满足 80 h 和安全约束，以及能耗、稳定性和响应性能如何”。")

    add_heading(document, "三 本次闭环升级新增的内容", 1)
    add_heading(document, "3.1 从功率指令推进状态改为实际功率驱动状态", 2)
    add_body(document, "升级前，模型依据阶段目标高度和速度直接形成下一时刻的飞行状态，再倒推出所需推进功率；即使发动机和电机的可用功率不足，飞行状态仍可能按目标继续前进，功率不足仅作为诊断量记录。该方式适合早期快速评估，但不能反映动力不足对爬升和速度跟踪的实际影响。")
    add_body(document, "升级后，控制器先根据实际高度和速度误差生成期望航迹角、迎角和推进功率请求；混合动力系统根据发动机目标功率、发动机功率上限、电机边界和 SOC 状态产生实际可用功率；螺旋桨模型将实际功率转换为实际推力；随后由二维纵向点质量方程积分速度、航迹角、高度和航程。因而，当发动机或电机能力不足时，实际推力下降，速度或爬升率随之下降，新的高度和速度误差会在下一步反馈给飞行控制器和能量管理器。")
    add_heading(document, "3.2 实际功率到螺旋桨推力的转换", 2)
    add_body(document, "当前默认螺旋桨模型采用动量理论。其输入为发动机发电机输出与电池贡献构成的可用推进功率，并考虑推进效率、飞行速度、空气密度、螺旋桨盘面积和诱导损失因子，求得实际推力及诱导速度。该过程使推进系统的功率边界能够直接影响飞行方程，而不再以预先假定的推力替代。")
    add_heading(document, "3.3 VSPAERO 气动校正与构型相关寄生阻力模型", 2)
    add_body(document, "当前“飞”模块采用 VSPAERO 气动代理模型与低阶实时气动方程相结合的方式。OpenVSP 根据翼面积、展弦比、后掠角、翼尖扭转和机身尺寸生成参数化构型；VSPAERO 对设计试验样本计算气动极线；代理模型据此为每个候选构型修正升力线斜率 CL_alpha 和奥斯瓦尔德效率 e。任务时域仿真不在每一个积分步直接调用 VSPAERO，而是读取代理模型输出的参数，以保证 MOEA/D 大量候选解评估时的计算效率。")
    add_body(document, "零升阻力系数 CD0 不再作为全程固定常数。首先，在参考飞行状态下采用部件阻力分解建立构型相关的基准 CD0：机翼、机身、水平尾翼和垂尾分别按湿面积、形状因子和湍流蒙皮摩擦系数计算，并加入等效螺旋桨安装、短舱/支架、冷却安装和部件干扰的预备阻力项；该基准值通过 ULTRA 72 h 基准点进行整体标定。随后，在每个飞行动力学积分步中依据实际空气密度、速度和平均气动弦长计算雷诺数，并用蒙皮摩擦系数比值更新 CD0。因而，翼面、机身、尾翼和螺旋桨安装尺寸的变化，以及爬升、留空和进近阶段的高度/速度变化，都会进入寄生阻力计算。")
    add_body(document, "当前气动力按 CL = CL_alpha * alpha 和 CD = CD0(z, Re) + CL^2 / (pi * AR * e) 计算，再由 L = q * S * CL 与 D = q * S * CD 得到实际升力和阻力。其中，CL_alpha 与 e 的构型趋势来自 VSPAERO 代理模型，CD0 来自经 ULTRA 基准校准的部件阻力分解与雷诺数修正模型；由于 VSPAERO 是无黏涡格法，当前不直接采用其阻力输出代替完整 CD0。实际升力和实际推力共同决定航迹角变化，实际推力、阻力和重力沿航迹分量共同决定速度变化。")
    add_body(document, "当前模型使用一致的自由来流动压计算控制器和气动力，避免未校核的滑流增升假设造成数值性的姿态超调。后续获得螺旋桨翼面干扰、部件外形和黏性 CFD/风洞数据后，可将等效安装阻力和自由来流假设替换为局部滑流场、部件干扰阻力及更高保真气动模型。")
    add_heading(document, "3.4 飞行控制与推进控制的真实反馈", 2)
    add_body(document, "飞行控制层以任务阶段的目标高度和速度为参考。高度误差形成受阶段最大航迹角限制的航迹角指令；实际航迹角与指令之间的误差形成法向加速度和迎角指令；速度误差形成有界的纵向加速度请求。迎角采用一阶响应和变化率限制，以避免在离散积分中出现无穷快的姿态建立。")
    add_body(document, "推进调度层读取当前 SOC、高度误差、速度误差、所需推进功率、发动机功率上限和电机边界。无外部调度器时，阶段功率表确定发动机与电机的基准分配；接入规则 SOC 反馈或模糊自适应调度器后，发动机目标功率会随 SOC 偏差和跟踪需求修正。发动机和电机实际响应后得到新的燃油流量和 SOC，下一时间步的调度决策因此具备能量状态反馈。")
    add_heading(document, "3.5 多速率积分与失效判定", 2)
    add_body(document, "模型对爬升、下降、进近和拉平等速度与航迹角变化较快的阶段采用不大于 20 s 的积分步长；长时间留空段在离散航迹角内环稳定后允许采用不大于 120 s 的步长。这一设置使 80 h 任务校准和验证保持可接受的计算时间，同时避免下降阶段因过大步长产生的数值振荡。")
    add_body(document, "功率不足不再在单个积分步立即中止任务，而是首先通过实际动力学作用于速度和高度。只有当速度已降至模型定义的安全下限且仍存在持续功率缺额时，仿真才以 propulsion_power_shortfall 终止；若迎角能力不足且同时到达速度下限，则以 lift_shortfall 终止。燃油耗尽和 SOC 安全下限仍为独立终止条件。")

    add_heading(document, "四 与 2.2.2 MOEA/D 的具体调用关系", 1)
    add_body(document, "在新的整体架构中，MOEA/D 的每一个候选解首先被解码为设计变量和控制变量。设计变量可包括翼面积、展弦比、后掠、扭转、机身尺寸、螺旋桨直径、载荷质量和混动系统质量；控制变量可包括各阶段速度、发动机功率上限、电机辅助能力，以及后续拟纳入优化的 SOC 参考轨迹、SOC 死区、充电偏置、助力偏置、功率变化率和模式切换阈值。")
    add_body(document, "候选解进入 1.2.1 模型后，构型变量经 OpenVSP/VSPAERO 代理模型改变 CL_alpha 与 e，经部件阻力分解模型改变基准 CD0，并同时改变质量分配、发动机能力和电池容量；控制变量改变阶段参考与功率分配。模型输出的不是预设结果，而是由实际功率、推力和纵向飞行动力学共同形成的时程，包括燃油流量、SOC、发动机和电机功率、推进功率请求、实际推进功率、推力、升力、阻力、CD0、高度、速度、航迹角、迎角和跟踪误差。")
    add_body(document, "随后 2.2.1 评价函数读取上述时程，计算综合能耗目标、稳定性目标和响应性能目标，并检查 80 h、SOC、燃油、功率、功率变化率、速度、高度、迎角和载荷等约束。MOEA/D 只接收这些目标和约束结果并更新种群。这样，MOEA/D 的 Pareto 前沿由真实飞推控闭环响应生成，而不再仅由阶段目标的代数功率估计生成。")
    add_body(document, "这一关系也界定了当前阶段的必要工作：旧 H=6、VSPAERO 修正的 MOEA/D 解集仍可作为构型和控制变量的初始样本，但其旧目标函数数值来自升级前的准稳态内层模型，不能再作为新闭环模型下的正式 Pareto 数值。应以旧解集为初始候选，在新 1.2.1 闭环模型上重新运行 MOEA/D，重新形成能耗、稳定性和响应性能的 Pareto 前沿。", "需要强调：")

    add_heading(document, "五 动能学院模型替换边界与接口", 1)
    add_heading(document, "5.1 仅替换发动机发电机模型", 2)
    add_body(document, "已定义 EngineGeneratorModel 接口。动能学院只需实现初始状态函数和步进函数：步进函数接收当前发动机内部状态、控制器要求的目标电功率、阶段功率上限、时间步长和飞机参数，返回下一内部状态、实际发电功率和燃油流量。发动机内部状态可以包含转速、扭矩、节气门、进气状态、燃烧状态、热状态、发电机状态和启停状态，而飞行仿真器不需要了解其内部实现。")
    add_body(document, "该接口的关键价值是，动能学院提供发动机的 BSFC 图、效率图、功率转速扭矩边界、动态响应、功率爬升率、启停约束和热限制后，只需替换发动机组件；任务剖面、飞行动力学、SOC 调度器、2.2.1 评价函数和 2.2.2 MOEA/D 均无需重写。")
    add_heading(document, "5.2 替换完整混合动力总成", 2)
    add_body(document, "已定义 HybridPowertrainModel 接口，用于后续接入串联、并联或功率分流构型的完整模型。完整总成需输出实际发动机功率、实际电池贡献功率、进入螺旋桨模型的实际可用功率、燃油流量、SOC 和电池电流。该接口可承载电机逆变器效率图、电池开路电压和内阻、极化和温度状态、电池 SOH、传动效率、轴系惯量及真实功率流拓扑。")
    add_body(document, "目前默认的 ReferenceHybridPowertrain 是等效直流母线模型：发动机功率表示发动机发电机的电母线输出，电机功率表示电池对母线的正向放电或负向吸收，二者构成可用推进功率。这是为了在缺少真实拓扑和部件图谱时保持统一的能量守恒和接口结构，不代表最终 PTDL 已确定串联或并联系统方案。")

    add_heading(document, "六 当前验证结果", 1)
    add_body(document, "为验证闭环升级和构型相关 CD0 模型没有破坏全任务仿真链路，选取既有 H=6、VSPAERO 修正后 Pareto 集中的综合能耗候选构型，在同一 80 h 任务剖面、同一 VSPAERO 气动修正、同一构型相关寄生阻力模型和同一阶段功率边界下重新运行三种调度策略。阶段定值调度、规则型 SOC 反馈调度和模糊自适应调度均完成 80.0 h 任务，说明新的实际功率到推力到飞行状态链路能够完成全剖面闭环仿真。")
    add_body(document, "重跑结果显示：阶段定值调度燃油消耗为 486.606 kg，终端 SOC 为 0.553，综合能耗目标为 20929.695 MJ；规则 SOC 反馈调度燃油消耗为 486.294 kg，终端 SOC 为 0.319，综合能耗目标为 20920.731 MJ；模糊自适应调度燃油消耗为 486.717 kg，终端 SOC 为 0.682，综合能耗目标为 20932.217 MJ。相较于固定 CD0 的旧重跑结果，能耗上升反映了寄生阻力随任务状态变化后的更保守估算。规则 SOC 反馈在当前低阶部件模型下具有略低能耗，但终端 SOC 更接近约束；模糊策略保留更高能量储备，尚未经过 MOEA/D 参数整定。")
    add_body(document, "这些数据用于证明新闭环模型的可运行性和控制链路的差异可被量化，不应解释为最终发动机或电池选型结论。发动机、电机、电池和螺旋桨的默认参数仍需由动能学院模型替换或校核后，才可形成工程性能结论。")
    if FIGURE.exists():
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(FIGURE), width=Cm(15.4))
        caption = document.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_spacing(caption, after=8, line=1.2)
        run = caption.add_run("图 1  闭环模型下三种能量管理策略的 SOC 与功率时程对比")
        set_run_font(run, "宋体", 9.5, color="555555")

    add_heading(document, "七 代码实现与测试状态", 1)
    add_body(document, "闭环主模型位于 ptdl_flight_propulsion_control.py；其中包含 EngineGeneratorModel、HybridPowertrainModel、ReferenceEngineGenerator 和 ReferenceHybridPowertrain。ptdl_80h_endurance_profile.py 和 ptdl_full_task_profile.py 已增加 engine_model 与 powertrain 参数，使两类任务适配器均可直接接入动能学院模型。ptdl_hybrid_coordination.py 和 ptdl_intelligent_energy_management.py 沿用既有 DispatchObservation 和 PowerDispatch 接口，无需改写。")
    add_body(document, "新增 test_ptdl_hybrid_flight_control_dynamics.py，使用零输出发动机替换默认发动机，验证发动机实际输出变化会造成推进功率缺额，并使实际高度和速度响应发生变化。这一测试直接验证了发动机模型不是事后燃油记账模块，而是真正参与飞行状态演化的内层组件。")
    add_body(document, "飞推控、协同调度、80 h 适配、全任务适配和 2.2 评价优化相关的 32 项测试已通过。完整回归中的一项既有 OpenVSP 几何生成测试在当前 Python 环境下因未安装 openvsp 而无法执行，该问题与本次闭环改造无关；OpenVSP/VSPAERO 气动数据的既有接口和结果文件未在本次修改中替换。")

    add_heading(document, "八 当前能力边界与下一步", 1)
    add_body(document, "当前 1.2.1 已具备作为 2.2.2 内层仿真对象和接收动能学院模型的程序能力，但仍是纵向任务级模型。它已经描述实际功率、实际推力、速度、航迹角、高度、燃油与 SOC 的闭环反馈；尚未描述完整横滚和偏航动力学、舵面与执行机构、螺旋桨桨距调节、发动机真实启停过程、电池极化和热状态、故障检测残差及完整航迹规划。")
    add_body(document, "下一步应首先由动能学院提供发动机发电机模型及其参数图谱，接入 EngineGeneratorModel 并在典型爬升、留空、下降和进近工况下校核功率、燃油和动态响应；随后视系统拓扑接入完整 HybridPowertrainModel。完成部件替换后，以当前闭环模型重新运行 MOEA/D，联合优化构型参数、阶段速度、功率上限和能量管理参数，并在扰动与部件退化场景下开展 1.2.3 多工况协同验证。")

    add_heading(document, "参考文件", 1)
    add_body(document, "[1] PTDL 项目组. PTDL 2.2.1 多目标性能指标体系与评价函数交付件. 2026.")
    add_body(document, "[2] PTDL 项目组. PTDL 2.2.2 多目标优化控制算法文献调研与适配报告. 2026.")
    add_body(document, "[3] PTDL 项目组. PTDL 1.2.2 协同控制与智能调度算法开发阶段完成情况说明. 2026.")
    add_body(document, "[4] PTDL 项目组. PTDL 80 h 任务剖面及 H=6 VSPAERO 修正候选构型重跑结果. 2026.")

    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
