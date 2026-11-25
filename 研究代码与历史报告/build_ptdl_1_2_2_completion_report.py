"""Build the PTDL 1.2.2 implementation-status report as a Word document."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PTDL 1.2.2 协同控制与智能调度算法开发阶段完成情况说明.docx"
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
    title.font.size = Pt(17)
    # Word's built-in Title style can carry a theme-colored bottom border.
    # Remove it so the cover remains a plain formal report title.
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
    run = header.add_run("PTDL 项目研究内容一 1.2.2 阶段说明")
    set_run_font(run, "宋体", 9, color="555555")
    add_page_number(section.footer.paragraphs[0])


def add_title(document: Document) -> None:
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(54)
    title.paragraph_format.space_after = Pt(16)
    run = title.add_run("PTDL 1.2.2 协同控制与智能调度算法开发阶段完成情况说明")
    set_run_font(run, "黑体", 17, bold=True)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(26)
    run = subtitle.add_run("模糊自适应调度  安全监督接口  初步仿真验证  后续高保真接入路径")
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
    add_body(document, "本报告说明 PTDL 项目研究内容一中 1.2.2 协同控制与智能调度算法开发的当前实际完成情况。围绕混合动力无人机飞推控耦合系统的实时功率分配需求，现已形成可运行的分层能量管理原型：以可解释的模糊自适应调度器作为在线主控制器，以神经网络 SOC 参考预测器作为预留接口，以安全约束投影和健康状态监督作为容错框架，并已在与 2.1 任务剖面、2.2 多目标评价体系和 H=6 MOEA/D 候选构型一致的 80 h 算例中完成初步对比验证。")
    add_body(document, "核心结论：当前已经完成了协同控制算法的原型实现、统一程序接口和无故障工况下的闭环仿真验证，证明调度器能够在全任务剖面内完成发动机、电机和电池的功率分配，并保持 80 h 任务可行；但神经网络尚未完成训练，真实部件故障尚未注入动力学对象，因而现阶段不能宣称已经获得经工程验证的智能控制性能或故障容错性能。", "核心结论：")

    add_heading(document, "二 申请书要求与当前研究定位", 1)
    add_body(document, "申请书要求针对飞推控系统中的非线性、时变和不确定性，开展多种控制方法的对比和融合，引入神经网络、模糊控制等人工智能技术实现智能能量管理，并在起降、低速高升力、巡航和滑翔等工况下评价响应速度、稳定性和能耗，最终形成自适应、容错的协同控制方法体系。")
    add_body(document, "当前 1.2.2 的定位不是重新建立飞行器动力学或外层多目标问题，而是在 1.2.1 协同模型上增加在线决策层。其输入是任务阶段、推进功率需求、SOC、速度和高度跟踪误差以及部件能力边界；其输出是发动机目标功率、SOC 目标和运行模式。实际执行时仍由飞推控仿真器实施功率平衡、发动机与电机功率限幅、SOC 安全边界和燃油状态更新。")
    add_bullet(document, "与 2.1 的关系：", "2.1 提供六阶段 80 h 任务剖面及纵向飞行状态和推进功率需求，1.2.2 的调度器在每一仿真步读取这些状态进行功率分配。")
    add_bullet(document, "与 2.2.1 的关系：", "2.2.1 提供综合能耗、稳定性和响应性能三个目标及 SOC、燃油、功率和飞行包线约束，1.2.2 的输出直接被评价函数接收。")
    add_bullet(document, "与 2.2.2 的关系：", "MOEA/D 是外层离线多目标优化器，用于搜索构型与控制器参数的 Pareto 组合；1.2.2 是内层在线协同控制器，两者构成离线整定与在线执行的分层关系。")

    add_heading(document, "三 已实现的协同控制架构", 1)
    add_heading(document, "3.1 总体分层结构", 2)
    add_body(document, "现有原型采用“任务与飞行层、模糊能量管理层、安全执行层、评价优化层”的分层架构。任务与飞行层根据高度、速度和航迹角状态计算瞬时推进功率需求；模糊能量管理层根据阶段名、SOC 偏差、功率需求比例和跟踪误差形成发动机目标功率；安全执行层将该目标投影到由发动机上限、电机助力能力和功率变化率共同定义的可行域；飞推控模型据此更新发动机功率、电机功率、燃油和 SOC；评价优化层计算三目标和约束，并可由 MOEA/D 离线搜索控制器参数。")
    add_body(document, "这种结构的关键边界是：人工智能或模糊模块只生成受约束的功率分配倾向，不直接向执行器写入无边界控制指令。即使未来接入神经网络，其输出也只能在名义 SOC 目标附近有限修正，最终仍由安全层进行可行性投影。因此，学习模块不承担安全约束本身，安全由模型边界和监督逻辑承担。")

    add_heading(document, "3.2 模糊自适应能量管理器", 2)
    add_body(document, "已实现 FuzzyAdaptiveCoordinator 模糊自适应调度器。该控制器以阶段 SOC 目标、当前 SOC、所需推进功率与发动机能力的比值、速度与高度跟踪误差为输入。其规则关系可概括为：SOC 低于目标时增加发动机充电偏置；SOC 高于目标且处于爬升阶段时允许有限电机助力；跟踪误差较大时保留一定功率储备；推进需求接近发动机能力上限时优先维持发动机供能，避免因主动放电造成后续 SOC 约束失效。")
    add_body(document, "针对 80 h 留空任务，控制器采用保守的阶段 SOC 参考轨迹：主爬升 0.76、转入 ISR 高度 0.70、长时间留空 0.62、长下降 0.55、进近下降 0.48、拉平 0.38。主动电机助力仅在主爬升和转入 ISR 高度阶段开放，留空、下降、进近与拉平阶段不再因 SOC 较高而主动放电，从控制结构上保护终端能量储备。")
    add_body(document, "已实现的工程化保护包括三项。第一，功率可行性投影确保发动机目标功率不低于“所需推进功率减去可用电机功率”的下界，避免发动机与电机总功率不足。第二，发动机目标功率受到 3.0 kW/s 的变化率限制，抑制离散时间步之间的突变。第三，调度结果依据电机功率平衡自动标记为模糊电机助力、模糊电池充电、模糊发动机跟随或安全受限四类模式，便于后续切换统计和模式优化。")

    add_heading(document, "3.3 神经网络预测接口", 2)
    add_body(document, "已定义 SOCReferencePredictor 接口，用于接入离线训练的神经网络预测器。该接口接收当前调度观测量和名义 SOC 目标，输出下一时段的建议 SOC 目标；主调度器将其限制在名义值上下 0.06 的范围内。因此，神经网络未来的功能是基于任务阶段、飞行状态、功率需求和环境信息预测更合适的 SOC 参考轨迹或等效能耗因子，而不是直接替代发动机、电机执行器控制。")
    add_body(document, "目前未训练神经网络。这是有意保留的阶段边界：现有发动机、电机和电池模型仍是低阶或预估模型，用其批量生成标签会使网络学习到未校核的部件特性。待动能学院提供经验证的效率图、动态响应和热安全边界后，可使用离线动态规划结果、MOEA/D 优化控制轨迹以及高保真仿真数据构建训练集；初期宜采用小型多层感知机预测短期 SOC 目标，具有足够时序样本后再比较 GRU 等循环网络。")

    add_heading(document, "3.4 健康状态监督与容错接口", 2)
    add_body(document, "已实现 FaultTolerantFuzzyCoordinator 健康监督包装器，建立发动机可用功率比例、电机可用功率比例、电池可用能量比例和传感器数据有效性四项健康状态接口。发动机或电机降额时，监督器在决策层同步收紧相应的功率边界；电池可用能量低于 75% 时，提高 SOC 保护目标；SOC 或关键传感器失效时，转入保守的传感器降级模式，由发动机尽可能跟踪当前推进需求且不再主动指令电池放电。")
    add_body(document, "该模块目前完成的是容错决策逻辑和接口，不是故障物理过程的仿真验证。现有飞推控对象尚未包含发动机故障后的转速扭矩响应、电机失效后的转矩缺失、电池内阻增长与温度升高等状态方程，因此健康状态仅能影响决策可用边界。真实故障注入、检测阈值整定、故障隔离率和任务成功率评价，应在 1.2.1 接入高保真部件模型后完成。")

    add_heading(document, "四 程序实现与可复现实验", 1)
    add_body(document, "1.2.2 的主体程序为 ptdl_intelligent_energy_management.py，其中包含模糊能量管理参数、SOC 参考预测器协议、模糊自适应调度器、健康状态数据结构和容错监督器。调度器遵循既有 DispatchObservation 与 PowerDispatch 接口，可直接传入 ptdl_flight_propulsion_control.py 的全任务仿真循环，无需改变 2.1 或 2.2.1 的评价函数字段。")
    add_body(document, "对比试验由 run_ptdl_1_2_2_fuzzy_comparison.py 完成。试验从 H=6、VSPAERO 修正后的 MOEA/D 可行 Pareto 解集中选择综合能耗最优构型，沿用同一 80 h 任务剖面、同一发动机功率上限、同一电机辅助能力和同一气动修正，分别运行阶段定值调度、规则型 SOC 反馈调度和模糊自适应调度。该设置使三种策略的差异只来自功率分配逻辑，而非构型或任务条件变化。")
    add_body(document, "已补充单元测试，覆盖 SOC 偏低时的充电倾向、跟踪误差增大时的电机助力倾向和传感器无效时的保守降级模式。当前智能调度模块与飞推控、协同模型、80 h 任务评价等相关测试共 12 项均通过。")

    add_heading(document, "五 初步结果与正确解读", 1)
    add_body(document, "在上述统一算例下，三种调度策略均完成 80.0 h 任务，说明当前候选构型和低阶飞推控模型能够支撑 80 h 约束。阶段定值策略的燃油消耗为 474.553 kg，综合能耗目标为 20411.618 MJ，终端 SOC 为 0.550；规则型 SOC 反馈策略的燃油消耗为 474.160 kg，综合能耗目标为 20399.398 MJ，终端 SOC 为 0.318；模糊自适应策略的燃油消耗为 474.693 kg，综合能耗目标为 20415.406 MJ，终端 SOC 为 0.684。")
    add_body(document, "与阶段定值策略相比，规则型 SOC 反馈策略在当前低阶模型中将综合能耗目标降低约 0.06%，但最低 SOC 已接近 0.25 的安全下限。模糊自适应策略的稳定性目标略有降低，为 62.5094，相比阶段定值的 62.5097 变化很小；其综合能耗略高，原因是当前保守 SOC 参考使其在全任务末端保留了较大电量。这个结果说明模糊控制链路已经可以安全运行，但参数尚未完成针对能耗与终端 SOC 的优化整定，不能把它描述为优于既有规则控制的最终方案。")
    if FIGURE.exists():
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(FIGURE), width=Cm(15.3))
        caption = document.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_spacing(caption, after=8, line=1.2)
        run = caption.add_run("图 1  三种能量管理策略的 SOC 与发动机电机功率时程对比")
        set_run_font(run, "宋体", 9.5, color="555555")
    add_body(document, "图中可见，模糊自适应策略的发动机功率变化更集中于阶段目标附近，电机仅在允许的爬升阶段有限参与助力，长时间留空段基本避免主动放电；这与其较高终端 SOC 和偏保守能耗表现一致。后续应由 MOEA/D 对 SOC 目标轨迹、充电偏置、助力偏置、跟踪储备和功率变化率进行联合整定，使安全裕度与能耗目标形成可量化的 Pareto 权衡。")

    add_heading(document, "六 如何形成自适应 容错协同控制方法体系", 1)
    add_body(document, "建议以“安全约束为底座，模型驱动调度为主，数据驱动预测为辅，外层多目标优化整定”的路线形成完整方法体系。在线层采用模糊自适应或后续的模糊自适应等效能耗最小化策略，保证规则可解释、可在线运行并便于接受硬约束。预测层采用神经网络根据任务剩余时间、飞行阶段、SOC、载荷和环境信息修正 SOC 参考或等效因子，改善长航时条件下的前瞻性。离线层由 MOEA/D 以综合能耗、稳定性和响应性能为目标，对控制器参数和参考轨迹进行 Pareto 搜索。")
    add_body(document, "容错层应独立于学习层。其职责包括基于模型残差、传感器一致性检查和部件状态估计识别发动机降额、电机能力下降、电池容量或热边界收紧以及传感器失真；一旦检测到异常，立即收紧可用功率或可用能量边界，并切换至满足任务安全优先级的降级模式。学习模块可辅助预测，但不应绕开故障监督器和约束投影层。")
    add_body(document, "对于强化学习，建议后续作为对比算法而非当前主控制器。原因是强化学习需在覆盖正常和故障场景的大规模、高可信度仿真数据上训练，且其探索性动作需要额外安全屏障；在动力系统高保真模型和故障模型尚未接入前，直接将其作为主控制器不利于结论可信度。")

    add_heading(document, "七 后续工作与动能学院接口", 1)
    add_body(document, "第一，接入并校核发动机发电机、电机逆变器、电池和螺旋桨模型。发动机侧至少需要功率转速扭矩边界、燃油消耗图、动态响应和启停限制；电池侧需要 OCV SOC 曲线、内阻或等效电路参数、电流和热边界、容量及 SOH 演化；电机侧需要效率图、转矩转速边界、热限制和响应时间。只有这些模型到位，健康降额和切换延迟才具有物理含义。")
    add_body(document, "第二，构建离线训练与整定数据集。在多任务剖面、载荷、温度、风扰和部件退化组合下，使用高保真模型、直接法或动态规划生成参考调度轨迹，并记录状态、预测特征、SOC 目标和功率分配。此后分别训练小型前馈网络与 GRU 预测器，比较预测误差、端部 SOC 偏差、计算时间和控制闭环收益。")
    add_body(document, "第三，完成控制器参数的 MOEA/D 联合优化。优先优化阶段 SOC 目标、SOC 死区、充电偏置上限、助力偏置上限、跟踪储备、发动机功率变化率和模式切换阈值；将 80 h 续航作为硬约束，在满足安全边界的可行解中分析能耗、稳定性和响应性能的 Pareto 前沿。")
    add_body(document, "第四，开展故障和扰动验证。建议至少设置发动机可用功率下降、电机峰值功率下降、电池容量衰减、温度限制收紧、SOC 传感器失效、风扰和任务载荷变化等工况，比较无监督控制、健康监督控制和智能预测辅助控制的任务完成率、约束违反度、燃油、SOC 裕度和响应指标。")

    add_heading(document, "八 当前完成边界汇总", 1)
    add_body(document, "1.2.2 已完成一个可运行、可测试、可接入 MOEA/D 的协同控制与智能调度原型。其实际成果是模糊自适应功率分配、神经网络参考预测预留接口、安全可行性投影、健康状态监督接口和统一 80 h 对比试验；其当前价值在于为真实动力模型接入后开展控制器整定、算法对比和容错验证提供稳定的软件骨架。神经网络训练、真实故障机理建模、部件动态验证及多工况鲁棒性结论仍属于下一阶段工作。")

    add_heading(document, "参考文献", 1)
    add_body(document, "[1] NASA. Turboelectric Energy Management for Aircraft Propulsion Technology. NASA Electrified Aircraft Propulsion Program, 2026.")
    add_body(document, "[2] Ji S, Wang J, Sun F. Reinforcement Learning Based Energy Management for Hybrid UAV. Electronics, 2021, 10(16): 1929.")
    add_body(document, "[3] PTDL 项目组. PTDL 1.2.1 混合动力与飞行控制协同模型完成情况说明. 2026.")
    add_body(document, "[4] PTDL 项目组. PTDL 2.2.1 多目标性能指标体系与评价函数交付件. 2026.")
    add_body(document, "[5] PTDL 项目组. PTDL 2.2.2 多目标优化控制算法文献调研与适配报告. 2026.")

    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
