from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PTDL 1.2.2 协同控制与智能调度算法开发 定位与阶段成果报告.docx"


def set_font(run, name: str = "宋体", size: float = 11, bold: bool = False, color: str = "000000") -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_spacing(paragraph, before: float = 0, after: float = 6, line: float = 1.45) -> None:
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = line


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_font(run, size=9, color="555555")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)
    run = paragraph.add_run(" 页")
    set_font(run, size=9, color="555555")


def configure(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.25)
    section.bottom_margin = Cm(2.1)
    section.left_margin = Cm(2.45)
    section.right_margin = Cm(2.25)
    section.header_distance = Cm(0.9)
    section.footer_distance = Cm(0.9)

    normal = document.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)

    title = document.styles["Title"]
    title.font.name = "黑体"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.font.size = Pt(18)
    title_ppr = title._element.get_or_add_pPr()
    title_borders = title_ppr.find(qn("w:pBdr"))
    if title_borders is not None:
        title_ppr.remove(title_borders)

    for level, size in ((1, 15), (2, 12.5), (3, 11.5)):
        style = document.styles[f"Heading {level}"]
        style.font.name = "黑体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.size = Pt(size)

    bullet = document.styles.add_style("PTDL Bullet", WD_STYLE_TYPE.PARAGRAPH)
    bullet.base_style = document.styles["Normal"]
    bullet.paragraph_format.left_indent = Cm(0.72)
    bullet.paragraph_format.first_line_indent = Cm(-0.5)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("PTDL 项目研究内容一 1.2.2 阶段报告")
    set_font(run, size=9, color="555555")
    add_page_number(section.footer.paragraphs[0])


def add_title(document: Document) -> None:
    paragraph = document.add_paragraph(style="Title")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(62)
    paragraph.paragraph_format.space_after = Pt(16)
    set_font(paragraph.add_run("PTDL 1.2.2 协同控制与智能调度算法开发"), "黑体", 18, True)

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_spacing(paragraph, after=18, line=1.2)
    set_font(paragraph.add_run("总体框架定位与阶段成果报告"), "黑体", 14, True)

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_spacing(paragraph, after=7, line=1.2)
    set_font(paragraph.add_run("面向导师汇报的当前实现范围、作用机制与后续接口说明"), size=10.5, color="555555")

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_spacing(paragraph, after=10, line=1.2)
    set_font(paragraph.add_run("PTDL 项目  2026 年 9 月"), size=10, color="555555")
    document.add_page_break()


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    set_spacing(paragraph, before=15 if level == 1 else 10, after=7, line=1.2)
    set_font(paragraph.add_run(text), "黑体", 15 if level == 1 else 12.5, True)


def add_body(document: Document, text: str, lead: str | None = None) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    set_spacing(paragraph, after=7, line=1.48)
    if lead:
        set_font(paragraph.add_run(lead), "宋体", 11, True)
    set_font(paragraph.add_run(text), "宋体", 11)


def add_bullet(document: Document, lead: str, text: str) -> None:
    paragraph = document.add_paragraph(style="PTDL Bullet")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_spacing(paragraph, after=4, line=1.4)
    set_font(paragraph.add_run("- "), "宋体", 11)
    set_font(paragraph.add_run(lead), "宋体", 11, True)
    set_font(paragraph.add_run(text), "宋体", 11)


def add_flow(document: Document) -> None:
    lines = (
        "候选构型与控制参数 z  由 2.2.2 MOEA/D 离线生成",
        "↓",
        "1.2.1 飞推控闭环物理仿真  任务参考  飞行控制  推进系统  气动与纵向动力学",
        "↓                         ↑",
        "1.2.2 在线协同控制与智能调度  根据 SOC 跟踪误差 功率需求与健康状态实时调度",
        "↓",
        "2.2.1 评价函数与约束判定  综合能耗 飞行稳定性 任务响应性能 80 h 续航硬约束",
        "↓",
        "目标函数值 约束违反度 可行性结果  返回 MOEA/D 更新 Pareto 解集",
    )
    for index, line in enumerate(lines):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_spacing(paragraph, after=1 if line == "↓" else 3, line=1.1)
        color = "C55A11" if "1.2.2" in line else "1F4E79" if "MOEA/D" in line or "2.2.1" in line else "000000"
        set_font(paragraph.add_run(line), "宋体", 10.5 if index not in (1, 3, 5, 7) else 12, "1.2.2" in line, color)


def main() -> None:
    document = Document()
    configure(document)
    add_title(document)

    add_heading(document, "一 报告结论与研究定位")
    add_body(document, "1.2.2 的核心定位是嵌入飞推控内层闭环仿真中的在线协同决策层。它不重新定义 2.2.1 的多目标评价函数，不替代 2.2.2 的 MOEA/D 外层寻优，也不重复建立 1.2.1 的发动机、电池、电机、螺旋桨和飞行动力学模型；它解决的是在飞行过程中，混合动力系统应如何依据实际状态分配功率和保持能量储备的问题。", "核心结论：")
    add_body(document, "在总体架构中，1.2.1 给出可计算的飞推控物理对象和状态反馈通道，1.2.2 利用这些反馈实时输出发动机目标功率、电机充放电倾向、SOC 参考和降级模式，2.2.1 将任务结果转化为目标函数与约束，2.2.2 再用 MOEA/D 对构型、阶段控制和调度参数进行离线联合寻优。因此，1.2.2 的作用是把原有的“阶段给定功率”扩展为“根据真实飞行状态动态修正的功率分配”，使飞行、推进和能量管理形成真正闭合的控制回路。")
    add_body(document, "当前已完成模糊自适应能量管理、功率可行性投影、健康状态监督、传感器失效保守回退、多工况对比与调度参数 MOEA/D 预寻优接口。神经网络训练、真实部件退化机理、MPC 与数据驱动控制的正式对比，以及基于真实动力部件参数的联合优化，仍属于下一阶段工作。")

    add_heading(document, "二 1.2.2 在总体框架中的工作位置")
    add_body(document, "PTDL 当前采用“外层离线优化、内层物理仿真、在线实时调度、统一评价反馈”的分层体系。外层优化负责在大量候选方案中搜索满足 80 h 续航约束的设计与控制参数组合；内层物理仿真负责计算候选方案在任务剖面下的真实状态演化；在线调度负责在每一个积分时间步把飞行需求转化为发动机、电机和电池的实际协同供能指令；评价层负责判断候选方案是否满足安全、能量与任务边界。")
    add_flow(document)
    add_body(document, "上图中的关键关系是：1.2.2 位于飞行控制器与混合动力推进对象之间，同时读取推进对象和飞行状态的反馈。飞行控制器根据高度、速度和航迹误差提出所需推力或推进功率；1.2.2 不直接改变飞行参考，而是决定这部分推进功率由发动机持续供给、电机短时助力还是发动机带动电池充电来满足。发动机和电机的实际输出再经螺旋桨、气动模型和纵向动力学作用于速度与高度，形成下一时刻的误差反馈。")

    add_heading(document, "三 1.2.2 为整体框架新增的能力")
    add_heading(document, "3.1 从阶段定值分配到状态反馈分配", 2)
    add_body(document, "原有全任务仿真能够按照阶段给定发动机功率上限、电机辅助能力、速度和高度参考完成任务计算，但该方式本质上是开环或弱反馈的阶段调度。1.2.2 新增后，SOC、发动机可用功率、实际推进功率需求、速度误差和高度误差均进入调度决策。即使任务阶段不变，发动机目标功率和电机功率也可以随当前状态变化，使调度决策具有在线适应性。")
    add_heading(document, "3.2 增加受约束的能量管理逻辑", 2)
    add_body(document, "1.2.2 不是仅用规则决定“发动机多发还是电机多发”，而是在调度结果之后设置安全执行边界。发动机目标功率首先受到发动机能力上限和发动机加电变化率限制；其次不得低于“推进所需功率减去可用电机最大功率”的可行性下界；电机指令再受到充电、放电、功率和 SOC 边界限制。这样，智能或模糊模块只能在物理可行域内调整功率分配，不能以违反功率平衡或耗尽电池为代价改善表面目标。")
    add_heading(document, "3.3 增加面向部件异常的监督与降级路径", 2)
    add_body(document, "1.2.2 增加了发动机可用功率比例、电机可用功率比例、电池可用能量比例和传感器有效性四类健康状态输入。发生发动机或电机降额时，调度层输出的功率边界会同步传递到实际动力系统，因而降额会改变真实推力、速度和高度状态，而不是停留在日志标记层面。若关键传感器数据无效，控制器进入保守回退模式：发动机在能力范围内跟踪当前推进需求，电池不再被主动用于助力。")
    add_heading(document, "3.4 将智能调度参数转化为可优化设计变量", 2)
    add_body(document, "1.2.2 将阶段 SOC 目标偏置、充电偏置上限、电机助力偏置上限、跟踪功率储备和发动机目标功率变化率定义为显式控制设计变量。这些变量不再只能凭经验手工设定，而可以由 2.2.2 的 MOEA/D 与翼面积、展弦比、质量分配、阶段速度和功率上限等变量共同搜索。由此建立了“离线整定控制器参数，在线执行调度策略”的闭环开发路径。")

    add_heading(document, "四 在线协同控制的输入 输出与作用机制")
    add_heading(document, "4.1 输入信息")
    add_body(document, "每一时间步，调度器接收任务阶段名称、推进功率需求、当前 SOC、阶段名义 SOC 目标、发动机和电机可用功率边界、速度误差、高度误差、时间步长及健康状态。任务阶段决定长期能量策略，例如爬升阶段允许有限电机助力，长时间留空阶段强调 SOC 保持；误差与功率需求决定短时功率储备；健康状态决定可用功率和可用能量边界。")
    add_heading(document, "4.2 模糊自适应调度")
    add_body(document, "当前 FuzzyAdaptiveCoordinator 以 SOC 偏差、推进需求比例和跟踪需求构成可解释的模糊规则。SOC 低于阶段目标时，增加发动机充电偏置；SOC 较高且处于主爬升或转入 ISR 高度阶段时，允许有限电机助力；当速度或高度跟踪误差增大时，保留一定的功率储备；当推进需求接近发动机能力上限时，优先让发动机满足推进需求，避免过早主动放电。控制器输出发动机目标功率、SOC 目标和运行模式，实际电机功率由功率平衡和电池边界共同确定。")
    add_heading(document, "4.3 飞推控闭环中的实际作用")
    add_body(document, "发动机与电机实际功率之和决定螺旋桨可用轴功率和推力；推力与气动阻力、重力沿航迹分量共同决定速度和爬升率；速度和高度的实际演化又产生新的跟踪误差。1.2.2 正是在这个循环中把 SOC 管理与飞行性能连接起来：它既不能只追求低燃油而忽略 SOC 储备，也不能为维持较高 SOC 而无限增加发动机功率。其输出最终会通过 2.2.1 的综合能耗、稳定性和响应性能目标被量化评价。")

    add_heading(document, "五 当前程序实现与已完成验证")
    add_body(document, "已将当前可运行代码整理为“PTDL 飞推控协同优化代码包”。其中“智能能量管理与容错调度.py”实现模糊自适应调度、SOCReferencePredictor 预留接口、健康状态结构和 FaultTolerantFuzzyCoordinator；“飞推控闭环物理模型.py”接收调度结果并计算实际发动机、电机、螺旋桨、气动和纵向状态；“MOEAD多目标联合优化.py”提供外层 MOEA/D；“指标体系与评价函数.py”实现 2.2.1 的三目标和约束判定。")
    add_body(document, "已完成四类标准工况的统一对比脚本，覆盖起飞后初始爬升、低速高升力、稳定巡航和无动力滑翔；每类工况比较阶段定值、规则 SOC 反馈、模糊自适应和发动机降额模糊调度。脚本输出燃油消耗、SOC 变化、速度与高度误差、调节时间、功率模式切换、载荷因子及气动阻力系数时程，为后续在真实部件模型下开展算法对比提供固定的数据接口。")
    add_body(document, "已完成一次固定 80 kg 预估混动系统质量下的模糊调度参数 MOEA/D 预寻优，采用 H=3、10 个权重子问题、16 代。10 个最终候选均满足 80 h 续航硬约束，但去重后只得到 1 个有效调度解。这一结果说明当前固定稳态任务对三项目标的调度差异激励不足，尚未形成有离散度的控制参数 Pareto 集；它验证了“调度参数可进入 MOEA/D”的程序链路，而不能被表述为最终最优控制参数已经确定。")
    add_body(document, "核心回归测试已通过 6 项，包括实际发动机输出改变飞行状态、任务时序记录请求和实际推进功率、部件降额传递至实际动力边界、低 SOC 充电倾向、跟踪误差助力倾向和传感器失效保守回退。")

    add_heading(document, "六 与 2.2.1 2.2.2 和 1.2.1 的协同关系")
    add_body(document, "四个部分不是并列的四套模型，而是同一技术链条的不同层级。2.2.1 给出统一的“如何评价”标准，即综合能耗、飞行稳定性、任务响应性能和 80 h 续航等硬约束；2.2.2 给出“如何搜索”方案，即 MOEA/D 对多目标问题进行权重分解、邻域更新和 Pareto 解集维护；1.2.1 给出“候选方案如何在物理上运行”的飞推控对象；1.2.2 给出“飞行过程中如何实时分配混合动力”的在线策略。")
    add_body(document, "因此，1.2.2 的引入使外层 MOEA/D 不再只优化静态构型和阶段给定功率，而能够优化具有反馈、限幅、SOC 保护和故障降级能力的控制器。对于同一个候选构型，改变 1.2.2 的 SOC 目标、充电偏置、助力偏置或功率变化率，会改变燃油消耗、电池能量轨迹、瞬态推力可用性和跟踪响应，继而改变 2.2.1 的目标值与约束可行性。")

    add_heading(document, "七 当前边界与动能学院模型替换接口")
    add_body(document, "目前 1.2.2 已完成的是控制软件骨架和低阶闭环验证，尚未完成真实部件参数支撑下的工程定型。发动机功率上限、燃油流量、动态响应，电机效率与热限制，电池容量、内阻、温度和衰减边界仍为可替换的预估模型。现有接口已将发动机发电机模型和混合动力系统模型设计为可替换对象，动能学院提供模型后不需要修改 2.2.1 的评价逻辑或 2.2.2 的 MOEA/D 框架。")
    add_bullet(document, "发动机侧接口：", "需要功率转速扭矩边界、燃油消耗图、功率动态响应、启停限制和性能退化规律，用于替换当前参考发动机模型。")
    add_bullet(document, "电池与电机侧接口：", "需要 OCV SOC 曲线、内阻或等效电路、电流与热边界、容量衰减、效率图、转矩转速边界和动态响应，用于替换当前参考混合动力模型。")
    add_bullet(document, "故障与环境侧接口：", "需要故障模式、检测阈值、风扰与温度场景，用于将当前的可用功率比例降额扩展为可验证的故障机理与鲁棒性试验。")

    add_heading(document, "八 下一阶段工作")
    add_body(document, "第一，基于动能学院的真实部件模型校准发动机、电池和电机动态，使功率变化率、效率、热约束、退化和故障降额具有物理意义。第二，重新设计具有充分动态激励的起飞、低速高升力、巡航、滑翔和扰动工况，并以调节时间、超调量、SOC 裕度、燃油和约束违反度形成标准对比指标。第三，在真实模型基础上实现 MPC 或自适应控制对照组，并将模糊调度参数纳入 H 更高、重复随机种子更多的 MOEA/D 正式联合优化。")

    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
