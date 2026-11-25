from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


OUT = r"C:\Users\23598\Documents\ChatGPT\LDTP\PTDL研究内容一二个人执行细化方案.docx"


def set_run(run, font="宋体", size=10.5, bold=False, color=None):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_para(p, before=0, after=6, line=1.10, left=None, first=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if left is not None:
        pf.left_indent = Cm(left)
    if first is not None:
        pf.first_line_indent = Cm(first)


def style_doc(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 16, 8),
        ("Heading 2", 13, "2E74B5", 12, 6),
        ("Heading 3", 12, "1F4D78", 8, 4),
    ]:
        st = doc.styles[name]
        st.font.name = "微软雅黑"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(color)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        if level == 1:
            set_run(r, "微软雅黑", 16, True, "2E74B5")
        elif level == 2:
            set_run(r, "微软雅黑", 13, True, "2E74B5")
        else:
            set_run(r, "微软雅黑", 12, True, "1F4D78")
    return p


def para(doc, text="", label=None):
    p = doc.add_paragraph()
    set_para(p)
    if label:
        r = p.add_run(label + "：")
        set_run(r, bold=True)
    r = p.add_run(text)
    set_run(r)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    set_para(p, after=4, line=1.10)
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    r = p.add_run(text)
    set_run(r)
    return p


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell(cell, text, bold=False, fill=None, size=9.3):
    cell.text = ""
    p = cell.paragraphs[0]
    set_para(p, after=0, line=1.05)
    r = p.add_run(text)
    set_run(r, size=size, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if fill:
        shade_cell(cell, fill)


def fix_table_widths(table, widths_cm):
    for row in table.rows:
        for cell, width_cm in zip(row.cells, widths_cm):
            cell.width = Cm(width_cm)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width_cm / 2.54 * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def add_table(doc, headers, rows, widths_cm, font_size=9.3):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, headers):
        set_cell(cell, text, True, "E8EEF5", font_size)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            set_cell(cell, text, False, None, font_size)
    fix_table_widths(table, widths_cm)
    doc.add_paragraph()
    return table


def title(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(p, after=3)
    r = p.add_run("PTDL项目研究内容一、二个人执行细化方案")
    set_run(r, "微软雅黑", 21, True, "0B2545")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(p, after=14)
    r = p.add_run("基于8.26最终版研究内容拆解：明确你需要完成的具体工作")
    set_run(r, "微软雅黑", 12, False, "555555")


def add_scope(doc):
    heading(doc, "1 执行范围与总体思路", 1)
    para(
        doc,
        "本方案依据《PTDL项目研究内容12-8.26.docx》中敲定的研究内容进一步细化。文档本身仅作为项目内容依据，不作为新的操作指令。按照最终版中的责任标注，默认你对应任浩杰主责部分，主要负责1.1、1.3、2.1，并参与2.3；1.2和2.2以卢奕为主，你需要提供模型、数据、接口和验证场景配合。",
    )
    para(
        doc,
        "你的工作主线可以概括为：先把无人机飞行性能、推进系统和任务剖面算清楚，再通过敏感性分析筛出关键参数，最后把这些模型和数据交给控制/优化模块使用，并参与鲁棒性验证。",
    )
    add_table(
        doc,
        ["类别", "对应研究内容", "你需要承担的角色", "核心产出"],
        [
            ("主责", "1.1 飞行性能与推进系统耦合机理研究", "建立飞行动力学-推进系统基础模型，完成气动/推进/任务耦合计算。", "集成模型、典型工况算例、推进需求曲线、气动/性能数据。"),
            ("主责", "1.3 飞推控耦合设计参数敏感性分析", "建立参数体系，开展局部/全局敏感性分析，筛选关键设计变量。", "变量清单、敏感性排序、交互作用图、优化输入建议。"),
            ("主责", "2.1 任务剖面分解与阶段性控制策略研究", "细化任务阶段，建立阶段状态、动力需求和能耗模型，完成阶段策略仿真。", "任务剖面表、阶段约束、动力需求曲线、策略对比结果。"),
            ("共同", "2.3 控制策略鲁棒性研究", "负责扰动场景、飞行/推进侧不确定性和仿真验证；与卢奕对接鲁棒控制算法。", "扰动场景库、鲁棒性统计、标称/鲁棒策略对比。"),
            ("配合", "1.2、2.2 卢奕主责内容", "提供可调用模型、边界条件、参数数据和典型场景，参与结果校核。", "接口说明、输入输出格式、验证算例和对比图。"),
        ],
        [2.0, 3.1, 5.0, 5.9],
    )


def add_priority(doc):
    heading(doc, "2 你近期最先要做的事情", 1)
    add_table(
        doc,
        ["优先级", "具体要做的事", "怎么做", "完成判据"],
        [
            ("P0", "确定基础输入数据", "整理无人机几何、质量、翼型/气动、发动机、电池、螺旋桨、任务剖面和环境条件。缺失数据先给出合理假设和来源说明。", "形成一张基础参数表，所有参数有单位、取值、来源和是否待确认标记。"),
            ("P0", "统一变量和单位", "统一高度、速度、迎角、航迹角、升阻系数、推力、功率、燃油、电池SOC等变量符号。", "形成变量字典，后续所有模型和图表使用同一套命名。"),
            ("P1", "跑通飞行性能-推进需求计算链", "用Python计算不同高度/速度/质量下的升阻、需用推力、需用功率和燃油/等效能耗。", "至少跑通巡航、爬升、下降三个典型工况，并输出曲线。"),
            ("P1", "建立任务剖面基准算例", "把起飞、爬升、巡航/侦察、返航、下降、着陆拆成阶段，设置初末状态和约束。", "形成一套可复现实验基准剖面，能输出阶段时间、航程、能耗和功率需求。"),
            ("P2", "开展敏感性分析", "先做单因素扫描，再对关键变量做Morris/Sobol或Monte Carlo分析。", "得到关键变量排序和至少3类主要耦合关系解释。"),
            ("P2", "与卢奕对接控制/优化接口", "把模型输入输出整理成固定格式，提供功率边界、气动数据、任务约束和评价指标。", "卢奕可以直接调用你的数据开展控制策略和多目标优化。"),
            ("P3", "开展鲁棒性验证", "设置风、温度、空气密度、发动机效率、电池衰减、任务距离变化等扰动，批量仿真。", "输出约束违反概率、燃油/能耗增量、航时变化和终端误差统计。"),
        ],
        [1.5, 4.5, 5.5, 5.0],
    )


def add_11(doc):
    heading(doc, "3 研究内容1.1：你要完成的细化任务", 1)
    heading(doc, "3.1 1.1.1 飞行动力学与推进系统集成建模", 2)
    para(doc, "这一部分是你后续所有分析的地基。重点不是一开始就做复杂模型，而是先搭出能跑通、能校核、能扩展的中低保真模型。")
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("数据整理", "整理几何、质量、气动极曲线、发动机功率/比油耗、电池容量/SOC边界、螺旋桨效率和任务剖面参数。", "按参数类别建表，缺失值先标记为假设值。", "Excel/CSV、Python", "基础参数表、变量字典。"),
            ("气动模型", "建立升力、阻力、升阻比与高度、速度、迎角、质量之间的关系。", "采用气动极曲线、RCAIDE快速分析和必要的CFD校核。", "RCAIDE、STAR-CCM+或Fluent、Python", "CL/CD数据库、典型工况气动结果。"),
            ("推进需求模型", "计算不同阶段的需用推力、需用功率、功率裕度和燃油/等效能耗。", "准定常纵向动力学为基础，爬升/下降考虑航迹角和能量变化。", "Python、NumPy/SciPy", "Treq/Preq曲线、功率包线。"),
            ("混动能量模型", "描述发动机、电池、电机/发电机和螺旋桨之间的能量流。", "建立功率平衡、SOC变化、效率修正和工作边界。", "Python、RCAIDE", "能量流模型、SOC变化曲线。"),
            ("模型校核", "检查各模块输入输出是否一致，选取典型工况进行手算或文献/软件结果对比。", "单位检查、边界检查、单点算例校核。", "Python、pytest", "验证算例、误差说明。"),
        ],
        [2.1, 4.5, 3.9, 3.0, 3.0],
        8.7,
    )
    bullet(doc, "注意：若使用Treq = D + W sin(gamma)等简化关系，需要注明适用条件为准定常飞行；若分析加速/减速或机动过程，要补充惯性项。")
    bullet(doc, "混合动力部分建议统一使用“燃油消耗 + 电能消耗/SOC变化 + 等效能耗”的口径，不只看燃油。")

    heading(doc, "3.2 1.1.2 参数敏感性与耦合效应定量分析", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("确定参数集", "把翼载荷、推重比、发动机功率、电池容量、螺旋桨直径/效率、巡航高度、巡航速度等列为候选变量。", "按气动/推进/能量/任务四类整理变量边界。", "Excel、Python", "参数边界表。"),
            ("局部敏感性", "逐一改变关键参数，观察航时、航程、能耗、功率裕度、稳定性指标的变化。", "单因素扫描、有限差分、局部梯度。", "Python、Matplotlib/Plotly", "单因素响应曲线。"),
            ("全局敏感性", "分析多参数同时变化时的主效应和交互效应。", "Morris筛选、Sobol指数、Monte Carlo。", "Python、SALib", "敏感性排序、交互作用图。"),
            ("典型场景扫描", "在低温、高空、低密度空气、不同载荷和不同巡航任务下重复分析。", "建立场景集并批量运行。", "Python、RCAIDE", "多工况性能响应图。"),
            ("解释关键耦合", "说明哪些参数影响最大，以及为什么影响大。", "结合物理机理解释，如翼载荷-阻力-功率、电池容量-重量-航时。", "Python、图表后处理", "关键参数建议。"),
        ],
        [2.1, 4.5, 3.9, 3.0, 3.0],
        8.7,
    )

    heading(doc, "3.3 1.1.3 仿真驱动的改进与优化策略", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("模型修正", "根据RCAIDE、CFD或试验/文献数据修正气动和推进效率参数。", "参数辨识、曲线拟合、误差最小化。", "Python、RCAIDE、CFD软件", "修正后的耦合模型。"),
            ("动力匹配建议", "给出发动机功率、电池容量、螺旋桨参数和任务阶段功率需求之间的匹配关系。", "基于功率裕度、能耗和重量影响进行权衡。", "Python、pymoo/SciPy", "动力匹配方案。"),
            ("能量分配策略", "提出爬升、巡航、侦察、返航、下降阶段的发动机/电池使用原则。", "规则策略加参数优化，比较不同策略能耗。", "Python", "阶段能量分配策略。"),
            ("优化结果整理", "把优化前后航时、能耗、功率裕度和约束满足情况做对比。", "统一指标和图表模板。", "Python、Plotly/Matplotlib", "优化对比图和说明。"),
        ],
        [2.1, 4.8, 4.0, 3.0, 2.7],
        8.7,
    )


def add_13(doc):
    heading(doc, "4 研究内容1.3：你要完成的细化任务", 1)
    heading(doc, "4.1 1.3.1 全剖面飞推控参数化模型构建", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("参数体系", "把飞行性能、推进系统、控制/任务、气动/几何参数统一到一张总表。", "按变量、固定参数、约束参数、评价指标分类。", "Excel、Python", "参数化模型说明和变量表。"),
            ("边界设定", "给每个设计变量设置上下界、单位和物理含义。", "结合工程约束、导师给定指标和已有模型可行范围。", "Excel、Python", "变量边界和约束表。"),
            ("指标定义", "定义航时、燃油消耗、等效能耗、稳定性、动力效率、功率裕度、终端状态误差等指标。", "统一指标方向，明确最大化/最小化。", "Python", "指标体系和计算口径。"),
        ],
        [2.0, 5.0, 4.2, 2.7, 3.1],
        8.8,
    )

    heading(doc, "4.2 1.3.2 关键参数局部与全局敏感性分析", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("实验设计", "确定样本数量、变量组合和典型任务场景。", "局部分析先少量扫描，全局分析再批量采样。", "Python、SALib", "敏感性实验方案。"),
            ("批量计算", "对每组参数运行任务剖面模型，记录航时、能耗、功率裕度和约束情况。", "自动化批量仿真，失败样本记录原因。", "Python", "样本结果数据库。"),
            ("结果分析", "计算局部斜率、Morris指标、Sobol一阶/总效应指数。", "排序并区分主效应和交互效应。", "Python、SALib", "敏感性分析报告。"),
            ("代理模型", "当仿真计算很慢时，用代理模型近似响应面。", "高斯过程、径向基函数或多项式响应面。", "scikit-learn、SciPy", "代理模型和误差评估。"),
        ],
        [2.0, 5.0, 4.2, 2.7, 3.1],
        8.8,
    )

    heading(doc, "4.3 1.3.3 敏感性驱动的设计优化与参数调控", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("变量筛选", "把变量分成固定参数、主优化变量、需保留裕度参数。", "综合敏感性、工程可调性和风险约束。", "Python、Excel", "变量筛选清单。"),
            ("耦合关系解释", "重点解释翼展-结构重量、机翼面积-翼载荷、功率-爬升能力、高度-推进效率、速度-燃油消耗等关系。", "用曲线和物理机理同时说明。", "Python", "关键耦合规律图。"),
            ("向内容二传递", "把筛选后的变量、边界和约束整理成控制优化可用的输入。", "与卢奕约定输入输出字段。", "Python、Excel/CSV", "内容二输入配置和接口说明。"),
        ],
        [2.0, 5.0, 4.2, 2.7, 3.1],
        8.8,
    )


def add_21(doc):
    heading(doc, "5 研究内容2.1：你要完成的细化任务", 1)
    heading(doc, "5.1 2.1.1 任务剖面精细化分解", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("阶段拆分", "把任务拆成起飞、爬升、巡航、侦察/待机、返航、下降、进近、着陆。", "为每个阶段定义起止高度、速度、航程/时间、载荷和动力需求。", "Python、Excel、RCAIDE", "任务剖面阶段表。"),
            ("状态定义", "统一描述高度、速度、航迹角、推力需求、燃油消耗、电池SOC和累计航程。", "建立状态变量和派生指标。", "Python", "状态变量清单。"),
            ("阶段衔接", "确定每个阶段的初末状态和约束传递关系。", "上一阶段终端状态作为下一阶段初始状态。", "Python、RCAIDE", "阶段衔接条件。"),
            ("基准工况", "建立标准任务场景，作为后续策略对比的统一基准。", "固定载荷、环境、任务距离和初始能量。", "Python", "标准工况集。"),
        ],
        [2.0, 5.0, 4.2, 2.7, 3.1],
        8.8,
    )

    heading(doc, "5.2 2.1.2 阶段控制模型与推进控制策略构建", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("控制变量整理", "区分优化控制变量和策略指令变量，例如推进功率/功率分配系数属于优化变量，速度/高度/爬升率指令属于策略变量。", "避免变量层级混乱。", "Python", "控制变量定义表。"),
            ("阶段目标", "为不同阶段设置目标：起飞安全、爬升功率裕度、巡航航时/能耗、侦察稳定、返航安全、着陆终端状态。", "阶段目标与全任务目标分开定义。", "Python、RCAIDE", "阶段目标函数。"),
            ("约束建模", "设置速度、高度、功率、SOC、爬升率、终端状态和控制变化率约束。", "用路径约束和边界约束表达。", "Python、CasADi/SciPy", "阶段约束清单。"),
            ("策略规则", "提出各阶段发动机/电池工作模式和功率分配规则。", "规则策略先行，再做参数优化。", "Python", "推进控制模式和切换规则。"),
        ],
        [2.0, 5.0, 4.2, 2.7, 3.1],
        8.8,
    )

    heading(doc, "5.3 2.1.3 阶段性控制策略仿真验证与优化", 2)
    add_table(
        doc,
        ["任务", "你具体做什么", "方法", "软件", "输出"],
        [
            ("策略仿真", "在同一基准任务下比较恒速、恒高、最小功率、最大航时、阶梯爬升、滑翔下降等策略。", "批量仿真并统一评价指标。", "Python、RCAIDE", "策略对比结果。"),
            ("动态响应", "分析阶段切换时速度、高度、功率、SOC和控制量的变化。", "检查是否有功率突变、终端偏差或约束违反。", "Python、Plotly/Matplotlib", "动态响应图。"),
            ("策略优化", "调整巡航高度、巡航速度、功率分配和阶段切换点。", "以能耗、航时、功率裕度和控制平滑性作为评价。", "Python、SciPy/pymoo", "优化策略参数。"),
            ("组合建议", "选出每个阶段的最优/近优策略，并形成全任务策略组合。", "从阶段最优过渡到全任务可行。", "Python", "全任务策略组合建议。"),
        ],
        [2.0, 5.0, 4.2, 2.7, 3.1],
        8.8,
    )


def add_23(doc):
    heading(doc, "6 研究内容2.3：共同部分中你要做的事", 1)
    para(
        doc,
        "2.3是共同负责内容。建议你负责扰动场景、飞行/推进侧不确定性建模和仿真统计；卢奕负责鲁棒控制算法主体。你们需要在输入输出接口和评价指标上提前统一。",
    )
    add_table(
        doc,
        ["子任务", "你具体负责", "卢奕主要负责", "共同确认"],
        [
            ("扰动因素建模", "整理风速/阵风、温度、空气密度、发动机效率下降、电池衰减、任务距离变化等扰动范围。", "确认控制算法需要哪些扰动输入。", "扰动变量、分布形式、强度等级。"),
            ("扰动场景库", "生成不同阶段、不同强度的扰动场景，并保证可批量仿真。", "提供控制器对扰动场景的调用方式。", "场景编号、输入字段、仿真次数。"),
            ("鲁棒仿真", "用你的任务剖面模型跑标称策略和鲁棒策略的对比仿真。", "提供鲁棒控制策略或补偿控制参数。", "对比指标和判定标准。"),
            ("统计分析", "统计约束违反概率、终端误差、燃油/等效能耗增量、航时变化和安全裕度。", "解释控制算法在不同扰动下的表现。", "鲁棒性报告和图表。"),
        ],
        [2.5, 5.2, 4.2, 4.6],
    )


def add_interface(doc):
    heading(doc, "7 你需要和卢奕对接的接口", 1)
    add_table(
        doc,
        ["接口内容", "你提供什么", "对方使用目的", "建议格式"],
        [
            ("飞行/推进模型", "给定高度、速度、质量、航迹角和控制量后，输出升阻、推力需求、功率需求、燃油/等效能耗和SOC变化。", "用于控制策略和多目标优化算法调用。", "Python函数或CSV样本表。"),
            ("气动数据库", "不同高度、速度、迎角下的CL、CD、升阻比和CFD修正系数。", "用于控制算法中的性能约束和代理模型训练。", "表格数据和插值说明。"),
            ("任务剖面约束", "每个阶段的初末状态、速度/高度/功率/SOC边界、终端状态要求。", "用于建立优化问题约束。", "阶段约束表。"),
            ("评价指标", "航时、燃油消耗、等效能耗、功率裕度、终端误差、控制平滑性、安全裕度。", "用于多目标评价和Pareto解筛选。", "指标定义表。"),
            ("验证场景", "标准工况、极端工况、扰动场景和不可行场景。", "用于算法验证和鲁棒性测试。", "场景库。"),
        ],
        [2.5, 5.8, 4.2, 3.7],
    )


def add_deliverables(doc):
    heading(doc, "8 最终你应形成的材料清单", 1)
    add_table(
        doc,
        ["材料类型", "具体内容", "用途"],
        [
            ("参数表", "基础参数表、变量字典、设计变量边界、扰动因素清单。", "支撑模型复现和导师检查。"),
            ("模型", "飞行动力学-推进系统集成模型、任务剖面模型、阶段控制模型、能量管理模型。", "支撑仿真、敏感性分析和控制优化。"),
            ("图表", "功率-速度-高度曲面、阶段能耗曲线、SOC变化曲线、敏感性排序图、Pareto前沿、鲁棒性统计图。", "支撑论文/汇报展示。"),
            ("算例", "巡航、爬升、下降、全任务剖面、典型扰动和极端工况算例。", "支撑模型校核和算法验证。"),
            ("报告", "模型说明、敏感性分析报告、任务剖面分析报告、鲁棒性验证报告。", "支撑阶段汇报和最终验收。"),
            ("接口说明", "输入输出字段、单位、约束、评价指标、与卢奕算法模块的对接说明。", "保证团队协同推进。"),
        ],
        [2.2, 9.5, 4.8],
    )

    heading(doc, "9 建议执行顺序", 1)
    for item in [
        "第一步：整理基础参数和任务剖面，先把变量、单位、边界和数据来源统一。",
        "第二步：用Python和RCAIDE跑通基础飞行性能-推进需求计算链，先覆盖巡航、爬升、下降三个典型阶段。",
        "第三步：补充关键工况CFD校核，优先校核对升阻特性和推进效率影响最大的工况，不必一开始做过宽的CFD矩阵。",
        "第四步：开展敏感性分析，筛出真正影响航时、能耗、功率裕度和稳定性的关键参数。",
        "第五步：建立全任务剖面基准算例，完成阶段策略仿真和优化前后对比。",
        "第六步：把模型、参数边界和评价指标交给卢奕做控制/优化算法，同时你负责验证数据和场景。",
        "第七步：共同完成鲁棒性仿真，形成标称策略与鲁棒策略的对比结论。",
    ]:
        bullet(doc, item)


def footer(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("PTDL项目研究内容一、二个人执行细化方案")
    set_run(r, size=9, color="777777")


def main():
    doc = Document()
    style_doc(doc)
    title(doc)
    add_scope(doc)
    add_priority(doc)
    add_11(doc)
    add_13(doc)
    add_21(doc)
    add_23(doc)
    add_interface(doc)
    add_deliverables(doc)
    footer(doc)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
