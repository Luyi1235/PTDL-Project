from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


OUT = r"C:\Users\23598\Documents\ChatGPT\LDTP\PTDL内容1-2研究规划细化稿_软件简化版.docx"


def set_font(run, name="宋体", size=11, bold=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_para(paragraph, before=0, after=6, line=1.15, first_line=None, left=None):
    pf = paragraph.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if first_line is not None:
        pf.first_line_indent = Cm(first_line)
    if left is not None:
        pf.left_indent = Cm(left)


def style_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.49)
    section.footer_distance = Inches(0.49)

    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for style_name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 16, 8),
        ("Heading 2", 13, "2E74B5", 12, 6),
        ("Heading 3", 12, "1F4D78", 8, 4),
    ]:
        st = doc.styles[style_name]
        st.font.name = "微软雅黑"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(color)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True


def add_title(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(p, after=3)
    r = p.add_run("PTDL项目研究内容一、二研究规划细化稿")
    set_font(r, "微软雅黑", 22, True, "0B2545")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(p, after=14)
    r = p.add_run("三级目录、研究方法与软件/程序实施方案")
    set_font(r, "微软雅黑", 12, False, "555555")


def add_text(doc, text, bold_label=None):
    p = doc.add_paragraph()
    set_para(p)
    if bold_label:
        r = p.add_run(bold_label)
        set_font(r, bold=True)
        r = p.add_run(text)
        set_font(r)
    else:
        r = p.add_run(text)
        set_font(r)
    return p


def add_label_items(doc, items):
    for label, text in items:
        add_text(doc, text, bold_label=f"{label}：")


def add_heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        if level == 1:
            set_font(r, "微软雅黑", 16, True, "2E74B5")
        elif level == 2:
            set_font(r, "微软雅黑", 13, True, "2E74B5")
        else:
            set_font(r, "微软雅黑", 12, True, "1F4D78")
    return p


def add_list_item(doc, text, level=0, ordered=False):
    style = "List Number" if ordered else "List Bullet"
    p = doc.add_paragraph(style=style)
    set_para(p, after=4, line=1.15)
    p.paragraph_format.left_indent = Cm(0.74 + level * 0.45)
    p.paragraph_format.first_line_indent = Cm(-0.37)
    r = p.add_run(text)
    set_font(r)
    return p


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, fill=None):
    cell.text = ""
    p = cell.paragraphs[0]
    set_para(p, after=0, line=1.08)
    r = p.add_run(text)
    set_font(r, "宋体", 9.5, bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if fill:
        shade_cell(cell, fill)


def add_program_table(doc):
    rows = [
        ("基础数据与配置", "Python、Excel/CSV/YAML", "实现无人机参数、任务剖面、变量边界、约束阈值和优化权重的统一管理。"),
        ("RCAIDE快速分析", "RCAIDE、Python", "实现无人机几何、气动、推进和任务剖面的快速耦合分析。"),
        ("CFD高保真校核", "STAR-CCM+、Fluent", "实现关键外形和关键工况流场仿真，校核并修正升阻极曲线和气动数据库。"),
        ("飞行性能/气动分析", "Python、RCAIDE、CFD结果", "实现标准大气、升阻特性、配平需求、需用推力和需用功率计算。"),
        ("推进/能量分析", "Python、RCAIDE", "实现发动机、电机/发电机、螺旋桨、储能与功率分配建模。"),
        ("任务剖面仿真", "Python、RCAIDE", "实现起飞、爬升、巡航、下降、进近和着陆等阶段动力学仿真。"),
        ("敏感性分析", "Python、SALib", "实现单因素、Morris、Sobol和代理模型敏感性分析。"),
        ("优化控制", "Python、CasADi/SciPy/pymoo", "实现直接配点、多目标优化、Pareto搜索和鲁棒优化。"),
        ("结果后处理", "Python、Matplotlib/Plotly/Jupyter", "实现轨迹、功率、燃油、约束裕度和Pareto前沿的可视化分析。"),
    ]
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for cell, text in zip(hdr, ["模块", "软件", "实现内容"]):
        set_cell_text(cell, text, bold=True, fill="E8EEF5")
    for module, program, purpose in rows:
        cells = table.add_row().cells
        set_cell_text(cells[0], module, bold=True)
        set_cell_text(cells[1], program)
        set_cell_text(cells[2], purpose)

    widths = [Cm(3.0), Cm(5.6), Cm(7.9)]
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width.cm / 2.54 * 1440)))
            tc_w.set(qn("w:type"), "dxa")
    doc.add_paragraph()


def add_toc(doc):
    add_heading(doc, "三级目录总览", 1)
    toc_lines = [
        "1 研究内容1：XXXX无人机FTK耦合机理研究",
        "1.1 飞行性能与推进系统耦合机理研究",
        "1.1.1 飞行状态-气动载荷-推进需求显式映射",
        "1.1.2 巡航、爬升、下降典型阶段耦合规律",
        "1.1.3 FTK耦合模型验证与接口固化",
        "1.2 推进系统与控制策略协同作用研究",
        "1.2.1 混合动力推进系统数学模型与工作边界",
        "1.2.2 控制策略库与功率分配逻辑",
        "1.2.3 多工况协同评价与规律提炼",
        "1.3 FTK耦合设计参数敏感性分析",
        "1.3.1 参数体系、变量边界与指标体系构建",
        "1.3.2 单因素与全局敏感性分析",
        "1.3.3 主变量筛选、裕度设置与优化输入固化",
        "2 研究内容2：全RW剖面下最优控制策略研究",
        "2.1 RW剖面分解与阶段性控制策略研究",
        "2.1.1 剖面阶段划分与状态/控制变量建模",
        "2.1.2 阶段约束、目标函数与可行性判据",
        "2.1.3 阶段策略对比与全任务策略组合",
        "2.2 多目标优化控制算法研究",
        "2.2.1 多目标指标体系与归一化评价函数",
        "2.2.2 优化算法选择、实现与对比",
        "2.2.3 Pareto解筛选与仿真分析平台",
        "2.3 控制策略鲁棒性研究",
        "2.3.1 环境、模型与任务不确定性建模",
        "2.3.2 标称最优策略扰动仿真与分系统评估",
        "2.3.3 鲁棒控制策略设计与迭代优化",
        "3 阶段成果与验收指标",
    ]
    for line in toc_lines:
        p = doc.add_paragraph()
        level = line.count(".")
        set_para(p, after=2, line=1.08, left=min(level * 0.45, 0.9))
        r = p.add_run(line)
        set_font(r, "宋体", 10.5, bold=(level == 0))


def add_intro(doc):
    add_heading(doc, "编制说明", 1)
    add_text(
        doc,
        "本细化稿在原有材料基础上，将PTDL项目研究内容一、研究内容二整理为三级目录，并在每个三级研究任务下明确研究目标、主要方法、拟采用软件/程序和阶段输出。整体思路是先建立飞行性能、推进系统和控制策略之间的飞推控耦合模型，再在全RW任务剖面下开展阶段控制、全任务优化和鲁棒性验证。",
    )
    add_text(
        doc,
        "软件路线建议采用Python作为主平台，用于模型计算、任务仿真、敏感性分析和优化控制；RCAIDE用于无人机概念设计、气动/推进/任务耦合快速分析；STAR-CCM+或Fluent用于关键外形和关键工况的高保真CFD校核，并将气动修正结果回写到Python分析平台。",
    )

    add_heading(doc, "总体技术路线与软件/程序体系", 1)
    add_text(
        doc,
        "总体技术路线为“参数建模-阶段仿真-耦合分析-敏感性筛选-最优控制-鲁棒验证”。输入包括无人机几何/质量参数、气动数据、推进系统部件效率图、任务剖面约束和控制策略参数；输出包括燃油消耗、航时、轨迹、功率需求、功率裕度、约束裕度、Pareto前沿和鲁棒性统计结果。",
    )
    add_program_table(doc)


def add_research_content_1(doc):
    add_heading(doc, "1 研究内容1：XXXX无人机FTK耦合机理研究", 1)
    add_text(
        doc,
        "研究内容一面向“飞行性能-推进系统-控制策略”之间的耦合机理。核心任务是把飞行状态、气动载荷、推进需求、燃油消耗和控制策略连接成可计算、可验证、可优化的模型，为后续全RW剖面优化控制提供低阶但可信的分析基础。",
    )

    add_heading(doc, "1.1 飞行性能与推进系统耦合机理研究", 2)
    add_heading(doc, "1.1.1 飞行状态-气动载荷-推进需求显式映射", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "建立高度h、速度V、迎角alpha、航迹角gamma、飞机质量m与升力L、阻力D、需用推力Treq、需用功率Preq、燃油消耗mf之间的显式映射关系，形成FTK耦合分析的基础计算链。"),
            ("主要方法", "采用标准大气模型计算密度和声速；采用点质量纵向动力学和力平衡关系计算升阻需求；基于气动极曲线或查表模型获得CL/CD；通过Treq = D + W sin(gamma)和Preq = Treq V / eta_prop计算推进需求，并结合发动机比油耗/效率图计算燃油流量。"),
            ("软件/程序", "Python实现飞行状态、气动载荷和推进需求的计算映射；RCAIDE实现气动/推进/任务快速耦合分析；STAR-CCM+或Fluent实现关键工况CFD校核和气动数据修正。"),
            ("阶段输出", "输出飞行状态到推进需求的映射函数、典型工况计算表、功率-速度-高度曲面图和燃油流量等值线图。"),
        ],
    )

    add_heading(doc, "1.1.2 巡航、爬升、下降典型阶段耦合规律", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "分别揭示巡航、爬升、下降和进近阶段中飞行性能变量对推进系统工作点的影响规律，识别功率需求峰值、经济工作区和约束边界。"),
            ("主要方法", "巡航阶段重点分析高度变化对空气密度、升力系数、阻力、螺旋桨效率和燃油消耗的影响；爬升阶段建立爬升率、爬升角、速度、最大功率、功重比和剩余爬升能力之间的约束关系；下降和进近阶段研究低功率/小推力状态下轨迹约束对推进系统工作点的反向限制。"),
            ("软件/程序", "Python实现巡航、爬升、下降等阶段仿真和性能图谱生成；RCAIDE用于快速剖面分析；Matplotlib或Plotly用于结果可视化。"),
            ("阶段输出", "输出各阶段的推进需求包线、爬升可行域、巡航经济高度/速度区间、下降能量管理边界和阶段耦合规律总结。"),
        ],
    )

    add_heading(doc, "1.1.3 FTK耦合模型验证与接口固化", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "保证飞行性能模型、推进模型和控制模型之间的变量定义、单位、接口和计算顺序一致，使该模型能够被内容二的最优控制程序直接调用。"),
            ("主要方法", "设置典型算例进行单点校核、剖面校核和极限工况校核；对输入输出进行量纲检查、边界检查和守恒关系检查；将模型封装为统一函数接口，避免不同阶段重复写公式。"),
            ("软件/程序", "Python实现模型接口封装、单位检查和典型算例验证；pytest用于核心计算函数测试；CSV/JSON用于保存算例输入和验证结果。"),
            ("阶段输出", "输出可复用的FTK耦合模型接口、验证算例集、误差/一致性检查表和后续优化调用说明。"),
        ],
    )

    add_heading(doc, "1.2 推进系统与控制策略协同作用研究", 2)
    add_heading(doc, "1.2.1 混合动力推进系统数学模型与工作边界", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "建立发动机、电机/发电机、螺旋桨、储能单元和功率分配之间的混合动力模型，明确最大功率、最小稳定功率、功率变化率、效率区间和经济工作区。"),
            ("主要方法", "采用部件级效率图或分段函数描述发动机比油耗、电机效率、螺旋桨效率和储能状态变化；建立功率流约束、SOC约束、热/功率裕度约束和部件工作边界；在多高度、多速度、多功率需求条件下计算推进系统可行工作点。"),
            ("软件/程序", "Python实现混合动力系统、功率平衡和能量管理计算；RCAIDE用于推进系统与任务剖面耦合建模；STAR-CCM+或Fluent用于螺旋桨滑流、进排气和局部干扰校核。"),
            ("阶段输出", "输出推进系统工作边界图、经济工作区定义、功率分配模型和可供控制策略调用的推进系统接口。"),
        ],
    )

    add_heading(doc, "1.2.2 控制策略库与功率分配逻辑", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "建立覆盖主要任务阶段的控制策略库，研究控制策略选择的速度、高度、姿态和功率分配如何改变推进系统工作点。"),
            ("主要方法", "构建定高度巡航、定速度巡航、最小功率巡航、最小燃油巡航、高度自由优化巡航、阶梯爬升、滑翔下降等策略；每种策略统一定义状态反馈、控制变量、约束触发条件和功率分配规则；通过相同任务剖面进行策略对比。"),
            ("软件/程序", "Python实现控制策略库、功率分配和策略切换逻辑；SciPy或CasADi用于控制参数优化；有限状态机用于规则型策略实现。"),
            ("阶段输出", "输出控制策略库、功率分配规则表、策略触发条件和各策略对应的推进系统工作点分布。"),
        ],
    )

    add_heading(doc, "1.2.3 多工况协同评价与规律提炼", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "建立推进-控制协同评价指标，定量比较不同控制策略在全任务燃油、阶段燃油、推进效率、功率裕度和约束满足方面的差异。"),
            ("主要方法", "开展高度、速度、质量、任务距离、风场和功率边界等多工况仿真；统计工作点在功率-速度-高度空间中的分布；计算平均推进效率、单位航程燃油消耗、单位时间燃油消耗、经济工作区占比和功率裕度。"),
            ("软件/程序", "Python实现多工况批量仿真、协同评价指标计算和结果后处理；CSV/Parquet用于结果存储；Matplotlib或Plotly用于图表输出。"),
            ("阶段输出", "输出协同评价指标体系、策略对比图、约束违反统计表和推进-控制协同规律总结。"),
        ],
    )

    add_heading(doc, "1.3 FTK耦合设计参数敏感性分析", 2)
    add_heading(doc, "1.3.1 参数体系、变量边界与指标体系构建", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "形成FTK耦合参数化分析模型，明确需要研究的推进参数、控制/任务参数、气动/几何参数和输出指标。"),
            ("主要方法", "建立参数分层：推进系统参数包括最大功率、螺旋桨盘面积、推进效率、比油耗、功率裕度和发动机部分功率效率；控制与任务参数包括巡航高度、巡航速度、爬升率、下降率、巡航距离、任务持续时间、终端高度、终端速度和功率变化率限制；气动/几何参数包括翼展、机翼面积、展弦比、阻力极曲线参数和结构重量估计参数。"),
            ("软件/程序", "Python实现设计变量管理、参数边界设置和样本生成；Excel/CSV用于参数表维护；NumPy、SciPy和pandas用于数据处理。"),
            ("阶段输出", "输出设计变量清单、变量边界、参数物理含义、约束条件和性能指标定义表。"),
        ],
    )

    add_heading(doc, "1.3.2 单因素与全局敏感性分析", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "识别影响续航时间、燃油消耗、功率裕度、爬升可行性和终端约束满足的关键参数，并区分主效应与交互效应。"),
            ("主要方法", "先开展单因素敏感性分析，获得参数变化对任务性能的基本趋势；再开展全局敏感性分析，采用Morris筛选法识别重要变量，采用Sobol方差分解量化一阶效应和总效应；当原模型计算成本较高时，采用高斯过程、径向基函数或多项式响应面构建代理模型。"),
            ("软件/程序", "Python和SALib实现单因素、Morris和Sobol敏感性分析；scikit-learn或SciPy用于代理模型和响应面分析。"),
            ("阶段输出", "输出敏感性排序、参数交互矩阵、响应面模型、关键参数解释和对后续优化变量选择的建议。"),
        ],
    )

    add_heading(doc, "1.3.3 主变量筛选、裕度设置与优化输入固化", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "根据敏感性结果确定可固定参数、需联合优化参数和需设置设计裕度的参数，降低后续优化问题维度并提高优化稳定性。"),
            ("主要方法", "结合敏感性指标、物理可解释性、工程可调性和约束风险进行变量筛选；重点分析翼展与结构重量、机翼面积与翼载荷、最大功率与爬升可行性、巡航高度与推进效率、巡航速度与燃油消耗之间的耦合关系；形成优化变量、固定参数和鲁棒裕度三类清单。"),
            ("软件/程序", "Python实现变量筛选、优化输入整理和设计裕度分析，并将筛选结果用于后续最优控制与多目标优化。"),
            ("阶段输出", "输出优化主变量集、参数冻结清单、设计裕度建议和内容二优化控制的输入配置文件。"),
        ],
    )


def add_research_content_2(doc):
    add_heading(doc, "2 研究内容2：全RW剖面下最优控制策略研究", 1)
    add_text(
        doc,
        "研究内容二面向全RW任务剖面下的最优控制策略。核心任务是在内容一建立的FTK耦合模型基础上，将起飞、爬升、巡航/待机、下降、进近和着陆等阶段统一到一个可优化的任务剖面模型中，形成兼顾燃油、航时、平滑性和安全裕度的控制策略。",
    )

    add_heading(doc, "2.1 RW剖面分解与阶段性控制策略研究", 2)
    add_heading(doc, "2.1.1 剖面阶段划分与状态/控制变量建模", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "将全RW剖面划分为起飞滑跑、主爬升、短平飞调整、长航时巡航/待机、长距离下降、进近下降、拉平和着陆滑跑等阶段，并建立统一状态空间描述。"),
            ("主要方法", "采用分段动态模型描述每一阶段，状态变量建议取x(t) = [s, h, Vx, Vh, mf, SOC]T，其中s为水平航程，h为高度，Vx为水平速度，Vh为垂向速度，mf为累计燃油消耗，SOC为储能状态；控制变量包括推进功率、推力/油门、航迹角、速度指令、爬升率指令和混动功率分配系数。"),
            ("软件/程序", "Python和RCAIDE实现RW任务剖面建模、阶段动力学描述和状态/控制变量定义。"),
            ("阶段输出", "输出RW剖面阶段划分表、状态变量/控制变量定义、阶段衔接条件和统一剖面仿真接口。"),
        ],
    )

    add_heading(doc, "2.1.2 阶段约束、目标函数与可行性判据", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "为每个飞行阶段明确初末状态约束、路径约束、控制约束、阶段目标和可行性判据，避免优化结果只在数学上可行而不满足飞行/推进边界。"),
            ("主要方法", "起飞段关注离地速度、滑跑距离和功率裕度；爬升段关注高度建立、爬升率、最大功率和最小速度；巡航段关注燃油、航时和推进经济工作区；下降段关注能量释放、进近入口状态和低功率边界；着陆段关注速度、高度、下沉率和安全边界。对每个阶段设置速度、高度、功率、SOC、爬升率、终端状态和控制变化率约束。"),
            ("软件/程序", "Python实现阶段约束、目标函数和可行性判据计算，并输出约束满足情况和不可行原因诊断。"),
            ("阶段输出", "输出各阶段目标函数、约束清单、可行性判据和不可行原因诊断规则。"),
        ],
    )

    add_heading(doc, "2.1.3 阶段策略对比与全任务策略组合", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "对每一阶段进行控制策略对比，形成“阶段最优/近优策略”和“全任务控制策略组合”。"),
            ("主要方法", "在相同初始质量、气象条件和任务约束下，对恒速度、恒高度、恒爬升率、最小功率、最大航程、最大航时、高度自由优化、阶梯爬升和滑翔下降等策略进行批量仿真；按照阶段燃油、时间、功率裕度、终端状态误差和控制平滑性筛选策略组合。"),
            ("软件/程序", "Python实现阶段策略扫描、全任务策略组合和批量剖面仿真；RCAIDE用于典型任务剖面快速验证。"),
            ("阶段输出", "输出阶段策略对比结果、全任务策略组合、典型剖面曲线和控制策略选择原则。"),
        ],
    )

    add_heading(doc, "2.2 多目标优化控制算法研究", 2)
    add_heading(doc, "2.2.1 多目标指标体系与归一化评价函数", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "建立面向全任务剖面的多目标评价函数，兼顾燃油经济性、任务时间、控制平滑性和安全裕度。"),
            ("主要方法", "将目标函数写为J = w_f J_fuel + w_t J_time + w_s J_smooth + w_m J_safety，其中J_fuel表示燃油消耗或等效能耗，J_time表示任务时间或航时指标，J_smooth表示控制变化率/轨迹平滑性，J_safety表示速度、高度、功率、SOC和终端状态约束裕度惩罚。所有指标先进行量纲统一和归一化，再进行权重扫描或Pareto分析。"),
            ("软件/程序", "Python实现多目标指标计算、归一化处理和综合评价函数构建。"),
            ("阶段输出", "输出指标定义、归一化方法、权重设置方案和目标函数计算程序。"),
        ],
    )

    add_heading(doc, "2.2.2 优化算法选择、实现与对比", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "选择并实现适用于全RW剖面最优控制的算法组合，比较不同算法在收敛性、计算成本和工程可解释性方面的表现。"),
            ("主要方法", "连续控制问题优先采用直接配点法或伪谱法，将状态和控制离散为有限维非线性规划；多目标权衡采用加权和法、epsilon约束法和NSGA-II/Pareto搜索；当单次剖面仿真计算成本较高时，采用贝叶斯优化或代理模型辅助搜索。算法选择原则为：阶段问题用快速局部优化，全任务剖面用直接配点，多目标方案集用NSGA-II或权重扫描，昂贵模型用代理模型/贝叶斯优化。"),
            ("软件/程序", "Python实现最优控制离散、多目标优化和Pareto解搜索；CasADi、SciPy和pymoo用于优化求解。"),
            ("阶段输出", "输出算法对比报告、收敛曲线、最优控制量序列、状态轨迹和Pareto候选解集。"),
        ],
    )

    add_heading(doc, "2.2.3 Pareto解筛选与仿真分析平台", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "建立可根据不同任务指标权重进行仿真和方案筛选的平台，形成最省油、最短时间、最平滑控制和综合折中等典型方案。"),
            ("主要方法", "对Pareto前沿进行非支配排序、膝点识别和约束裕度筛选；对典型解进行剖面对比，分析速度、高度、功率、燃油、SOC和控制变化率随时间/航程的变化；将方案评价从单一最优扩展为任务需求驱动的可选方案集。"),
            ("软件/程序", "Python实现仿真分析平台、Pareto解后处理和报告生成；Plotly、Dash或Jupyter Notebook用于交互式结果展示。"),
            ("阶段输出", "输出多目标优化控制仿真平台、典型Pareto方案、剖面对比图和任务需求-控制策略映射表。"),
        ],
    )

    add_heading(doc, "2.3 控制策略鲁棒性研究", 2)
    add_heading(doc, "2.3.1 环境、模型与任务不确定性建模", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "建立影响控制策略有效性的三类不确定因素模型，为鲁棒性评估和鲁棒优化提供输入。"),
            ("主要方法", "环境扰动包括风速、阵风、气温、空气密度偏差、巡航高度变化和起降场高度变化；模型不确定性包括气动系数误差、推进效率误差、比油耗误差、质量估计误差和结构重量估计误差；任务不确定性包括巡航距离变化、任务时间变化、返航高度变化、进近入口约束变化和剩余燃油限制。各参数可采用区间、正态/均匀分布或场景集描述。"),
            ("软件/程序", "Python实现不确定参数建模、扰动场景生成和风场/环境扰动描述；采样方法采用拉丁超立方、Monte Carlo或Sobol序列。"),
            ("阶段输出", "输出不确定参数清单、分布/区间设定、扰动场景库和鲁棒性仿真输入文件。"),
        ],
    )

    add_heading(doc, "2.3.2 标称最优策略扰动仿真与分系统评估", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "检验标称最优控制策略在扰动条件下是否违反速度、高度、功率、SOC和终端状态约束，并识别对性能退化最敏感的子系统。"),
            ("主要方法", "对标称最优策略开展扰动仿真，统计约束违反概率、终端状态偏差、燃油增量、航时变化和安全裕度下降；分别对气动、动力、轨迹/任务等子系统添加扰动，分析其对性能指标的影响贡献；用箱线图、概率分布和灵敏度条形图展示鲁棒性。"),
            ("软件/程序", "Python实现扰动仿真、分系统鲁棒性评估、Monte Carlo批量计算和鲁棒性结果统计。"),
            ("阶段输出", "输出标称最优策略鲁棒性报告、分系统扰动影响排序、约束违反统计和关键风险场景。"),
        ],
    )

    add_heading(doc, "2.3.3 鲁棒控制策略设计与迭代优化", 3)
    add_label_items(
        doc,
        [
            ("研究目标", "在燃油经济性和鲁棒安全性之间建立折中关系，形成适用于不确定环境和模型偏差的鲁棒控制策略。"),
            ("主要方法", "在目标函数中加入约束违反概率、最坏情形惩罚或安全裕度惩罚；采用场景优化、机会约束或保守裕度设计方法；对比标称最优与鲁棒最优在燃油、航时、控制平滑性和安全裕度上的差异，并根据鲁棒性结果迭代调整控制策略、功率裕度和终端约束。"),
            ("软件/程序", "Python实现鲁棒优化、机会约束处理和鲁棒/标称策略对比；CasADi、SciPy和pymoo用于优化求解。"),
            ("阶段输出", "输出鲁棒最优控制策略、标称/鲁棒方案对比、鲁棒性-经济性折中曲线和最终迭代优化建议。"),
        ],
    )


def add_milestones(doc):
    add_heading(doc, "3 阶段成果与验收指标", 1)
    add_text(
        doc,
        "为便于导师检查研究规划落地性，建议按模型、分析、优化和验证四个阶段组织成果。每一阶段均应形成可运行程序、可复现实验配置和图表化结果。",
    )
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    headers = ["阶段", "主要任务", "软件与实现内容", "验收指标"]
    for cell, text in zip(table.rows[0].cells, headers):
        set_cell_text(cell, text, bold=True, fill="E8EEF5")
    rows = [
        ("阶段1", "完成FTK基础模型和剖面分段模型", "Python和RCAIDE实现基础性能、推进系统和任务剖面计算。", "典型工况可重复计算；单位、边界和接口检查通过。"),
        ("阶段2", "完成耦合机理和敏感性分析", "Python、SALib和CFD结果实现耦合规律分析、敏感性排序和变量筛选。", "得到关键变量排序、耦合规律图和优化主变量清单。"),
        ("阶段3", "完成全RW剖面最优控制", "Python、CasADi/SciPy/pymoo实现全剖面最优控制和多目标Pareto搜索。", "得到至少4类典型Pareto方案并完成剖面对比。"),
        ("阶段4", "完成鲁棒性评估和鲁棒优化", "Python实现不确定性场景仿真、鲁棒性评估和鲁棒优化。", "形成扰动场景库、约束违反统计和鲁棒/标称策略对比。"),
    ]
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            set_cell_text(cell, text)
    widths = [Cm(2.0), Cm(4.4), Cm(5.4), Cm(4.7)]
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width.cm / 2.54 * 1440)))
            tc_w.set(qn("w:type"), "dxa")

    add_heading(doc, "建议需进一步确认的输入", 2)
    for item in [
        "XXXX无人机的几何、质量、气动极曲线、动力系统效率图和燃油/储能参数是否已有确定版本。",
        "RW剖面中各阶段的初末状态、跑道/起降场约束、巡航距离、巡航高度和终端约束是否已有导师指定指标。",
        "课题组是否指定RCAIDE、STAR-CCM+或Fluent中的某一个作为重点软件；若无指定，建议以Python+RCAIDE为主线，STAR-CCM+/Fluent作为关键工况高保真校核工具。",
    ]:
        add_list_item(doc, item, ordered=False)


def add_footer(doc):
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("PTDL项目研究内容一、二研究规划细化稿")
    set_font(r, "宋体", 9, False, "777777")


def main():
    doc = Document()
    style_document(doc)
    add_title(doc)
    add_intro(doc)
    add_toc(doc)
    add_research_content_1(doc)
    add_research_content_2(doc)
    add_milestones(doc)
    add_footer(doc)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
