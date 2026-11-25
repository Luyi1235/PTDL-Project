from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


OUT = r"C:\Users\23598\Documents\ChatGPT\LDTP\PTDL_2.2多目标优化控制算法前期调研与适配方案.docx"


def set_run(run, font="宋体", size=10.5, bold=False, color=None):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_para(p, before=0, after=6, line=1.15, left=None, first=None):
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
    normal.paragraph_format.line_spacing = 1.15

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
    set_para(p, after=4, line=1.12)
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    r = p.add_run(text)
    set_run(r)


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell(cell, text, bold=False, fill=None, size=9.0):
    cell.text = ""
    p = cell.paragraphs[0]
    set_para(p, after=0, line=1.05)
    r = p.add_run(text)
    set_run(r, size=size, bold=bold)
    cell.vertical_alignment = 1
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


def table(doc, headers, rows, widths_cm, size=9.0):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    for c, h in zip(tbl.rows[0].cells, headers):
        set_cell(c, h, True, "E8EEF5", size)
    for row in rows:
        cells = tbl.add_row().cells
        for c, text in zip(cells, row):
            set_cell(c, text, False, None, size)
    fix_table_widths(tbl, widths_cm)
    doc.add_paragraph()
    return tbl


def title(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(p, after=3)
    r = p.add_run("PTDL 2.2 多目标优化控制算法前期调研与适配方案")
    set_run(r, "微软雅黑", 20, True, "0B2545")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(p, after=14)
    r = p.add_run("面向混合动力无人机飞推控一体化控制问题")
    set_run(r, "微软雅黑", 12, False, "555555")


def add_summary(doc):
    heading(doc, "1 报告目的与结论摘要", 1)
    para(
        doc,
        "本报告面向研究内容2.2“多目标优化控制算法研究”的前期调研阶段，围绕遗传算法、粒子群算法、NSGA-II、MOEA/D、代理模型/Kriging、直接配点法和伪谱法等算法开展梳理，并结合混合动力无人机飞推控一体化控制问题提出算法适配路线。",
    )
    para(
        doc,
        "在当前尚未完全获得1.1、1.3、2.1详细模型和任务剖面参数的条件下，建议先采用“分层算法路线”：以NSGA-II作为多目标优化基线，以MOEA/D作为高维目标或分解式优化对照，以PSO/GA作为参数调优和策略组合搜索补充，以Kriging等代理模型降低高成本仿真计算量，以直接配点法或伪谱法处理连续控制变量，并用局部优化对进化算法得到的候选方案进行精修。",
    )
    table(
        doc,
        ["结论", "推荐做法", "理由"],
        [
            ("基线算法", "NSGA-II", "实现成熟、适合非线性多目标问题，能形成Pareto解集，便于与文献和后续算法对比。"),
            ("重点发展", "NSGA-II + 代理模型 + 局部精修", "兼顾全局搜索、计算效率和连续控制精度，适合飞推控耦合模型计算成本较高的情况。"),
            ("对照算法", "MOEA/D、MOPSO/PSO、GA", "用于比较不同搜索机制在收敛速度、解分布和稳定性方面的差异。"),
            ("连续控制处理", "直接配点法/伪谱法 + CasADi/SciPy", "适合将全任务剖面的状态和控制时程转化为有限维优化问题。"),
            ("最终平台", "Python + pymoo + CasADi/SciPy + scikit-learn", "覆盖多目标进化优化、非线性规划、代理模型和结果后处理。"),
        ],
        [2.6, 5.2, 8.6],
    )


def add_problem(doc):
    heading(doc, "2 飞推控一体化多目标优化问题特征", 1)
    para(
        doc,
        "混合动力无人机全任务剖面优化控制问题不是单纯的轨迹优化，也不是单纯的动力系统匹配，而是飞行状态、推进系统、能量管理和控制策略同时耦合的多目标、多约束优化问题。算法选择应服务于问题特征，而不是简单罗列算法。",
    )
    table(
        doc,
        ["问题特征", "具体表现", "对算法的要求"],
        [
            ("多目标冲突", "燃油/等效能耗、航时、任务响应速度、控制平滑性、安全裕度之间存在权衡。", "算法应输出Pareto解集，而不只是单一最优解。"),
            ("强非线性", "气动载荷、推进效率、电池SOC、功率分配和飞行状态相互影响。", "需要能处理非凸、非线性、不可导或近似黑箱模型。"),
            ("多约束", "速度、高度、功率、SOC、爬升率、终端状态和控制变化率均有限制。", "需支持约束惩罚、可行性优先或约束支配机制。"),
            ("混合变量", "连续控制量、离散模式切换、阶段策略组合可能同时存在。", "适合采用进化算法进行全局搜索，再用局部优化精修连续变量。"),
            ("仿真成本", "RCAIDE、CFD修正或高保真任务剖面仿真可能较耗时。", "需要代理模型、样本复用和分阶段优化降低计算量。"),
            ("全剖面耦合", "起飞、爬升、巡航、侦察、返航、下降、着陆阶段存在状态传递。", "需要阶段优化与全任务优化相结合。"),
        ],
        [2.7, 7.0, 6.7],
    )


def add_metrics(doc):
    heading(doc, "3 多目标指标体系建议", 1)
    para(doc, "建议先建立清晰的指标口径，再选择算法。若目标函数采用最小化形式，所有指标应统一转化为成本项或惩罚项。")
    table(
        doc,
        ["指标", "建议定义", "优化方向", "说明"],
        [
            ("燃油消耗", "全任务或分阶段燃油消耗量", "最小化", "反映发动机经济性。"),
            ("等效能耗", "燃油能量与电池能量折算后的综合能耗", "最小化", "适合混合动力系统，避免只优化燃油而过度消耗电池。"),
            ("电池能量管理效率", "SOC保持、SOC终端偏差、电池功率峰值或循环损耗", "最小化惩罚/最大化效率", "约束电池使用强度和任务结束剩余能量。"),
            ("航时能力", "实际航时或航时不足惩罚", "最大化/最小化惩罚", "若采用最小化目标，建议写为航时不足惩罚。"),
            ("任务响应速度", "阶段切换响应时间、终端状态收敛时间或任务完成时间", "最小化", "与长航时目标存在冲突，应结合任务类型设置权重。"),
            ("飞行稳定性", "高度/速度偏差、姿态变化、终端误差或约束裕度", "最小化偏差/最大化裕度", "可作为目标或硬约束。"),
            ("控制平滑性", "油门、功率分配系数、速度/高度指令变化率", "最小化", "避免控制抖振和频繁模式切换。"),
            ("安全裕度", "功率裕度、SOC裕度、速度/高度边界裕度", "最大化/最小化惩罚", "建议在约束惩罚中显式体现。"),
        ],
        [2.8, 5.3, 2.6, 5.7],
        8.8,
    )
    para(
        doc,
        "初始数学模型可写为：min F(x,u,p) = [J_fuel, J_energy, J_time_penalty, J_smooth, J_safety_penalty]，满足飞行动力学约束、推进系统约束、SOC约束、阶段衔接约束和终端状态约束。其中x为状态变量，u为控制变量，p为设计或策略参数。",
    )


def add_algorithms(doc):
    heading(doc, "4 候选算法调研", 1)
    heading(doc, "4.1 遗传算法及其适用性", 2)
    para(
        doc,
        "遗传算法通过选择、交叉和变异进行群体搜索，适合处理非凸、不可导和混合变量优化问题。对于本项目，普通遗传算法适合作为策略组合、模式切换阈值和控制参数的单目标或加权多目标基线，但不宜作为最终唯一算法，因为加权和方法难以完整描述非凸Pareto前沿。",
    )
    bullet(doc, "适配方向：用于规则策略参数、阶段切换点、功率分配阈值的初步搜索。")
    bullet(doc, "主要风险：收敛速度较慢，参数设置影响较大，需结合精英保留、约束惩罚和局部优化。")

    heading(doc, "4.2 粒子群算法及MOPSO", 2)
    para(
        doc,
        "粒子群算法通过群体位置和速度更新进行搜索，参数较少、实现简单，适合连续控制参数优化。对多目标问题可采用MOPSO或将多目标加权后进行PSO搜索。本项目可将PSO用于控制器参数、MPC权重、功率分配系数和模式切换阈值调优。",
    )
    bullet(doc, "适配方向：作为快速参数寻优工具，用于与GA、NSGA-II结果进行对比。")
    bullet(doc, "主要风险：容易早熟收敛，对约束处理和Pareto多样性维护要求较高。")

    heading(doc, "4.3 NSGA-II", 2)
    para(
        doc,
        "NSGA-II采用快速非支配排序、拥挤距离和精英保留机制，是工程多目标优化中最常用的基线算法之一。它不要求目标函数可导，能直接处理仿真驱动问题，适合本项目早期建立Pareto解集和开展算法对比。",
    )
    bullet(doc, "适配方向：作为2.2.2的主基线算法，用于能耗、航时、平滑性和安全裕度之间的Pareto权衡。")
    bullet(doc, "改进方向：加入约束支配规则、自适应交叉变异概率、基于阶段的编码方式，以及局部搜索精修。")

    heading(doc, "4.4 MOEA/D", 2)
    para(
        doc,
        "MOEA/D将多目标优化问题分解为一组标量子问题，通过邻域协同优化逼近Pareto前沿。与NSGA-II相比，MOEA/D在目标较多或需要均匀分布解集时具有优势，但权重向量、聚合函数和邻域参数设置较关键。",
    )
    bullet(doc, "适配方向：用于与NSGA-II对比，尤其适合燃油、SOC、稳定性、响应速度、安全裕度等目标同时考虑的场景。")
    bullet(doc, "主要风险：若目标尺度归一化不合理，分解效果会受到明显影响。")

    heading(doc, "4.5 代理模型与Kriging", 2)
    para(
        doc,
        "当每次任务剖面仿真或高保真校核成本较高时，可以采用Kriging/高斯过程、径向基函数、随机森林或多项式响应面建立目标和约束的近似模型。代理模型不应完全替代真实仿真，而应服务于样本筛选、预搜索和高价值候选解精修。",
    )
    bullet(doc, "适配方向：采用“初始采样-代理训练-候选搜索-真实仿真校正-代理更新”的循环。")
    bullet(doc, "主要风险：代理模型外推不可靠，需要保留真实模型复核和误差评估。")

    heading(doc, "4.6 直接配点法与伪谱法", 2)
    para(
        doc,
        "直接配点法和伪谱法适合连续最优控制问题，可将状态轨迹和控制时程离散为有限维非线性规划。对于本项目，它们适合处理巡航、爬升、返航等连续阶段的控制时程优化，但对模型光滑性、初值和约束表达要求较高。",
    )
    bullet(doc, "适配方向：用于进化算法给出的候选方案精修，或用于单阶段连续控制问题。")
    bullet(doc, "主要风险：全局性较弱，复杂非凸问题可能依赖初值，建议与NSGA-II/MOEA/D结合。")

    heading(doc, "4.7 算法对比表", 2)
    table(
        doc,
        ["算法", "优势", "不足", "本项目定位"],
        [
            ("GA", "实现简单，适合离散/混合变量和策略组合搜索。", "Pareto表达能力有限，收敛较慢。", "辅助算法，用于参数和策略组合初筛。"),
            ("PSO/MOPSO", "连续参数寻优快，参数较少。", "容易早熟，对多目标多样性维护较敏感。", "对照算法，用于控制参数和权重调优。"),
            ("NSGA-II", "成熟稳健，适合仿真驱动多目标问题，Pareto结果直观。", "高维目标和昂贵仿真下计算量较大。", "主基线算法。"),
            ("MOEA/D", "分解式搜索，适合多目标和均匀前沿。", "权重向量和归一化设置影响较大。", "重点对照算法。"),
            ("代理模型/Kriging", "减少高成本仿真次数，提高搜索效率。", "存在模型误差和外推风险。", "效率提升模块。"),
            ("直接配点/伪谱法", "适合连续控制时程优化，约束表达清晰。", "依赖初值，非凸问题全局性不足。", "连续控制精修模块。"),
        ],
        [2.4, 4.3, 4.3, 5.4],
        8.7,
    )


def add_route(doc):
    heading(doc, "5 推荐算法适配路线", 1)
    para(
        doc,
        "建议不要一开始把所有算法都做成完整平台，而是采用“三层推进”的方式：第一层建立基线可运行算法，第二层做问题适配和效率改进，第三层形成算法库和仿真平台。",
    )
    table(
        doc,
        ["阶段", "算法路线", "你具体做什么", "输出"],
        [
            ("阶段A：基线建立", "NSGA-II + 加权和/SciPy局部优化", "建立目标函数和约束函数，跑通典型阶段和简化全剖面算例。", "基线Pareto解集、收敛曲线、策略对比。"),
            ("阶段B：对照研究", "MOEA/D、PSO/MOPSO、GA", "在相同算例上比较收敛速度、可行解比例、Pareto分布和计算耗时。", "算法对比报告。"),
            ("阶段C：效率提升", "Kriging/高斯过程代理模型 + 主动采样", "针对高成本仿真建立代理模型，用真实仿真校核候选解。", "代理模型误差、加速比、校核结果。"),
            ("阶段D：连续控制精修", "直接配点法/伪谱法 + CasADi/SciPy", "对NSGA-II/MOEA/D得到的典型解进行连续控制时程精修。", "控制量序列、状态轨迹、约束满足情况。"),
            ("阶段E：平台化", "统一算法库 + 可视化后处理", "封装算法接口、结果存储和Pareto筛选流程。", "多目标优化控制仿真分析平台。"),
        ],
        [2.5, 4.2, 6.1, 3.6],
    )
    heading(doc, "5.1 推荐的算法组合", 2)
    bullet(doc, "主算法：NSGA-II，用于形成可解释的Pareto解集和早期算法基线。")
    bullet(doc, "对照算法：MOEA/D，用于检验分解式多目标优化在多指标场景下的表现。")
    bullet(doc, "参数调优算法：PSO/GA，用于控制器参数、切换阈值和策略组合搜索。")
    bullet(doc, "效率提升：Kriging或高斯过程代理模型，用于减少高成本仿真次数。")
    bullet(doc, "精修算法：直接配点法或伪谱法，用于连续控制时程优化和典型Pareto方案二次优化。")


def add_tasks(doc):
    heading(doc, "6 前期调研与实现任务分解", 1)
    table(
        doc,
        ["任务编号", "任务内容", "具体工作", "完成标志"],
        [
            ("T1", "文献调研", "阅读GA、PSO、NSGA-II、MOEA/D、Kriging/EGO、直接配点/伪谱法等经典文献，整理算法原理、适用条件和局限。", "形成算法调研表和参考文献清单。"),
            ("T2", "问题建模", "把飞推控问题抽象为状态变量、控制变量、设计参数、目标函数和约束条件。", "形成数学模型草案。"),
            ("T3", "指标量化", "明确燃油、等效能耗、SOC、航时、响应速度、稳定性、安全裕度、控制平滑性等指标计算方式。", "形成指标定义表。"),
            ("T4", "基线算法实现", "优先实现NSGA-II，并在简化算例上跑通Pareto搜索。", "输出第一版Pareto前沿和收敛曲线。"),
            ("T5", "算法对照", "实现或调用MOEA/D、PSO/GA进行同算例对比。", "形成算法性能对比表。"),
            ("T6", "代理模型验证", "用采样数据训练Kriging/高斯过程或其他代理模型，比较预测误差和优化加速效果。", "输出代理模型误差和真实仿真校核结果。"),
            ("T7", "连续控制适配", "选取典型阶段，用直接配点或伪谱法离散控制过程。", "输出状态轨迹和控制量序列。"),
            ("T8", "算法库设计", "统一算法输入输出、约束处理、结果保存和图表模板。", "形成可复用的算法库框架。"),
        ],
        [1.6, 3.0, 7.8, 4.0],
        8.8,
    )

    heading(doc, "6.1 近期两周建议完成内容", 2)
    bullet(doc, "整理算法文献表：每种算法至少包括核心思想、变量类型、约束处理方式、优缺点、适配场景。")
    bullet(doc, "建立简化测试问题：先不等待完整飞推控模型，用简化任务剖面和假设参数跑通多目标优化流程。")
    bullet(doc, "完成NSGA-II基线：至少输出Pareto前沿、可行解比例、收敛曲线和典型解剖面对比。")
    bullet(doc, "准备与任浩杰对接清单：列出你需要的模型接口、气动/推进数据、任务阶段约束和扰动场景。")


def add_platform(doc):
    heading(doc, "7 仿真平台与算法库初步设计", 1)
    para(
        doc,
        "仿真平台建议采用Python实现，核心不是界面复杂，而是接口统一、结果可复现、算法可替换。早期可用Jupyter Notebook组织实验，后期再用Dash或轻量GUI展示结果。",
    )
    table(
        doc,
        ["模块", "实现内容", "建议软件"],
        [
            ("问题定义模块", "读取任务剖面、变量边界、目标函数和约束条件。", "Python、pandas、YAML/CSV"),
            ("模型调用模块", "调用飞行性能、推进系统、能量管理和阶段控制模型。", "Python、RCAIDE接口"),
            ("优化算法模块", "封装NSGA-II、MOEA/D、PSO/GA、代理模型辅助优化和直接配点法。", "pymoo、CasADi、SciPy、scikit-learn"),
            ("约束处理模块", "处理硬约束、软惩罚、可行性优先和安全裕度惩罚。", "Python"),
            ("后处理模块", "输出Pareto前沿、收敛曲线、时程曲线、雷达图和算法对比表。", "Matplotlib、Plotly"),
            ("报告输出模块", "保存典型方案、参数配置、图表和算法比较结果。", "Python、Word/PPT模板"),
        ],
        [3.0, 7.5, 5.9],
    )


def add_risks(doc):
    heading(doc, "8 风险点与应对措施", 1)
    table(
        doc,
        ["风险点", "表现", "应对措施"],
        [
            ("目标方向混乱", "航时希望越长越好，任务时间有时希望越短越好。", "统一最小化/最大化方向，必要时改为航时不足惩罚或任务时间约束。"),
            ("只优化燃油", "算法可能过度消耗电池，导致终端SOC不可接受。", "引入等效能耗、SOC终端约束和电池损耗指标。"),
            ("约束处理不足", "Pareto解看似优越但违反速度、高度、功率或SOC边界。", "采用可行性优先、约束支配或强惩罚机制。"),
            ("计算成本过高", "全剖面仿真导致进化算法迭代缓慢。", "先用简化模型筛选，再用代理模型和真实仿真校核。"),
            ("算法比较不公平", "不同算法参数和预算不同，结论不可靠。", "统一函数评价次数、种群规模、终止条件和测试场景。"),
            ("模型接口不稳定", "任浩杰模型更新导致算法输入输出变化。", "提前约定接口字段、单位、版本号和测试样本。"),
        ],
        [3.2, 5.6, 7.6],
    )


def add_refs(doc):
    heading(doc, "9 参考文献与调研来源", 1)
    refs = [
        "Holland, J. H. Adaptation in Natural and Artificial Systems. University of Michigan Press, 1975.",
        "Kennedy, J., and Eberhart, R. Particle Swarm Optimization. Proceedings of ICNN'95, 1995.",
        "Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 2002.",
        "Zhang, Q., and Li, H. MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition. IEEE Transactions on Evolutionary Computation, 2007.",
        "Coello Coello, C. A., Pulido, G. T., and Lechuga, M. S. Handling Multiple Objectives With Particle Swarm Optimization. IEEE Transactions on Evolutionary Computation, 2004.",
        "Jones, D. R., Schonlau, M., and Welch, W. J. Efficient Global Optimization of Expensive Black-Box Functions. Journal of Global Optimization, 1998.",
        "Betts, J. T. Survey of Numerical Methods for Trajectory Optimization. Journal of Guidance, Control, and Dynamics, 1998.",
        "Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., and Diehl, M. CasADi: A Software Framework for Nonlinear Optimization and Optimal Control. Mathematical Programming Computation, 2019.",
        "Blank, J., and Deb, K. pymoo: Multi-Objective Optimization in Python. IEEE Access, 2020.",
    ]
    for ref in refs:
        bullet(doc, ref)


def footer(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("PTDL 2.2 多目标优化控制算法前期调研与适配方案")
    set_run(r, size=9, color="777777")


def main():
    doc = Document()
    style_doc(doc)
    title(doc)
    add_summary(doc)
    add_problem(doc)
    add_metrics(doc)
    add_algorithms(doc)
    add_route(doc)
    add_tasks(doc)
    add_platform(doc)
    add_risks(doc)
    add_refs(doc)
    footer(doc)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
