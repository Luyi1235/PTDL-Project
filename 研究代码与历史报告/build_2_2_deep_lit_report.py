from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = r"C:\Users\23598\Documents\ChatGPT\LDTP\PTDL_2.2多目标优化控制算法文献调研与适配报告_80小时目标修订版.docx"


def set_east_asian_font(run, font_name="宋体"):
    run.font.name = font_name
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.append(r_fonts)
    r_fonts.set(qn("w:eastAsia"), font_name)
    r_fonts.set(qn("w:ascii"), "Calibri")
    r_fonts.set(qn("w:hAnsi"), "Calibri")


def style_paragraph(paragraph, before=0, after=6, line=1.18, first_line=True):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    if first_line:
        fmt.first_line_indent = Pt(22)


def add_para(doc, text, bold_prefix=None, after=6):
    p = doc.add_paragraph()
    style_paragraph(p, after=after)
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        r1.bold = True
        set_east_asian_font(r1)
        r2 = p.add_run(text[len(bold_prefix):])
        set_east_asian_font(r2)
    else:
        r = p.add_run(text)
        set_east_asian_font(r)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_east_asian_font(r, "黑体")
    r.bold = True
    if level == 1:
        r.font.size = Pt(16)
        r.font.color.rgb = RGBColor(46, 116, 181)
    elif level == 2:
        r.font.size = Pt(13)
        r.font.color.rgb = RGBColor(46, 116, 181)
    else:
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(31, 77, 120)
    return p


def add_reference(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, after=4, first_line=False)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-18)
    r = p.add_run(text)
    set_east_asian_font(r)
    r.font.size = Pt(9.5)


def add_note(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, before=4, after=8, line=1.15, first_line=False)
    p.paragraph_format.left_indent = Pt(12)
    p.paragraph_format.right_indent = Pt(12)
    r = p.add_run(text)
    set_east_asian_font(r, "宋体")
    r.font.color.rgb = RGBColor(31, 77, 120)
    r.bold = True
    return p


def set_page(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)


def set_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.18

    for name in ("Heading 1", "Heading 2", "Heading 3"):
        st = styles[name]
        st.font.name = "Calibri"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.font.bold = True
    styles["Heading 1"].font.size = Pt(16)
    styles["Heading 1"].font.color.rgb = RGBColor(46, 116, 181)
    styles["Heading 2"].font.size = Pt(13)
    styles["Heading 2"].font.color.rgb = RGBColor(46, 116, 181)
    styles["Heading 3"].font.size = Pt(12)
    styles["Heading 3"].font.color.rgb = RGBColor(31, 77, 120)


def add_footer(doc):
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("PTDL项目 2.2 多目标优化控制算法文献调研与适配报告")
        set_east_asian_font(r)
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(100, 100, 100)


doc = Document()
set_page(doc)
set_styles(doc)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(4)
r = title.add_run("PTDL 2.2 多目标优化控制算法文献调研与适配报告")
set_east_asian_font(r, "黑体")
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = RGBColor(11, 37, 69)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_after = Pt(14)
r = sub.add_run("面向混合动力无人机飞推控一体化控制 | 卢奕主责方向")
set_east_asian_font(r)
r.font.size = Pt(11)
r.font.color.rgb = RGBColor(85, 85, 85)

add_note(
    doc,
    "报告定位：本报告围绕研究内容2.2中“多目标优化控制算法开发”展开前期文献调研和项目适配论证，重点回答“为什么选这些算法、各算法适合解决哪类飞推控问题、如何嵌入PTDL项目仿真流程”。",
)

add_heading(doc, "一、研究背景与问题定位", 1)
add_para(
    doc,
    "PTDL项目的研究内容2.2旨在解决混合动力无人机在全任务剖面下的多性能指标并行优化问题，其关键系统目标是在统一任务定义下将续航性能由ULTRA的72小时提升至80小时。该问题的核心并不是把80小时与其他性能相互交换，而是在首先达到80小时的前提下，在燃油消耗、电池能量管理、飞行稳定性、任务响应速度、控制平滑性和安全裕度之间寻找可解释、可复现、可部署的折中解。由于研究内容1.2涉及推进系统和飞行控制协同模型，需要与动能学院进行模型接口和参数边界对接，现阶段先开展算法侧调研是合理的：算法框架可以先在简化模型、公开算例和可控的Python仿真环境中完成原型验证，待耦合模型成熟后再迁移到高保真仿真平台。",
)
add_para(
    doc,
    "从优化问题性质看，飞推控一体化控制不是一个标准的低维凸优化问题。控制变量可能同时包含连续变量和离散变量，例如巡航速度、高度、爬升率、发动机功率、电机功率、电池SOC参考轨迹、控制器增益、能量模式切换阈值等；约束条件则覆盖飞行动力学约束、推进系统功率边界、电池SOC上下限、终端航程或航时要求、姿态稳定边界、执行机构速率限制以及典型扰动下的安全裕度。该类问题具有非线性、强耦合、多约束、目标冲突明显和仿真代价较高等特征，因此不宜只依赖单一算法，而应建立“全局多目标搜索—代理模型加速—连续最优控制精修—高保真验证”的组合式技术路线。",
)
add_para(
    doc,
    "本报告依据多目标进化优化、群智能优化、代理模型优化和最优控制离散化等方向的代表性文献，对遗传算法、粒子群算法、NSGA-II、MOEA/D、Kriging代理模型以及直接配点法/伪谱法进行深入分析，并在此基础上给出PTDL项目的算法选型建议。报告中的软件支撑保持工程实现粒度：Python用于算法、数据处理和仿真调度；pymoo用于多目标进化算法实现；SciPy和CasADi用于局部优化与直接配点建模；scikit-learn或Gaussian process相关工具用于代理模型；RCAIDE用于任务级飞机/推进/能量网络分析；Fluent或STAR-CCM+用于必要的气动高保真校核与代理样本补充。",
)

add_heading(doc, "二、PTDL多目标优化控制问题的数学特征", 1)
add_para(
    doc,
    "为使算法调研直接服务于项目，首先需要把2.2问题抽象为一个可计算的约束多目标优化控制问题。设决策变量为z，其中既可以包含任务阶段参数，也可以包含离散化后的控制量序列。典型变量包括爬升段速度指令、爬升率、巡航高度、巡航速度、发动机输出功率、电机辅助功率、功率分配系数、模式切换阈值、控制律权重和若干关键节点处的状态/控制参数。所有正式候选解首先应满足g_80(z) = 80 h - T_endurance(z) <= 0，以及统一任务剖面、有效载荷功率、环境假设和终端能量储备。满足该门槛后，耗电型基准任务采用min F_base(z) = [J_E, J_stab, J_resp]，并把终端SOC储备、SOC下限、功率边界和控制变化率作为约束；只有在必须比较不同储备策略或执行荷电维持任务时，才采用min F_ext(z) = [J_E, J_SOC,res, J_stab, J_resp]。J_smooth优先作为路径约束或修复规则，不宜在算法原型阶段与上述目标并列，以避免不必要地增加目标维数。",
)
add_para(
    doc,
    "这些目标之间存在内在冲突。例如，为降低燃油消耗而提高电助推比例可能造成SOC下降过快，从而压缩后续任务阶段的能量裕度；为提升任务响应速度而增大功率输出或控制增益，可能带来能耗上升和控制振荡风险；为提高稳定性而保守限制状态变化率，又可能降低爬升、转场或应急任务响应能力。因此，2.2不应被写成“求一个最优值”的问题，而应建立Pareto前沿，通过不同折中解揭示任务需求变化下的控制策略选择空间。",
)
add_para(
    doc,
    "从算法角度看，PTDL问题同时包含三类难点。第一，目标评价高度依赖仿真：即使初期使用低阶动力学模型和RCAIDE任务级分析，后期也可能需要调用气动/推进高保真结果，因此每次函数评价的成本不可忽视。第二，约束多且工程含义强：不可行解不能仅作为数学异常丢弃，而要通过约束违反度、罚函数或修复算子反馈给搜索过程。第三，控制问题具有时间结构：简单把全任务剖面离散为大量节点后直接交给进化算法，会导致维数膨胀和收敛变慢，因此需要把阶段参数优化、控制序列优化和局部最优控制精修分层处理。",
)

add_heading(doc, "三、遗传算法（GA）的文献基础与项目适配", 1)
add_para(
    doc,
    "遗传算法由Holland提出，并经Goldberg等工作发展为工程优化中常用的随机全局搜索方法。其基本思想是把候选解编码为“个体”，通过选择、交叉和变异形成新一代种群，使适应度较高的解在迭代中保留并扩散。对于非线性、非凸、含离散变量或模型不可微的问题，GA的优势在于不依赖梯度信息，能够在较宽设计空间内进行鲁棒探索。对于飞推控问题中的模式切换阈值、阶段策略参数和控制器参数组合，GA具备天然适配性。",
)
add_para(
    doc,
    "但从2.2的核心需求看，传统GA不宜作为最终主算法。原因在于传统GA通常以单目标适应度为核心，如果采用简单加权和方式处理多目标，就会把Pareto折中关系压缩为一个人为权重下的单点结果；同时，GA在连续控制序列维数较高时搜索效率会下降，容易出现大量不可行解。对于混合动力无人机的全任务剖面控制，如果把每个时间节点的发动机功率、电机功率、速度指令和姿态控制参数全部放入染色体，GA的维数会迅速膨胀，导致收敛曲线不稳定、计算量过高，也难以解释结果。",
)
add_para(
    doc,
    "因此，GA在PTDL项目中更适合作为基线算法和外层策略搜索工具，而不是单独承担完整多目标优化控制求解。可将GA用于低维、阶段级变量优化，例如巡航高度、巡航速度、SOC阈值、发动机启停阈值、电助推比例上限、爬升段功率分配系数和控制器增益初值。其适配方式是先构造长度较短的策略染色体，每个基因对应一个工程可解释参数；仿真评价时将该策略映射到任务剖面，计算总能耗、SOC保持、稳定性指标和响应指标；在约束处理上采用可行性优先规则或罚函数，把SOC越界、功率越界、终端任务未完成、姿态超限等问题转化为适应度惩罚。GA输出可用于形成初始样本库，也可作为NSGA-II和MOEA/D初始种群的一部分，以提高后续多目标搜索的起点质量。",
)

add_heading(doc, "四、粒子群算法（PSO/MOPSO）的文献基础与项目适配", 1)
add_para(
    doc,
    "粒子群算法由Kennedy和Eberhart提出，来源于群体协同行为的启发式建模。PSO中每个粒子表示一个候选解，粒子根据自身历史最优位置和群体最优位置更新速度与位置。与GA相比，PSO的结构更简洁，连续变量搜索效率较高，参数数量较少，较适合控制器参数、权重系数和连续功率分配参数的快速整定。在工程优化中，PSO常用于模型不完全可微、评价函数存在噪声或局部不规则的场景。",
)
add_para(
    doc,
    "多目标粒子群算法（MOPSO）将Pareto支配关系引入PSO框架。Coello Coello等提出的MOPSO使用外部档案保存非支配解，并从档案中选择领导粒子引导搜索，同时通过变异机制改善探索能力。这一思想对PTDL问题有直接启发：飞推控策略的多目标结果不应被单个全局最优粒子代表，而应通过外部档案保留一组能耗、稳定性和响应速度不同侧重的候选策略。",
)
add_para(
    doc,
    "PSO/MOPSO的优势在于适配连续变量和调参问题，缺点则是多约束、多峰和强离散变量问题下容易早熟收敛，且对约束处理、领导粒子选择和档案维护策略较敏感。对于PTDL项目，PSO更适合承担两个角色。第一，作为控制器参数和权重的快速整定算法，例如对纵向控制律权重、MPC代价函数权重、SOC参考轨迹参数、功率分配比例和模式切换滞环宽度进行连续优化。第二，作为NSGA-II或MOEA/D之后的局部改进工具，对已获得的Pareto候选解进行邻域微调，提高响应速度或控制平滑性。",
)
add_para(
    doc,
    "在项目适配上，不建议将MOPSO作为唯一主算法，而应将其定位为“连续参数优化模块”。具体而言，可将粒子位置定义为一组控制策略参数，例如z = [K_h, K_V, K_SOC, SOC_ref, alpha_engine, alpha_motor, P_bat_max, threshold_mode]。每个粒子进入仿真评价函数后，平台计算能耗、SOC偏差、速度/高度跟踪误差、控制变化率和约束违反度。外部档案保存非支配策略，领导粒子从档案稀疏区域中选择，以避免所有粒子集中到某一类能耗最低但稳定性较差的方案。PSO结果可与NSGA-II结果交叉验证：若两类算法在某些折中区域给出相近策略，说明该区域更具工程可信度。",
)

add_heading(doc, "五、NSGA-II的文献基础与项目适配", 1)
add_para(
    doc,
    "NSGA-II由Deb、Pratap、Agarwal和Meyarivan提出，是多目标进化优化领域最具代表性的基准算法之一。其核心机制包括快速非支配排序、拥挤距离维护和精英保留。快速非支配排序用于将种群划分为不同Pareto层级；拥挤距离用于保持解集在目标空间中的分布多样性；精英保留使优秀解不会在迭代中丢失。这三个机制使NSGA-II能够在一次运行中输出一组折中解，而不是单一最优解，非常符合PTDL项目需要分析“能耗—稳定性—响应速度—SOC保持”权衡关系的目标。",
)
add_para(
    doc,
    "NSGA-II对PTDL项目的适配价值主要体现在三方面。第一，它可以自然处理多个相互冲突目标，不需要在算法入口处过早设定主观权重。第二，pymoo等Python框架已经提供成熟实现，便于快速建立可复现的算法验证流程。第三，NSGA-II可以通过自定义采样、交叉、变异和修复算子嵌入工程约束，使不可行解逐步向可行域移动。对于卢奕主责的2.2研究，NSGA-II应作为第一阶段主算法和基准算法，用于形成项目的标准Pareto前沿、收敛曲线和候选控制策略集。",
)
add_para(
    doc,
    "具体到飞推控问题，NSGA-II的个体编码不宜直接覆盖所有仿真时间节点，而应采用“阶段参数+关键节点控制量”的紧凑编码方式。一个适合初期研究的个体可包含巡航速度、巡航高度、爬升段功率分配系数、巡航段发动机/电机功率比例、SOC参考水平、SOC下限保护阈值、发动机介入阈值、电机辅助阈值、纵向控制器增益和控制平滑权重。若后续需要优化连续控制序列，可先用少量样条控制点描述P_engine(t)、P_bat(t)或V_cmd(t)，再由插值生成完整任务剖面控制量。这样既能保留控制问题的时间结构，又能避免维度过高导致NSGA-II收敛变慢。",
)
add_para(
    doc,
    "NSGA-II在PTDL项目中的目标函数可设置为五类。第一类为等效能耗指标，将燃油消耗和电池放电量统一折算为任务能量成本；第二类为SOC管理指标，度量任务结束SOC、SOC波动幅度和关键阶段能量储备；第三类为稳定性指标，统计高度、速度、姿态角或角速度偏差及其峰值；第四类为响应指标，度量爬升、转场、扰动恢复或动力模式切换后的过渡时间；第五类为控制平滑性指标，度量功率变化率、舵面变化率或控制输入二范数。约束方面，可采用Deb提出的可行性规则：可行解优先于不可行解；同为不可行解时约束违反度小者优先；同为可行解时再比较非支配等级和拥挤距离。该处理方式比简单大罚函数更稳健，也更容易解释不可行策略被淘汰的原因。",
)
add_para(
    doc,
    "NSGA-II的局限在于计算评价次数较多，对高成本仿真并不友好；当目标数量增加到四个以上时，非支配关系区分能力下降，种群中大量个体可能互不支配，导致选择压力不足。因此本项目不应停留在“直接调用NSGA-II跑仿真”的层面，而应将其与代理模型、目标归一化、约束修复和后处理决策方法结合。推荐做法是以NSGA-II建立主Pareto前沿，以Kriging或高斯过程代理减少真实仿真调用，以膝点选择、TOPSIS或模糊满意度方法从Pareto解集中选择适合不同任务偏好的代表方案。",
)

add_heading(doc, "六、MOEA/D的文献基础与项目适配", 1)
add_para(
    doc,
    "MOEA/D由Zhang和Li提出，其基本思想是把一个多目标优化问题分解为一组标量子问题，并在邻域内协同优化这些子问题。与NSGA-II通过非支配排序维护Pareto前沿不同，MOEA/D通过权重向量、Tchebycheff函数或PBI函数把目标空间划分为多个搜索方向，每个方向对应一个局部子问题。该机制使MOEA/D在多目标数量较多、需要均匀覆盖Pareto前沿、以及并行计算条件较好时具有优势。",
)
add_para(
    doc,
    "PTDL项目的完整评价维度包括能耗、SOC、稳定性、响应速度、控制平滑性、扰动鲁棒性和热管理，但早期正式Pareto目标应控制在三目标或四目标，80小时续航、SOC安全、控制平滑性和热管理优先作为约束或诊断量。在模型成熟、确有必要扩展到四目标以上时，NSGA-II可能出现选择压力下降，而MOEA/D的分解思想更适合保持目标空间的覆盖。特别是对于全任务剖面问题，可以把不同权重向量理解为在“均满足80小时”条件下的不同任务偏好：某些子问题偏重更低能耗，某些子问题偏重动力响应，某些子问题偏重稳定裕度，某些子问题偏重电池储备。这样输出的控制策略不仅是数学上的非支配解，也更容易对应到工程场景。",
)
add_para(
    doc,
    "MOEA/D对PTDL项目的适配方式应强调目标归一化和场景分解。由于能耗、响应时间、姿态误差、SOC偏差等指标量纲差异很大，直接使用权重分解会导致某些目标主导搜索。因此，应先用初始样本估计每个目标的理想点和近似最差点，对目标进行归一化，再执行Tchebycheff或PBI分解。邻域更新时，可让相邻权重方向共享相似控制策略，例如同样偏重节能的子问题共享低功率巡航策略，同样偏重响应速度的子问题共享高功率过渡策略。若平台具备并行计算条件，MOEA/D的子问题评价也便于分发到多个仿真进程。",
)
add_para(
    doc,
    "MOEA/D不应完全替代NSGA-II，而应作为重点对比算法。原因是NSGA-II更成熟、解释成本低、适合作为第一版基线；MOEA/D对权重向量、分解函数、目标归一化和邻域大小更敏感，参数设计不当会导致前沿覆盖偏斜。对本项目而言，合理策略是以NSGA-II建立可复现的基准结果，再引入MOEA/D比较收敛速度、前沿均匀性和高维目标适应性。如果MOEA/D在五目标指标体系下能够以更少评价次数获得更均匀的解集，则可升级为算法库中的主算法之一；如果其对参数过于敏感，则保留为复杂指标体系下的增强算法。",
)

add_heading(doc, "七、Kriging代理模型与仿真加速机制", 1)
add_para(
    doc,
    "代理模型的引入是PTDL项目能否把算法从“可跑”推进到“可用”的关键。Jones、Schonlau和Welch提出的Efficient Global Optimization将Kriging模型与期望改进准则结合，用于昂贵黑箱函数优化。Kriging或高斯过程模型不仅给出预测均值，还能给出预测不确定性，因此特别适合仿真代价高、样本数量有限、需要在探索和利用之间平衡的工程问题。对于飞推控一体化优化，真实评价可能包含任务级动力学仿真、推进系统模型、控制闭环仿真，甚至气动高保真校核，若每一代算法都完全调用真实仿真，将很难形成足够丰富的Pareto前沿。",
)
add_para(
    doc,
    "代理模型在本项目中的适配方式应是“辅助进化算法”，而不是替代真实仿真。首先用拉丁超立方采样或Sobol采样生成覆盖设计空间的初始样本，调用低保真或中保真仿真得到目标和约束值；随后训练Kriging/高斯过程代理模型，分别拟合能耗、SOC终值、稳定性误差、响应时间和约束违反度；再让NSGA-II或MOEA/D在代理模型上进行快速搜索，筛选出预测性能优良且不确定性较高的候选解；最后只把少量候选解送回真实仿真验证，并用新增真实样本更新代理模型。该循环能够显著降低真实仿真调用次数，同时避免代理模型长期脱离真实物理。",
)
add_para(
    doc,
    "对于多目标问题，代理模型可采用两类集成方式。第一类是标量化代理优化，例如ParEGO思想，即每次用不同权重把多目标转化为标量目标，在Kriging模型上进行期望改进搜索，逐步逼近Pareto前沿。第二类是直接的代理辅助多目标进化算法，即为每个目标和关键约束建立代理，在进化迭代中结合预测均值、不确定性和非支配关系决定哪些个体值得真实仿真。结合PTDL项目的工程需求，推荐优先采用第二类思路，因为它能保留NSGA-II/MOEA/D的Pareto解释框架，也便于把不同任务偏好的候选策略同时保留下来。",
)
add_para(
    doc,
    "代理模型的风险主要来自外推误差和高维稀疏样本。飞推控问题一旦决策变量过多，Kriging模型所需样本量会迅速上升；若将代理模型用于未覆盖区域，可能得到物理上不可行但预测上很优的策略。因此，项目适配中应设置三个保护机制：其一，决策变量先控制在工程可解释的低维参数上，不在早期直接代理完整高维控制序列；其二，对代理预测最优解必须进行真实仿真回代确认；其三，将不确定性、约束违反概率和样本距离纳入候选解筛选，避免算法过度利用不可靠区域。",
)

add_heading(doc, "八、直接配点法与伪谱法在连续最优控制中的作用", 1)
add_para(
    doc,
    "直接配点法和伪谱法属于数值最优控制中的直接法。Betts对轨迹优化数值方法进行了系统综述，Hargraves和Paris的直接配点工作则展示了通过多项式近似状态、插值控制并将微分方程约束转化为非线性规划问题的基本思路。与GA、PSO、NSGA-II和MOEA/D不同，直接配点法本身更接近连续最优控制求解器：它将状态轨迹和控制轨迹在离散节点上作为优化变量，并显式满足动力学约束、路径约束和边界条件。",
)
add_para(
    doc,
    "在PTDL项目中，直接配点法不宜作为最初的全局主搜索算法，但非常适合作为Pareto候选解的连续控制精修工具。原因在于飞推控问题存在明显的时间连续性，例如发动机功率和电池功率不应在相邻时刻剧烈跳变，SOC演化必须满足能量平衡，速度、高度和姿态状态也必须满足动力学方程。进化算法擅长全局搜索和处理多峰、离散与黑箱特征，但对连续轨迹细节的精修效率不高；直接配点法擅长在给定初值附近求解高精度连续控制轨迹，但对初值、模型光滑性和局部可行域较敏感。因此，二者应形成互补关系。需要明确的是，直接配点通常经由IPOPT等非线性规划求解器求解单一标量目标；它不自动给出Pareto前沿。工程上应对NSGA-II/MOEA/D给出的代表解采用epsilon约束或参考点标量化，例如固定可接受的J_stab和J_resp上限后最小化J_E，再以该代表解作为初值进行连续轨迹精修。",
)
add_para(
    doc,
    "项目适配上，可先用NSGA-II或MOEA/D获得若干代表性阶段策略，例如节能优先、响应优先、SOC保护优先和综合折中策略。随后以这些策略作为初值，在CasADi中建立阶段级直接配点问题。状态变量可包括飞行高度h、速度V、航程s、电池SOC、燃油质量或剩余燃油、姿态相关状态；控制变量可包括发动机功率P_eng、电机功率P_bat、速度指令V_cmd、爬升角或高度变化率指令以及关键控制律参数。动力学约束由简化飞行动力学和能量平衡方程给出，路径约束包括SOC上下限、功率边界、爬升率边界、速度包线和稳定性限制。这样，直接配点法负责把进化算法给出的“阶段策略”细化为可执行的连续控制量序列。",
)
add_para(
    doc,
    "伪谱法与直接配点法相比，通常使用全局正交多项式和特殊配点节点，能够在光滑轨迹问题上获得较高精度。对于PTDL早期研究，不建议立即把伪谱法作为主线，因为其理论和实现门槛高于普通直接配点法，且对模型光滑性要求更强。更稳妥的做法是先用CasADi实现直接配点原型，验证飞推控能量管理和姿态/速度约束能否稳定求解；在此基础上，如果后续任务剖面较长且控制轨迹较平滑，再把伪谱法作为高精度轨迹优化补充。",
)

add_heading(doc, "九、算法验证设计与选型判据", 1)
add_para(
    doc,
    "算法选型不能仅凭一次运行的Pareto图形判断。NSGA-II、MOEA/D、GA/MOPSO等算法必须在完全相同的决策变量边界、任务剖面、80小时续航门槛、约束处理、真实仿真评价函数和评价次数预算下比较；每种算法应使用多个随机种子重复运行，并报告中位数与离散性，而不是只展示最好的一次结果。第一阶段应先以最大化T_endurance或最小化g_80违反度确认80小时可行性；达标后再在F_base = [J_E, J_stab, J_resp]上比较，以确认算法在可行域内的解集质量；只有在基准问题稳定后，再切换到包含J_SOC,res的四目标扩展问题。这样可以避免把尚未达标的物理设计问题或指标体系本身的不稳定误判为算法优劣。",
)
add_para(
    doc,
    "选型评价应同时覆盖可行性、解集质量、计算效率和工程可执行性。可行性层面首先关注80小时达标率、达标解比例、任务完成率、最大约束违反度和关键安全裕度；解集质量层面只对满足g_80(z) <= 0的解，在目标归一化后采用Hypervolume评价收敛与覆盖，用IGD+或与合并非支配参考集的距离评价接近程度，并检查解集在各任务偏好方向上的空洞；效率层面比较达到相同Hypervolume所需的真实仿真次数和墙钟时间；工程层面检查控制轨迹是否平滑、功率分配是否具有物理可解释性、同类解在不同随机种子下是否稳定出现。最终算法库应保留不同任务偏好的80小时以上代表策略，而不是强行给出一个加权总分最高的“唯一最优解”。",
)
add_para(
    doc,
    "对进入候选库的Pareto解，还应执行独立场景复算。建议在标称环境外，至少改变起飞质量、风场/阵风、环境温度或密度高度、推进效率偏差和电池可用容量等影响因素，分别检查任务完成、终端SOC、最大功率和控制误差是否仍满足边界。若模型接口暂未齐全，可先用参数扰动形式完成这一验证；待动能学院模型接入后，再将参数扰动替换为真实推进性能图和环境模型。此步骤不改变算法的优化目标，但可避免选择只在标称工况下看起来很好的脆弱策略。",
)

add_heading(doc, "十、综合比较与算法选型结论", 1)
add_para(
    doc,
    "基于上述文献调研和PTDL问题特征，本项目不宜采用“单算法包打天下”的路线。纯GA和纯PSO虽然实现简单、对模型可微性要求低，但难以充分表达多目标Pareto权衡，且在高维控制序列下收敛效率不足。NSGA-II成熟可靠、文献基础扎实、Python工具链完善，应作为2.2的第一主算法和结果基准。MOEA/D对目标数量扩展和前沿均匀覆盖更有优势，应作为第二主算法，与NSGA-II共同构成多目标进化算法的核心对比。Kriging代理模型不是独立的控制算法，而是用于降低仿真成本的加速层，应嵌入NSGA-II/MOEA/D评价流程。直接配点法或伪谱法不是替代进化算法的全局搜索工具，而是用于将Pareto候选策略转化为连续控制轨迹的精修层。",
)
add_para(
    doc,
    "因此，推荐的PTDL算法体系为“约束NSGA-II基准算法 + MOEA/D对比增强算法 + Kriging代理模型加速 + CasADi直接配点局部精修”。在这一体系中，NSGA-II负责稳定地产生可解释Pareto前沿；MOEA/D负责验证高维目标体系下的前沿覆盖能力；Kriging代理负责减少昂贵仿真调用次数；直接配点负责将离散阶段策略平滑化并满足动力学约束。GA和PSO不作为最终主算法，但应保留为基线和局部调参工具，用于控制器增益、权重和阈值参数的快速整定。",
)
add_para(
    doc,
    "这一选型也符合卢奕主责部分的可执行性。当前阶段无需等待完整高保真飞推控耦合模型，可以先搭建Python算法原型：用简化纵向动力学、能量平衡模型和任务剖面分段模型构造可验证算例；用pymoo实现NSGA-II和MOEA/D；用SciPy或自写评价函数计算目标和约束；用scikit-learn或Gaussian process工具训练代理模型；用CasADi实现直接配点精修。待动能学院提供推进系统模型接口和气动数据后，再把评价函数从简化模型替换为RCAIDE/联合仿真接口，算法主体无需推倒重建。",
)

add_heading(doc, "十一、面向PTDL项目的具体适配方案", 1)
add_para(
    doc,
    "PTDL项目的算法适配应从问题抽象开始。首先将全任务剖面划分为爬升、巡航、任务执行、返航和下降等典型阶段，并为每个阶段定义可控参数和状态约束。对于每个候选策略，评价函数应完整运行一次任务剖面仿真，输出能耗、SOC演化、稳定性误差、响应时间和约束违反度。这样可以保证算法优化的不是孤立工况下的瞬时性能，而是全任务剖面下的综合控制效果。",
)
add_para(
    doc,
    "在变量设计上，初期应优先采用低维策略变量。推荐把决策变量限制在10到25维范围内，包含巡航速度、巡航高度、各阶段功率分配比例、SOC参考水平、SOC保护阈值、发动机介入阈值、电机辅助阈值、控制器权重和少量控制轨迹样条节点。这样既能覆盖飞推控耦合的主要调节自由度，又能保证NSGA-II/MOEA/D在有限仿真次数内形成有效前沿。若后续模型稳定，再逐步增加控制节点数量或加入更多阶段变量。",
)
add_para(
    doc,
    "在目标函数设计上，应避免把所有指标强行合成为一个综合评分。推荐保留至少三个核心目标：等效能耗最小、稳定性误差最小、任务响应时间最小；SOC管理可根据研究重点作为独立目标或作为强约束处理。若SOC作为目标，可用于分析“更充分使用电池”和“保留任务末端能量裕度”的权衡；若SOC作为约束，则可突出满足安全边界前提下的能耗和响应优化。项目早期建议采用四目标形式：等效能耗、SOC偏差、稳定性误差和响应时间；控制平滑性先作为约束或惩罚项，待控制轨迹优化阶段再作为独立目标。",
)
add_para(
    doc,
    "在约束处理上，推荐采用“可行性优先 + 约束违反度分级 + 工程修复算子”的方式。可行性优先用于保证算法不会把明显违反SOC、功率或速度包线的解误判为优解；约束违反度分级用于区分轻微越界和严重不可行；工程修复算子用于把部分可修复变量拉回边界内，例如功率比例归一化、SOC阈值排序、控制变化率限幅和模式切换滞环修正。这种方式比单纯大罚函数更适合飞推控问题，因为它保留了不可行解向可行域演化的信息。",
)
add_para(
    doc,
    "在仿真平台适配上，Python应作为统一调度层。算法端由pymoo实现NSGA-II和MOEA/D，SciPy用于基线优化和局部搜索，CasADi用于直接配点最优控制；模型端早期使用Python自建的简化飞行动力学与能量模型，中期接入RCAIDE进行任务级飞机性能、推进和能量网络分析，后期在关键气动参数或典型工况上使用Fluent或STAR-CCM+进行高保真校核，并把校核结果回填到代理模型或修正低阶模型。这样的软件组合与项目现状相匹配：不依赖Matlab/Simulink，也保留了航空气动软件的接口空间。",
)
add_para(
    doc,
    "在算法运行流程上，建议先执行NSGA-II基准实验，得到第一版Pareto前沿和代表性控制策略；随后执行MOEA/D对比实验，观察在四目标或五目标设置下的收敛速度和前沿分布；接着引入Kriging代理模型，将真实仿真评价次数压缩到可接受范围；最后选取Pareto前沿上的膝点解、节能优先解、响应优先解和SOC保护优先解，分别用CasADi直接配点进行连续控制轨迹精修。精修后的策略再回到统一仿真平台进行全任务剖面验证，比较精修前后的能耗、稳定性、响应和控制平滑性变化。",
)
add_para(
    doc,
    "在结果表达上，本研究最终不应只给出“某算法最好”的笼统结论，而应给出分场景选型。例如，在任务需求以超长航时和能量管理为主时，优先选择NSGA-II或MOEA/D得到的节能/SOC保护折中解；在任务需求以快速响应和扰动恢复为主时，选择响应优先解并通过直接配点控制功率变化率；在模型评价成本较高时，启用Kriging代理辅助搜索；在控制器参数整定阶段，使用PSO作为快速局部调参工具。这样形成的算法库既能服务论文/报告中的方法论，也能服务后续仿真迭代。",
)

add_heading(doc, "十二、软件/程序支撑建议", 1)
add_para(
    doc,
    "Python用于建立统一算法平台，承担数据管理、目标函数评价、约束处理、结果绘图和批量仿真调度。pymoo用于实现NSGA-II、MOEA/D及多目标性能评价指标，便于输出Pareto前沿、收敛曲线和非支配解集。SciPy用于单目标基线、局部搜索和数值插值。CasADi用于直接配点或伪谱法形式的连续最优控制建模，重点生成候选控制量序列和状态轨迹。scikit-learn或高斯过程工具用于Kriging/GP代理模型，实现高成本仿真的样本学习与快速预测。",
)
add_para(
    doc,
    "RCAIDE可作为中期任务级飞行性能与推进能量分析平台，用于构建飞机、推进系统、能量网络和任务剖面，并向优化算法返回能耗、航程、航时和关键状态量。Fluent或STAR-CCM+可作为后期高保真气动校核工具，用于典型工况下的升阻特性、螺旋桨/机体干扰或关键气动参数修正；其结果不直接替代优化主循环，而是用于修正低阶模型、训练代理模型或验证关键Pareto解。这样的软件分工能够兼顾当前可实现性和后续高保真扩展空间。",
)

add_heading(doc, "十三、结论", 1)
add_para(
    doc,
    "综合文献调研和PTDL项目需求，2.2多目标优化控制算法研究应以NSGA-II作为主基准算法，以MOEA/D作为高维多目标增强和对比算法，以Kriging代理模型降低昂贵仿真的评价成本，以直接配点法/伪谱法完成连续控制轨迹精修。GA和PSO具有工程调参价值，但不宜单独作为最终多目标优化控制主算法。该组合路线能够同时覆盖全局搜索、目标权衡、约束处理、仿真加速和连续控制可执行性，适合混合动力无人机飞推控一体化控制问题从算法原型走向项目仿真验证。",
)
add_para(
    doc,
    "对于卢奕主责工作，近期最关键的不是追求复杂算法数量，而是建立一套清晰可复现的算法验证框架：明确目标函数和约束定义，建立简化可运行的全任务剖面评价函数，完成NSGA-II与MOEA/D的基准对比，引入代理模型降低仿真成本，并用CasADi对代表性Pareto解进行控制轨迹精修。这样形成的报告和程序成果能够直接支撑后续与动能学院模型对接后的算法迁移。",
)

add_heading(doc, "参考文献", 1)
refs = [
    "[1] Holland, J. H. Adaptation in Natural and Artificial Systems. University of Michigan Press, 1975.",
    "[2] Goldberg, D. E. Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley, 1989.",
    "[3] Kennedy, J., and Eberhart, R. Particle swarm optimization. Proceedings of ICNN'95 - International Conference on Neural Networks, 1995. DOI: 10.1109/ICNN.1995.488968.",
    "[4] Coello Coello, C. A., Pulido, G. T., and Lechuga, M. S. Handling multiple objectives with particle swarm optimization. IEEE Transactions on Evolutionary Computation, 8(3), 256-279, 2004. DOI: 10.1109/TEVC.2004.826067.",
    "[5] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. A fast and elitist multi-objective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197, 2002. DOI: 10.1109/4235.996017.",
    "[6] Zhang, Q., and Li, H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 712-731, 2007. DOI: 10.1109/TEVC.2007.892759.",
    "[7] Jones, D. R., Schonlau, M., and Welch, W. J. Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13, 455-492, 1998. DOI: 10.1023/A:1008306431147.",
    "[8] Díaz-Manríquez, A., Toscano, G., Barron-Zambrano, J. H., and Tello-Leal, E. A review of surrogate assisted multiobjective evolutionary algorithms. Mathematical Problems in Engineering, 2016. DOI: 10.1155/2016/9420460.",
    "[9] Betts, J. T. Survey of numerical methods for trajectory optimization. Journal of Guidance, Control, and Dynamics, 21(2), 193-207, 1998. DOI: 10.2514/2.4231.",
    "[10] Hargraves, C. R., and Paris, S. W. Direct trajectory optimization using nonlinear programming and collocation. Journal of Guidance, Control, and Dynamics, 10(4), 338-342, 1987. DOI: 10.2514/3.20223.",
    "[11] Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., and Diehl, M. CasADi: A software framework for nonlinear optimization and optimal control. Mathematical Programming Computation, 11(1), 1-36, 2019. DOI: 10.1007/s12532-018-0139-4.",
    "[12] Blank, J., and Deb, K. pymoo: Multi-objective optimization in Python. IEEE Access, 8, 89497-89509, 2020. DOI: 10.1109/ACCESS.2020.2990567.",
    "[13] RCAIDE Documentation. Open-source software for aircraft design and analysis, including mission-level analysis, propulsion/energy networks and design space exploration. https://www.docs.rcaide.leadsresearchgroup.com/",
    "[14] Ansys PyFluent Documentation. Pythonic access to Ansys Fluent for meshing, solver control and field data extraction. https://fluent.docs.pyansys.com/",
    "[15] Siemens Simcenter STAR-CCM+ product documentation. Multiphysics CFD simulation platform with automated workflows and design exploration. https://www.siemens.com/en-us/products/simcenter/fluids-thermal-simulation/star-ccm/",
    "[16] Deb, K. An efficient constraint handling method for genetic algorithms. Computer Methods in Applied Mechanics and Engineering, 186(2-4), 311-338, 2000. DOI: 10.1016/S0045-7825(99)00389-8.",
    "[17] Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., and da Fonseca, V. G. Performance assessment of multiobjective optimizers: an analysis and review. IEEE Transactions on Evolutionary Computation, 7(2), 117-132, 2003. DOI: 10.1109/TEVC.2003.810758.",
]
for ref in refs:
    add_reference(doc, ref)

add_footer(doc)
doc.core_properties.title = "PTDL 2.2 多目标优化控制算法文献调研与适配报告"
doc.core_properties.subject = "混合动力无人机飞推控一体化控制，多目标优化算法，文献调研"
doc.core_properties.author = "Codex"
doc.save(OUT)
print(OUT)
