from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = r"C:\Users\23598\Documents\ChatGPT\LDTP\PTDL_2.1多目标性能优化指标体系文献调研与建模报告_80小时目标与阶段输出修订版.docx"


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
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.25

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


def style_body(p, first_line=True, before=0, after=8, line=1.25):
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    if first_line:
        p.paragraph_format.first_line_indent = Pt(22)


def add_para(doc, text, first_line=True):
    p = doc.add_paragraph()
    style_body(p, first_line=first_line)
    r = p.add_run(text)
    set_east_asian_font(r)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    p.paragraph_format.space_before = Pt(16 if level == 1 else 10)
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


def add_note(doc, text):
    p = doc.add_paragraph()
    style_body(p, first_line=False, before=4, after=10, line=1.18)
    p.paragraph_format.left_indent = Pt(12)
    p.paragraph_format.right_indent = Pt(12)
    r = p.add_run(text)
    set_east_asian_font(r)
    r.bold = True
    r.font.color.rgb = RGBColor(31, 77, 120)


def add_ref(doc, text):
    p = doc.add_paragraph()
    style_body(p, first_line=False, after=4, line=1.12)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-18)
    r = p.add_run(text)
    set_east_asian_font(r)
    r.font.size = Pt(9.5)


def add_footer(doc):
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("PTDL项目 2.1 多目标性能优化指标体系文献调研与建模报告")
        set_east_asian_font(r)
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(100, 100, 100)


doc = Document()
set_page(doc)
set_styles(doc)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(4)
r = title.add_run("PTDL 2.1 多目标性能优化指标体系文献调研与建模报告")
set_east_asian_font(r, "黑体")
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = RGBColor(11, 37, 69)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_after = Pt(14)
r = sub.add_run("面向混合动力超长航时无人机飞推控一体化控制")
set_east_asian_font(r)
r.font.size = Pt(11)
r.font.color.rgb = RGBColor(85, 85, 85)

add_note(
    doc,
    "报告定位：本报告围绕“研究无人机飞推控一体化控制问题的多目标性能优化指标体系”展开，结合飞机性能与概念设计、多学科设计优化、混合动力能量管理、飞行控制品质和多目标优化理论，给出PTDL项目中可量化、可仿真、可用于NSGA-II/MOEA/D等算法求解的指标体系与数学建模建议。",
)

add_heading(doc, "一、项目背景与指标体系建设逻辑", 1)
add_para(
    doc,
    "PTDL项目面向对标美国ULTRA无人机战术指标的混合动力超长航时无人机，关键可量化目标是在统一任务定义下将续航性能由ULTRA的72小时提升至80小时，并兼顾高效动力输出、低可探测性和任务适应性。该类无人机的核心矛盾不是单纯“飞得越久越好”，而是在固定重量、气动布局、推进功率、电池容量、燃油携带量、控制稳定性和任务响应能力之间寻找最优折中。飞机设计文献中，航程、续航、升阻比、功率需求、单位重量能量利用率和约束边界是评价长航时飞行器性能的基本指标；多学科设计优化文献进一步指出，飞机总体、气动、推进、结构和控制之间存在强耦合，单学科最优往往不能保证系统级最优。",
)
add_para(
    doc,
    "对于混合动力无人机，指标体系必须比传统燃油飞机更丰富。传统螺旋桨飞机续航分析通常围绕最小需用功率、最大升阻比和燃油消耗率展开，而混合动力系统还需要考虑电池SOC演化、发动机与电机功率分配、充放电效率、功率部件热限制、电池寿命保护和任务末端能量储备。如果只把燃油消耗作为目标，算法可能倾向于过度使用电池，使任务后段缺乏功率余量；如果只追求SOC保持，算法可能让发动机长期处于不经济工况，反而降低总续航。因此，PTDL的性能指标体系应同时覆盖任务完成能力、能量效率、控制品质和安全约束。",
)
add_para(
    doc,
    "多目标优化理论为这一问题提供了适当表达方式。Miettinen和Deb等关于多目标优化的研究表明，当多个目标互相冲突时，不应把问题简单压缩为一个综合评分，而应先构建Pareto意义下的非支配解集，再根据任务偏好选择折中方案。对于PTDL项目，这意味着应将“最长续航”“最低等效能耗”“最高能量管理效率”“最小稳定误差”“最快任务响应”和“最大安全裕度”等指标转化为统一的目标函数和约束集合，使算法输出能反映工程权衡，而不是只产生一个缺乏解释性的最优值。",
)
add_para(
    doc,
    "本报告将续航/任务完成、燃油与等效能耗、SOC管理、飞行性能与气动效率、飞行稳定性与控制品质、任务响应速度与动力可用性统称为“六类候选评价维度”。这六类维度共同构成完整的工程评价框架，但不意味着六者必须同时进入Pareto目标向量。其后按“Pareto目标 + 强约束 + 诊断评价指标”划分角色：Pareto目标直接驱动非支配解集形成；强约束保证飞行、推进、电池和控制系统处于可行安全范围；诊断评价指标不参与支配关系计算，但用于解释某个策略为什么有效或失效。这样的划分能够避免目标过多导致算法选择压力下降，同时保留后续工程分析所需的信息。",
)
add_para(
    doc,
    "需要特别避免把所有可测性能都并列设为优化目标。目标数增加会削弱非支配排序的区分能力，也会使结果难以解释。建议PTDL在算法原型阶段采用“3个核心目标 + 强约束”的最小可用配置：以等效任务能耗J_E、飞行控制品质J_stab和任务响应J_resp构成Pareto目标；将80小时续航目标、任务完成、终端SOC储备、SOC下限、功率边界、飞行包线和控制变化率作为强约束。只有当任务明确要求在飞行过程中维持某条SOC轨迹，或必须比较不同电池使用深度时，才把SOC管理指标提升为第4个独立目标。低可探测性、热管理和寿命损耗在缺乏可信模型时先作为诊断项或场景约束，不应为了“指标完整”而过早加入目标向量。",
)

add_heading(doc, "二、飞行器设计文献对指标选择的启示", 1)
add_para(
    doc,
    "飞机性能与概念设计理论通常从任务剖面、重量估算、气动效率和推进效率出发评价飞行器。Raymer、Roskam、Torenbeek、Anderson等飞机设计与性能教材均强调，航程和续航不是孤立指标，而是由重量、升阻比、燃油或能量消耗率、推进效率和任务剖面共同决定。对于螺旋桨飞机，最大续航通常对应最小需用功率附近的飞行状态；最大航程则更接近最大升阻比或单位距离能耗最低状态。这一结论对PTDL项目非常关键：若项目目标是超长航时，优化指标不能只看总距离，还必须区分“单位时间能耗最低”和“单位距离能耗最低”两种不同飞行状态。",
)
add_para(
    doc,
    "Breguet航程/续航关系虽然最初用于传统动力飞机，但其物理含义仍可作为指标体系构建的基础。对长航时无人机而言，续航能力可以理解为可用能量、推进效率、气动效率和重量变化共同作用的结果。若采用等效能量表达，可把燃油化学能、电池可用电能和推进链效率统一到任务能量消耗中，从而建立跨动力源的比较基准。该思路适合混合动力无人机，因为发动机燃油和电池电能虽然能量密度、效率链路和动态响应不同，但最终都服务于克服阻力、维持高度/速度、完成任务载荷和机动响应。",
)
add_para(
    doc,
    "多学科设计优化文献则提示，应避免把飞行器性能指标孤立地归入某一学科。例如，降低巡航速度可能降低需用功率并延长续航，但也可能改变螺旋桨效率、发动机工作点、控制裕度和任务响应时间；提高展弦比可能改善升阻比，但带来结构重量和动态特性变化；增大电池容量可提高电助推能力，但增加起飞重量并可能提高巡航需用功率。因此，PTDL指标体系要体现飞行性能、推进能量和控制策略之间的耦合关系，至少在目标函数中保留能量、稳定和响应三个维度。",
)

add_heading(doc, "三、候选评价维度一：续航时间与任务完成能力", 1)
add_para(
    doc,
    "优化角色：80小时续航目标是本项目的一级可行性门槛，应写为g_80(z) = 80 h - T_endurance(z) <= 0，而不是与能耗、稳定性和响应速度可相互交换的一般Pareto目标。固定任务距离、驻留时间、有效载荷工作时间和返航储备后，只有满足g_80(z) <= 0的策略才进入正式Pareto前沿；续航时间仍作为诊断输出，用于比较80小时以上方案的裕度。只有在前期“能否达标”的可行性攻关阶段，才将J_T = -T_endurance作为单独的主导目标。",
)
add_para(
    doc,
    "续航时间是PTDL项目最直接的牵引指标。其基本定义为无人机在满足任务剖面、飞行包线、推进系统和能量安全约束条件下能够持续飞行的总时间，可记为T_endurance。若任务剖面包含爬升、巡航、任务区滞空、返航和下降，则总续航时间应写为T_endurance = t_climb + t_cruise + t_loiter + t_return + t_descent，其中任务区滞空时间和全剖面可持续时间应作为重点评价对象。对于对标ULTRA类超长航时指标的项目，80小时必须指向统一任务剖面、统一有效载荷功率、统一环境假设和统一终端能量储备下的全任务续航；不得以仅改变巡航段理论滞空时间、降低载荷功率或取消返航储备来替代该定义。",
)
add_para(
    doc,
    "围绕80小时目标，优化应采用分阶段表达。阶段A为可行性攻关阶段：在任务剖面和安全边界固定的前提下，以最大化T_endurance或最小化g_80的违反度为主，定位气动、推进和能量管理中的达标瓶颈。阶段B为正式多目标控制优化阶段：将g_80(z) <= 0与任务完成约束共同施加，仅在80小时及以上的可行策略之间，以J_E、J_stab和J_resp形成Pareto前沿。这样的处理避免出现“79小时但响应更快”被误选为折中最优的情况，也避免算法通过降低任务强度、缩短有效任务内容来虚假提高续航。",
)
add_para(
    doc,
    "与续航时间配套的任务完成能力指标包括终端航程误差、任务点到达误差、任务区驻留时间、有效载荷工作时间和返航能量储备。正式模型至少应包含续航门槛g_80(z) = 80 h - T_endurance(z) <= 0、航程约束s(tf) >= S_req、任务区驻留约束T_loiter >= T_loiter,min、终端能量约束SOC(tf) >= SOC_res和m_fuel(tf) >= m_res。若项目后续引入侦察、通信中继或载荷供电任务，还应把载荷工作功率P_payload纳入能量平衡，否则算法会低估任务真实能耗。",
)

add_heading(doc, "四、候选评价维度二：燃油消耗与等效能耗", 1)
add_para(
    doc,
    "优化角色：等效任务能耗J_E是基准Pareto优化的第一目标。燃油消耗、燃油流量和电池净能量消耗作为J_E的组成与诊断分量同步输出，用于解释能耗改善来自发动机工况、气动工况还是电池使用深度。",
)
add_para(
    doc,
    "燃油消耗是传统飞机性能优化中的核心指标，也是混合动力无人机评价经济性和续航潜力的基础。燃油消耗可直接定义为m_fuel,used = m_fuel(0) - m_fuel(tf)，也可通过发动机燃油流量积分得到m_fuel,used = ∫ dot_m_f(t) dt。若发动机模型已知，燃油流量可由功率和比油耗近似计算，例如dot_m_f = BSFC(P_eng, n_eng, h, V) · P_eng；若早期缺乏详细发动机图谱，可先用常值等效比油耗或分段效率模型近似，待动能学院提供推进系统参数后替换为发动机工作图。",
)
add_para(
    doc,
    "对于混合动力无人机，单独最小化燃油消耗会带来指标偏置，因为算法可能用电池能量替代燃油而不考虑电池后续任务价值。因此更合理的核心指标是等效任务能耗，并应首先在统一能量单位下表达。建议定义J_E = m_fuel,used·LHV_fuel + Delta_E_bat,net，其中Delta_E_bat,net = ∫[P_bat,dis(t)/eta_dis(t) - eta_chg(t)·P_bat,chg(t)]dt，表示电池在任务中的净化学能消耗；P_bat,dis和P_bat,chg均取非负值。若只掌握SOC而没有完整电流/效率图，可用Delta_E_bat,net约等于[SOC(0)-SOC(tf)]·E_bat,usable作早期近似，但应在报告中标注该近似未体现倍率、温度与老化对可用能量的影响。只有在必须把结果换算为“等效燃油质量”时，才使用J_ECF = J_E/LHV_fuel；不建议把经验权重k_soc直接写成物理能量折算系数，以免混淆物理守恒与任务偏好。",
)
add_para(
    doc,
    "等效能耗指标的优势在于能够统一比较不同功率分配策略。例如，策略A燃油消耗较低但SOC下降严重，策略B燃油消耗略高但保留较多电池余量，如果只看燃油，策略A可能被误判为更优；若采用等效能耗和终端SOC约束，算法会识别出两者在续航与能量安全上的真实权衡。该指标也便于与多目标优化算法结合：J_E作为主要最小化目标，SOC管理指标作为独立目标或约束，二者共同塑造能量利用Pareto前沿。",
)

add_heading(doc, "五、候选评价维度三：电池能量管理效率与SOC健康", 1)
add_para(
    doc,
    "优化角色：对耗电型超长航时基准任务，SOC下限、终端SOC储备、充放电功率和电流倍率应作为强约束，SOC最低值和轨迹用于诊断；对荷电维持任务或需要比较不同储备策略的任务，终端/关键节点SOC偏差J_SOC,res才作为第四个可选Pareto目标。",
)
add_para(
    doc,
    "混合动力无人机的电池并不只是辅助能源，而是影响动力响应、低可探测运行、应急机动和任务后段安全裕度的关键资源。不过SOC的优化表达必须服从任务能量策略。对“耗电型”超长航时任务，电池允许在全程或指定阶段单调消耗，此时SOC(tf) >= SOC_res和SOC(t) >= SOC_safe应作为硬约束，J_E已包含电池净能量消耗，不宜再把SOC轨迹跟踪重复设为目标。对“荷电维持型”或要求中途保留机动电量的任务，可定义J_SOC1 = |SOC(tf) - SOC_ref,tf|，并用J_SOC2 = ∫[SOC(t)-SOC_ref(t)]^2dt度量能量管理偏差。无论采用何种模式，均应记录SOC最低值、SOC波动幅度和关键阶段储备，以便解释策略是否透支了后段安全裕度。",
)
add_para(
    doc,
    "电池能量管理效率还应考虑充放电损失和高倍率使用惩罚。可定义电池损耗指标J_bat_loss = ∫ [(1/eta_dis - 1)P_bat,dis(t) + (1 - eta_chg)P_bat,chg(t)] dt，用于度量电池能量在转换过程中的损失；定义J_Crate = ∫ max(0, |I_bat(t)| - I_nom)^2 dt，用于惩罚高倍率充放电；若后续获得电池热模型，还可加入温度约束T_bat(t) <= T_bat,max或温度惩罚J_Tbat。这样可以避免优化算法为了短期能耗最优而频繁大功率充放电，损害电池寿命和安全性。",
)
add_para(
    doc,
    "从飞推控耦合角度看，SOC指标与控制响应有显著冲突。电机响应快，适合爬升、突防、抗扰和动力模式切换，但过度使用会降低任务后段SOC；发动机能量密度高，适合长时间巡航供能，但响应较慢且可能带来噪声、红外和振动特征。PTDL指标体系应明确这种冲突：在常规长航时工况下，SOC轨迹应平稳下降并保留末端储备；在快速响应工况下，允许短时电助推但必须设置SOC恢复或保护机制。算法输出的Pareto解应能展示“节能巡航策略”“电池保护策略”和“响应优先策略”之间的差异。",
)

add_heading(doc, "六、候选评价维度四：飞行性能与气动效率", 1)
add_para(
    doc,
    "优化角色：升阻比、需用功率、爬升率和单位时间能耗是J_E和任务可行性的物理支撑，主要作为模型输入、任务约束与诊断输出。除非后续研究专门比较不同总体/气动构型，否则不将气动效率与J_E并列为独立Pareto目标，以避免对同一能耗收益重复计量。",
)
add_para(
    doc,
    "飞行性能指标连接飞机设计和控制策略，是续航优化的物理基础。对于固定翼长航时无人机，气动效率可用升阻比L/D、阻力D、需用功率P_req = D·V、单位距离能耗E/S和单位时间能耗E/T来表达。巡航段若以续航为主，应关注最小需用功率附近的状态；若以航程为主，应关注单位距离能耗最低或接近最大升阻比的状态。由于PTDL项目强调超长航时，指标体系应优先关注P_req、P_req/W、E/T和任务区驻留效率，而不是只看最大速度或最大航程。",
)
add_para(
    doc,
    "飞行性能目标可定义为J_P = ∫ P_req(t) dt或J_D = ∫ D(t)V(t) dt，其实质与能耗目标相关，但保留气动/飞行性能指标有助于解释优化结果。例如，某一策略能耗降低可能来自更优巡航高度和速度，也可能来自过度使用电池；通过同时观察P_req、L/D和功率分配，可以判断性能提升来源。若后续接入RCAIDE，飞行性能指标可由任务分析输出；若接入Fluent或STAR-CCM+，典型工况下的升阻特性可用于修正低阶极曲线C_D = C_D0 + kC_L^2。",
)
add_para(
    doc,
    "飞行性能还包括爬升能力、速度包线和高度保持能力。爬升能力可用最大爬升率ROC、爬升段能耗E_climb和达到任务高度时间t_climb表示；速度包线约束可写为V_min(h,W) <= V(t) <= V_max(h,P_avail)；高度保持误差可写为J_h = ∫ [h(t)-h_cmd(t)]^2 dt。对于超长航时无人机，爬升段虽然时间占比可能低于巡航段，但对电池和发动机峰值功率需求影响显著，因此不能只优化巡航段而忽略爬升能量成本。",
)

add_heading(doc, "七、候选评价维度五：飞行稳定性与控制品质", 1)
add_para(
    doc,
    "优化角色：归一化飞行跟踪误差、峰值偏差和扰动恢复品质共同构成基准Pareto目标J_stab。失速、姿态角、角速度和载荷等绝对安全边界仍必须作为硬约束，而不能被较小的平均误差掩盖。",
)
add_para(
    doc,
    "飞行稳定性指标用于保证优化得到的节能策略不是以牺牲可控性和任务安全为代价。飞行控制文献通常通过跟踪误差、超调量、调节时间、阻尼特性、稳定裕度、控制输入幅值和执行机构饱和程度评价控制品质。对于PTDL项目，稳定性指标应覆盖速度、高度、姿态和动力模式切换过程。可定义J_track = ∫ [w_h(h-h_cmd)^2 + w_V(V-V_cmd)^2 + w_theta(theta-theta_cmd)^2 + w_q q^2] dt，用于描述纵向飞行状态跟踪误差。",
)
add_para(
    doc,
    "稳定性还应包括峰值偏差和恢复能力。均方误差较小并不代表安全，因为短时间的大姿态偏差或速度跌落可能触发失速、载荷限制或任务失败。因此可定义e_h,max、e_V,max、theta_max、q_max等峰值指标，并设置硬约束。对于扰动场景，可定义恢复时间t_rec，即遭遇风扰、功率突变或任务指令变化后，状态误差重新进入允许带并保持稳定的时间。这样，稳定性指标不仅评价稳态巡航，还评价飞推控耦合系统在动态过程中的安全边界。",
)
add_para(
    doc,
    "控制品质指标与能耗目标存在明显冲突。为了降低能耗，算法可能选择较小控制输入、较慢速度变化和较低功率响应，但这会增加任务转换时间和扰动恢复时间；为了提高稳定性和响应速度，控制器可能频繁调节舵面和功率，导致执行机构负担增加、能耗上升并引入振荡风险。因此，控制平滑性应作为指标体系的重要组成部分。可定义J_u = ∫ ||u(t)||^2 dt评价控制努力，定义J_du = ∫ ||du/dt||^2 dt评价控制变化率。对于发动机和电机功率，还应分别设置|dP_eng/dt|和|dP_bat/dt|限制，体现推进系统动态响应约束。",
)

add_heading(doc, "八、候选评价维度六：任务响应速度与动力可用性", 1)
add_para(
    doc,
    "优化角色：任务阶段转换时间、功率建立时间、扰动恢复时间和模式切换时间共同构成基准Pareto目标J_resp。最大可用功率、功率爬升率和执行机构速率属于支撑J_resp的路径约束与诊断量，用于防止算法用不可实现的瞬态功率换取表面上的快速响应。",
)
add_para(
    doc,
    "任务响应速度反映无人机在任务指令变化、航迹调整、突发扰动和动力模式切换时的动态适应能力。对于战术无人机，超长航时并不意味着始终低速、低功率、慢响应；在侦察转场、规避、返航或应急处置中，动力系统必须提供足够响应余量。因此，响应速度应作为独立性能指标，而不是完全包含在稳定性误差中。",
)
add_para(
    doc,
    "响应速度可用调节时间、上升时间、任务阶段转换时间、扰动恢复时间和可用功率裕度来量化。典型定义包括t_settle，即速度、高度或姿态误差进入给定误差带并保持的时间；t_switch，即发动机/电机模式切换到新稳态所需时间；P_margin = P_available - P_required，用于衡量当前工况下的动力余量；E_boost_available，用于衡量在未来一段时间内可用于电助推的电池能量。对于PTDL优化模型，可将J_resp = t_settle + beta_switch t_switch + beta_rec t_rec作为响应目标，也可将P_margin >= P_margin,min作为约束。",
)
add_para(
    doc,
    "响应指标与电池管理和低可探测性均存在冲突。电机具有快速响应优势，能缩短t_settle和t_switch，但会消耗SOC；发动机长时间高功率运行可提供持续动力余量，但可能增加燃油消耗、热特征和噪声特征。若项目强调低可探测运行，则在某些任务阶段可能倾向于电驱或低功率发动机工况，但这又会限制长时间任务中的能量储备。因此，任务响应速度指标需要与SOC储备、功率模式和可探测性代理指标联合解释。",
)

add_heading(doc, "九、扩展指标：低可探测性、热管理与系统可靠性", 1)
add_para(
    doc,
    "项目简介强调混合动力无人机具有低可探测性优势，因此指标体系中应保留低可探测性相关扩展指标。由于早期可能缺乏完整声学、红外和电磁特征模型，不建议一开始把低可探测性作为高保真目标直接优化，而应采用代理指标。声学代理指标可用发动机工作时间占比、发动机高功率工作时间、螺旋桨转速变化率和功率谱特征近似；红外代理指标可用发动机功率、排气热流或关键部件温度近似；电磁或通信可探测性可在任务载荷层面另行建模。对于现阶段2.1，建议把低可探测性作为诊断评价指标或阶段约束，而不是主目标。",
)
add_para(
    doc,
    "热管理指标同样重要。混合动力系统中的电机、电控、电池和发动机都有温度边界，高功率爬升或频繁电助推可能导致热积累。若早期缺乏详细热模型，可先设置功率持续时间约束和C-rate约束；后期可引入电池温度T_bat、电机温度T_motor和控制器温度T_inv，并设置T_i(t) <= T_i,max。热约束的工程意义在于防止算法给出短期性能很强但无法长期执行的控制策略。",
)
add_para(
    doc,
    "系统可靠性可通过能量储备、冗余动力可用性和约束裕度体现。对于超长航时无人机，任务后段的不确定性较大，保留燃油、电池和功率余量比单点最优更重要。因此可定义综合安全裕度M_safe，由SOC裕度、燃油裕度、速度失速裕度、功率裕度和热裕度共同构成。该指标可作为鲁棒性研究2.3的接口，也可在2.1中作为多目标优化的约束基础。",
)

add_heading(doc, "十、约束条件体系", 1)
add_para(
    doc,
    "目标函数决定算法追求什么，约束条件决定哪些策略具有工程可行性。PTDL项目的约束体系应包括任务约束、飞行包线约束、推进系统约束、电池约束、控制约束和安全裕度约束。任务约束包括起降条件、任务航程、任务区驻留时间、终端位置误差和返航能量储备；飞行包线约束包括速度上下限、高度范围、迎角限制、载荷因数限制、爬升率限制和失速裕度；推进系统约束包括发动机功率边界、电机功率边界、螺旋桨转速边界、发动机动态响应限制和功率变化率限制。",
)
add_para(
    doc,
    "电池约束应至少包括SOC_min <= SOC(t) <= SOC_max，|I_bat(t)| <= I_max，P_bat(t) <= P_bat,max，以及任务结束SOC(tf) >= SOC_res。若考虑电池寿命和安全，应增加充放电倍率、温度和循环深度约束。控制约束应包括舵面偏角、舵面角速度、控制输入幅值、控制输入变化率和姿态边界。对于算法实现，所有约束可统一转化为g_i(z) <= 0形式，并计算总约束违反度CV(z) = Σ max(0, g_i(z))，用于NSGA-II、MOEA/D或其他多目标算法中的可行性优先排序。",
)
add_para(
    doc,
    "约束设置必须避免过松或过严。约束过松会使算法产生物理上不可执行的节能策略，例如SOC过低、功率突变过大或速度接近失速；约束过严则可能使可行域过小，导致算法难以收敛。合理做法是在早期简化模型中先设置保守但可行的边界，通过初始仿真样本观察可行解比例，再逐步调整约束。对于关键安全约束，如SOC下限、速度包线、功率上限和姿态边界，应作为硬约束；对于控制平滑性、末端SOC偏好和能量储备水平，可根据研究目的在目标函数和软约束之间切换。",
)

add_heading(doc, "十一、六类候选评价维度的优化角色与进入规则", 1)
add_para(
    doc,
    "为使指标体系能够直接进入2.2算法而不产生重复计量，建议按任务模式配置目标向量。基准配置用于耗电型长航时任务：F_base(z) = [J_E, J_stab, J_resp]，并统一施加80小时门槛g_80(z) <= 0、SOC(tf) >= SOC_res、SOC(t) >= SOC_safe等约束。扩展配置用于需要比较电池使用深度或中途储备的任务：F_ext(z) = [J_E, J_SOC,res, J_stab, J_resp]，其中J_SOC,res只评价终端或关键节点的储备偏差，不与J_E重复计算同一段电池放电。若引入电池寿命或热管理，优先以倍率、温度和功率边界约束开始；只有在有电池热-老化模型及可靠数据支撑时，才将J_bat_loss或寿命损耗升级为独立目标。",
)
add_para(
    doc,
    "诊断指标应与优化指标同步输出，但不参与支配关系计算。建议每个Pareto解至少附带：续航时间或任务区驻留时间、燃油消耗、电池净能量消耗、终端SOC、最低SOC、最大功率、最大功率变化率、最大高度/速度/姿态误差、约束裕度以及气动工作点。这样能够区分“能耗低是因为气动工况更优”与“能耗低只是因为更多消耗了电池”两种完全不同的结论，也便于后续与动能学院模型对接时追溯参数来源。",
)

add_heading(doc, "十二、多目标数学模型建议", 1)
add_para(
    doc,
    "综合上述指标，PTDL可建立如下约束多目标优化模型：在给定任务剖面M、飞机/推进/电池参数Theta和环境条件E下，寻找控制策略参数和阶段变量z，使目标向量F(z)最小，同时满足动力学、能量、任务和安全约束。耗电型基准任务采用F_base(z) = [J_E, J_stab, J_resp]；需要显式比较能量储备策略时采用F_ext(z) = [J_E, J_SOC,res, J_stab, J_resp]。两种正式模型均必须满足g_80(z) = 80 h - T_endurance(z) <= 0，以及统一的任务、飞行包线、推进、电池、控制和安全约束。控制平滑性J_smooth、低可探测性J_detect、热管理和可靠性指标在模型和数据成熟前作为诊断指标或软约束，避免在早期把不确定性很大的估计量变成主导优化方向。",
)
add_para(
    doc,
    "各目标的推荐定义为：J_E = m_fuel,used·LHV_fuel + Delta_E_bat,net，用于统一燃油与电能的任务消耗；J_SOC,res = |SOC(tf)-SOC_ref,tf|，或在存在关键机动节点时加入相应节点偏差；J_stab = ∫[w_h e_h,norm^2 + w_V e_V,norm^2 + w_theta e_theta,norm^2 + w_q q_norm^2]dt + gamma_peak·J_peak，用于评价跟踪品质和峰值风险；J_resp = t_settle + beta_rec·t_rec + beta_switch·t_switch，用于评价任务转换、扰动恢复和动力模式切换速度。误差量在进入J_stab前应按允许误差或工程限值归一化，J_peak应由高度、速度、姿态等峰值超限的归一化最大值构成。续航时间既可作为J_T = -T_endurance加入探索性目标，也可在固定任务距离与驻留时间条件下通过J_E和终端储备间接体现；正式对比时必须固定任务要求，不能以缩短任务来换取更优指标。",
)
add_para(
    doc,
    "变量z应采用分层表达。早期算法研究中，z可包含巡航速度、巡航高度、爬升速度、爬升率、发动机/电机功率分配系数、SOC参考值、模式切换阈值和控制器权重，维度控制在10到25维；中期可引入少量样条节点描述P_eng(t)、P_bat(t)、V_cmd(t)和h_cmd(t)；后期在CasADi直接配点中将状态和控制在时间节点上离散化，形成连续控制轨迹优化问题。分层变量设计的好处是既保留工程解释性，又避免早期多目标算法因维度过高而失效。",
)
add_para(
    doc,
    "指标归一化是多目标优化中容易被忽略但非常关键的步骤。能耗、SOC偏差、姿态误差和响应时间具有不同量纲和数量级，若直接进入权重分解、距离评价或后处理决策，会导致某些目标支配结果。建议在初始样本或基准策略上估计每个指标的理想值f_i,min和参考最差值f_i,max，采用f_i,norm = (f_i - f_i,min)/(f_i,max - f_i,min + epsilon)进行归一化。对于必须满足的安全指标，不应只归一化后参与目标比较，而应保留硬约束形式。",
)

add_heading(doc, "十三、指标冲突与权衡关系", 1)
add_para(
    doc,
    "PTDL指标体系的价值在于揭示冲突，而不是掩盖冲突。第一类冲突是续航时间与任务响应速度之间的冲突。低功率、低阻巡航有利于延长续航，但会降低加速、爬升和扰动恢复能力；快速响应需要更高功率和更积极控制，通常带来能耗上升。第二类冲突是燃油消耗与电池SOC保持之间的冲突。减少燃油消耗往往意味着更多使用电池，而保持SOC则需要发动机承担更多功率或进行充电，可能偏离发动机经济工作点。",
)
add_para(
    doc,
    "第三类冲突是飞行稳定性与控制能耗之间的冲突。更高控制增益和更频繁功率调节可以降低状态误差，但可能造成执行机构频繁动作、功率变化率增大和能耗增加；过度平滑的控制策略虽然节能，却可能在风扰或任务切换时恢复缓慢。第四类冲突是气动效率与任务剖面要求之间的冲突。最小需用功率状态未必满足任务速度、任务高度或通信/侦察覆盖要求；最优升阻比状态也未必对应推进系统最高效率工作点。",
)
add_para(
    doc,
    "第五类冲突是低可探测性与续航/响应能力之间的冲突。电驱或低功率发动机模式可能降低声学和红外特征，但会消耗电池储备或限制持续功率输出；发动机经济巡航有利于长航时，却可能增加声学和热特征。第六类冲突是模型精度与优化效率之间的冲突。高保真气动和推进模型能提高指标可信度，但会显著增加算法评价成本；低阶模型便于搜索，但可能漏掉关键气动/推进耦合效应。因此，指标体系还应记录模型来源和置信度，配合代理模型和分层验证方法使用。",
)
add_para(
    doc,
    "这些冲突决定了PTDL项目应采用“80小时可行性筛选后的Pareto前沿”而非单一综合评分作为主要结果表达。对于导师和项目组而言，更有价值的结果不是宣称某个策略“最优”，而是在全部达到80小时的前提下展示若干代表性策略：节能优先策略、SOC保护优先策略、快速响应策略、稳定裕度优先策略和综合折中策略。每类策略都应给出对应的续航裕度、能耗、SOC轨迹、稳定误差、响应时间和约束裕度，从而支持后续控制策略选择和飞推控耦合设计迭代。",
)

add_heading(doc, "十四、软件实现与数据来源建议", 1)
add_para(
    doc,
    "Python应作为指标体系计算和优化算法调度的统一平台。早期可用Python建立简化飞行动力学、能量平衡和任务剖面模型，计算J_E、J_SOC、J_stab、J_resp和约束违反度；pymoo用于NSGA-II、MOEA/D等多目标算法调用；SciPy用于基线优化、插值和局部搜索；CasADi用于后续直接配点形式的连续最优控制；scikit-learn或Gaussian process工具用于代理模型训练和指标快速预测。",
)
add_para(
    doc,
    "RCAIDE可用于中期任务级飞机性能、推进系统、能量网络和任务剖面分析，为指标体系提供更可信的航时、航程、能耗、功率需求和SOC演化数据。Fluent或STAR-CCM+可用于典型工况气动校核，输出升阻特性、螺旋桨/机体干扰或关键气动参数修正结果。高保真CFD不建议直接嵌入每一次优化迭代，而应作为低阶模型修正、代理模型训练和关键Pareto解验证的数据来源。",
)
add_para(
    doc,
    "在项目数据尚不完整的情况下，指标体系可先采用分级模型。第一级为理论/经验模型，用于快速跑通算法；第二级为RCAIDE任务级模型，用于形成正式仿真结果；第三级为Fluent或STAR-CCM+关键点校核，用于提高气动参数可信度。每个指标都应记录其计算来源，例如理论公式、RCAIDE输出、CFD修正或试验/文献参数。这样后续与动能学院对接推进系统模型时，只需替换评价函数内部的数据来源，而不需要重构整个多目标优化框架。",
)

add_heading(doc, "十五、结论", 1)
add_para(
    doc,
    "PTDL项目的多目标性能优化指标体系应围绕“由72小时提升至80小时”的硬目标建立，但不能只用续航时间或燃油消耗单一指标描述。正式Pareto优化应以T_endurance >= 80 h为先决可行性条件，在此基础上将等效能耗、飞行稳定性和任务响应速度作为初期核心目标；将SOC管理按任务模式作为强约束或可选第四目标；将控制平滑性、低可探测性、热管理和安全裕度作为诊断指标或阶段性约束。该体系能够较好覆盖飞推控一体化问题中的气动、推进、能量和控制耦合关系。",
)
add_para(
    doc,
    "对于后续2.2算法研究，最适合的数学表达是分任务模式的约束多目标优化问题：耗电型基准任务采用F_base(z) = [J_E, J_stab, J_resp]，能量储备需要显式权衡时采用F_ext(z) = [J_E, J_SOC,res, J_stab, J_resp]，并统一施加任务、飞行包线、推进、电池、控制和安全约束。该模型既能被NSGA-II和MOEA/D直接调用，也能与Kriging代理模型和CasADi直接配点方法衔接。更重要的是，它能够把算法输出解释为工程折中：某个Pareto解为什么更省能、为什么响应更快、为什么SOC更安全，以及它牺牲了哪些性能。这样的指标体系可以作为PTDL多目标优化控制研究的正式研究基础。",
)

add_heading(doc, "十六、2.2.1阶段输出与验收形式", 1)
add_para(
    doc,
    "本节研究完成时，应形成可直接交付给2.2.2算法开发工作的标准化输入，而不是停留在指标文字说明。阶段输出以“指标定义文件 + 约束条件集 + 归一化与偏好分析方案 + 可调用评价函数 + 图形化验证结果”为主体，并统一采用80小时续航门槛作为正式任务可行性判据。",
)
add_para(
    doc,
    "第一，形成多目标性能指标体系。应给出六类候选评价维度及其优化角色，其中基准Pareto目标为等效任务能耗J_E、飞行控制品质J_stab和任务响应J_resp；SOC管理J_SOC,res为可选第四目标；续航/任务完成、气动性能、控制平滑性、低可探测性和热管理按任务需要分别归入强约束或诊断评价指标。每个指标应明确物理含义、计算公式、量纲、输入变量、输出变量和适用任务模式。",
)
add_para(
    doc,
    "第二，形成约束条件集。约束条件至少包括80小时续航门槛g_80(z) = 80 h - T_endurance(z) <= 0，任务航程与驻留时间、终端SOC和燃油储备、速度/高度/迎角/载荷因子等飞行包线、发动机与电机功率边界、电池SOC/电流/温度边界、控制输入与变化率边界，以及终端状态条件。每项约束应标明硬约束或软约束属性、上下限来源和约束违反度的计算方法。",
)
add_para(
    doc,
    "第三，形成归一化方法、权重扫描方案和Pareto评价规则。对J_E、J_stab和J_resp按基准样本估计理想值与参考最差值，采用统一归一化公式后再用于MOEA/D分解、解集距离计算和可视化。权重扫描仅用于考察不同任务偏好下的解集变化，或用于MOEA/D子问题设置与Pareto解后处理选择；正式结论仍以80小时以上可行解的非支配关系为基础，不能以单一加权总分替代Pareto分析。",
)
add_para(
    doc,
    "第四，形成可复用的Python评价程序。建议以NumPy实现指标和约束的数值计算，以pandas记录任务工况、参数样本、约束违反度和评价结果，以Matplotlib或Plotly输出SOC轨迹、功率分配曲线、约束裕度、Pareto前沿和权重扫描结果。程序至少提供统一的evaluate_mission(z, scenario)评价入口，输入控制策略参数与任务场景，输出目标向量、约束违反度、80小时达标标志及诊断指标；该接口将直接供2.2.2中的NSGA-II、MOEA/D等算法调用。",
)
add_para(
    doc,
    "第五，形成阶段验收材料。最终提交内容应包括：多目标指标体系说明；约束条件集与参数来源说明；归一化和权重扫描说明；评价函数源程序及调用示例；标称任务下的80小时达标检查结果；以及能够展示能耗、SOC、稳定性、响应速度和约束裕度的图形化结果。这样，2.2.1的输出既可作为研究报告的独立成果，也可作为后续多目标算法对比、代理模型训练和连续控制轨迹优化的统一数据接口。",
)

add_heading(doc, "参考文献", 1)
refs = [
    "[1] Raymer, D. P. Aircraft Design: A Conceptual Approach. AIAA Education Series, 6th ed., 2018.",
    "[2] Roskam, J. Airplane Design, Parts I-VIII. DARcorporation, 1985-2000.",
    "[3] Torenbeek, E. Synthesis of Subsonic Airplane Design. Delft University Press, 1982.",
    "[4] Anderson, J. D. Aircraft Performance and Design. McGraw-Hill, 1999.",
    "[5] Gudmundsson, S. General Aviation Aircraft Design: Applied Methods and Procedures. Butterworth-Heinemann, 2014.",
    "[6] Martins, J. R. R. A., and Lambe, A. B. Multidisciplinary design optimization: A survey of architectures. AIAA Journal, 51(9), 2049-2075, 2013. DOI: 10.2514/1.J051895.",
    "[7] Sobieszczanski-Sobieski, J., and Haftka, R. T. Multidisciplinary aerospace design optimization: Survey of recent developments. Structural Optimization, 14, 1-23, 1997. DOI: 10.1007/BF01197554.",
    "[8] Friedrich, C., and Robertson, P. A. Hybrid-electric propulsion for aircraft. Journal of Aircraft, 52(1), 176-189, 2015. DOI: 10.2514/1.C032660.",
    "[9] Pornet, C., Kaiser, S., Isikveren, A. T., and Hornung, M. Integrated fuel-battery hybrid for a narrow-body sized transport aircraft. Aircraft Engineering and Aerospace Technology, 86(6), 568-574, 2014. DOI: 10.1108/AEAT-05-2014-0062.",
    "[10] Donateo, T., Ficarella, A., Spedicato, L., Arista, A., and Ferraro, M. A new approach to calculating endurance in electric flight and comparing fuel cells and batteries. Applied Energy, 187, 807-819, 2017. DOI: 10.1016/j.apenergy.2016.11.100.",
    "[11] Miettinen, K. Nonlinear Multiobjective Optimization. Kluwer Academic Publishers, 1999. DOI: 10.1007/978-1-4615-5563-6.",
    "[12] Deb, K. Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.",
    "[13] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. A fast and elitist multi-objective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197, 2002. DOI: 10.1109/4235.996017.",
    "[14] Zhang, Q., and Li, H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 712-731, 2007. DOI: 10.1109/TEVC.2007.892759.",
    "[15] Stevens, B. L., Lewis, F. L., and Johnson, E. N. Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems. Wiley, 3rd ed., 2015.",
    "[16] Etkin, B., and Reid, L. D. Dynamics of Flight: Stability and Control. Wiley, 3rd ed., 1996.",
    "[17] Nelson, R. C. Flight Stability and Automatic Control. McGraw-Hill, 2nd ed., 1998.",
]
for ref in refs:
    add_ref(doc, ref)

add_footer(doc)
doc.core_properties.title = "PTDL 2.1 多目标性能优化指标体系文献调研与建模报告"
doc.core_properties.subject = "混合动力超长航时无人机，飞推控一体化，多目标性能指标体系"
doc.core_properties.author = "Codex"
doc.save(OUT)
print(OUT)
