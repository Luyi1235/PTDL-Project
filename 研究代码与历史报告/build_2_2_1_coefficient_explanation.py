import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = r"C:\Users\23598\Documents\ChatGPT\LDTP\PTDL_2.2.1交付1至交付3公式项与系数释义.docx"


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
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.25
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.bold = True
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    doc.styles["Heading 1"].font.size = Pt(16)
    doc.styles["Heading 1"].font.color.rgb = RGBColor(46, 116, 181)
    doc.styles["Heading 2"].font.size = Pt(13)
    doc.styles["Heading 2"].font.color.rgb = RGBColor(46, 116, 181)
    doc.styles["Heading 3"].font.size = Pt(12)
    doc.styles["Heading 3"].font.color.rgb = RGBColor(31, 77, 120)


def add_heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    p.paragraph_format.space_before = Pt(16 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_east_asian_font(run, "黑体")
    run.bold = True
    return p


def add_para(doc, text, first_line=True, after=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.25
    if first_line:
        p.paragraph_format.first_line_indent = Pt(22)
    run = p.add_run(text)
    set_east_asian_font(run)
    return p


def add_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.left_indent = Pt(12)
    p.paragraph_format.right_indent = Pt(12)
    run = p.add_run(text)
    set_east_asian_font(run)
    run.bold = True
    run.font.color.rgb = RGBColor(31, 77, 120)
    return p


def _math_run(text):
    run = OxmlElement("m:r")
    text_node = OxmlElement("m:t")
    text_node.text = text
    run.append(text_node)
    return run


def _math_script(base, sub=None, sup=None):
    if sub is not None and sup is not None:
        node = OxmlElement("m:sSubSup")
        base_node = OxmlElement("m:e")
        base_node.append(_math_run(base))
        sub_node = OxmlElement("m:sub")
        sub_node.append(_math_run(sub))
        sup_node = OxmlElement("m:sup")
        sup_node.append(_math_run(sup))
        node.extend([base_node, sub_node, sup_node])
        return node
    if sub is not None:
        node = OxmlElement("m:sSub")
        base_node = OxmlElement("m:e")
        base_node.append(_math_run(base))
        sub_node = OxmlElement("m:sub")
        sub_node.append(_math_run(sub))
        node.extend([base_node, sub_node])
        return node
    if sup is not None:
        node = OxmlElement("m:sSup")
        base_node = OxmlElement("m:e")
        base_node.append(_math_run(base))
        sup_node = OxmlElement("m:sup")
        sup_node.append(_math_run(sup))
        node.extend([base_node, sup_node])
        return node
    return _math_run(base)


SCRIPT_TOKEN = re.compile(
    r"(?P<base>[A-Za-zΑ-Ωα-ω]+)_(?:\((?P<sub_paren>[^)]+)\)|(?P<sub_plain>[A-Za-z0-9]+))(?P<sup>\^[0-9]+)?"
    r"|(?P<base_sup>[A-Za-zΑ-Ωα-ω]+)\^(?P<sup_only>[0-9]+)"
)


def add_equation(doc, equation):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    math = OxmlElement("m:oMath")
    position = 0
    for match in SCRIPT_TOKEN.finditer(equation):
        if match.start() > position:
            math.append(_math_run(equation[position:match.start()]))
        if match.group("base") is not None:
            sub = match.group("sub_paren") or match.group("sub_plain")
            sup = match.group("sup")
            math.append(_math_script(match.group("base"), sub=sub, sup=sup[1:] if sup else None))
        else:
            math.append(_math_script(match.group("base_sup"), sup=match.group("sup_only")))
        position = match.end()
    if position < len(equation):
        math.append(_math_run(equation[position:]))
    p._p.append(math)
    return p


def add_footer(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("PTDL项目 2.2.1 交付1至交付3公式项与系数释义")
    set_east_asian_font(run)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(100, 100, 100)


doc = Document()
set_page(doc)
set_styles(doc)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(4)
run = title.add_run("PTDL 2.2.1 交付1至交付3公式项与系数释义")
set_east_asian_font(run, "黑体")
run.bold = True
run.font.size = Pt(20)
run.font.color.rgb = RGBColor(11, 37, 69)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(14)
run = subtitle.add_run("多目标性能指标体系、约束条件集、归一化与权重扫描")
set_east_asian_font(run)
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(85, 85, 85)

add_note(doc, "使用说明：本文件只解释2.2.1交付1至交付3中的公式、变量、系数和取值原则。80小时为项目已明确的硬门槛；其余具体飞机、推进和电池边界必须以真实模型、规格书或试验数据为准。文中给出的权重、允许误差和SOC数值仅作为算法原型的初始设置，不应直接写入最终设计指标。")

add_heading(doc, "一、符号、单位与取值状态", 1)
add_para(doc, "所有目标按最小化处理，所有约束采用g(z) <= 0形式。J_E的单位为MJ，J_stab为无量纲控制品质指标，J_resp单位为s，J_SOC,res为无量纲SOC偏差。时间t采用s作为计算单位，报告中的80 h在程序中转换为288000 s；功率采用kW，功率积分后除以1000换算为MJ。")
add_para(doc, "取值状态分为三类。第一类是已确定项目要求：T_endurance >= 80 h。第二类是算法初始值：控制品质权重w_h = 0.30、w_V = 0.30、w_theta = 0.20、w_q = 0.20，峰值惩罚gamma_peak = 0.50，beta_rec = 1.0，beta_switch = 1.0，归一化epsilon = 1e-12。第三类是待模型确认参数：LHV_fuel、eta_dis、eta_chg、SOC_res、SOC_safe、功率边界、飞行包线和允许跟踪误差。")

add_heading(doc, "二、交付一：多目标性能指标的公式项与系数", 1)
add_heading(doc, "2.1 等效任务能耗J_E", 2)
add_equation(doc, "J_E = m_(fuel,used) · LHV_(fuel) + ΔE_(bat,net)")
add_equation(doc, "ΔE_(bat,net) = ∫ [ P_(bat,dis) / η_(dis) - η_(chg) · P_(bat,chg) ] dt")
add_para(doc, "m_fuel,used表示全任务燃油消耗质量，单位kg，由发动机燃油流量积分得到。LHV_fuel表示燃油低位发热值，单位MJ/kg，负责把燃油质量转换为可比较的化学能；其取值必须按实际燃油种类确定，不能由优化算法任意调节。Delta_E_bat,net表示电池净化学能消耗，单位MJ。")
add_para(doc, "P_bat,dis和P_bat,chg分别为非负的放电功率和充电功率，单位kW。eta_dis为放电效率，位于0到1之间；在放电项中位于分母，效率越低，同样的输出电功率需要消耗更多电池化学能。eta_chg为充电效率，位于充电项的乘数位置；效率越低，回充到电池的有效化学能越少。二者应由电池电流、温度和SOC相关效率图给出；算法原型可用0.95作为演示值，但正式模型必须替换为真实数据或分段拟合。")
add_para(doc, "该目标中不再额外设置燃油与电池的经验交换系数。这样可以避免把主观任务偏好误写成物理能量守恒。若项目需要强调任务末端电量，应通过SOC约束或J_SOC,res表达，而不是修改LHV_fuel或效率参数。")

add_heading(doc, "2.2 飞行控制品质J_stab", 2)
add_equation(doc, "J_(stab) = (1 / T) ∫ [ w_h e_(h,norm)^2 + w_V e_(V,norm)^2 + w_θ e_(θ,norm)^2 + w_q q_(norm)^2 ] dt + γ_(peak) · J_(peak)")
add_equation(doc, "e_(h,norm) = e_h / e_(h,allow),    e_(V,norm) = e_V / e_(V,allow)")
add_equation(doc, "e_(θ,norm) = e_θ / e_(θ,allow),    q_(norm) = q / q_(allow)")
add_para(doc, "T为评价区间时长，除以T的作用是消除任务长度对积分值的直接影响，使不同阶段或不同任务时长的控制品质可以比较。e_h、e_V、e_theta和q分别为高度误差、速度误差、俯仰角误差和俯仰角速度；allow后缀表示相应允许误差。原型程序的初始允许值为30 m、3 m/s、3 deg和5 deg/s，作用是尺度归一化，不等于最终飞行安全边界。正式值应由任务精度、飞行品质要求和控制律能力共同确定。")
add_para(doc, "w_h、w_V、w_theta和w_q为四类误差的相对重要性系数，要求非负且建议归一化为和等于1。初始设置为0.30、0.30、0.20、0.20，表示超长航时任务中高度和速度保持对能耗与任务稳定性的影响略高于短时俯仰动态；若任务包含强机动、抗风或突防阶段，可提高w_theta和w_q，并在任务阶段之间采用不同权重。")
add_para(doc, "J_peak为归一化高度、速度和姿态误差的最大值，用于防止平均误差较小却存在瞬态严重超限。gamma_peak为峰值惩罚系数，初始取0.50，表示峰值风险以中等强度参与控制品质评价。gamma_peak增大时，算法会更偏向抑制突变；取值过大可能使策略过度保守，因此应通过标称/扰动场景扫描确定。绝对失速、姿态、载荷等安全边界仍由硬约束保证，不能只依赖J_peak。")

add_heading(doc, "2.3 任务响应J_resp与SOC储备J_SOC,res", 2)
add_equation(doc, "J_(resp) = t_(settle) + β_(rec) · t_(recovery) + β_(switch) · t_(switch)")
add_equation(doc, "J_(SOC,res) = | SOC(t_f) - SOC_(ref,t_f) |")
add_para(doc, "t_settle表示任务指令变化后的稳定时间，t_recovery表示风扰、模型偏差或任务扰动后的恢复时间，t_switch表示发动机/电机动力模式切换的完成时间，单位均为s。beta_rec和beta_switch用于表达恢复与切换相对普通稳态响应的重要程度。原型的无偏好设置为beta_rec = beta_switch = 1.0；若任务更重视抗扰，可令beta_rec大于1；若任务频繁切换动力模式或重视低可探测窗口，可令beta_switch大于1。")
add_para(doc, "J_SOC,res衡量终端SOC相对参考储备的偏差，范围通常在0到1之间。SOC_ref,t_f不是任意调参量，而是由返航、应急机动、低可探测飞行和电池安全所需的末端储备确定。对耗电型80小时任务，SOC储备通常先作为硬约束；仅在需要比较多种储备策略时才作为第四个Pareto目标。")

add_heading(doc, "三、交付二：约束条件的公式项与边界含义", 1)
add_heading(doc, "3.1 任务与80小时续航约束", 2)
add_equation(doc, "g_(80) = 80 h - T_(endurance) ≤ 0")
add_equation(doc, "g_(range) = S_(req) - s(t_f) ≤ 0,    g_(loiter) = T_(loiter,min) - T_(loiter) ≤ 0")
add_equation(doc, "g_(position) = e_(position)(t_f) - e_(position,max) ≤ 0")
add_para(doc, "g_80是第一优先级约束。其数值小于或等于0表示续航达到或超过80小时，正值表示未达标小时数。T_endurance必须在同一任务剖面、同一有效载荷功率、同一环境假设和同一终端储备下计算。S_req、T_loiter,min和e_position,max分别由任务需求方确定，用于防止算法通过缩短航程、压缩任务区驻留或放宽终端精度换取表面续航。")

add_heading(doc, "3.2 能量、推进与电池约束", 2)
add_equation(doc, "g_(SOC,end) = SOC_(res) - SOC(t_f) ≤ 0,    g_(SOC,min) = SOC_(safe) - min[ SOC(t) ] ≤ 0")
add_equation(doc, "g_(fuel) = m_(res) - m_(fuel)(t_f) ≤ 0")
add_equation(doc, "0 ≤ P_(eng)(t) ≤ P_(eng,max),    P_(motor,min) ≤ P_(motor)(t) ≤ P_(motor,max)")
add_equation(doc, "| dP_(eng) / dt | ≤ r_(eng,max),    | dP_(motor) / dt | ≤ r_(motor,max)")
add_equation(doc, "P_(bat,dis) ≤ P_(bat,dis,max),    P_(bat,chg) ≤ P_(bat,chg,max),    | I_(bat) | ≤ I_(bat,max)")
add_equation(doc, "T_(bat,min) ≤ T_(bat) ≤ T_(bat,max)")
add_para(doc, "SOC_res为任务结束时的最低储备，原型默认0.20；SOC_safe为飞行全程允许的最低SOC，原型默认0.15。二者仅用于程序接口验证，正式数值需要依据电池放电曲线、内阻上升、末端机动功率需求和寿命要求确认。m_res为燃油末端储备，若任务采用燃油完全耗尽的理论续航分析可暂不启用，但正式任务方案建议保留。")
add_para(doc, "P_eng,max、P_motor,min/max、P_bat,dis/max、P_bat,chg/max由推进系统和电池能力边界给出；r_eng,max和r_motor,max描述可实现的功率爬升/下降速率，防止算法使用物理上不可达到的瞬态功率。I_bat,max和电池温度范围分别来源于电池规格书、热模型或试验；这些参数接入后才能对高倍率放电和热风险作出可信判断。")

add_heading(doc, "3.3 飞行包线与控制边界", 2)
add_equation(doc, "V_(min) ≤ V(t) ≤ V_(max),    h_(min) ≤ h(t) ≤ h_(max)")
add_equation(doc, "α_(min) ≤ α(t) ≤ α_(max),    n(t) ≤ n_(max)")
add_para(doc, "V_min由失速速度、阵风裕度和控制能力共同确定，V_max由气动/结构限制、推进可用功率和任务要求共同确定。h_min/max由起降、地形、通信/侦察覆盖和发动机/螺旋桨高空性能确定。alpha边界来自气动极线和失速裕度，n_max来自结构载荷限制。上述边界均为硬约束，不能通过提高J_stab权重替代。")

add_heading(doc, "四、交付三：归一化与权重扫描的系数含义", 1)
add_heading(doc, "4.1 固定参考归一化", 2)
add_equation(doc, "f_(i,norm) = [ f_i - f_(i,min) ] / [ f_(i,ref) - f_(i,min) + ε ]")
add_para(doc, "f_i是第i个原始目标值；f_i,min为初始样本或经确认基准策略中的最优参考值；f_i,ref为参考最差值，建议取同一批样本的95%分位数，避免极端异常值主导尺度。epsilon用于防止分母为零，程序中取1e-12，仅承担数值稳定功能，不对物理结果产生可见影响。归一化参考点在同一次算法比较中必须冻结，否则不同代或不同算法之间的距离、权重和Hypervolume将失去可比性。")

add_heading(doc, "4.2 MOEA/D权重与后处理偏好权重", 2)
add_equation(doc, "w_E + w_(stab) + w_(resp) = 1,    w_E, w_(stab), w_(resp) ≥ 0")
add_equation(doc, "N_(weights) = (H + 1)(H + 2) / 2")
add_equation(doc, "S = Σ_i w_i f_(i,norm)")
add_para(doc, "w_E、w_stab和w_resp是目标空间偏好方向系数，不是控制品质中的w_h、w_V、w_theta和w_q。前者用于MOEA/D把多目标问题分解为多个偏好子问题，或用于对Pareto解进行后处理；后者用于构造J_stab内部的误差组成。两类权重必须在程序变量名和论文表述中严格区分。")
add_para(doc, "H为三维单纯形网格的分辨率，N_weights为对应权重向量数。H = 4时，N_weights = 15，适合原型阶段的快速权重扫描；H增大时，权重方向覆盖更细，但MOEA/D子问题数与真实仿真计算量同时增加。建议从H = 4开始，在模型运行稳定后再评估H = 6或更高分辨率。")
add_para(doc, "后处理评分S只用于从已达到80小时、且全部硬约束可行的Pareto解中选择代表方案。能耗优先可取(w_E, w_stab, w_resp) = (0.60, 0.20, 0.20)，稳定优先为(0.20, 0.60, 0.20)，响应优先为(0.20, 0.20, 0.60)，均衡偏好为(1/3, 1/3, 1/3)。这四组仅用于报告展示和敏感性分析，不代表项目最终只选择某一组权重。")

add_heading(doc, "五、参数确认顺序", 1)
add_para(doc, "在进入2.2.2前，应优先确认与80小时可行性直接相关的参数：任务剖面与有效载荷功率、起飞重量与燃油/电池容量、发动机燃油流量图、发动机/电机功率与速率边界、电池可用能量及SOC安全边界。随后确认飞行控制品质参数：速度高度包线、允许跟踪误差、姿态/载荷边界和扰动场景。最后再根据标称与扰动仿真结果调整w_h、w_V、w_theta、w_q、gamma_peak、beta_rec和beta_switch。")
add_para(doc, "因此，本文件中的公式可立即用于搭建算法与评价程序，但系数不应一次性“拍死”。物理参数必须由模型或试验给定；偏好参数必须通过任务场景和敏感性分析校核；80小时续航门槛则始终保持不变。")

add_footer(doc)
doc.core_properties.title = "PTDL 2.2.1 交付1至交付3公式项与系数释义"
doc.core_properties.subject = "多目标性能指标、约束条件、归一化与权重扫描系数说明"
doc.core_properties.author = "Codex"
doc.save(OUT)
print(OUT)
