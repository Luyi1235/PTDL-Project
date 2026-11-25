import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = r"C:\Users\23598\Documents\ChatGPT\LDTP\PTDL_2.2.1多目标性能指标体系与评价函数交付件_OMML公式版.docx"


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


def add_para(doc, text, first_line=True, after=8, code=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.25
    if first_line:
        p.paragraph_format.first_line_indent = Pt(22)
    run = p.add_run(text)
    set_east_asian_font(run, "等线" if code else "宋体")
    if code:
        run.font.size = Pt(9.5)
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.first_line_indent = Pt(0)
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


_SCRIPT_TOKEN = re.compile(
    r"(?P<base>[A-Za-zΑ-Ωα-ω]+)_(?:\((?P<sub_paren>[^)]+)\)|(?P<sub_plain>[A-Za-z0-9]+))(?P<sup>\^[0-9]+)?"
    r"|(?P<base_sup>[A-Za-zΑ-Ωα-ω]+)\^(?P<sup_only>[0-9]+)"
)


def add_equation(doc, equation):
    """Insert an editable Office Math (OMML) equation paragraph with scripts."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    math = OxmlElement("m:oMath")
    position = 0
    for match in _SCRIPT_TOKEN.finditer(equation):
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


def add_ref(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.12
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-18)
    run = p.add_run(text)
    set_east_asian_font(run)
    run.font.size = Pt(9.5)


def add_footer(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("PTDL项目 2.2.1 多目标性能指标体系与评价函数交付件")
    set_east_asian_font(run)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(100, 100, 100)


doc = Document()
set_page(doc)
set_styles(doc)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(4)
run = title.add_run("PTDL 2.2.1 多目标性能指标体系与评价函数交付件")
set_east_asian_font(run, "黑体")
run.bold = True
run.font.size = Pt(20)
run.font.color.rgb = RGBColor(11, 37, 69)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(14)
run = subtitle.add_run("面向混合动力超长航时无人机飞推控一体化控制")
set_east_asian_font(run)
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(85, 85, 85)

add_note(doc, "交付定位：本文件直接给出2.2.1可供算法开发调用的指标定义、约束条件、归一化和权重扫描规则、评价函数输入输出规范。80小时续航是正式Pareto结果的硬可行性门槛，而不是可由其他指标补偿的普通优化目标。")

add_heading(doc, "一、交付范围与统一约定", 1)
add_para(doc, "本交付件以统一任务剖面M、飞机/推进/电池参数Theta、环境场景E和策略参数z为输入，输出目标向量F(z)、约束违反度G(z)、80小时达标标志及诊断指标。评价函数只承担“统一度量”职责，不替代飞行动力学、推进系统或能量管理模型。上游模型可以是Python简化模型、RCAIDE任务分析或后续飞推控耦合模型；只要能输出规定的任务时序数据，即可调用同一评价函数。")
add_para(doc, "统一约定如下：全部优化目标均按最小化处理；约束写为g_j(z) <= 0；正式候选解必须同时满足全部硬约束；未达到80小时的策略只能用于前期可行性攻关和瓶颈诊断，不进入正式Pareto集。燃油低位发热值暂采用LHV_fuel表示，实际数值由所选发动机燃油类型确认后写入配置。")

add_heading(doc, "二、交付一：多目标性能指标体系", 1)
add_para(doc, "指标体系采用“六类候选评价维度 + 明确优化角色”的结构。续航/任务完成、燃油与等效能耗、SOC管理、飞行性能与气动效率、飞行稳定性与控制品质、任务响应速度与动力可用性共同构成完整评价框架，但不等同于六目标同时优化。基准Pareto目标仅为J_E、J_stab和J_resp；SOC储备在需要比较能量策略时升为可选第四目标J_SOC,res；其他维度作为任务约束或诊断输出。")
add_para(doc, "等效任务能耗由燃油化学能消耗与电池净化学能消耗共同构成：")
add_equation(doc, "J_E = m_(fuel,used) · LHV_(fuel) + ΔE_(bat,net)")
add_equation(doc, "ΔE_(bat,net) = ∫ [ P_(bat,dis) / η_(dis) - η_(chg) · P_(bat,chg) ] dt")
add_para(doc, "其中m_fuel,used由燃油流量积分得到。该定义使燃油消耗与电池使用深度在同一能量单位下比较，同时分别保留燃油质量和电池净能量作为诊断输出。")
add_para(doc, "飞行控制品质由全程归一化跟踪误差和峰值风险共同度量：")
add_equation(doc, "J_(stab) = (1 / T) ∫ [ w_h e_(h,norm)^2 + w_V e_(V,norm)^2 + w_θ e_(θ,norm)^2 + w_q q_(norm)^2 ] dt + γ_(peak) · J_(peak)")
add_para(doc, "高度、速度、俯仰角和俯仰角速度误差分别用允许误差归一化；J_peak为各类归一化误差的最大值。该目标同时反映全程平均跟踪品质与瞬态峰值风险。")
add_para(doc, "任务响应由任务指令变化、扰动恢复和动力模式切换的时间代价构成：")
add_equation(doc, "J_(resp) = t_(settle) + β_(rec) · t_(recovery) + β_(switch) · t_(switch)")
add_para(doc, "t_settle用于任务指令变化后的稳定时间，t_recovery用于扰动后的恢复时间，t_switch用于发动机/电机模式切换的完成时间。上述时间由任务阶段或扰动事件识别模块提供，评价函数以统一规则汇总。")
add_para(doc, "SOC储备的可选目标和耗电型任务的硬约束分别为：")
add_equation(doc, "J_(SOC,res) = | SOC(t_f) - SOC_(ref,t_f) |")
add_equation(doc, "SOC(t_f) ≥ SOC_(res),    SOC(t) ≥ SOC_(safe)")
add_para(doc, "必要时可叠加关键机动节点的SOC偏差。对典型耗电型超长航时任务，J_SOC,res不进入基准目标向量，以避免与J_E重复计量。")

add_heading(doc, "三、交付二：约束条件集", 1)
add_para(doc, "任务可行性约束包括80小时续航、航程、任务区驻留和终端位置误差：")
add_equation(doc, "g_(80) = 80 h - T_(endurance) ≤ 0")
add_equation(doc, "g_(range) = S_(req) - s(t_f) ≤ 0,    g_(loiter) = T_(loiter,min) - T_(loiter) ≤ 0")
add_equation(doc, "g_(position) = e_(position)(t_f) - e_(position,max) ≤ 0")
add_para(doc, "80小时约束应在统一任务剖面、有效载荷功率、环境假设和终端能量储备下计算，不能通过减少任务段、降低载荷功率或取消返航储备规避。")
add_para(doc, "能量与推进约束包括终端和全程能量储备、发动机/电机功率及其变化率：")
add_equation(doc, "g_(SOC,end) = SOC_(res) - SOC(t_f) ≤ 0,    g_(SOC,min) = SOC_(safe) - min[ SOC(t) ] ≤ 0")
add_equation(doc, "g_(fuel) = m_(res) - m_(fuel)(t_f) ≤ 0")
add_equation(doc, "0 ≤ P_(eng)(t) ≤ P_(eng,max),    P_(motor,min) ≤ P_(motor)(t) ≤ P_(motor,max)")
add_equation(doc, "| dP_(eng) / dt | ≤ r_(eng,max),    | dP_(motor) / dt | ≤ r_(motor,max)")
add_equation(doc, "P_(bat,dis) ≤ P_(bat,dis,max),    P_(bat,chg) ≤ P_(bat,chg,max),    | I_(bat) | ≤ I_(bat,max)")
add_equation(doc, "T_(bat,min) ≤ T_(bat) ≤ T_(bat,max)")
add_para(doc, "飞行与控制约束为：")
add_equation(doc, "V_(min) ≤ V(t) ≤ V_(max),    h_(min) ≤ h(t) ≤ h_(max)")
add_equation(doc, "α_(min) ≤ α(t) ≤ α_(max),    n(t) ≤ n_(max)")
add_para(doc, "安全边界必须保持硬约束属性；控制平滑性在早期可通过功率变化率和执行机构速率限制实现，而不与J_E、J_stab、J_resp并列增加目标维度。")
add_para(doc, "实际数值边界不在本交付件中虚构指定。它们应由飞机总体设计、气动数据、推进工作图、电池规格书和控制律设计参数填入Python配置对象。评价函数已将这些边界参数化：未配置的边界不计算相应约束，已配置的边界自动输出g_j(z)及约束裕度。")

add_heading(doc, "四、交付三：归一化方法与权重扫描规则", 1)
add_para(doc, "采用固定参考归一化，而不在每一代算法中随意改变尺度。先用拉丁超立方、Sobol样本或基准控制策略在同一任务场景下得到初始评价样本；对第i个最小化目标，取理想值f_i,min为样本最小值，取参考最差值f_i,ref为样本95%分位数或经工程确认的最差可接受值：")
add_equation(doc, "f_(i,norm) = [ f_i - f_(i,min) ] / [ f_(i,ref) - f_(i,min) + ε ]")
add_para(doc, "归一化参考点在一次算法比较中固定，只有模型版本或任务定义改变时才重新建立。")
add_para(doc, "对于基准三目标问题，权重扫描采用三维单纯形网格。分辨率H下，全部非负权重组合应满足：")
add_equation(doc, "w_E + w_(stab) + w_(resp) = 1,    w_E, w_(stab), w_(resp) ≥ 0")
add_para(doc, "例如H=4可产生15组均匀权重，供MOEA/D子问题或偏好敏感性分析使用。建议至少保留能耗优先、稳定优先、响应优先和均衡偏好四类代表方向。权重不用于把所有目标压缩为唯一结论，正式结果仍以80小时达标解之间的非支配关系为依据。")
add_para(doc, "Pareto解后处理可使用偏好加权评分选择代表策略：")
add_equation(doc, "S = Σ_i w_i f_(i,norm)")
add_para(doc, "该评分仅用于从已获得的可行Pareto集中挑选“节能优先”“稳定优先”“响应优先”和“综合折中”方案。任何不满足硬约束的解均不参与权重评分和正式比较。")

add_heading(doc, "五、交付四：评价函数与程序接口", 1)
add_para(doc, "随本交付件提供的ptdl_2_2_1_evaluator.py已实现评价层。其核心入口为evaluate_mission(profile, config, response_metrics, terminal_fuel_kg, range_m, loiter_s, terminal_position_error_m)。其中profile为按时间排序的pandas.DataFrame，至少包含time_s、fuel_flow_kgps、battery_discharge_kw、battery_charge_kw、soc、h_m、v_mps、p_engine_kw、p_motor_kw、e_h_m、e_v_mps、e_theta_rad和q_radps；可选列包括alpha_rad、load_factor、battery_current_a、battery_temperature_c。")
add_para(doc, "函数输出EvaluationResult对象，包含三个部分：objectives返回J_energy_mj、J_stability和J_response_s；constraints返回每个g_j(z)的数值；diagnostics返回续航裕度、燃油消耗、电池净能量消耗、终端和最低SOC、功率峰值及跟踪误差峰值。feasible属性由全部g_j(z) <= 0自动判定，total_violation提供不可行策略的总违反度，可直接供NSGA-II和MOEA/D实施可行性优先约束处理。")
add_para(doc, "程序还提供ObjectiveNormalizer.from_samples，用于从初始样本建立固定参考归一化；generate_simplex_weights(resolution)，用于生成MOEA/D三目标权重扫描；weighted_preference_score，用于对已达标Pareto解进行后处理偏好选择。NumPy承担数值积分与数组计算，pandas承担时序数据和结果记录；Matplotlib或Plotly可在算法主程序中读取结果记录，绘制SOC轨迹、功率分配、约束裕度和Pareto前沿。")
add_para(doc, "程序内的demo_configuration仅用于单元测试，绝不是PTDL的设计参数。接入真实模型时，应由动能学院和总体/控制方向提供实际任务剖面、P_eng/P_motor边界、发动机燃油流量图、电池效率/热边界、速度高度包线及允许跟踪误差，然后填入EvaluationConfig。")

add_heading(doc, "六、交付五：阶段验收结果与使用方式", 1)
add_para(doc, "本阶段的可交付成果现已明确为：一套可计算的多目标指标体系；一套以80小时为硬门槛的约束条件集；一套固定参考归一化和三目标单纯形权重扫描规则；以及一个可被优化算法直接调用的Python评价函数模块。该模块已包含80小时达标标志、目标值、约束违反度和诊断量的标准输出形式。")
add_para(doc, "在当前尚未获得真实飞推控耦合参数的阶段，评价函数可用简化任务时序数据跑通接口、约束处理和图形化流程；待模型对接后，只需将真实仿真输出映射到profile字段并填入实际边界，即可进入2.2.2的NSGA-II、MOEA/D、代理模型与直接配点优化。验收时应至少给出一组标称任务结果，证明代码能够正确识别80小时达标与未达标策略，并输出目标向量、约束裕度、SOC轨迹和功率分配结果。")

add_heading(doc, "参考文献", 1)
for ref in [
    "[1] Miettinen, K. Nonlinear Multiobjective Optimization. Kluwer Academic Publishers, 1999. DOI: 10.1007/978-1-4615-5563-6.",
    "[2] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. A fast and elitist multi-objective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197, 2002. DOI: 10.1109/4235.996017.",
    "[3] Zhang, Q., and Li, H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 712-731, 2007. DOI: 10.1109/TEVC.2007.892759.",
    "[4] Deb, K. An efficient constraint handling method for genetic algorithms. Computer Methods in Applied Mechanics and Engineering, 186(2-4), 311-338, 2000. DOI: 10.1016/S0045-7825(99)00389-8.",
    "[5] Stevens, B. L., Lewis, F. L., and Johnson, E. N. Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems. Wiley, 3rd ed., 2015.",
]:
    add_ref(doc, ref)

add_footer(doc)
doc.core_properties.title = "PTDL 2.2.1 多目标性能指标体系与评价函数交付件"
doc.core_properties.subject = "多目标性能指标体系、约束条件、归一化、权重扫描与Python评价函数"
doc.core_properties.author = "Codex"
doc.save(OUT)
print(OUT)
