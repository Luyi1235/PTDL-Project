"""Build the PTDL 2.2.2 delivery report from the executed prototype outputs."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "PTDL_2.2.2算法交付结果"
OUTPUT = ROOT / "PTDL_2.2.2多目标优化控制算法开发交付件.docx"

BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
INK = RGBColor(11, 37, 69)
MUTED = RGBColor(89, 99, 112)


def set_run_font(run, name: str = "Calibri", size: float | None = None, color=None, bold=None, italic=None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(*_args, **_kwargs) -> None:
    """Kept intentionally unused: this deliverable contains no tables."""


def add_math_paragraph(doc: Document, expression: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(6)
    math = OxmlElement("m:oMath")
    run = OxmlElement("m:r")
    text = OxmlElement("m:t")
    text.text = expression
    run.append(text)
    math.append(run)
    paragraph._p.append(math)


def add_body(doc: Document, text: str, bold_lead: str | None = None) -> None:
    paragraph = doc.add_paragraph(style="Body")
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, size=11, color=INK, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest, size=11, color=INK)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, size=11, color=INK)


def add_heading(doc: Document, text: str, level: int) -> None:
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.add_run(text)


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    set_run_font(run, size=9.5, color=MUTED, italic=True)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)

    body = styles.add_style("Body", WD_STYLE_TYPE.PARAGRAPH)
    body.font.name = "Calibri"
    body._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    body.font.size = Pt(11)
    body.font.color.rgb = INK
    body.paragraph_format.space_before = Pt(0)
    body.paragraph_format.space_after = Pt(6)
    body.paragraph_format.line_spacing = 1.10

    for level, size, color, before, after in (
        (1, 16, BLUE, 16, 8),
        (2, 13, BLUE, 12, 6),
        (3, 12, DARK_BLUE, 8, 4),
    ):
        style = styles[f"Heading {level}"]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    run = header.add_run("PTDL 2.2.2 | 多目标优化控制算法开发交付件")
    set_run_font(run, size=8.5, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("PTDL 项目 | 卢奕主责方向 | 算法原型交付")
    set_run_font(run, size=8.5, color=MUTED)


def metric(row: pd.Series, key: str, digits: int = 2) -> str:
    return f"{float(row[key]):.{digits}f}"


def main() -> None:
    comparison = pd.read_csv(RESULTS / "algorithm_comparison.csv")
    representatives = pd.read_csv(RESULTS / "representative_solutions.csv")
    nsga = comparison.loc[comparison["algorithm"] == "NSGA-II"].iloc[0]
    moead = comparison.loc[comparison["algorithm"] == "MOEA/D"].iloc[0]
    energy = representatives.loc[representatives["preference"] == "energy_priority"].iloc[0]
    stability = representatives.loc[representatives["preference"] == "stability_priority"].iloc[0]
    response = representatives.loc[representatives["preference"] == "response_priority"].iloc[0]

    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(18)
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("PTDL 2.2.2 多目标优化控制算法开发交付件")
    set_run_font(run, size=23, color=INK, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(14)
    run = subtitle.add_run("约束 NSGA-II 基准算法、MOEA/D 对比算法与任务剖面控制策略原型")
    set_run_font(run, size=13, color=MUTED)
    metadata = doc.add_paragraph()
    metadata.paragraph_format.space_after = Pt(14)
    run = metadata.add_run("项目：PTDL 混合动力超长航时无人机飞推控耦合优化设计 | 负责人：卢奕 | 版本：算法原型交付版")
    set_run_font(run, size=10, color=MUTED)

    add_heading(doc, "一、交付结论与适用边界", 1)
    add_body(doc, "本交付件在已完成的 2.2.1 指标体系、约束集与评价器基础上，形成了可直接运行的多目标优化控制原型。原型以约束 NSGA-II 作为基准算法，以 MOEA/D 作为分解式对比算法，统一调用 ptdl_2_2_1_evaluator.py 计算等效能耗、飞行稳定性、任务响应时间和全部约束违反度；程序可自动导出收敛曲线、Pareto 候选、代表控制量序列、状态轨迹及算法比较结果。")
    add_body(doc, "80 小时是本项目的刚性可行性门槛，而不是可与能耗、稳定性和响应速度相互交换的 Pareto 目标。仅当候选策略满足续航、SOC、速度高度包线、功率边界和功率变化率等约束后，才进入正式的 Pareto 比较与偏好筛选。")
    add_math_paragraph(doc, "min F_base(z) = [J_E(z), J_stab(z), J_resp(z)] ;  g_80(z) = 80 h - T_endurance(z) <= 0")
    add_body(doc, "适用边界：本次交付的数值算例使用低阶分段任务代理模型，目的仅是验证算法、评价器、约束处理和结果导出链路，不能作为 PTDL 飞行性能、推进效率、燃油消耗或电池容量的设计预测。后续接入 RCAIDE、动能学院推进系统模型或飞推控联合仿真后，应以真实模型重新运行本算法框架。", "适用边界：")

    add_heading(doc, "二、算法选型与PTDL适配方案", 1)
    add_heading(doc, "2.1 算法体系", 2)
    add_body(doc, "NSGA-II 被确定为主基准算法。它通过非支配排序、拥挤距离和精英保留生成分布较均匀的 Pareto 解集，不需要在搜索开始前把能耗、稳定性和响应速度压缩为单一主观权重，适合形成 PTDL 的首版标准前沿和可解释候选策略库。")
    add_body(doc, "MOEA/D 被确定为对比增强算法。它将多目标问题按权重向量分解为一组标量子问题，适合检验不同任务偏好方向的覆盖能力，并为后续扩展至四目标或更多目标提供算法基础。GA 与 PSO 保留为低维策略参数的基线或局部调参工具，不作为当前交付版的最终主算法。")
    add_body(doc, "代理模型与连续最优控制采用分层接入：当前低阶任务代理模型用于验证算法流程；真实高成本仿真接入后，使用 Kriging/高斯过程模型筛选值得复算的候选解；从 Pareto 前沿选出的代表策略再通过 CasADi 直接配点进行连续控制轨迹精修。此分工避免把高维全时间节点控制序列直接交给进化算法造成维数膨胀。")

    add_heading(doc, "2.2 决策变量、评价接口与约束处理", 2)
    add_body(doc, "当前原型采用 10 维工程可解释决策变量：任务持续时间、巡航速度、巡航高度、爬升/巡航/返航三个阶段的发动机功率、对应阶段的电机辅助功率，以及控制增益。阶段功率在任务时间轴上插值得到连续控制序列，满足“阶段参数加关键节点控制量”的紧凑编码原则。")
    add_body(doc, "每个候选策略经分段任务模型生成按时间排序的 profile，并进入 2.2.1 的 evaluate_mission 接口。接口输出 J_energy_mj、J_stability、J_response_s、各项 g_j(z)、续航裕度、SOC、功率峰值和跟踪误差峰值。NSGA-II 与 MOEA/D 均采用 Deb 可行性规则：可行解优先于不可行解；均不可行时总违反度更小者优先；均可行时再按 Pareto 支配或分解标量值比较。")
    add_body(doc, "为避免未接入发动机发电燃油图时出现“负电机功率产生无成本充电”的伪物理现象，本代理算例仅优化电机辅助功率，不优化充电模式；真实模型接入后，应将发动机发电、充电效率、电池热状态与燃油消耗共同纳入能量平衡，再恢复完整的充放电策略搜索。")

    add_heading(doc, "三、可运行算法与参数配置", 1)
    add_heading(doc, "3.1 NSGA-II基准配置", 2)
    add_body(doc, "正式算例中，NSGA-II 种群规模设为 36，迭代 35 代，随机种子为 20260831，合计执行 1296 次任务评价。交叉采用有界模拟二进制交叉，分布指数 eta 为 15；逐变量变异概率为 1/10，变异标准差为变量取值范围的 8%，所有变量均在工程边界内截断。该配置适用于当前 10 维低阶策略空间，后续接入高保真模型后应根据单次仿真时间调整评价次数预算。")
    add_heading(doc, "3.2 MOEA/D对比配置", 2)
    add_body(doc, "MOEA/D 使用三目标权重单纯形分辨率 H=5，因此生成 21 个权重子问题；每个子问题保留 5 个最近邻权重方向，迭代 60 代，随机种子为 20260832，合计执行 1281 次任务评价。分解函数采用归一标度下的 Tchebycheff 形式，理想点仅由可行候选的目标值更新。该评价次数与 NSGA-II 接近，便于开展公平的原型级比较。")
    add_math_paragraph(doc, "N_weights = (H + 1)(H + 2) / 2 = 21 ;  g_te(x|lambda,z*) = max_i { lambda_i |(f_i(x)-z_i*)/s_i| }")
    add_heading(doc, "3.3 结果文件与运行方式", 2)
    add_body(doc, "主程序为 ptdl_2_2_2_optimizer.py，执行后自动生成 nsga_ii_convergence.csv、moea_d_convergence.csv、两类 Pareto 候选文件、algorithm_comparison.csv、representative_solutions.csv、candidate_control_sequences.csv、candidate_state_trajectories.csv、PNG/SVG 收敛曲线和运行元数据文件。测试程序为 test_ptdl_2_2_2_optimizer.py，使用 Python 标准库 unittest，无需额外安装 pytest。")
    add_body(doc, "正式运行命令为：python ptdl_2_2_2_optimizer.py --output-dir PTDL_2.2.2算法交付结果。快速回归验证命令为：python ptdl_2_2_2_optimizer.py --quick --output-dir ptdl_2_2_2_quick_results。")

    add_heading(doc, "四、正式原型算例与算法比较结果", 1)
    add_body(doc, f"本次正式算例在相同的变量边界、80 小时约束、评价器和低阶任务代理模型下运行。NSGA-II 最终种群中可行解为 {int(nsga['feasible_count'])}/{int(nsga['population_size'])}，可行 Pareto 解为 {int(nsga['pareto_count'])} 个；MOEA/D 最终种群中可行解为 {int(moead['feasible_count'])}/{int(moead['population_size'])}，可行 Pareto 解为 {int(moead['pareto_count'])} 个。两类算法的最小总约束违反度均为 0，说明在该原型算例中均已越过 80 小时和运行边界构成的可行性门槛。")
    add_body(doc, f"NSGA-II 在最终解集中得到的最小等效能耗为 {metric(nsga, 'best_J_energy_mj')} MJ，最小稳定性指标为 {metric(nsga, 'best_J_stability', 4)}，最小响应时间为 {metric(nsga, 'best_J_response_s')} s；MOEA/D 的对应数值为 {metric(moead, 'best_J_energy_mj')} MJ、{metric(moead, 'best_J_stability', 4)} 和 {metric(moead, 'best_J_response_s')} s。这些数值仅用于说明算法对同一代理算例的搜索表现，不可外推为 PTDL 的真实任务能耗或响应能力。")
    image_path = RESULTS / "convergence_feasibility.png"
    doc.add_picture(str(image_path), width=Inches(6.25))
    add_caption(doc, "图 1  NSGA-II 与 MOEA/D 在代理算例中的可行解比例收敛曲线。曲线数据同时以 CSV 文件交付。")
    add_body(doc, "本次单次固定随机种子算例显示，两种算法均能稳定保留可行解并输出 Pareto 集。尚不能据此宣称任一算法在 PTDL 真实问题上绝对更优；真实模型接入后应采用多个随机种子、相同真实仿真预算，并以可行解比例、Hypervolume、IGD+、达到给定 Hypervolume 的真实仿真次数和控制轨迹可执行性进行统计比较。")

    add_heading(doc, "五、代表控制策略与输出解释", 1)
    add_body(doc, f"能耗偏好代表解来自 {energy['algorithm']}，任务时长为 {metric(energy, 'duration_h')} h，发动机功率在爬升/巡航/返航阶段分别为 {metric(energy, 'engine_power_climb_kw')}、{metric(energy, 'engine_power_cruise_kw')}、{metric(energy, 'engine_power_return_kw')} kW，电机辅助功率分别为 {metric(energy, 'motor_power_climb_kw')}、{metric(energy, 'motor_power_cruise_kw')}、{metric(energy, 'motor_power_return_kw')} kW，控制增益为 {metric(energy, 'control_gain', 3)}；其三个目标值为 {metric(energy, 'J_energy_mj')} MJ、{metric(energy, 'J_stability', 4)} 和 {metric(energy, 'J_response_s')} s。该解按 0.60/0.20/0.20 的偏好权重从可行 Pareto 集中筛选，不等同于单独最小能耗解。")
    add_body(doc, f"稳定性偏好代表解来自 {stability['algorithm']}，任务时长为 {metric(stability, 'duration_h')} h，三个阶段发动机功率为 {metric(stability, 'engine_power_climb_kw')}、{metric(stability, 'engine_power_cruise_kw')}、{metric(stability, 'engine_power_return_kw')} kW，电机辅助功率为 {metric(stability, 'motor_power_climb_kw')}、{metric(stability, 'motor_power_cruise_kw')}、{metric(stability, 'motor_power_return_kw')} kW，控制增益为 {metric(stability, 'control_gain', 3)}；其稳定性指标为 {metric(stability, 'J_stability', 4)}。")
    add_body(doc, f"响应偏好代表解来自 {response['algorithm']}，任务时长为 {metric(response, 'duration_h')} h，爬升/巡航/返航电机辅助功率为 {metric(response, 'motor_power_climb_kw')}、{metric(response, 'motor_power_cruise_kw')}、{metric(response, 'motor_power_return_kw')} kW，控制增益为 {metric(response, 'control_gain', 3)}；其响应时间为 {metric(response, 'J_response_s')} s。与能耗偏好解相比，该解通过增加过渡阶段电机辅助和控制增益换取更快的响应，同时付出更高的能耗和更大的稳定性代价，体现了本问题的 Pareto 权衡。")
    add_body(doc, "代表解的完整时序数据位于 candidate_control_sequences.csv 与 candidate_state_trajectories.csv。前者可直接绘制发动机功率、电机功率与电池功率轨迹；后者可直接绘制 SOC、高度、速度和跟踪误差轨迹。真实模型接入后，这两个文件将成为与动能学院和总体/控制方向核对策略可执行性的统一交换格式。")

    add_heading(doc, "六、与真实PTDL模型的接入方案", 1)
    add_body(doc, "第一步，保留 2.2.1 的 evaluate_mission 评价接口不变，将当前 simulate_surrogate_mission 替换为 RCAIDE 或飞推控联合仿真适配器；适配器必须返回按时间排序的 profile，以及调节、恢复和模式切换时间等 response_metrics。这样所有目标、约束、归一化、可行性规则和结果导出机制无需重写。")
    add_body(doc, "第二步，由动能学院提供发动机燃油流量图、发动机/电机功率与变化率边界、电池容量与 SOC-效率-温度特性、充放电倍率和热边界；由总体与控制方向提供任务分段、有效载荷功率、速度高度包线、允许跟踪误差、扰动场景及终端状态要求。将这些真实参数填入 EvaluationConfig 和仿真适配器后，重新冻结 2.2.1 的归一化参考值。")
    add_body(doc, "第三步，先以真实模型执行可行性搜索，确保 T_endurance >= 80 h；再开展 NSGA-II 与 MOEA/D 的多随机种子对比；当单次高保真评价成本较高时，引入 Kriging/高斯过程代理模型，仅对预测优良或不确定性较高的候选策略执行真实复算；最后用 CasADi 直接配点对节能优先、稳定性优先、响应优先和综合折中解进行连续轨迹精修。")

    add_heading(doc, "七、测试与验收结果", 1)
    add_body(doc, "已新增 test_ptdl_2_2_2_optimizer.py 并执行 3 项自动化测试，结果全部通过。测试一验证代理模型生成的 profile 与 2.2.1 评价器接口一致，且目标计算有限有效；测试二执行缩小规模的完整优化试验，验证全部结果文件和四类代表策略均能成功导出；测试三验证 NSGA-II 与 MOEA/D 的最终 Pareto 候选均为可行解，且均满足 g_80 <= 0。")
    add_body(doc, "验收结论：2.2.2 已形成算法层面的可运行交付版，包括约束多目标算法、可复现参数配置、收敛曲线、候选控制量序列、状态轨迹、算法比较结果、测试文件和接入真实模型的替换边界。尚未完成的工作不是算法框架，而是使用真实 PTDL 飞推控模型进行物理性能定型与多工况统计验证。")

    add_heading(doc, "参考文献", 1)
    references = [
        "[1] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 2002.",
        "[2] Zhang, Q., and Li, H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 2007.",
        "[3] Deb, K. An efficient constraint handling method for genetic algorithms. Computer Methods in Applied Mechanics and Engineering, 186(2-4), 2000.",
        "[4] Kennedy, J., and Eberhart, R. Particle swarm optimization. Proceedings of ICNN'95, 1995.",
        "[5] Coello Coello, C. A., Pulido, G. T., and Lechuga, M. S. Handling multiple objectives with particle swarm optimization. IEEE Transactions on Evolutionary Computation, 8(3), 2004.",
        "[6] Jones, D. R., Schonlau, M., and Welch, W. J. Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13, 1998.",
        "[7] Betts, J. T. Survey of numerical methods for trajectory optimization. Journal of Guidance, Control, and Dynamics, 21(2), 1998.",
        "[8] Hargraves, C. R., and Paris, S. W. Direct trajectory optimization using nonlinear programming and collocation. Journal of Guidance, Control, and Dynamics, 10(4), 1987.",
        "[9] Andersson, J. A. E., et al. CasADi: A software framework for nonlinear optimization and optimal control. Mathematical Programming Computation, 11(1), 2019.",
        "[10] Blank, J., and Deb, K. pymoo: Multi-objective optimization in Python. IEEE Access, 8, 2020.",
    ]
    for reference in references:
        paragraph = doc.add_paragraph(style="Body")
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.first_line_indent = Inches(-0.18)
        run = paragraph.add_run(reference)
        set_run_font(run, size=10, color=INK)

    doc.core_properties.title = "PTDL 2.2.2 多目标优化控制算法开发交付件"
    doc.core_properties.subject = "PTDL 混合动力无人机飞推控一体化多目标优化控制算法交付"
    doc.core_properties.author = "卢奕"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
