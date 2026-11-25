from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, r"D:\LDTP\report_formula_deps")
from latex2mathml.converter import convert
from lxml import etree
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph


SOURCE = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_章节编号统一.docx")
OUT = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_算法原理扩充.docx")
XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
transform = etree.XSLT(etree.parse(str(XSL)))
doc = Document(SOURCE)
anchor = next(p for p in doc.paragraphs if p.text.startswith("图112给出"))
body_template = next(p for p in doc.paragraphs if p.text.startswith("遗传算法通过选择、交叉和变异"))
equation_index = 0


def insert_paragraph(text="", kind="body"):
    global anchor
    node = OxmlElement("w:p")
    source_properties = body_template._p.pPr
    if source_properties is not None:
        node.append(deepcopy(source_properties))
    anchor._p.addnext(node)
    p = Paragraph(node, anchor._parent)
    anchor = p
    if kind == "title":
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
    elif kind == "equation":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
    if text:
        run = p.add_run(text)
        run.font.name = "宋体"
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        run.font.size = Pt(10.5)
        if kind == "title":
            run.bold = True
    return p


def paragraph(text):
    insert_paragraph(text)


def title(text):
    insert_paragraph(text, "title")


def equation(latex):
    global equation_index
    equation_index += 1
    p = insert_paragraph(kind="equation")
    mathml = convert(latex)
    math = transform(etree.fromstring(mathml.encode("utf-8"))).getroot()
    if math.tag == qn("m:oMathPara"):
        for child in list(math):
            p._p.append(child)
    else:
        p._p.append(math)
    suffix = p.add_run(f"    （3.5-A{equation_index}）")
    suffix.font.name = "Times New Roman"
    suffix.font.size = Pt(10.5)


title("（1）非支配排序遗传算法的基本原理")
paragraph("设候选设计向量为x，其三个性能目标分别为任务能量消耗、飞行跟踪误差和响应时间，统一按最小化处理。对于两个满足约束的候选方案，若方案a在各目标上均不劣于方案b，且至少一个目标严格优于方案b，则称方案a支配方案b，其判定关系为")
equation(r"x_a\prec x_b\iff f_m(x_a)\leq f_m(x_b),\ \forall m,\quad \exists j:f_j(x_a)<f_j(x_b)")
paragraph("NSGA-II依据上述关系逐层提取非支配集合。第一层中的方案在当前集合内不被其他方案支配，其余方案依次形成后续层。约束处理采用可行性优先规则，可行方案优于不可行方案；两个不可行方案按约束总违背程度比较；两个可行方案按目标支配关系比较。由此，能源储备、飞行状态和部件能力约束进入种群排序。")
paragraph("同一非支配层内以拥挤距离衡量方案附近目标空间的稀疏程度。对第m个目标排序后，以相邻两个方案的目标差值除以该目标在当前层内的跨度，并对各目标求和，得到内部方案i的拥挤距离")
equation(r"d_i=\sum_{m=1}^{M}{\frac{f_m^{i+1}-f_m^{i-1}}{f_m^{\max}-f_m^{\min}}}")
paragraph("目标取值位于边界的方案赋予较大的保留优先级。父代与交叉变异所得子代共同参加环境选择，先按非支配层次保留方案，再在容量不足的层内优先保留拥挤距离较大的方案。该过程使能耗、跟踪和响应具有不同取舍的候选同时参与下一代搜索。")

title("（2）多目标粒子群算法的基本原理")
paragraph("粒子群方法将每个候选方案表示为设计空间中的粒子，位置对应几何、动力和控制参数，速度表示参数的更新方向与步长。第i个粒子的速度和位置按下式更新")
equation(r"v_i^{t+1}=\omega v_i^t+c_1r_1\odot(p_i^t-x_i^t)+c_2r_2\odot(l_i^t-x_i^t)")
equation(r"x_i^{t+1}=\Pi_{\Omega}(x_i^t+v_i^{t+1})")
paragraph("式中，ω为惯性系数，c₁和c₂为学习系数，r₁和r₂为逐维生成的随机数向量，pᵢ为粒子历史位置，lᵢ为从外部解集中选取的引导方案，⊙表示逐元素乘法，ΠΩ表示将参数限制在允许范围内。本项目实现对速度按各变量取值跨度限幅，再对位置进行边界截断。")
paragraph("多目标问题中的引导方案来自可行非支配档案。档案由历代候选更新，超过容量时按拥挤距离保留分布较分散的方案；尚无可行方案时，以当前约束违背较小的候选提供搜索方向。个体历史位置按可行性和支配关系更新，对互不支配的可行方案采用随机选择。")

title("（3）基于分解的多目标进化算法")
paragraph("MOEA/D将多目标问题分解为一组相互关联的标量子问题。每个子问题对应一个权重向量，表示三个性能目标的一种相对关注方向。对三目标问题，在单位单纯形上构造离散权重")
equation(r"\lambda^i=(h_1/H,h_2/H,h_3/H),\quad h_1+h_2+h_3=H,\quad h_m\in\{0,\ldots,H\}")
paragraph("式中H为权重划分数，非负整数h₁、h₂、h₃表示三个目标的离散份额。满足份额之和为H的组合数给出子问题数量，亦即该实现中的种群规模")
equation(r"N=\binom{H+2}{2}=\frac{(H+1)(H+2)}{2}")
paragraph("例如H取6时得到28组权重。单个权重对应一个子问题，权重之间的距离决定信息交换范围。对权重向量计算欧氏距离，按距离从小到大选取若干相邻权重，构成子问题i的邻域")
equation(r"B(i)=\{r_i(1),\ldots,r_i(T)\}")
paragraph("式中rᵢ(k)为距离权重向量λⁱ第k近的权重向量的序号，T为邻域大小。邻域表示目标偏好相近的一组子问题。每轮从当前子问题的邻域中选取两个父代，对设计参数进行交叉和变异，生成新候选；新候选经过气动、动力与飞行控制联合评价后，返回三个目标值及约束违背程度。")
paragraph("考虑各目标量纲和数值尺度不同，先以固定尺度sₘ对目标与参考点的差值进行缩放，再采用加权切比雪夫函数衡量候选相对各目标方向的最大偏离")
equation(r"g_i(x\mid\lambda^i,z)=\max_{m=1,\ldots,M}\lambda_m^i\left|\frac{f_m(x)-z_m}{s_m}\right|")
paragraph("式中z为目标参考点，sₘ为第m个目标的固定缩放尺度。函数值由加权偏离最大的目标决定，降低该值要求减少该权重方向上的主要偏离。不同权重改变三个目标在子问题中的相对作用，使搜索覆盖不同的能源、跟踪与响应取舍。权重取零时，对应目标在该子问题的标量值中不直接贡献。")
paragraph("程序初始化时，有可行方案则以可行方案的各目标最小值组成参考点；初始种群没有可行方案时暂以初始种群的目标最小值确定参考点。随后每出现可行候选，按各目标更新参考点")
equation(r"z_m^{t+1}=\min(z_m^t,f_m(x_{\mathrm{new}}))")
paragraph("新候选依次与邻域内当前方案比较。若仅新候选可行，则用其替换原方案；若两个候选均不可行，则保留约束总违背较小者；若二者均可行，则在该邻域成员自身的权重下比较标量函数值。以xʲ表示子问题j的当前方案，可行方案之间的替换规则为")
equation(r"g_j(x_{\mathrm{new}}\mid\lambda^j,z)<g_j(x^j\mid\lambda^j,z)\quad\Rightarrow\quad x^j\leftarrow x_{\mathrm{new}}")
paragraph("同一个新候选可以改善多个相邻子问题，因此一次联合仿真评价产生的信息可在多个相近权重方向上使用。等值时保留原方案，避免相同标量值造成邻域内方案被反复覆盖。程序同时记录历代可行非支配方案，供目标权衡和后续方案筛选使用。")

title("（4）遗传搜索与粒子更新的协同机制")
paragraph("混合算法在每一代将候选生成分为遗传搜索与粒子更新两条支路。遗传支路利用当前种群的非支配层次和拥挤距离，通过锦标赛选择父代，再进行模拟二进制交叉与随机变异；粒子支路依据粒子速度、个体历史位置和外部档案引导方案更新位置。两条支路生成的候选均接受相同的飞推控联合评价。")
paragraph("对两个父代的第k个设计变量，模拟二进制交叉以随机系数βₖ组合生成子代；随后按变量范围施加随机变异，并将结果截断到允许区间")
equation(r"y_k=\Pi_{[L_k,U_k]}\left[\frac{1}{2}((1+\beta_k)x_{a,k}+(1-\beta_k)x_{b,k})+\delta_k\right]")
paragraph("式中βₖ由随机数和交叉分布参数确定；δₖ在该变量被选中变异时按变量范围生成高斯扰动，否则为零。遗传支路的作用是重组和扰动几何、动力及控制参数。粒子支路按式（3.5-A3）和式（3.5-A4）更新速度及位置，结合粒子自身搜索历史和可行非支配档案中的引导方案调整候选。")
paragraph("完成候选评价后，父代与两条支路的子代合并。程序依据可行性和非支配层次选择下一代，在最后一个需截断的层内按拥挤距离排序。所选候选的位置、速度和个体历史状态同步进入下一代；遗传支路产生的新候选以自身为初始历史位置。外部档案根据新候选持续更新，向后续粒子更新提供引导。")
paragraph("两条支路的计算份额按有效改进情况调整。有效改进按可行性和支配关系判定：候选由不可行变为可行，或者两个候选均不可行时约束违背减少，或者两个候选均可行时新候选支配对照候选，记为一次有效改进。遗传子代分别与两个父代比较，任一比较获得有效改进即计入；粒子子代与其更新前的方案比较。")
equation(r"R_{\mathrm{GA}}^t=\frac{1}{n_{\mathrm{GA}}}\sum I_{\mathrm{GA}},\qquad R_{\mathrm{PSO}}^t=\frac{1}{n_{\mathrm{PSO}}}\sum I_{\mathrm{PSO}}")
equation(r"q_{\mathrm{GA}}^{t+1}=\operatorname{clip}\!\left[q_{\mathrm{GA}}^t+\eta(R_{\mathrm{GA}}^t-R_{\mathrm{PSO}}^t),q_{\min},q_{\max}\right]")
paragraph("式中I为有效改进的指示量，n为相应支路本代候选数，η为比例调整系数，q为遗传支路在下一代中的候选比例。两支路的平均改进率之差改变候选生成比例，比例边界使每一代仍保留两类搜索操作。非支配排序和拥挤距离负责下一代的环境选择，遗传交叉变异负责参数重组，粒子速度更新负责结合个体历史与档案信息移动候选，上述作用通过统一评价结果和种群状态连接。")

title("算法原理参考文献")
paragraph("[1] Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 2002, 6(2): 182–197. DOI: 10.1109/4235.996017.")
paragraph("[2] Zhang Q, Li H. MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition. IEEE Transactions on Evolutionary Computation, 2007, 11(6): 712–731. DOI: 10.1109/TEVC.2007.892759.")
paragraph("[3] Coello Coello C A, Pulido G T, Lechuga M S. Handling multiple objectives with particle swarm optimization. IEEE Transactions on Evolutionary Computation, 2004, 8(3): 256–279. DOI: 10.1109/TEVC.2004.826067.")

doc.save(OUT)
print("Output:", OUT)
print("Inserted editable equations:", equation_index)
print("Paragraphs:", len(doc.paragraphs))
