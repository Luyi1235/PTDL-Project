from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, r"D:\LDTP\report_formula_deps")
from latex2mathml.converter import convert
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph
from lxml import etree

src = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_算法原理扩充.docx")
out = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_原理深化版.docx")
doc = Document(src)
transform = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))
body = next(p for p in doc.paragraphs if p.text.startswith("MOEA/D将多目标问题分解"))
counter = 0


def find(prefix):
    matches = [p for p in doc.paragraphs if p.text.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError((prefix, len(matches)))
    return matches[0]


def insert_after(anchor, text=None, latex=None):
    global counter
    node = OxmlElement("w:p")
    if body._p.pPr is not None:
        node.append(deepcopy(body._p.pPr))
    anchor._p.addnext(node)
    p = Paragraph(node, anchor._parent)
    if text is not None:
        r = p.add_run(text)
        r.font.name = "宋体"
        r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        r.font.size = Pt(10.5)
    else:
        counter += 1
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        math = transform(etree.fromstring(convert(latex).encode("utf-8"))).getroot()
        if math.tag == qn("m:oMathPara"):
            for child in list(math):
                p._p.append(child)
        else:
            p._p.append(math)
        r = p.add_run(f"    （3.5-B{counter}）")
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
    return p


def block(prefix, parts):
    p = find(prefix)
    for typ, content in parts:
        p = insert_after(p, text=content if typ == "p" else None,
                         latex=content if typ == "e" else None)


block("（3）MOEA/D算法的基本原理", [
    ("p", "多目标优化通常不存在使全部目标同时达到各自最小值的单一方案。MOEA/D为不同目标分配不同权重，将原问题转化为一系列标量子问题；每个子问题保留一个当前候选，所有子问题共享同一任务模型和约束判定。设可行域为Ω，目标向量为F(x)，第i个子问题可表示为"),
    ("e", r"\min_{x\in\Omega}\ g_i(x\mid\lambda^i,z),\qquad F(x)=(f_1(x),f_2(x),f_3(x))"),
    ("p", "其中目标分量分别对应任务能量、跟踪误差和响应指标。权重并不是任务仿真中的物理参数，而是定义搜索方向的参数；不同子问题对同一候选的评价结果相同，但按照各自权重比较标量值。"),
])
block("    （3.5-A6）", [
    ("p", "组合数可由隔板法得到：将H份权重分配到三个目标，相当于在H个份额与两个分隔符组成的H＋2个位置中选出两个位置，故共有C(H＋2,2)组。增大H能够增加权重方向，同时也增加种群规模和每代候选评价次数，不能仅将权重加密理解为精度必然提高。"),
])
block("式中rᵢ(k)为距离权重向量", [
    ("p", "权重间距离在三目标情形下由各分量之差计算，距离较小表示两个子问题的目标偏好相近"),
    ("e", r"d_{ij}=\Vert\lambda^i-\lambda^j\Vert_2=\sqrt{\sum_{m=1}^{3}(\lambda_m^i-\lambda_m^j)^2}"),
    ("p", "实现中邻域包含当前权重自身，父代从该邻域有放回地抽取，因此两个父代也可能相同。邻域交叉使偏好相近的子问题交换设计信息，但邻域相近不等于设计变量也一定相近；生成的新候选仍需通过完整的目标和约束计算确定其价值。"),
])
block("式中z为目标参考点", [
    ("p", "标量函数可以从各目标的归一化偏差出发理解。记第m个目标的偏差为dₘ，标量值τ同时不小于各项加权偏差，则最小化τ与求取最大加权偏差等价"),
    ("e", r"d_m(x)=\left|\frac{f_m(x)-z_m}{s_m}\right|,\qquad g_i(x)=\max_m\lambda_m^i d_m(x)"),
    ("e", r"\min\ \tau\quad\mathrm{s.t.}\quad \lambda_m^i d_m(x)\leq\tau,\ \ m=1,2,3"),
    ("p", "因此，τ反映该权重下最大的目标偏差。与简单加权求和不同，某一项偏差过大时不能直接由另一项偏差较小抵消。上述等价关系用于说明标量函数的含义；程序仍按式（3.5-A8）直接计算，并未额外引入τ求解。固定尺度在各子问题中一致，改变尺度会影响目标偏差的相对大小，因此不同算法比较时应使用一致的目标和评价口径。"),
])
block("同一个新候选可以改善多个相邻子问题", [
    ("p", "以第j个邻域成员为例，可将其更新理解为三个依次执行的判断：先比较可行性，再比较不可行方案的违约程度，最后才比较可行方案的标量值。记V(x)为程序返回的总约束违背程度，候选y替换当前解xʲ的条件可写为"),
    ("e", r"\mathrm{feas}(y)>\mathrm{feas}(x^j)"),
    ("e", r"\mathrm{feas}(y)=\mathrm{feas}(x^j)=0,\quad V(y)<V(x^j)"),
    ("e", r"\mathrm{feas}(y)=\mathrm{feas}(x^j)=1,\quad g_j(y)<g_j(x^j)"),
    ("p", "这里feas取1表示可行、取0表示不可行；满足上述任一条件即接受替换。实际代码在可行候选的标量值比较中设置数值容差，以减少浮点计算中的近似相等引起的重复替换。对子问题分别更新并共享候选评价，可以复用一次飞推控耦合计算结果；但权重分布、邻域规模和有限评价预算都会影响最终解集，不能仅凭分解形式保证获得完整的Pareto前沿。"),
])
block("式中βₖ由随机数和交叉分布参数确定", [
    ("p", "模拟二进制交叉的系数由均匀随机数uₖ及分布参数ηc确定，其分段关系为"),
    ("e", r"\beta_k=\begin{cases}(2u_k)^{1/(\eta_c+1)},&u_k\leq 0.5\\(2(1-u_k))^{-1/(\eta_c+1)},&u_k>0.5\end{cases}"),
    ("p", "当βₖ较接近1时，新变量接近选定父代；βₖ小于1时，新值位于两个父代之间；βₖ大于1时，也可能在父代区间外形成候选。各维变异独立触发，触发后叠加高斯扰动，最后截断到允许区间。该操作改变的是设计参数本身，适于重组已有几何、动力配置及控制参数组合，并不直接判断新方案是否满足任务约束。"),
    ("p", "粒子支路先计算未限幅速度，再按照各变量的取值跨度Δₖ限制步长。代码中第k维的处理可写为"),
    ("e", r"\bar v_{i,k}^{t+1}=\mathrm{clip}(v_{i,k}^{t+1},-0.2\Delta_k,0.2\Delta_k)"),
    ("e", r"x_{i,k}^{t+1}=\mathrm{clip}(x_{i,k}^{t}+\bar v_{i,k}^{t+1},L_k,U_k)"),
    ("p", "其中Δₖ＝Uₖ－Lₖ。限幅约束单次参数移动的幅度，位置截断保证参数不越出给定区间；飞行包线、能源储备等任务约束仍由后续评价检查。粒子所用的引导方案从当前种群与外部档案形成的候选集合中随机抽取，个体历史位置按可行性和支配关系更新。"),
])
block("完成候选评价后，父代与两条支路的子代合并", [
    ("p", "统一环境选择同时处理两条支路产生的方案：将N个父代与N个子代组成大小为2N的集合，按约束支配关系划分层次，并依次填入下一代。若当前层容纳不下剩余名额，则按该层拥挤距离从大到小截取"),
    ("e", r"P^{t+1}=\mathrm{Select}_{N}(P^t\cup Q_{\mathrm{GA}}^t\cup Q_{\mathrm{PSO}}^t)"),
    ("p", "选择时位置、速度、个体历史位置及其评价结果按同一索引保留，避免方案与粒子状态错配。遗传子代初始化速度为零、历史位置为自身；它被选入下一代后，可以在随后迭代中由粒子更新使用。这样，遗传操作产生的参数组合能够进入粒子历史和档案引导过程，而粒子产生的候选也可以成为后续遗传选择的父代。"),
])
block("式中I为有效改进的指示量", [
    ("p", "记G(y,x)为新候选相对对照方案的有效改进指示量，可行性优先规则将其写为"),
    ("e", r"G(y,x)=\begin{cases}1,&\mathrm{feas}(y)>\mathrm{feas}(x)\\1,&\mathrm{feas}(y)=\mathrm{feas}(x)=0,\ V(y)<V(x)\\1,&\mathrm{feas}(y)=\mathrm{feas}(x)=1,\ y\prec x\\0,&\mathrm{otherwise}\end{cases}"),
    ("p", "对遗传子代y，以两个父代中较大的改进指示量计数；对粒子子代，以该粒子更新前的位置作为对照"),
    ("e", r"I_{\mathrm{GA}}(y)=\max\{G(y,x_a),G(y,x_b)\},\quad I_{\mathrm{PSO}}(y)=G(y,x_i)"),
    ("p", "遗传候选数由当前比例乘以种群规模后取整，并限制在1至N－1之间，其余名额分配给粒子支路；具体哪些位置执行哪一种操作由随机排列决定。式（3.5-A13）的比例更新发生在本代评价和选择之后，因而影响下一代候选生成。该改进率衡量的是各支路本代候选相对其对照方案的有效改进频次，并非超体积增益，也不能单独证明某一支路的全局搜索能力更强。"),
    ("p", "两个分支的贡献通过统一评价、环境选择和档案更新共同作用：遗传操作提供重新组合参数的候选，粒子操作利用历史位置与档案信息调整搜索方向，非支配排序和拥挤距离决定哪些候选保留，改进率则调节下一代两类候选的数量。算法的实际效果仍应在一致评价预算、相同约束和多次独立运行条件下，与各基线比较解集质量、可行率及计算代价。"),
])

# Preserve the user's four subheading texts and their direct formatting.
doc.save(out)
print("Saved", out, "new editable equations:", counter)
