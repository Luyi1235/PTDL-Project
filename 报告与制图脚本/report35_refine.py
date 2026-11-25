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

source = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容 修改版.docx")
target = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_原理补充与语言优化版.docx")
doc = Document(source)
transform = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))
template = next(p for p in doc.paragraphs if p.text.startswith("MOEA/D将多目标问题分解"))

replacements = {
    "多目标优化通常不存在使全部目标同时达到各自最小值的单一方案。": "多目标优化需要协调各项目标之间的性能取舍。",
    "权重并不是任务仿真中的物理参数，而是定义搜索方向的参数；": "权重用于定义搜索方向；",
    "不能仅将权重加密理解为精度必然提高。": "权重划分密度可结合可用评价预算与所需的目标空间覆盖程度确定。",
    "但邻域相近不等于设计变量也一定相近；": "邻域关系由权重空间距离确定；",
    "权重取零时，对应目标在该子问题的标量值中不直接贡献。": "权重取零时，该目标在对应子问题中的标量项为零。",
    "与简单加权求和不同，某一项偏差过大时不能直接由另一项偏差较小抵消。": "最大偏差项决定标量值，使该权重方向上偏离较大的目标得到关注。",
    "程序仍按式（3.5-A8）直接计算，并未额外引入": "程序按式（3.5-A8）直接计算，等价推导使用",
    "求解。固定尺度在各子问题中一致": "表示最大加权偏差。固定尺度在各子问题中一致",
    "但权重分布、邻域规模和有限评价预算都会影响最终解集，不能仅凭分解形式保证获得完整的Pareto前沿。": "权重分布、邻域规模及评价预算共同影响解集在Pareto前沿附近的覆盖程度。",
    "并不直接判断新方案是否满足任务约束。": "新方案的任务约束由后续统一评价判定。",
    "该改进率衡量的是各支路本代候选相对其对照方案的有效改进频次，并非超体积增益，也不能单独证明某一支路的全局搜索能力更强。": "该改进率反映各支路本代候选相对其对照方案的有效改进频次，解集整体质量则结合超体积等指标评价。",
}

changes = {}
for p in doc.paragraphs:
    for node in p._p.xpath(".//w:t"):
        for old, new in replacements.items():
            if old in (node.text or ""):
                node.text = node.text.replace(old, new)
                changes[old] = changes.get(old, 0) + 1
if set(changes) != set(replacements):
    raise ValueError(("replacement not found", set(replacements) - set(changes)))

counter = 0

def find(prefix):
    matches = [p for p in doc.paragraphs if p.text.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError((prefix, len(matches)))
    return matches[0]

def after(anchor, text=None, latex=None):
    global counter
    node = OxmlElement("w:p")
    if template._p.pPr is not None:
        node.append(deepcopy(template._p.pPr))
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
        omml = transform(etree.fromstring(convert(latex).encode("utf-8"))).getroot()
        if omml.tag == qn("m:oMathPara"):
            for child in list(omml):
                p._p.append(child)
        else:
            p._p.append(omml)
        r = p.add_run(f"    （3.5-C{counter}）")
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
    return p

def block(prefix, entries):
    p = find(prefix)
    for kind, value in entries:
        p = after(p, text=value if kind == "p" else None, latex=value if kind == "e" else None)

block("因此，反映该权重下最大的目标偏差", [
    ("p", "从几何意义看，当权重分量均为正时，给定标量值τ对应一组归一化目标偏差的允许范围。由λₘdₘ≤τ可得第m个目标的偏差上界；权重越大，该目标在相同τ下允许的偏差越小。三个偏差范围共同构成以参考点为中心的目标空间区域。"),
    ("e", r"d_m(x)\leq \frac{\tau}{\lambda_m^i},\qquad \lambda_m^i>0,\quad m=1,2,3"),
    ("p", "优化子问题时，算法比较候选所需的最大加权偏差，较小值对应较紧的目标偏差范围。单位单纯形的边界包含权重为零的情形，此时相应目标的权重项为零；式（3.5-C1）的除法关系适用于该权重分量为正的目标。"),
    ("p", "本项目实现对任务能量、跟踪误差和响应指标分别采用3000、1和400作为固定缩放尺度。尺度用于计算目标与参考点的相对偏差，使数值量级不同的三个目标能够共同进入标量函数；尺度与权重共同决定各目标差异对比较结果的影响。"),
])
block("这里feas取1表示可行", [
    ("p", "邻域更新的计算顺序体现了分解搜索的信息传递方式。对子问题i生成候选y并完成一次评价后，先更新参考点，再使用各邻域成员各自的权重λʲ计算比较值。因此，同一候选可以在某些权重方向上替换当前方案，在另一些方向上保留原有方案。"),
    ("p", "设N为权重个数、T为邻域大小。一代中按子问题顺序生成N个候选，每个候选进行一次联合评价，并最多与T个邻域位置比较；目标计算复用于这些比较。若计入初始种群，则完成G代后，该实现的候选评价次数为"),
    ("e", r"N_{\mathrm{eval}}=N+GN=N(G+1)"),
    ("p", "式中统计的是优化循环中调用候选评价的次数；最终方案的独立复算等额外操作应另行计数。报告算法比较结果时，采用相同的评价次数口径能够对应不同算法实际调用任务模型的计算负担。"),
])
block("选择时位置、速度、个体历史位置及其评价结果", [
    ("p", "粒子历史位置的更新采用分情况比较：若新位置与历史位置的可行性不同，则选取可行位置；二者均不可行时比较约束违背程度；二者均可行时按目标支配关系比较。对互不支配的可行位置，程序通过随机选择保留其中之一，使具有不同目标取舍的历史位置都有机会参与后续速度更新。"),
    ("p", "外部档案按可行性和非支配关系组织：已有可行方案时，收集可行非支配候选并按设计变量去重；数量超过容量时依据拥挤距离保留分布较分散的方案。尚未得到可行方案时，以约束违背较小的方案作为引导。每代的粒子引导集合由当前种群与已有档案共同形成，并从中随机选取引导方案。"),
])
block("遗传候选数由当前比例乘以种群规模后取整", [
    ("p", "候选数分配可以表示为下式，其中qᵗ为本代遗传支路比例，取整后再通过边界限制保证每条支路至少生成一个候选"),
    ("e", r"n_{\mathrm{GA}}^t=\mathrm{clip}(\mathrm{round}(Nq^t),1,N-1),\quad n_{\mathrm{PSO}}^t=N-n_{\mathrm{GA}}^t"),
    ("p", "支路比例由上一代记录的平均有效改进率更新。当遗传支路的有效改进率较高时，下一代增加遗传候选份额；当粒子支路的有效改进率较高时，下一代增加粒子候选份额；两者相等时维持当前比例。比例上下界控制调整范围，取整和候选数边界决定每代实际分配数量。"),
    ("p", "对于混合算法，初始种群评价N次，此后每代生成并评价N个子代；在完成G代时，优化循环内的评价次数同样为N(G＋1)。按评价次数组织比较，可以将搜索操作的差异与任务模型调用次数对应起来；实际运行时间还与模型计算开销及算法自身操作有关。"),
])

doc.save(target)
print("Saved:", target, "reworded:", len(changes), "new equations:", counter)
