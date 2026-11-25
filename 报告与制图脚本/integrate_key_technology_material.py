from copy import deepcopy
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

src = r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_优化问题公式修订版.docx"
out = r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_图文融合版.docx"
doc = Document(src)
ps = doc.paragraphs
assert "建立基于改进率" in ps[2].text
assert "有效改进" in ps[3].text
assert len(ps[5]._p.xpath(".//m:oMath")) == 1
assert len(ps[13]._p.xpath(".//w:drawing")) == 1

second = ps[2]
second.clear()
second.add_run(
    "针对这一问题，建立基于改进率的自适应遗传粒子群混合算法，将构型、动力和控制"
    "参数纳入统一优化过程。联合评价以设计变量为输入，统一计算任务目标及约束，"
    "其优化问题可表示为"
)


def place_after(anchor, paragraph):
    anchor._p.addnext(paragraph._p)
    return paragraph


def paragraph_after(anchor, text):
    element = deepcopy(second._p)
    anchor._p.addnext(element)
    p = Paragraph(element, anchor._parent)
    p.clear()
    p.add_run(text)
    return p


anchor = place_after(second, ps[5])
anchor = place_after(anchor, ps[6])
anchor = paragraph_after(
    anchor,
    "算法根据当前种群的非支配层次和拥挤程度选择遗传操作的父代，通过交叉与变异"
    "重组设计参数，形成新的候选方案；同时利用粒子速度、个体历史位置和档案引导"
    "方案更新粒子位置，使搜索过程能够利用已有方案的演化信息。粒子更新的基本关系为",
)
anchor = place_after(anchor, ps[8])
anchor = place_after(anchor, ps[9])
anchor = paragraph_after(
    anchor,
    "两类候选均通过同一飞推控任务仿真，计算任务能量消耗、飞行跟踪误差和响应时间，"
    "并检查部件功率、能源储备与飞行状态等约束。统一的任务评价使两类候选在相同条件"
    "下进行比较，为判断搜索效果和协调评价资源提供依据。算法总体流程如",
)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), " REF _AlgoOverview \\h ")
run = OxmlElement("w:r")
text = OxmlElement("w:t")
text.text = "图1"
run.append(text)
field.append(run)
anchor._p.append(field)
anchor.add_run("所示。")
anchor = place_after(anchor, ps[13])
anchor = place_after(anchor, ps[14])

third = ps[3]
third.clear()
third.add_run(
    "在此基础上，算法依据候选方案相对于参考方案的可行性、约束违反程度和目标支配关系，"
    "判断两类搜索产生的有效改进。其中，方案进入可行区域、约束违反程度降低或可行方案"
    "在目标上形成支配关系，分别反映搜索在可行性恢复和多目标优化方面的进展。"
    "两条支路分别统计有效改进情况，并按改进率调整遗传候选比例"
)
anchor = place_after(third, ps[11])
anchor = place_after(anchor, ps[12])
paragraph_after(
    anchor,
    "算法根据两条支路平均有效改进率的差异调整下一代候选生成比例，使评价资源随优化"
    "进程向当前改进效果较好的方式倾斜，同时保持两类搜索持续参与。经过统一评价的候选"
    "与已有种群共同参与选择，外部档案保存可行非支配方案，并通过拥挤距离维护解集"
    "的分布。这些方案既呈现不同目标之间的取舍，也为后续粒子搜索提供引导。由此形成"
    "“候选生成—任务评价—改进反馈—比例调整”的自适应优化过程，为长航时无人机"
    "飞推控一体化协同设计提供方法支撑。",
)
for i in (4, 7, 10):
    ps[i]._p.getparent().remove(ps[i]._p)
doc.save(out)
print(out)
