from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

source = Path(r"C:\Users\pc\Desktop\多目标优化成果应用方向及效益分析需修改稿.docx")
output = Path(r"C:\Users\pc\Desktop\多目标优化成果应用方向及效益分析_扩充版.docx")
doc = Document(source)
body = doc._element.body
template = deepcopy(doc.paragraphs[6]._p.pPr)
for child in list(body):
    if child.tag != qn("w:sectPr"):
        body.remove(child)

title = doc.add_paragraph("成果应用方向及效益分析", style="Title")

def heading(text, new_page=False):
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = new_page
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    r.font.name = "宋体"
    r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")

def paragraph(text):
    p = doc.add_paragraph()
    if template is not None:
        p._p.insert(0, deepcopy(template))
    p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    r.font.name = "宋体"
    r.font.size = Pt(12)
    r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")

heading("一、成果应用方向")
paragraph("本成果面向超长航时混合动力无人机总体设计与任务运用，将气动构型、动力配置和飞行控制参数纳入统一优化过程，综合考虑任务能量消耗、轨迹跟踪和动态响应，形成满足动力能力、能源储备及飞行状态要求的候选方案。在方案设计阶段，可依据任务航程、留空时间及飞行状态要求，联合分析机翼与机身几何参数、动力系统配置和各任务阶段的功率分配，比较不同方案的能源消耗与控制表现，为构型选择和动力配置提供依据。在控制策略设计阶段，可结合任务剖面分析发动机与电池的供能安排，协调各飞行阶段的推进需求和能源使用。通过改善气动效率、推进效率和能源分配，可为提升无人机续航能力、扩大任务范围和延长目标区域持续留空时间提供技术支持；随着平台续航和远程任务能力提高，相关成果可为突破岛链封锁、拓展远海任务空间提供能力支撑。")
paragraph("本成果还可推广应用于其他类型飞行器的总体设计和性能优化。针对不同机型的气动特性、动力形式与使用任务，建立相应的模型、目标函数和约束条件，将构型、推进系统与控制策略纳入协同设计。对于承担不同航程、载荷或飞行任务的飞机，可调整评价指标和任务条件，比较各设计方案在航程、能耗、载荷和飞行品质等方面的表现，并从可行方案中选择符合任务侧重点的设计。相关方法可用于无人机及有人驾驶飞机的方案论证、动力系统匹配和设计改进，为不同类型飞行器提升综合效能提供支持。")

heading("二、效益分析", new_page=True)
paragraph("在飞机性能方面，成果可服务于现有飞机的改进设计和性能挖掘。通过分析构型参数、动力匹配和控制策略之间的相互影响，设计人员能够在既有方案基础上识别影响续航、能耗和飞行品质的关键因素，比较不同改进方案的综合表现。优化结果可为动力系统选型、任务功率分配和控制参数调整提供依据，有助于提高现有飞机的能源利用水平与任务执行能力，并提升既有装备的使用价值。对于执行长期任务的机型，单位任务能源消耗的改善还可随着使用架次增加逐步体现，形成运行成本方面的收益。")
paragraph("在研发效率和科研成本方面，多学科协同优化将气动、动力和飞行控制等专业纳入统一评价流程，在方案论证阶段统筹分析各专业的相互影响。研究人员可利用代理模型和任务仿真快速筛选候选方案，将后续高精度计算、样机研制和试验资源集中于具有发展潜力的设计，减少跨专业反复调整、重复建模和低效试验所需的人力、计算与材料投入。模型、评价流程及优化结果还可在相关型号和相近任务中复用，进一步降低方案论证成本，缩短研制周期，提高研发资源的使用效率。")
paragraph("在任务应用方面，超长航时无人机的航时、任务范围和留空能力提升后，可在更长时间内获取和更新目标区域信息，为远海监视、战场态势感知和信息保障提供持续的平台支持。飞行器性能的改善有助于提高任务覆盖的连续性和信息获取的及时性，增强对广域、长时间任务的适应能力，并促进相关任务能力的发挥。成果应用效益将随具体机型、任务环境和实际部署方式体现，可结合飞行试验及后续使用情况，从性能改善、研制投入和任务运行等方面进行综合评价。")

doc.save(output)
print(output)
