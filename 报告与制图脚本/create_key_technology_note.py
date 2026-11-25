from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

out = r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_扩展版.docx"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.4)
sec.left_margin = sec.right_margin = Cm(2.8)
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(0)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(14)
pbdr = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
bottom.set(qn("w:val"), "nil")
pbdr.append(bottom)
title._p.get_or_add_pPr().append(pbdr)
r = title.add_run("飞推控一体化多目标自适应协同优化技术")
r.font.name = "黑体"
r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
r.font.size = Pt(16)
r.font.bold = True
r.font.color.rgb = RGBColor(0, 0, 0)


def para(text, bold_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Pt(24)
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    return p


para(
    "长航时无人机的气动构型、动力配置与控制策略共同影响能源消耗、飞行跟踪误差和动态响应。"
    "这些设计参数相互关联，优化过程中既要处理多个目标之间的取舍，也要满足动力部件、"
    "能源储备和飞行任务约束。为此，本研究围绕飞推控一体化多目标自适应协同优化技术开展攻关，"
    "重点解决不同搜索机制如何根据任务评价结果动态协同的问题。"
)
para(
    "针对这一问题，建立基于改进率的自适应遗传粒子群混合算法，将构型、动力和控制参数"
    "纳入统一优化过程。算法根据当前种群的非支配层次和拥挤程度选择遗传操作的父代，"
    "通过交叉与变异重组设计参数，形成新的候选方案；同时利用粒子速度、个体历史位置和"
    "档案引导方案更新粒子位置，使搜索过程能够利用已有方案的演化信息。"
    "两类候选均通过同一飞推控任务仿真，计算任务能量消耗、飞行跟踪误差和响应时间，"
    "并检查部件功率、能源储备与飞行状态等约束。统一的任务评价使两类候选在相同条件下"
    "进行比较，为判断搜索效果和协调评价资源提供依据。"
)
para(
    "在此基础上，算法依据候选方案相对于参考方案的可行性、约束违反程度和目标支配关系，"
    "判断两类搜索产生的有效改进。其中，方案进入可行区域、约束违反程度降低或可行方案"
    "在目标上形成支配关系，分别反映搜索在可行性恢复和多目标优化方面的进展。"
    "算法统计两类候选的平均有效改进率，根据两者差异调整下一代候选生成比例，"
    "使评价资源随优化进程向当前改进效果较好的方式倾斜，同时保持两类搜索持续参与。"
    "经过统一评价的候选与已有种群共同参与选择，外部档案保存可行非支配方案，"
    "并通过拥挤距离维护解集的分布。这些方案既呈现不同目标之间的取舍，也为后续粒子"
    "搜索提供引导。由此形成“候选生成—任务评价—改进反馈—比例调整”的"
    "自适应优化过程，为长航时无人机飞推控一体化协同设计提供方法支撑。"
)
doc.save(out)
print(out)
