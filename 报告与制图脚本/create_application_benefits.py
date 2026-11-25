from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

out = r"C:\Users\pc\Desktop\多目标优化成果应用方向及经济效益分析.docx"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.4)
sec.left_margin = sec.right_margin = Cm(2.8)
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing = 1.3
normal.paragraph_format.space_after = Pt(0)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(12)
borders = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
bottom.set(qn("w:val"), "nil")
borders.append(bottom)
title._p.get_or_add_pPr().append(borders)
run = title.add_run("成果应用方向及经济效益分析")
run.font.name = "黑体"
run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
run.font.size = Pt(16)
run.font.bold = True
run.font.color.rgb = RGBColor(0, 0, 0)


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    r.font.size = Pt(12)
    r.bold = True


def para(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Pt(24)
    p.add_run(text)


heading("一、成果应用方向")
para(
    "本成果面向长航时无人机飞推控一体化协同设计，形成基于改进率的自适应遗传粒子群"
    "多目标优化方法。该方法将飞行器构型、动力配置与控制参数纳入统一设计过程，"
    "结合任务剖面开展气动、动力和飞行控制联合评价，在任务能量消耗、飞行跟踪误差"
    "及响应时间等目标之间寻求协调，形成满足部件能力、能源储备和飞行状态等约束的"
    "候选方案，可为长航时无人机总体设计和方案论证提供方法支持。"
)
para(
    "在方案设计阶段，可依据任务航程、留空时间及飞行状态要求，联合分析机翼与机身"
    "几何参数、混合动力配置以及任务阶段功率分配，比较不同设计方案的能源消耗与"
    "控制表现，为构型选择和动力系统配置提供依据。在控制策略设计阶段，可结合给定"
    "任务剖面与动力系统工作条件，研究控制参数和能量分配对飞行轨迹、能源消耗及"
    "响应特性的影响，为任务需求下的协同控制策略调整提供参考。"
)
para(
    "该方法还可应用于无人机任务方案比较。通过调整任务剖面和评价条件，分析"
    "不同任务阶段对能源供给和飞行性能的要求，从可行解集中选择符合任务偏好的"
    "设计方案。通过对接相应的飞行器和动力系统模型，方法可进一步用于其他"
    "具有多任务、多约束和多目标特征的飞行器设计研究。"
)
heading("二、经济效益分析")
para(
    "本成果的经济效益主要体现在研发设计和任务运行两个环节。在研发设计阶段，"
    "采用统一的飞推控联合评价方法，可以在方案前期分析构型、动力和控制参数的"
    "相互影响，及早筛选符合任务要求的方案，减少各专业反复修改和重新计算"
    "所需的人力与计算资源。设计人员可依据优化得到的候选方案开展后续论证，"
    "有助于降低方案筛选和技术协调的研发成本。"
)
para(
    "在仿真与试验安排方面，通过对多个设计方案进行任务仿真和综合筛选，"
    "可为高精度分析、样机研制及后续试验提供更明确的方案范围，"
    "有望减少低效方案进入后续研制环节所带来的计算、材料和试验投入。"
    "同时，统一的评价流程有助于积累和复用模型与分析结果，"
    "降低相近任务或后续型号开展方案论证时的重复工作量。"
)
para(
    "在任务运行阶段，将任务能源消耗纳入构型、动力配置及控制策略的协同权衡，"
    "有助于筛选满足飞行任务要求的低能耗方案，具有降低燃油消耗和能源使用费用"
    "的潜力。对于需要长期执行任务的无人机，单次任务的能源费用改善可随运行"
    "架次增加逐步体现。实际经济收益可结合具体机型、任务频次、燃油及电力价格，"
    "以及优化前后的研发投入和能源消耗进行核算。"
)
doc.save(out)
print(out)
