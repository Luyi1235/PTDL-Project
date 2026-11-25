from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

out = r"C:\Users\pc\Desktop\多目标自适应混合算法创新点.docx"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec.left_margin = sec.right_margin = Cm(2.8)
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
style.font.size = Pt(12)
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.space_after = Pt(0)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(14)
pbdr = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
bottom.set(qn("w:val"), "nil")
pbdr.append(bottom)
title._p.get_or_add_pPr().append(pbdr)
r = title.add_run("多目标自适应混合算法创新点")
r.font.name = "黑体"
r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
r.font.size = Pt(16)
r.font.bold = True
r.font.color.rgb = RGBColor(0, 0, 0)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.first_line_indent = Pt(24)
p.add_run(
    "国内外多目标优化研究广泛采用NSGA-II、MOEA/D和多目标粒子群等方法，分别通过非支配排序、"
    "目标分解和粒子位置更新搜索兼顾多个目标的方案。针对长航时无人机飞行、推进与控制相互耦合的"
    "优化需求，本研究提出基于改进率的自适应遗传粒子群混合算法。该方法在每代设置遗传搜索和粒子"
    "更新两条候选生成支路：遗传支路利用非支配层次与拥挤距离选择父代，通过交叉和变异产生新方案；"
    "粒子支路结合个体历史位置和外部档案中的引导方案更新粒子位置。两类候选方案经过统一的飞推控"
    "任务仿真，评价能源消耗、飞行跟踪误差、响应时间及约束满足情况，再根据各支路相对参考方案的"
    "有效改进率，自适应调整下一代候选生成比例，并通过种群选择和外部档案更新保留可行的非支配方案。"
    "在相同任务模型、种群规模和迭代代数下开展的8次独立运行对比中，混合算法的平均解集超体积为"
    "0.885，较NSGA-II的0.694提高约27.5%，平均计算时间由17.6 min降至14.5 min，缩短约17.5%；"
    "混合算法终代平均可行率为100%，较多目标粒子群算法的80.4%提高19.6个百分点，"
    "其平均解集超体积较粒子群算法的0.972低约9.0%。"
)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(12)
p.paragraph_format.line_spacing = 1.2
run = p.add_run("资料依据  ")
run.bold = True
run.font.size = Pt(9)
run = p.add_run(
    "报告3.5节算法对比结果及工作区8次独立运行统计；"
    "Deb等，IEEE Transactions on Evolutionary Computation，2002；"
    "Zhang和Li，IEEE Transactions on Evolutionary Computation，2007。"
)
run.font.size = Pt(9)
doc.save(out)
print(out)
