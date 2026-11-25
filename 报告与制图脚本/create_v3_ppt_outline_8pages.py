import ast
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

source = Path(r"D:\LDTP\create_v3_ppt_outline.py").read_text(encoding="utf-8")
tree = ast.parse(source)
slides = next(ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "slides" for t in n.targets))
chosen = [list(slides[i]) for i in (0, 1, 2, 3, 5, 7, 8, 10)]

chosen[3] = [
    "3.2.3", "各工况关键点流场机理与全工况气动响应公式", "典型工况流场与全工况气动响应",
    "通过关键工况流场分析与迎角响应建模，描述翼桨布局对气动载荷的影响。",
    ["对爬升、降落和机动工况的典型布局进行配对分析，比较局部速度、压力与气动力变化。",
     "相同翼桨垂直位置变化在爬升和机动工况下可产生不同的截面载荷调整。",
     "结合巡航位置响应和关键点迎角修正，建立连续气动响应，再转换为飞行动力学所需载荷。"],
    "建立从翼桨布局和飞行工况到气动载荷的计算关系。",
    "左侧选取图29、图33中共面与上移的代表性流场，右侧选图35的一组气动系数曲线；版面拥挤时仅保留图35及一组流场。",
    "总报告图29“爬升工况不同垂直布置的局部速度与压力分布”、图33“机动工况不同垂直布置的局部速度与压力分布”、图35“代表布局的连续迎角气动响应”。图35是模型构建结果。",
    "",
]
chosen[4] = [
    "4.2.1", "多目标性能优化指标体系与约束条件构建", "多目标指标与任务约束",
    "围绕任务能源、飞行跟踪和响应过程，建立协同优化目标及可行性约束。",
    ["三个正式目标为任务能量、飞行跟踪误差和响应时间，统一由候选任务仿真计算。",
     "燃油消耗、电池转换效率及内部损失作为辅助评价，剩余燃油与电量用于能源储备约束。",
     "几何、动力和控制参数共同参与设计；约束覆盖部件能力、飞行状态及起飞着陆。"],
    "形成设计变量、优化目标与飞行任务约束相互关联的评价体系。",
    "中央用三列显示优化目标，底部用四项简列部件、能源、飞行和起降约束；根据空间放入一行辅助评价指标。",
    "总报告表102“优化目标与其他评价量的作用”、表104“协同优化的主要约束关系”；优先精简原表。",
    "",
]
chosen[7] = [
    "4.2.3", "多目标优化控制仿真分析平台构建与算法性能评估", "联合优化平台与算法性能评估",
    "依托联合仿真平台，对各类多目标算法进行任务评价和性能比较。",
    ["平台贯通参数配置、气动与飞行推进联算、多目标搜索、可行方案选择和结果分析。",
     "共同计算预算下，报告中MOPSO平均超体积为0.97，终代平均可行率为80.4%。",
     "混合算法终代平均可行率为100%，平均耗时14.5 min；算法选择综合考虑解集质量、可行性与耗时。"],
    "形成从候选评价到算法比较和方案选择的仿真分析流程。",
    "左侧简化图115的平台评价流程，右侧放图116的收敛曲线或重复运行分布；用简短数据条展示表108的三项评价指标。",
    "总报告图115“联合优化平台中的模型与任务评价”、图116“四算法的收敛过程与重复运行分布”、表108。超体积、可行率和时间属于报告中对应的比较算例。",
    "",
]

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(.62), Inches(.57)
sec.left_margin, sec.right_margin = Inches(.8), Inches(.8)
normal = doc.styles["Normal"]
normal.font.name = "宋体"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(11.5)
normal.font.color.rgb = RGBColor.from_string("202020")
title = doc.styles["Title"]
title.font.name = "微软雅黑"
title._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
title.font.size = Pt(16)
title.font.color.rgb = RGBColor(0, 0, 0)
for border in title._element.get_or_add_pPr().findall(qn("w:pBdr")):
    title._element.get_or_add_pPr().remove(border)

def line(label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.17
    p.add_run(label + "  ").bold = True
    p.add_run(text)

for index, (number, src_title, short_title, opening, points, closing, layout, sources, prompt) in enumerate(chosen, 1):
    if index > 1:
        doc.add_page_break()
    if index == 1:
        p = doc.add_paragraph("中期汇报 PPT大纲 八页精简版", style="Title")
        p.paragraph_format.space_after = Pt(3)
        p = doc.add_paragraph("依据 V3 技术进展总报告3.2和4.2节编排，共8页")
        p.paragraph_format.space_after = Pt(8)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(f"第{index}页  {number}  {short_title}")
    r.bold = True
    r.font.size = Pt(14)
    line("一级标题", "研究内容一" if number.startswith("3") else "研究内容二")
    line("二级标题", "3.2 全工况气动响应与系统动力学描述" if number.startswith("3") else "4.2 多目标优化控制算法研究")
    line("三级标题", short_title + "（对应报告 " + number + " " + src_title + "）")
    line("开头总领句", opening)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.add_run("中间展示内容").bold = True
    for text in points:
        p = doc.add_paragraph(text, style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.12
    line("结尾总结句", closing)
    line("版面安排", layout)
    line("优先使用的报告图表", sources)
    if prompt:
        line("补充配图提示词", prompt)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run("PPT制作参考")
output = r"C:\Users\pc\Desktop\PTDL中期汇报PPT大纲_V3总报告_8页精简版.docx"
doc.save(output)
print(output)
