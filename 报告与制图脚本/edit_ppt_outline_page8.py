from docx import Document
from docx.shared import Pt

path = r"C:\Users\pc\Desktop\PTDL中期汇报PPT大纲_V3总报告_8页精简版.docx"
doc = Document(path)
start = next(i for i, p in enumerate(doc.paragraphs) if p.text.startswith("第8页"))
paragraphs = doc.paragraphs[start:]
replacements = [
    ("第8页", "第8页  4.2.3  飞推控协同设计与多目标优化验证"),
    ("三级标题", "三级标题  飞推控协同设计与多目标优化验证（对应报告4.2.3）"),
    ("开头总领句", "开头总领句  围绕长航时任务，开展飞推控一体化协同设计与多目标优化仿真验证。"),
    ("平台贯通", "贯通任务与模型确定、控制参数整定、多目标搜索、可行方案选择和任务复算。"),
    ("结尾总结句", "结尾总结句  形成面向长航时无人机的飞推控一体化协同设计与多目标优化仿真验证方法体系。"),
    ("版面安排", "版面安排  左侧选用图118的协同设计流程，右侧选用图116的收敛曲线或重复运行分布；用简短数据条展示表108的三项评价指标。"),
    ("优先使用的报告图表", "优先使用的报告图表  总报告图118“推进系统与飞行控制协同设计流程”、图116“四算法的收敛过程与重复运行分布”、表108。超体积、可行率和时间属于报告中对应的比较算例。"),
]
for prefix, text in replacements:
    p = next(p for p in paragraphs if p.text.startswith(prefix))
    p.clear()
    if prefix == "第8页":
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(14)
    elif prefix == "平台贯通":
        p.add_run(text)
    else:
        label, content = text.split("  ", 1)
        p.add_run(label + "  ").bold = True
        p.add_run(content)

assert all("平台" not in p.text for p in paragraphs)
doc.save(r"C:\Users\pc\Desktop\PTDL中期汇报PPT大纲_V3总报告_8页修订版.docx")
print("第八页已修改。")
