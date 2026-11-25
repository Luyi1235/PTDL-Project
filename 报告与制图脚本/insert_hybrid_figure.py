from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from docx.text.paragraph import Paragraph


source = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_方法机理扩充版.docx")
output = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_混合算法框图版.docx")
image = Path(r"C:\Users\pc\AppData\Local\Temp\codex-clipboard-8a580a23-d8fd-4912-a306-4d2ea46dba89.png")
doc = Document(source)
paragraphs = doc.paragraphs
anchor = next(p for p in paragraphs if p.text.startswith("混合算法在每一代将候选生成分为遗传搜索与粒子更新两条支路"))
original_caption = next(p for p in paragraphs if p.text.startswith("图112 多目标搜索与候选方案评价"))
original_ref = next(p for p in paragraphs if p.text.startswith("图112给出多目标搜索与候选方案评价"))
first_table = next(p for p in paragraphs if p.text.startswith("表102 "))

ref_xml = deepcopy(original_ref._p)
for element in ref_xml.xpath(".//w:instrText"):
    if "REF" in (element.text or ""):
        element.text = " REF CodexRefHybridFlow \\h \\* MERGEFORMAT "
for element in ref_xml.xpath(".//w:t"):
    if element.text == "112":
        element.text = "113"
    elif element.text and element.text.startswith("给出多目标搜索"):
        element.text = "展示混合算法从两条支路生成候选，经统一评价、种群选择和档案更新，再按有效改进调整候选生成比例的过程。"
anchor._p.addnext(ref_xml)

image_xml = deepcopy(original_caption._p)
for child in list(image_xml):
    if child.tag.endswith("}pPr"):
        continue
    image_xml.remove(child)
ref_xml.addnext(image_xml)
image_paragraph = Paragraph(image_xml, anchor._parent)
image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
image_paragraph.paragraph_format.first_line_indent = Inches(0)
image_paragraph.paragraph_format.keep_with_next = True
image_paragraph.add_run().add_picture(str(image), width=Inches(5.1))

caption_xml = deepcopy(original_caption._p)
all_ids = [int(x.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id"))
           for x in doc.element.xpath(".//w:bookmarkStart")]
next_id = max(all_ids, default=0) + 1
id_map = {}
for index, element in enumerate(caption_xml.xpath(".//w:bookmarkStart")):
    old_id = element.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id")
    new_id = str(next_id + index)
    id_map[old_id] = new_id
    element.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id", new_id)
    element.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}name",
                ("CodexRefHybridFlow", "PTDLHybridFigureCaption")[index])
for element in caption_xml.xpath(".//w:bookmarkEnd"):
    key = element.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id")
    if key in id_map:
        element.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id", id_map[key])
for element in caption_xml.xpath(".//w:t"):
    if element.text == "112":
        element.text = "113"
    elif element.text == "多目标搜索与候选方案评价":
        element.text = "混合算法求解流程"
image_xml.addnext(caption_xml)

original_caption._p.xpath(".//w:instrText")[-1].text = " \\r 112 \\* ARABIC "
first_table._p.xpath(".//w:instrText")[-1].text = " \\r 102 \\* ARABIC "
doc.save(output)
print(output)
