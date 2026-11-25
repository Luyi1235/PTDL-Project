from pathlib import Path
import re
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

path = Path(r"C:\Users\pc\Desktop\【V3】附件3-技术进展报告总_交叉引用宋体修正版.docx")
doc = Document(path)
current = ""
caption = ""
wanted = False
output = []
for block in doc.iter_inner_content():
    if isinstance(block, Paragraph):
        text = block.text.strip()
        if block.style.name == "GF报告二级标题":
            wanted = bool(re.match(r"^[34]\.2\s", text))
        if not wanted:
            continue
        if re.match(r"^[34]\.2\.[1-4]\s", text):
            current = text
            output.append("\n### " + current)
        if re.match(r"^表\s*\d+", text):
            caption = text
        ids = block._p.xpath(".//a:blip/@r:embed")
        if ids:
            names = [str(doc.part.related_parts[i].partname) for i in ids]
            output.append("IMAGE " + " ".join(names))
        if text:
            output.append("P " + text)
    elif wanted and isinstance(block, Table):
        output.append("TABLE " + caption)
        for row in block.rows:
            output.append(" | ".join(c.text.replace("\n", " / ") for c in row.cells))
        caption = ""
Path(r"D:\LDTP\midterm_sections_inspection.txt").write_text("\n".join(output), encoding="utf-8")
print("wrote inspection with", len(output), "lines")
