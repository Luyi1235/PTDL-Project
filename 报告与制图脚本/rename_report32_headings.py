import re
from docx import Document

source = r"C:\Users\pc\Desktop\3.2节报告.docx"
destination = r"C:\Users\pc\Desktop\3.2节报告_标题编号修改版.docx"
doc = Document(source)
count = 0
for paragraph in doc.paragraphs:
    if not re.match(r"^2(?:\.\d+)*[ \u3000]", paragraph.text):
        continue
    text_elements = paragraph._p.xpath(".//w:t")
    if not text_elements or not text_elements[0].text.startswith("2"):
        raise ValueError(f"Unexpected heading structure: {paragraph.text}")
    text_elements[0].text = "3.2" + text_elements[0].text[1:]
    count += 1
assert count == 13, count
doc.save(destination)
print(f"Updated {count} headings: {destination}")
