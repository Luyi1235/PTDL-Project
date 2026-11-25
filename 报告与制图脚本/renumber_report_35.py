from pathlib import Path
from zipfile import ZipFile
import re
from lxml import etree

source = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容.docx")
destination = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_章节编号统一.docx")
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def replace_range(nodes, start, end, replacement):
    positions = []
    offset = 0
    for node in nodes:
        text = node.text or ""
        positions.append((offset, offset + len(text)))
        offset += len(text)
    selected = [i for i, (a, b) in enumerate(positions) if a < end and b > start]
    if not selected:
        raise ValueError(f"Cannot locate text at {start}:{end}")
    first, last = selected[0], selected[-1]
    first_offset = start - positions[first][0]
    last_offset = end - positions[last][0]
    prefix = (nodes[first].text or "")[:first_offset]
    suffix = (nodes[last].text or "")[last_offset:]
    nodes[first].text = prefix + replacement + (suffix if first == last else "")
    for i in range(first + 1, last):
        nodes[i].text = ""
    if first != last:
        nodes[last].text = suffix


with ZipFile(source, "r") as zin:
    xml = zin.read("word/document.xml")
    root = etree.fromstring(xml)
    changes = []
    for paragraph in root.xpath(".//w:p", namespaces=ns):
        nodes = paragraph.xpath(".//w:t", namespaces=ns)
        text = "".join(n.text or "" for n in nodes)
        replacements = []
        replacements.extend((m.start(), m.end(), "3.5") for m in
                            re.finditer(r"(?<![\d.])4\.2(?=\.|\s|节|$)", text))
        replacements.extend((m.start(), m.end(), "3.5." + m.group(1)) for m in
                            re.finditer(r"(?<![\d.])3\.2\.([123])(?=节|建立|发展|\.|\s|$)", text))
        if text.startswith("仿真分析平台按"):
            for m in re.finditer(r"(?<![\d.])2\.([12])节", text):
                replacements.append((m.start(), m.end(), "3.5." + m.group(1) + "节"))
        for start, end, replacement in sorted(replacements, reverse=True):
            replace_range(nodes, start, end, replacement)
        if replacements:
            changes.append((text, "".join(n.text or "" for n in nodes)))
    new_xml = etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
    with ZipFile(destination, "w") as zout:
        for entry in zin.infolist():
            zout.writestr(entry, new_xml if entry.filename == "word/document.xml"
                          else zin.read(entry.filename))
for before, after in changes:
    print(before[:125], "=>", after[:125])
print("Changed paragraphs:", len(changes))
print("Output:", destination)
