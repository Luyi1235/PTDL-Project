from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

source = Path(r"C:\Users\pc\Desktop\报告3.5节 最终修改版V2_公式格式修正版.docx")
out = Path(r"C:\Users\pc\Desktop\报告3.5节 最终修改版V2_表102优化问题定义版.docx")
doc = Document(source)
old = doc.tables[0]
assert "设计向量" in old.cell(1, 1).text

table = doc.add_table(rows=0, cols=3)
table.autofit = False
table.alignment = old.alignment
grid = table._tbl.tblGrid.gridCol_lst
for col, width in zip(grid, (Cm(5.0), Cm(9.2), Cm(1.7))):
    col.set(qn("w:w"), str(width.twips))
pr = table._tbl.tblPr
for element in pr.xpath("./w:tblBorders"):
    pr.remove(element)
borders = OxmlElement("w:tblBorders")
for name in ("top", "bottom", "left", "right", "insideH", "insideV"):
    edge = OxmlElement("w:" + name)
    edge.set(qn("w:val"), "double" if name in ("top", "bottom") else "nil")
    if name in ("top", "bottom"):
        edge.set(qn("w:sz"), "8")
        edge.set(qn("w:color"), "000000")
    borders.append(edge)
pr.append(borders)


def border_row(row, name="top"):
    for cell in row.cells:
        tcpr = cell._tc.get_or_add_tcPr()
        cell_borders = tcpr.find(qn("w:tcBorders"))
        if cell_borders is None:
            cell_borders = OxmlElement("w:tcBorders")
            tcpr.append(cell_borders)
        edge = OxmlElement("w:" + name)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "5")
        edge.set(qn("w:color"), "000000")
        cell_borders.append(edge)


def add_row(first, description="", quantity="", section=False, header=False):
    row = table.add_row()
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    if header:
        row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    if section:
        border_row(row)
    for cell, width in zip(row.cells, (Cm(5.0), Cm(9.2), Cm(1.7))):
        cell.width = width
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.12
        p.paragraph_format.first_line_indent = Pt(0)
    for col, content in enumerate((first, description, quantity)):
        p = row.cells[col].paragraphs[0]
        if col == 2:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        chunks = content if isinstance(content, list) else [(content, None)]
        for text, position in chunks:
            r = p.add_run(text)
            r.font.name = "Times New Roman"
            r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
            r.font.size = Pt(10.5)
            r.bold = header or section
            r.italic = section
            if position == "sub":
                r.font.subscript = True
            elif position == "sup":
                r.font.superscript = True
    return row


add_row("定义", "说明", "数量", header=True)
add_row("Minimize", section=True)
add_row([("J", None), ("e", "sub"), ("(x)", None)], "任务能量消耗", "1")
add_row([("J", None), ("s", "sub"), ("(x)", None)], "飞行跟踪误差", "1")
add_row([("J", None), ("r", "sub"), ("(x)", None)], "响应时间指标", "1")
add_row("With regard to", section=True)
add_row([("x", None), ("g", "sub")], "机翼、机身及螺旋桨几何参数", "8")
add_row([("x", None), ("m", "sub")], "载荷质量与混合动力系统质量", "2")
add_row([("v", None), ("c", "sub")], "巡航速度", "1")
add_row([("P", None), ("e,i", "sub")], "各任务阶段发动机功率上限", "6")
add_row([("P", None), ("b,i", "sub")], "各任务阶段电池辅助功率", "6")
add_row("", "设计参数总数", "23")
subject_start = len(table.rows)
add_row("Subject to", section=True)
add_row([("x", None), ("L", "sup"), (" ≤ x ≤ x", None), ("U", "sup")], "设计参数取值范围", "23")
add_row([("g", None), ("T", "sub"), ("(x) ≤ 0", None)], "任务续航时间约束", "1")
add_row([("g", None), ("E,j", "sub"), ("(x) ≤ 0", None)], "电池电量与燃油储备约束", "3")
add_row([("g", None), ("F,j", "sub"), ("(x) ≤ 0", None)], "速度、高度、迎角、过载与失速约束", "8")
add_row([("g", None), ("P,j", "sub"), ("(x) ≤ 0", None)], "功率、电流、功率变化及下降供能约束", "10")
add_row([("g", None), ("G,j", "sub"), ("(x) ≤ 0", None)], "几何尺寸、质量及燃油容积约束", "7")
add_row("", "评价约束小计", "29")
add_row("", "约束总数（含设计参数范围）", "52")
for row in table.rows[subject_start:-1]:
    for cell in row.cells:
        cell.paragraphs[0].paragraph_format.keep_with_next = True

old._tbl.addprevious(table._tbl)
old._tbl.getparent().remove(old._tbl)
note = doc.add_paragraph()
note._p.insert(0, deepcopy(doc.paragraphs[6]._p.pPr))
note.paragraph_format.space_before = Pt(2)
note.paragraph_format.space_after = Pt(2)
note.paragraph_format.first_line_indent = Pt(0)
note.paragraph_format.keep_together = True
note_run = note.add_run("注：数量按本报告算法比较所用配置统计。23个设计参数中含固定参数；设计参数上下界按每组区间计1项，评价约束按单侧不等式计29项。")
note_run.font.size = Pt(9)
table._tbl.addnext(note._p)
doc.save(out)
print(out)
