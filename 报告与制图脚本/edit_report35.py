from copy import deepcopy
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


SOURCE = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_最新修改版.docx")
OUTPUT = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_问题定义与算法完善版.docx")
IMAGE = Path(r"C:\Users\pc\Desktop\MOEA算法配图.jpg")
d = Document(SOURCE)
p = d.paragraphs
body_style = deepcopy(p[6]._p.pPr)
caption_style = deepcopy(p[8]._p.pPr)


def fresh_para(properties=None):
    el = OxmlElement("w:p")
    if properties is not None:
        el.append(deepcopy(properties))
    return el


def run(text, math=False):
    r = OxmlElement("m:r" if math else "w:r")
    t = OxmlElement("m:t" if math else "w:t")
    t.text = text
    if text.startswith(" ") or text.endswith(" "):
        t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def field(parent, instruction, result):
    for tag, content in [("begin", None), ("instr", instruction),
                         ("separate", None), ("result", result), ("end", None)]:
        r = OxmlElement("w:r")
        if tag in ("begin", "separate", "end"):
            x = OxmlElement("w:fldChar")
            x.set(qn("w:fldCharType"), tag)
        elif tag == "instr":
            x = OxmlElement("w:instrText")
            x.set(qn("xml:space"), "preserve")
            x.text = content
        else:
            x = OxmlElement("w:t")
            x.text = content
        r.append(x)
        parent.append(r)


def add_ref(parent, bookmark, number):
    field(parent, " REF " + bookmark + r" \h ", number)


def caption(kind, title, bookmark, ident, first=False):
    el = fresh_para(caption_style)
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(ident))
    start.set(qn("w:name"), bookmark)
    el.append(start)
    el.append(run(kind))
    field(el, " SEQ " + kind + (r" \r 102" if first else "") + r" \* ARABIC ", "102" if first else "0")
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(ident))
    el.append(end)
    el.append(run(" " + title))
    return el


def new_table(headers, rows, widths):
    table = d.add_table(rows=0, cols=len(headers))
    table.autofit = False
    table.style = d.tables[0].style
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "bottom", "insideH"):
        border = OxmlElement("w:" + name)
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "6" if name != "insideH" else "3")
        border.set(qn("w:color"), "000000")
        borders.append(border)
    table._tbl.tblPr.append(borders)
    for n, rowdata in enumerate([headers] + rows):
        cells = table.add_row().cells
        trpr = cells[0]._tc.getparent().get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if n == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for cell, text, width in zip(cells, rowdata, widths):
            cell.width = Cm(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            para = cell.paragraphs[0]
            para.paragraph_format.space_before = Pt(0)
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.line_spacing = 1.15
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER if n == 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = para.add_run(text)
            r.font.name = "宋体"
            r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
            r.font.size = Pt(9)
            r.bold = n == 0
    return table._tbl


# The user's placeholder is removed without rebuilding its cross-reference field.
for r in p[6].runs:
    if r.text in ("XXXXX", "定义如下", " +", "公式", "/", "表"):
        r.text = ""

intro = fresh_para(body_style)
intro.append(run("飞推控协同设计以设计变量、优化目标和约束条件共同定义多目标优化问题。设计变量覆盖气动构型、动力配置与控制参数；任务仿真计算各候选方案的能量消耗、飞行跟踪误差和响应时间，并依据部件能力、能源储备及飞行任务要求判断可行性。优化问题的基本组成见"))
add_ref(intro, "PTDLProblemDefinition", "表102")
intro.append(run("，各目标及约束的具体形式在本节后续内容中给出。"))
p[5]._p.addnext(intro)
cap_problem = caption("表", "飞推控协同优化问题的基本定义", "PTDLProblemDefinition", 18001, True)
intro.addnext(cap_problem)
tab_problem = new_table(
    ["组成", "定义", "主要内容"],
    [
        ["设计变量", "设计向量 x = (x气动, x动力, x控制)，各变量在给定上下界内取值", "气动构型和几何参数、动力部件配置、飞行控制与能量管理参数"],
        ["优化目标", "min F(x) = [Je(x), Js(x), Jr(x)]ᵀ", "分别评价任务能量消耗、飞行跟踪误差与响应时间"],
        ["约束条件", "gj(x) ≤ 0（j = 1, …, m）", "部件功率与电流能力、能源储备、飞行状态及起降与任务完成条件"],
        ["求解结果", "满足约束的非支配方案集合", "给出不同目标之间的性能取舍，供后续任务偏好决策"],
    ], [2.3, 6.6, 7.2])
cap_problem.addnext(tab_problem)

# Move the original seeded table sequence forward after the newly inserted table.
for t in p[8]._p.xpath(".//w:instrText"):
    if t.text and r"\r 102" in t.text:
        t.text = t.text.replace(r"\r 102", "")

# Combine the original editable equations in a left-aligned OMML matrix.
maths = [deepcopy(x._p.xpath("./m:oMath")[0]) for x in p[47:50]]
target = p[47]._p
for x in list(target):
    if x.tag != qn("w:pPr") and x.tag not in (qn("w:commentRangeStart"),):
        target.remove(x)
p[47].alignment = WD_ALIGN_PARAGRAPH.CENTER
if p[47].paragraph_format.first_line_indent is not None:
    p[47].paragraph_format.first_line_indent = Pt(0)
omath = OxmlElement("m:oMath")
delimiter = OxmlElement("m:d")
dp = OxmlElement("m:dPr")
for tag, value in (("m:begChr", "{"), ("m:endChr", ""), ("m:sepChr", "")):
    item = OxmlElement(tag)
    item.set(qn("m:val"), value)
    dp.append(item)
delimiter.append(dp)
inside = OxmlElement("m:e")
matrix = OxmlElement("m:m")
mp = OxmlElement("m:mPr")
mcs = OxmlElement("m:mcs")
for alignment in ("left", "right"):
    mc = OxmlElement("m:mc")
    mcpr = OxmlElement("m:mcPr")
    count = OxmlElement("m:count")
    count.set(qn("m:val"), "1")
    mcpr.append(count)
    jc = OxmlElement("m:mcJc")
    jc.set(qn("m:val"), alignment)
    mcpr.append(jc)
    mc.append(mcpr)
    mcs.append(mc)
mp.append(mcs)
matrix.append(mp)
for num, equation in zip(("（2.5）", "（2.6）", "（2.7）"), maths):
    row = OxmlElement("m:mr")
    expr = OxmlElement("m:e")
    for x in list(equation):
        expr.append(x)
    row.append(expr)
    tag = OxmlElement("m:e")
    tag.append(run(num, True))
    row.append(tag)
    matrix.append(row)
inside.append(matrix)
delimiter.append(inside)
omath.append(delimiter)
target.append(omath)
for node in p[49]._p.xpath("./w:commentRangeEnd | ./w:r[w:commentReference]"):
    target.append(deepcopy(node))
for x in (p[48], p[49]):
    x._p.getparent().remove(x._p)

# First complete name and subsequent short form; keep references intact.
for r in p[63].runs:
    if "并组合遗传重组和粒子更新形成混合算法" in r.text:
        r.text = r.text.replace("并组合遗传重组和粒子更新形成混合算法", "并提出基于改进率的自适应遗传粒子群混合算法，简称改进率自适应混合算法")
for par in p:
    if par is p[63] or par is p[124]:
        continue
    for r in par.runs:
        if "混合算法" in r.text and "改进率自适应混合算法" not in r.text:
            r.text = r.text.replace("混合算法", "改进率自适应混合算法")
for r in p[124].runs:
    if r.text == "4）":
        r.text = "（4）"

algorithm_intro = fresh_para(body_style)
algorithm_intro.append(run("四种算法的搜索机制、主要优势及应用中需要关注的因素见"))
add_ref(algorithm_intro, "PTDLAlgorithmComparison", "表105")
algorithm_intro.append(run("。"))
p[63]._p.addnext(algorithm_intro)
cap_alg = caption("表", "多目标优化算法的搜索特点比较", "PTDLAlgorithmComparison", 18002)
algorithm_intro.addnext(cap_alg)
tab_alg = new_table(
    ["算法", "搜索机制与主要优势", "应用中需要关注的因素"],
    [
        ["NSGA-II", "非支配排序结合拥挤距离和精英保留，兼顾解集收敛与分布。", "排序与拥挤距离计算开销随种群规模和目标数量增加；搜索效果与遗传算子设置有关。"],
        ["MOEA/D", "利用权重将多目标问题分解为子问题，通过邻域协作搜索不同性能取舍。", "权重分布、邻域规模及目标尺度影响前沿覆盖和搜索效率。"],
        ["MOPSO", "依据个体历史与档案引导方案更新粒子，利用群体信息推进搜索。", "引导方案选择、速度限幅和参数设置影响搜索过程与解集分布。"],
        ["改进率自适应混合算法", "遗传支路进行重组与变异，粒子支路利用历史及档案引导，依据改进率调整两支路比例。", "需设置比例更新与档案维护规则，并结合评价预算考察两类搜索操作的作用。"],
    ], [3.1, 6.7, 6.3])
cap_alg.addnext(tab_alg)

fig_explain = fresh_para(body_style)
fig_explain.append(run("MOEA/D的分解思想如"))
add_ref(fig_explain, "PTDLMoeadDiagram", "图113")
fig_explain.append(run("所示。图中以两个目标为例，理想点作为目标空间的参考，不同权重方向对应不同的目标取舍；各子问题协同搜索，形成对Pareto前沿不同区域的覆盖。下文将这一思想应用于任务能量、飞行跟踪误差和响应时间三个目标。"))
p[84]._p.addnext(fig_explain)
image_p = d.add_paragraph()
image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
image_p.paragraph_format.space_before = Pt(3)
image_p.paragraph_format.space_after = Pt(0)
image_p.paragraph_format.keep_with_next = True
image_p.add_run().add_picture(str(IMAGE), width=Cm(12.0))
fig_explain.addnext(image_p._p)
cap_fig = caption("图", "MOEA/D多目标分解及Pareto前沿示意", "PTDLMoeadDiagram", 18003)
image_p._p.addnext(cap_fig)

# Update SEQ and REF field display values while retaining live Word fields.
allpars = d.element.body.xpath(".//w:p")
seq = {"表": 0, "图": 0}
bookmarks = {}
field_records = []
for par in allpars:
    nodes = list(par)
    active = None
    for node in nodes:
        if node.tag != qn("w:r"):
            continue
        chars = node.xpath("./w:fldChar")
        if chars:
            typ = chars[0].get(qn("w:fldCharType"))
            if typ == "begin":
                active = {"instr": [], "results": [], "separated": False}
            elif typ == "separate" and active:
                active["separated"] = True
            elif typ == "end" and active:
                ins = "".join(active["instr"])
                m = re.search(r"\bSEQ\s+([图表])", ins)
                if m:
                    kind = m.group(1)
                    reset = re.search(r"\\r\s+(\d+)", ins)
                    seq[kind] = int(reset.group(1)) if reset else seq[kind] + 1
                    for j, text in enumerate(active["results"]):
                        text.text = str(seq[kind]) if j == 0 else ""
                    for b in par.xpath("./w:bookmarkStart"):
                        bookmarks[b.get(qn("w:name"))] = kind + str(seq[kind])
                else:
                    ref = re.search(r"\bREF\s+(\S+)", ins)
                    if ref:
                        field_records.append((ref.group(1), active["results"]))
                active = None
            continue
        if active:
            if active["separated"]:
                active["results"].extend(node.xpath(".//w:t"))
            else:
                active["instr"].extend(node.xpath(".//w:instrText/text()"))
missing = []
for name, texts in field_records:
    if name in bookmarks:
        for j, text in enumerate(texts):
            text.text = bookmarks[name] if j == 0 else ""
    elif name.startswith(("CodexRef", "PTDL")):
        missing.append(name)
if missing:
    raise RuntimeError("Unresolved figure/table references: " + repr(missing))
d.save(OUTPUT)
print("Saved", OUTPUT)
print("Last sequences", seq, "updated references", len(field_records))
