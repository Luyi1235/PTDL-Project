from copy import deepcopy
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.text.paragraph import Paragraph

ROOT = Path(r"D:\LDTP")
SRC = Path(r"C:\Users\pc\Desktop\3.2节报告_标题编号修改版.docx")
OUT = Path(r"C:\Users\pc\Desktop\3.2节报告_推控方法与交叉引用完善版.docx")
DATA = Path(r"D:\LDTP\LDTP\PTDL飞推控协同优化代码包\运行结果\JSBSim地面起降80h联算_20261012\JSBSim候选\candidate_0001\PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
FIG = ROOT / "report32_mission_supply.png"

data = pd.read_csv(DATA)
assert len(data) == 493 and data.time_s.is_monotonic_increasing
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                     "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9})
fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)
hours = data.time_s / 3600
axs[0].plot(hours, data.altitude_target_m, color="#cf826b", lw=1.4, ls="--", label="高度指令")
axs[0].plot(hours, data.altitude_m, color="#2c668d", lw=1.35, label="实际高度")
axs[0].set_ylabel("高度 / m")
axs[0].legend(loc="upper right", ncol=2, frameon=False)
axs[1].plot(hours, data.engine_power_kw, color="#2c668d", lw=1.35, label="发动机—发电机供电")
axs[1].set_ylabel("供电功率 / kW")
axs[1].legend(loc="upper right", frameon=False)
axs[2].axhline(0, color="#555555", lw=0.7)
axs[2].plot(hours, data.battery_terminal_power_kw, color="#d27662", lw=1.2, label="电池端功率")
axs[2].set_ylabel("电池功率 / kW")
axs[2].set_xlabel("空中任务时间 / h")
axs[2].legend(loc="upper right", frameon=False)
for ax in axs:
    ax.grid(axis="y", alpha=0.23)
    ax.set_xlim(0, 82)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.savefig(FIG, dpi=240, facecolor="white")
plt.close(fig)

doc = Document(SRC)
ps = doc.paragraphs
captions = {}
for p in ps:
    m = re.match(r"^([图表])(\d+) ", p.text)
    if m and p._p.xpath(".//w:instrText[contains(.,'SEQ')]"):
        captions[(m.group(1), int(m.group(2)))] = p
assert len(captions) == 11, captions.keys()

ids = [int(x.get(qn("w:id"))) for x in doc.element.xpath(".//w:bookmarkStart")]
next_id = max(ids, default=0) + 1
bookmark_names = {}
for (kind, number), p in captions.items():
    label_run = next(r for r in p._p.xpath("./w:r") if "".join(r.xpath("./w:t/text()")) == kind)
    seq_end = next(r for r in p._p.xpath("./w:r") if r.xpath("./w:fldChar[@w:fldCharType='end']"))
    name = ("PTDL32Fig" if kind == "图" else "PTDL32Table") + str(number)
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(next_id))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(next_id))
    label_run.addprevious(start)
    seq_end.addnext(end)
    bookmark_names[(kind, number)] = name
    next_id += 1
captions[("图", 15)]._p.xpath(".//w:instrText")[0].text = " SEQ 图 \\r 15 \\* ARABIC "
captions[("表", 28)]._p.xpath(".//w:instrText")[0].text = " SEQ 表 \\r 28 \\* ARABIC "

def field_run(kind, value, template):
    r = OxmlElement("w:r")
    props = template.find(qn("w:rPr"))
    if props is not None:
        r.append(deepcopy(props))
    if kind == "instruction":
        item = OxmlElement("w:instrText")
        item.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        item.text = value
    else:
        item = OxmlElement("w:fldChar")
        item.set(qn("w:fldCharType"), value)
    r.append(item)
    return r

converted = 0
for p in ps:
    if p in captions.values():
        continue
    runs = list(p._p.xpath("./w:r"))
    for prev, curr in zip(runs, runs[1:]):
        prev_t = prev.xpath("./w:t")
        curr_t = curr.xpath("./w:t")
        if len(prev_t) != 1 or len(curr_t) != 1:
            continue
        previous, number = prev_t[0].text or "", curr_t[0].text or ""
        if not previous or previous[-1] not in "图表" or not number.isdigit():
            continue
        key = (previous[-1], int(number))
        if key not in bookmark_names:
            continue
        kind = previous[-1]
        prev_t[0].text = previous[:-1]
        label = deepcopy(curr)
        label.xpath("./w:t")[0].text = kind
        prev.addnext(field_run("field", "begin", curr))
        begin = prev.getnext()
        begin.addnext(field_run("instruction", f" REF {bookmark_names[key]} \\h \\* MERGEFORMAT ", curr))
        instruction = begin.getnext()
        instruction.addnext(field_run("field", "separate", curr))
        instruction.getnext().addnext(label)
        curr.addnext(field_run("field", "end", curr))
        converted += 1

def new_text_after(anchor, text):
    node = deepcopy(ps[4]._p)
    for child in list(node):
        if child.tag != qn("w:pPr"):
            node.remove(child)
    anchor._p.addnext(node)
    paragraph = Paragraph(node, anchor._parent)
    paragraph.add_run(text)
    return paragraph

p = new_text_after(ps[47], "任务剖面将航点、高度和速度组织为随阶段变化的参考序列，并与实际位置、速度和姿态持续比较。轨迹外环根据高度与速度偏差计算推进需求及姿态目标，能量管理依据发动机与电池的可用能力分配供电，动力执行环节将实际功率转换为螺旋桨推力。推力与气动力、重力共同决定飞机的加速与爬升，姿态及航向控制共同调整推力方向和飞行方向，使飞机沿任务给定的轨迹前进。")
new_text_after(p, "在这一闭环中，推进功率的调整直接影响沿航迹速度和高度变化率，进而影响航点到达时刻与阶段转换；轨迹偏差又进入下一控制周期，修正推进需求与能源分配。任务剖面因此同时承担参考轨迹、阶段调度和性能评价的作用，飞行控制与混合动力调度通过实际推力和状态反馈形成连续的协同关系。")
p = new_text_after(ps[60], "混合动力调度以推进需求和能源状态为输入，分别形成发动机—发电机供电指令与电池充放电指令。发动机按响应特性建立母线供电，电池根据当前差额及荷电状态提供辅助或吸收可用余电；两支路的实际功率共同作用于推进电机，功率及能源状态随任务连续更新。控制器将供能变化与速度、高度误差联合评估，使当前动力响应与后续阶段的能源储备相协调。")

lead = new_text_after(ps[151], "同一长航时算例中，图19进一步展示任务高度与两类电源的供电时序。发动机功率采用发电机母线输出口径，电池功率采用端口输出口径，正值表示放电、负值表示充电；两者共同参与推进电机供能。图中曲线依据记录的任务采样点绘制，空中时序采样间隔约10 min，覆盖至82 h。")
img_node = deepcopy(ps[149]._p)
for child in list(img_node):
    if child.tag != qn("w:pPr"):
        img_node.remove(child)
lead._p.addnext(img_node)
img_p = Paragraph(img_node, lead._parent)
img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
img_p.paragraph_format.first_line_indent = Inches(0)
img_p.paragraph_format.keep_with_next = True
img_p.add_run().add_picture(str(FIG), width=Inches(5.6))

caption = deepcopy(captions[("图", 18)]._p)
for b in list(caption.xpath(".//w:bookmarkStart | .//w:bookmarkEnd")):
    b.getparent().remove(b)
for t in caption.xpath(".//w:t"):
    if t.text == "18":
        t.text = "19"
    elif t.text and "长航时任务中的飞行状态与能源变化" in t.text:
        t.text = t.text.replace("长航时任务中的飞行状态与能源变化", "任务剖面与发动机、电池供电变化")
caption.xpath(".//w:instrText")[0].text = " SEQ 图 \\* ARABIC "
label_run = next(r for r in caption.xpath("./w:r") if "".join(r.xpath("./w:t/text()")) == "图")
seq_end = next(r for r in caption.xpath("./w:r") if r.xpath("./w:fldChar[@w:fldCharType='end']"))
start, end = OxmlElement("w:bookmarkStart"), OxmlElement("w:bookmarkEnd")
start.set(qn("w:id"), str(next_id))
start.set(qn("w:name"), "PTDL32FigPower")
end.set(qn("w:id"), str(next_id))
label_run.addprevious(start)
seq_end.addnext(end)
img_node.addnext(caption)

# Insert a Word cross-reference in the newly added paragraph.
run = lead.runs[0]._r
text = run.xpath("./w:t")[0]
before, after = text.text.split("图19", 1)
text.text = before
for elem in [field_run("field", "begin", run),
             field_run("instruction", " REF PTDL32FigPower \\h \\* MERGEFORMAT ", run),
             field_run("field", "separate", run)]:
    run.addnext(elem)
    run = elem
result = deepcopy(run.getprevious().getprevious().getprevious())
result.xpath("./w:t")[0].text = "图19"
run.addnext(result)
result.addnext(field_run("field", "end", result))
last = result.getnext()
tail = deepcopy(result)
tail.xpath("./w:t")[0].text = after
last.addnext(tail)

assert converted >= 20, converted
doc.save(OUT)
print(f"Saved {OUT}; converted {converted} references.")
