from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from lxml import etree
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

src = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_最新版.docx")
dest = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_公式示意图版.docx")
figure_path = Path(r"D:\LDTP\key_tech_algorithm_flow.png")
font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
boldfont = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")
fig, ax = plt.subplots(figsize=(15, 9.6), dpi=210)
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")


def box(x, y, w, h, title, detail, color="#163a78"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.006",
                                fc="#f5f9fc", ec=color, lw=1.3))
    ax.add_patch(FancyBboxPatch((x, y + h * .59), w, h * .41,
                                boxstyle="round,pad=0.004,rounding_size=0.006",
                                fc=color, ec=color, lw=1))
    ax.text(x+w/2, y+h*.805, title, ha="center", va="center", fontproperties=boldfont,
            fontsize=17, color="white")
    ax.text(x+w/2, y+h*.295, detail, ha="center", va="center", fontproperties=font,
            fontsize=12.5, color="#182333", linespacing=1.16)


def arrow(start, end, color="#254c83", style="-|>", lw=1.8, connection="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                                color=color, linewidth=lw, connectionstyle=connection))


box(.32, .885, .36, .09, "初始化种群与任务状态", "设计参数、粒子状态与初始评价")
box(.32, .747, .36, .09, "确定候选生成比例", "按当前比例分配遗传与粒子候选数量")
box(.065, .565, .39, .125, "遗传搜索", "非支配层次与拥挤距离选择父代\n交叉、变异生成遗传候选", "#145b98")
box(.545, .565, .39, .125, "粒子更新", "利用速度、历史位置和档案引导\n更新粒子位置并限制取值范围", "#18845b")
box(.265, .395, .47, .105, "统一飞推控任务评价", "计算任务目标并检查约束")
box(.065, .225, .39, .105, "种群环境选择", "合并父代与子代\n按排序和拥挤距离选择", "#145b98")
box(.545, .225, .39, .105, "外部档案更新", "保留可行非支配方案\n维护解集分布", "#18845b")
box(.32, .075, .36, .095, "计算改进率并调整比例", "比较两支路有效改进，更新下一代比例")
arrow((.50, .882), (.50, .84))
arrow((.40, .745), (.26, .695))
arrow((.60, .745), (.74, .695))
arrow((.26, .56), (.39, .505))
arrow((.74, .56), (.61, .505))
arrow((.39, .392), (.26, .335))
arrow((.61, .392), (.74, .335))
arrow((.26, .22), (.40, .175))
arrow((.74, .22), (.60, .175))
arrow((.50, .075), (.50, .022))
ax.text(.515, .037, "满足终止条件，输出候选方案与可行解集", fontproperties=font,
        fontsize=13, va="center", color="#263b52")
ax.plot([.32, .025, .025, .315], [.12, .12, .79, .79], color="#254c83", lw=1.7)
arrow((.285, .79), (.315, .79))
ax.text(.035, .48, "未终止，进入下一代", rotation=90, ha="center", va="center",
        fontproperties=font, fontsize=13, color="#254c83",
        bbox=dict(facecolor="white", edgecolor="none", pad=2))
ax.plot([.935, .975, .975], [.275, .275, .625], color="#18845b", lw=1.5, ls="--")
arrow((.975, .625), (.94, .625), color="#18845b", lw=1.5)
ax.text(.972, .45, "档案引导", rotation=90, ha="center", va="center",
        fontproperties=font, fontsize=13, color="#18845b",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))
fig.savefig(figure_path, bbox_inches="tight", pad_inches=.06, facecolor="white")
plt.close(fig)

doc = Document(src)
paragraphs = doc.paragraphs
assert paragraphs[5].text == "关键公式+技术图"
placeholder = paragraphs[5]
placeholder._element.getparent().remove(placeholder._element)
if not paragraphs[4].text.strip():
    paragraphs[4]._element.getparent().remove(paragraphs[4]._element)

reference = doc.paragraphs[3]
reference.add_run(" 算法总体流程如")
f = OxmlElement("w:fldSimple")
f.set(qn("w:instr"), " REF _AlgoOverview \\h ")
rr = OxmlElement("w:r")
tt = OxmlElement("w:t")
tt.text = "图1"
rr.append(tt)
f.append(rr)
reference._p.append(f)
reference.add_run("所示。")

intro = doc.add_paragraph()
intro.paragraph_format.first_line_indent = Pt(24)
intro.add_run("联合评价以设计变量为输入，统一计算任务目标及约束，其优化问题可表示为")

math_xsl = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))


def add_math(content, number):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    root = etree.fromstring(('<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
                            + content + '</mrow></math>').encode("utf-8"))
    result = math_xsl(root).getroot()
    # MML2OMML can return either a display container or an inline math object.
    if result.tag.endswith("}oMathPara"):
        item = result.find(qn("m:oMath"))
        p._p.append(deepcopy(item))
    else:
        p._p.append(deepcopy(result))
    p.add_run("    （%s）" % number)


add_math(
    '<munder><mi>min</mi><mi>x</mi></munder><mrow><mo>[</mo>'
    '<msub><mi>J</mi><mi>e</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo>'
    '<msub><mi>J</mi><mi>s</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo>'
    '<msub><mi>J</mi><mi>r</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><msup><mo>]</mo><mi>T</mi></msup>'
    '<mo>,</mo><mtext> s.t. </mtext><msub><mi>g</mi><mi>j</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo>'
    '<mo>≤</mo><mn>0</mn><mo>,</mo><msup><mi>x</mi><mi>L</mi></msup><mo>≤</mo><mi>x</mi>'
    '<mo>≤</mo><msup><mi>x</mi><mi>U</mi></msup></mrow>', "1")

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(24)
p.add_run("式中，x为设计变量；Jₑ、Jₛ和Jᵣ分别为任务能源消耗、飞行跟踪误差及响应时间指标；"
          "gⱼ为第j项约束函数，xᴸ和xᵁ分别为设计变量的下限和上限。")

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(24)
p.add_run("粒子支路利用自身历史位置和引导方案更新速度与位置，其基本关系为")
add_math(
    '<msubsup><mi>v</mi><mi>i</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msubsup>'
    '<mo>=</mo><mi>ω</mi><msubsup><mi>v</mi><mi>i</mi><mi>t</mi></msubsup><mo>+</mo>'
    '<msub><mi>c</mi><mn>1</mn></msub><msub><mi>r</mi><mn>1</mn></msub>'
    '<mo>(</mo><msub><mi>p</mi><mi>i</mi></msub><mo>−</mo><msubsup><mi>x</mi><mi>i</mi><mi>t</mi></msubsup><mo>)</mo>'
    '<mo>+</mo><msub><mi>c</mi><mn>2</mn></msub><msub><mi>r</mi><mn>2</mn></msub>'
    '<mo>(</mo><mi>l</mi><mo>−</mo><msubsup><mi>x</mi><mi>i</mi><mi>t</mi></msubsup><mo>)</mo>'
    '<mo>,</mo><mspace width="1em"/>'
    '<msubsup><mi>x</mi><mi>i</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msubsup>'
    '<mo>=</mo><msubsup><mi>x</mi><mi>i</mi><mi>t</mi></msubsup><mo>+</mo>'
    '<msubsup><mi>v</mi><mi>i</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msubsup>', "2")
p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(24)
p.add_run("式中，vᵢ和xᵢ为第i个粒子的速度和位置，pᵢ为其历史位置，l为从当前种群和外部"
          "档案构成的引导集合中选择的方案；ω、c₁和c₂为搜索参数，r₁和r₂为随机数。"
          "实际更新时对速度和位置进行限幅。")

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(24)
p.add_run("两条支路分别统计候选方案相对参考方案的有效改进情况，并按改进率调整遗传候选比例")
add_math(
    '<msubsup><mi>R</mi><mi>k</mi><mi>t</mi></msubsup><mo>=</mo>'
    '<mfrac><mn>1</mn><msubsup><mi>N</mi><mi>k</mi><mi>t</mi></msubsup></mfrac>'
    '<munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow>'
    '<msubsup><mi>N</mi><mi>k</mi><mi>t</mi></msubsup></munderover>'
    '<msubsup><mi>δ</mi><mrow><mi>k</mi><mo>,</mo><mi>i</mi></mrow><mi>t</mi></msubsup>'
    '<mo>,</mo><mspace width="1em"/><msup><mi>q</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msup>'
    '<mo>=</mo><mi>clip</mi><mo>(</mo><msup><mi>q</mi><mi>t</mi></msup><mo>+</mo><mi>η</mi>'
    '<mo>(</mo><msubsup><mi>R</mi><mi>G</mi><mi>t</mi></msubsup><mo>−</mo>'
    '<msubsup><mi>R</mi><mi>P</mi><mi>t</mi></msubsup><mo>)</mo><mo>,</mo>'
    '<msub><mi>q</mi><mi>min</mi></msub><mo>,</mo><msub><mi>q</mi><mi>max</mi></msub><mo>)</mo>', "3")
p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(24)
p.add_run("式中，k表示遗传或粒子支路，Nₖ为该支路当代候选数；δₖ,ᵢ为有效改进指示量，"
          "当候选由不可行转为可行、双方均不可行且候选约束违反程度降低，或双方均可行且"
          "候选支配参考方案时取1，其余情况取0。遗传候选分别与两个父代比较，取两个"
          "指示量的较大值；粒子候选与更新前粒子比较。q为遗传候选比例，η为调整系数，"
          "clip表示将比例限制在预设范围内，其余候选由粒子支路生成。")

picture = doc.add_paragraph()
picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
picture.paragraph_format.first_line_indent = Pt(0)
picture.paragraph_format.space_before = Pt(5)
picture.paragraph_format.keep_with_next = True
picture.add_run().add_picture(str(figure_path), width=Cm(15.1))
caption = doc.add_paragraph()
caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
caption.paragraph_format.first_line_indent = Pt(0)
start = OxmlElement("w:bookmarkStart")
start.set(qn("w:id"), "1234")
start.set(qn("w:name"), "_AlgoOverview")
caption._p.append(start)
caption.add_run("图")
seq = OxmlElement("w:fldSimple")
seq.set(qn("w:instr"), " SEQ Figure \\* ARABIC ")
r = OxmlElement("w:r")
t = OxmlElement("w:t")
t.text = "1"
r.append(t)
seq.append(r)
caption._p.append(seq)
end = OxmlElement("w:bookmarkEnd")
end.set(qn("w:id"), "1234")
caption._p.append(end)
caption.add_run("  飞推控多目标自适应混合算法总体流程")
doc.save(dest)
print(dest)
