from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from docx import Document
from docx.shared import Cm

source = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_图文融合版.docx")
output = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_线框图版.docx")
image = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法线框示意图.png")
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
fig, ax = plt.subplots(figsize=(14, 7.5), dpi=240)
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")


def box(x, y, title, text):
    w, h = .235, .23
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#fafafa",
                           edgecolor="#252525", linewidth=1.5))
    ax.text(x + w/2, y + h*.69, title, ha="center", va="center",
            fontproperties=font, fontsize=20, color="#191919")
    ax.text(x + w/2, y + h*.30, text, ha="center", va="center",
            fontproperties=font, fontsize=15.7, color="#191919", linespacing=1.35)


def arrow(start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                linewidth=1.5, color="#252525"))


box(.08, .62, "初始化", "设计变量与种群状态")
box(.385, .62, "候选方案生成", "遗传重组与变异\n粒子速度与位置更新")
box(.69, .62, "任务联合评价", "能耗、跟踪与响应\n任务约束")
box(.69, .25, "种群与档案更新", "选择下一代种群\n保留可行非支配方案")
box(.385, .25, "自适应比例调整", "统计两支路有效改进\n更新候选生成比例")
box(.08, .25, "终止判断", "检查迭代终止条件")

arrow((.315, .735), (.383, .735))
arrow((.62, .735), (.688, .735))
arrow((.8075, .615), (.8075, .485))
arrow((.688, .365), (.622, .365))
arrow((.383, .365), (.317, .365))

# The loop returns to candidate generation, while initialization occurs once.
ax.plot([.08, .033, .033, .5025], [.365, .365, .955, .955], color="#252525", lw=1.5)
arrow((.5025, .955), (.5025, .855))
ax.text(.048, .57, "未终止", fontproperties=font, fontsize=14, rotation=90,
        ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=2))
arrow((.1975, .245), (.1975, .12))
ax.text(.215, .14, "满足终止条件，输出可行非支配解集",
        fontproperties=font, fontsize=15, ha="left", va="center")
fig.savefig(image, bbox_inches="tight", pad_inches=.06, facecolor="white")
plt.close(fig)

doc = Document(source)
assert len(doc.inline_shapes) == 1
shape = doc.inline_shapes[0]
rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
doc.part.related_parts[rid]._blob = image.read_bytes()
with Image.open(image) as im:
    ratio = im.height / im.width
shape.width = Cm(15.1)
shape.height = int(shape.width * ratio)
doc.save(output)
print(output)
print(image)
