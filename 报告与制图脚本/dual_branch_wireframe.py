from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from docx import Document
from docx.shared import Cm

source = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_线框图版.docx")
output = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_双支路线框图版.docx")
image = Path(r"C:\Users\pc\Desktop\多目标自适应混合算法双支路线框图.png")

regular = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
bold = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
fig, ax = plt.subplots(figsize=(12, 9.2), dpi=240)
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")


def box(x, y, w, h, title, description=""):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#fafafa",
                           edgecolor="#292929", linewidth=1.4))
    if description:
        ax.text(x+w/2, y+h*.69, title, ha="center", va="center",
                fontproperties=bold, fontsize=16.5, color="#1c1c1c")
        ax.text(x+w/2, y+h*.30, description, ha="center", va="center",
                fontproperties=regular, fontsize=13.3, linespacing=1.25, color="#1c1c1c")
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center",
                fontproperties=regular, fontsize=16.5, color="#1c1c1c")


def arrow(start, end, color="#292929"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                linewidth=1.4, color=color))


box(.34, .905, .32, .065, "初始化种群与状态")
box(.34, .775, .32, .080, "按比例分配候选数量")
box(.09, .565, .37, .135, "遗传搜索", "父代选择、交叉与变异\n产生遗传候选方案")
box(.54, .565, .37, .135, "粒子更新", "历史位置与档案引导\n更新粒子速度和位置")
box(.29, .420, .42, .080, "飞推控任务联合评价", "计算多目标与任务约束")
box(.29, .275, .42, .080, "种群选择与档案更新", "保留下一代及可行非支配方案")
box(.29, .130, .42, .080, "改进率反馈与比例调整", "比较两支路改进效果，更新候选比例")
arrow((.50, .902), (.50, .858))
arrow((.42, .772), (.275, .705))
arrow((.58, .772), (.725, .705))
arrow((.275, .562), (.41, .505))
arrow((.725, .562), (.59, .505))
arrow((.50, .417), (.50, .358))
arrow((.50, .272), (.50, .213))

# Only candidate allocation repeats; initialization occurs once.
ax.plot([.29, .04, .04, .335], [.17, .17, .815, .815], color="#292929", lw=1.4)
arrow((.305, .815), (.335, .815))
ax.text(.052, .49, "未达到终止条件", fontproperties=regular, fontsize=12.5,
        rotation=90, ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=2))
arrow((.50, .127), (.50, .058))
ax.text(.515, .075, "达到终止条件，输出可行解集",
        fontproperties=regular, fontsize=13, va="center")
fig.savefig(image, bbox_inches="tight", pad_inches=.06, facecolor="white")
plt.close(fig)

doc = Document(source)
assert len(doc.inline_shapes) == 1
shape = doc.inline_shapes[0]
rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
doc.part.related_parts[rid]._blob = image.read_bytes()
with Image.open(image) as im:
    ratio = im.height / im.width
shape.width = Cm(13.3)
shape.height = int(shape.width * ratio)
doc.save(output)
print(output)
print(image)
