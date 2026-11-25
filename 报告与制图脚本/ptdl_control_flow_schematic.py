from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for file in ("msyh.ttc", "msyhbd.ttc"):
    font_manager.fontManager.addfont(str(Path(r"C:\Windows\Fonts") / file))
plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.weight": "bold",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(18, 10))
ax.set(xlim=(0, 100), ylim=(0, 100))
ax.axis("off")
blue = "#163d7a"
line = "#326da3"
feedback = "#248875"


def box(x, y, w, h, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor="white", edgecolor=blue, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 5), w, 5,
                               boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor=blue, edgecolor=blue))
    ax.text(x + w/2, y + h - 2.5, title, color="white", size=18,
            ha="center", va="center")
    if lines:
        ax.text(x + w/2, y + (h-5)/2, "\n".join(lines), color="#172f49",
                size=14.5, ha="center", va="center", linespacing=1.65)


def arrow(points, color=line):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2.5)
    start, end = points[-2], points[-1]
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20,
                                 lw=2.5, color=color))


def label(x, y, text, color=line, size=13):
    ax.text(x, y, text, color=color, size=size, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))


box(24, 88, 52, 10, "任务指令", ["飞行阶段　｜　高度、速度与航迹指令"])
box(7, 62, 35, 19, "飞行控制", ["高度、速度、航迹与姿态控制", "计算操纵指令与推进功率需求"])
box(58, 62, 35, 19, "能量管理", ["根据任务阶段、功率需求与能源状态", "计算发动机与电池功率分配指令"])
box(7, 18, 35, 19, "飞行动力学与状态更新", ["推进力、气动力与操纵作用", "更新速度、高度、位置和姿态"])
box(58, 18, 35, 19, "混合动力系统与螺旋桨", ["发动机、电池、电动机及螺旋桨", "更新部件输出、燃油与电池状态"])

arrow([(36, 88), (36, 84), (24, 84), (24, 81)])
arrow([(64, 88), (64, 84), (76, 84), (76, 81)])
arrow([(42, 72), (58, 72)])
label(50, 74.2, "推进功率需求", size=12)
arrow([(17, 62), (17, 37)])
label(17, 49.5, "操纵指令")
arrow([(34, 37), (34, 62)], feedback)
label(34, 49.5, "飞行状态反馈", feedback)
arrow([(68, 62), (68, 37)])
label(68, 49.5, "功率分配指令")
arrow([(85, 37), (85, 62)], feedback)
label(85, 49.5, "能源与部件状态反馈", feedback, 12)
arrow([(58, 27), (42, 27)])
label(50, 29.5, "螺旋桨推力", size=12)

ax.text(50, 8, "蓝色：指令与作用　　绿色：状态反馈", size=14,
        ha="center", va="center", color="#4e6175")
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)
for ext in ("png", "svg"):
    fig.savefig(Path(r"C:\Users\pc\Desktop") / f"PTDL_分层协同控制_箭头修改示意.{ext}",
                dpi=240, bbox_inches="tight", pad_inches=0.12, facecolor="white")
