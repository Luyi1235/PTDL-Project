from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

font_path = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15.5, 12), dpi=220)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.axis("off")
navy, blue, teal, ink = "#203D70", "#456EA5", "#3F8B83", "#202D3A"
border, pale = "#B7C6D4", "#F5F8FB"

def box(x, y, w, h, title, lines=(), color=blue, fill=pale, title_h=3.8):
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7",
                           linewidth=1.25, edgecolor=border, facecolor=fill)
    ax.add_patch(panel)
    ax.add_patch(FancyBboxPatch((x, y+h-title_h), w, title_h,
                 boxstyle="round,pad=0.15,rounding_size=0.65",
                 linewidth=0, facecolor=color))
    ax.text(x+w/2, y+h-title_h/2, title, color="white", fontsize=12.5,
            ha="center", va="center", fontweight="bold")
    if lines:
        gap = (h-title_h-1.1)/len(lines)
        for i, line in enumerate(lines):
            ax.text(x+1.5, y+h-title_h-0.7-(i+0.5)*gap, line, color=ink,
                    fontsize=11.1, ha="left", va="center")

def arrow(points, color=blue, lw=1.7, z=4):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z))
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                 mutation_scale=12, color=color, lw=lw, zorder=z))

ax.text(50, 100, "NSGA-II / MOPSO 混合算法求解流程", ha="center",
        va="top", fontsize=18, color=navy, fontweight="bold")
box(25, 87.5, 50, 8, "初始化种群与状态",
    ["生成设计参数，评价目标与约束；初始化速度、个体历史及档案"], navy)
box(27, 76, 46, 7.5, "确定本代候选生成比例",
    ["按 q 分配遗传与粒子候选数；计算排序层次和拥挤距离"], navy)
box(7, 51.5, 40, 20.5, "遗传支路：交叉与变异生成候选", [
    "① 依据排序层次和拥挤距离锦标赛选父代",
    "② 模拟二进制交叉，重组设计参数",
    "③ 逐维随机变异，截断至参数范围",
    "④ 子代速度置零，历史位置设为自身"], blue)
box(53, 51.5, 40, 20.5, "粒子支路：生成位置更新候选", [
    "① 从种群与外部档案构成的集合选引导方案",
    "② 结合当前速度、个体历史和引导方案",
    "③ 更新并限幅速度，截断粒子位置",
    "④ 记录候选位置和限幅后的速度"], teal)
box(25, 39, 50, 8, "统一飞推控任务评价",
    ["计算三个优化目标及约束违背程度，形成子代评价结果"], navy)
box(7, 24.5, 40, 10, "种群环境选择", [
    "评价后更新粒子历史，合并父代与两类子代",
    "按拥挤距离截取，保留位置、速度和个体历史"], blue)
box(53, 24.5, 40, 10, "外部档案更新", [
    "合并新子代，筛选可行非支配方案并去重",
    "超过容量时按拥挤距离保留；无可行解时取最小违约"], teal)
box(27, 11, 46, 8, "计算有效改进并更新比例",
    ["分别比较子代与对照方案，按两支路改进率调整 q"], navy)
ax.text(50, 3.5, "达到终止代数：输出候选方案与可行非支配解集",
        ha="center", va="center", fontsize=12.5, color=navy, fontweight="bold")

arrow([(50,87.5),(50,83.5)])
arrow([(38,76),(38,74),(27,74),(27,72)])
arrow([(62,76),(62,74),(73,74),(73,72)], teal)
arrow([(27,51.5),(27,49),(36,49),(36,47)])
arrow([(73,51.5),(73,49),(64,49),(64,47)], teal)
arrow([(36,39),(36,37),(27,37),(27,34.5)])
arrow([(64,39),(64,37),(73,37),(73,34.5)], teal)
arrow([(50,39),(50,19)], "#777D88", 1.4)
arrow([(27,24.5),(27,22),(12,22),(12,15),(3,15)], "#777D88", 1.4)
arrow([(27,15),(3,15),(3,79.5),(27,79.5)], "#777D88", 1.4)
ax.text(3.8, 46, "下一代", rotation=90, fontsize=10.5, color="#626B76", va="center")
arrow([(93,29.5),(97,29.5),(97,64),(93,64)], teal, 1.5)
ax.text(97.7, 43, "档案引导", rotation=90, fontsize=10.5, color=teal, va="center")
arrow([(50,11),(50,6.5)], navy)
fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
out = Path(r"C:\Users\pc\Desktop\混合算法原理框图.png")
fig.savefig(out, dpi=220, facecolor="white")
print(out)
