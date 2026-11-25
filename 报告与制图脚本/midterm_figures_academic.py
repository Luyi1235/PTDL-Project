"""Redraw the report's data figures in a restrained, consistent journal style."""
from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.text import Text

spec = importlib.util.spec_from_file_location("report_figures", r"D:\LDTP\midterm_figures_v3.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)

figures.ROOT = Path(r"C:\Users\pc\Desktop\PTDLfigure\总报告3.2与4.2_学术配色版")
figures.INK = "#252525"
figures.BLUE = "#7188d8"
figures.TEAL = "#e38170"
figures.ORANGE = "#c85f81"
figures.RED = "#80749f"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "Times New Roman"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.frameon": False,
    "svg.fonttype": "none",
})

original_imshow = Axes.imshow
original_scatter = Axes.scatter
soft_blue = colors.LinearSegmentedColormap.from_list(
    "paper_blue", ["#f5f6fb", "#c3cee9", "#7f98cb", "#4269a5"]
)
blue_rose = colors.LinearSegmentedColormap.from_list(
    "paper_blue_rose", ["#506ca9", "#b5c4e2", "#f1edf0", "#e9a8b6", "#c75e7a"]
)


def paper_imshow(self, *args, **kwargs):
    if kwargs.get("cmap") == "YlGnBu":
        kwargs["cmap"] = soft_blue
    elif isinstance(kwargs.get("cmap"), colors.ListedColormap):
        kwargs["cmap"] = colors.ListedColormap(["#aab9e1"])
    return original_imshow(self, *args, **kwargs)


def paper_scatter(self, *args, **kwargs):
    if kwargs.get("cmap") == "viridis":
        kwargs["cmap"] = blue_rose
    return original_scatter(self, *args, **kwargs)


Axes.imshow = paper_imshow
Axes.scatter = paper_scatter


def header(fig, title, subtitle):
    fig.text(.065, .975, title, fontsize=13, color="#202020", va="top")


def footer(fig, text):
    # Figure-level provenance is retained in the accompanying index.
    return None


def style(ax, grid="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#e3e3e3", lw=.45, alpha=.6)
    ax.tick_params(direction="in", length=3, width=.65, pad=4, labelsize=10,
                   colors="#222222", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(.7)


base_save = figures.save


def save(fig, section, name, source, detail, derived=True):
    # Use more of the canvas for the actual measurements and comparison plots.
    if name == "数据_02_四算法逐代收敛与单次分布":
        lines = fig.axes[0].lines
        for line, linestyle, marker in zip(lines, ["-", "--", "-.", ":"], ["o", "s", "^", "D"]):
            line.set_linestyle(linestyle)
            line.set_marker(marker)
            line.set_markevery(3)
            line.set_markersize(4)
            line.set_linewidth(1.5)
        fig.axes[0].legend(frameon=False, fontsize=10)
    for ax in fig.axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, .11 + (pos.y0-.13)*1.13, pos.width, pos.height*1.13])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(.65)
        ax.tick_params(direction="in", length=3, width=.6, top=True, right=True,
                       colors="#252525")
        for patch in ax.patches:
            if patch.get_facecolor()[-1] > 0:
                patch.set_edgecolor("#555555")
                patch.set_linewidth(.4)
    for text in fig.findobj(match=Text):
        size = text.get_fontsize()
        if size > 13:
            text.set_fontsize(12 if size >= 15 else 11.5)
        text.set_fontweight("normal")
        if text.get_color() in ("#263b49", "#647783"):
            text.set_color("#252525")
    base_save(fig, section, name, source, detail, derived)


figures.header = header
figures.footer = footer
figures.style = style
figures.save = save

if __name__ == "__main__":
    figures.extract_report()
    figures.layout_and_cases()
    figures.cruise()
    figures.case_heatmaps()
    figures.candidate_figures()
    figures.algorithm_results()
    figures.index()
    print("总图数", len(figures.ALL))
    for section in figures.SECTIONS:
        print(section, sum(item[0] == section for item in figures.ALL))
