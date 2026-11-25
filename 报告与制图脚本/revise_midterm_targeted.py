from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator

spec = importlib.util.spec_from_file_location("report_figures", r"D:\LDTP\midterm_figures_v3.py")
src = importlib.util.module_from_spec(spec)
spec.loader.exec_module(src)
root = Path(r"C:\Users\pc\Desktop\PTDLfigure\总报告3.2与4.2_学术配色版\本次重点修改")
plt.rcParams.update({
    "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
    "font.size": 17, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.linewidth": .85, "svg.fonttype": "none", "savefig.facecolor": "white",
})
ink = "#30343b"
blue, coral, rose, purple = "#7188d8", "#e38170", "#c85f81", "#80749f"


def save(fig, section, name):
    path = root / src.SECTIONS[section] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=210, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axes(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#404040")
        sp.set_linewidth(.8)
    ax.tick_params(direction="in", top=True, right=True, length=4, pad=5)
    ax.set_axisbelow(True)


def arrangement():
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=.5, wspace=.2,
                          left=.05, right=.97, top=.9, bottom=.12)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(-.8, 8.8); ax.set_ylim(-2, 4.2); ax.set_aspect("equal"); ax.axis("off")
    # Shape is an illustrative wing planform, with only the report's station values asserted.
    ax.add_patch(Polygon([[0, .0], [7.7, .2], [7.7, 2.0], [0, 1.5]], closed=True,
                         facecolor="#e4e8f1", edgecolor="#545c72", lw=1.4))
    ax.plot([0, 7.7], [1.5, 2], color="#545c72", lw=1.4)
    for x in range(1, 8):
        ax.axvline(x, ymin=.30, ymax=.70, lw=.7, ls=":", color="#acb2c0")
        ax.text(x, -.55, str(x), ha="center", va="top", fontsize=15)
    ax.text(4, -1.2, "展向站位 / m", ha="center", fontsize=18)
    for x, label, color in [(4, "内侧桨", blue), (7, "外侧桨", rose)]:
        ax.add_patch(Circle((x, 1.1), .53, ec=color, fc="white", lw=2.2, zorder=4))
        ax.plot([x-.35, x+.35], [1.1,1.1], color=color, lw=1.4, zorder=5)
        ax.plot([x,x], [.75,1.45], color=color, lw=1.4, zorder=5)
        ax.annotate(label, (x, 1.64), xytext=(x-.5, 3.15), fontsize=17,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.2))
    ax.text(-.35, 1.03, "机身侧", ha="right", va="center", fontsize=15)
    ax.text(.1, 3.85, "(a) 翼桨展向位置示意", fontsize=20)
    ax.text(.1, -1.75, "示例站位（4 m，7 m）", fontsize=16, color="#5c6170")

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.text(0, 2.75, "(b) 流向安装位置", fontsize=20)
    ax.add_patch(Polygon([[.8,.65],[8.9,.65],[8.9,.95],[.8,.95]], closed=True,
                         fc="#e4e8f1", ec="#626b7d", lw=1))
    for x, lab, c in [(1.6,"0.10c",blue),(4.9,"0.50c",coral),(8.2,"0.90c",rose)]:
        ax.plot([x,x],[.5,1.35], color=c, lw=2.2)
        ax.text(x,1.55,lab,ha="center",fontsize=17,color=c)
    ax.annotate("",(.8,.3),(8.9,.3),arrowprops=dict(arrowstyle="<->",lw=1))
    ax.text(4.85,.03,"参考弦长 c",ha="center",fontsize=16)
    ax.text(.8,2.0,"前缘",fontsize=15); ax.text(8.3,2.0,"后缘",fontsize=15)

    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-1, 4)
    ax.text(0,3.6,"(c) 垂直安装位置",fontsize=20)
    ax.axhline(1.15,xmin=.07,xmax=.91,color="#687084",lw=2)
    ax.text(8.0,1.3,"机翼基准面",fontsize=15)
    for y,lab,c in [(2.65,"上移  +0.40R",rose),(1.15,"共面  0",blue),(-.35,"下移  −0.40R",purple)]:
        ax.add_patch(Circle((2.3,y),.30,edgecolor=c,facecolor="white",lw=2,zorder=4))
        ax.text(3.05,y+.23,lab,va="center",fontsize=17,color=c)
    ax.text(6.5,-.1,"R  为桨半径\nc  为参考弦长",fontsize=15,linespacing=1.6)
    fig.text(.05,.025,"依据总报告表21；机翼为参数示意，圆形表示桨轴位置，形状与尺寸不代表实际几何。",fontsize=13,color="#666666")
    save(fig,"3.2.1","数据_01_翼桨位置与典型工况")


def phases():
    # The three CFD phases contain 18 reported configurations each. Cruise has only
    # minimum/mean/maximum values in table 22; plot those summaries explicitly.
    figs, axs = plt.subplots(1, 3, figsize=(17, 7.4))
    figs.subplots_adjust(left=.07,right=.98,top=.86,bottom=.21,wspace=.33)
    metrics=[("升力系数 $C_L$",1),("阻力系数 $C_D$",2),("力矩系数 $C_M$",3)]
    phase_names=["巡航\n1°","爬升\n6°","降落\n1°","机动\n−2°"]
    for k,(ax,(title,col)) in enumerate(zip(axs,metrics)):
        cr=src.TABLES[22][k+1]
        low,high,mean=map(float,cr[1:4])
        ax.vlines(0,low,high,color=blue,lw=3)
        ax.scatter([0],[mean],s=72,color=blue,zorder=4,label="巡航均值及范围")
        ax.hlines([low,high],-.08,.08,color=blue,lw=2)
        for i,num in enumerate([25,26,27],start=1):
            arr=np.array([float(r[col]) for r in src.TABLES[num][1:]])
            bp=ax.boxplot([arr],positions=[i],widths=.40,patch_artist=True,
                          showfliers=False,medianprops={"color":"#303030","linewidth":1.5},
                          boxprops={"facecolor":[coral,rose,purple][i-1],"alpha":.34,"linewidth":1},
                          whiskerprops={"linewidth":1},capprops={"linewidth":1})
            # Evenly arranged columns reflect individual cases, not altered data values.
            offsets=np.linspace(-.19,.19,len(arr))
            ax.scatter(i+offsets,arr,s=19,color=[coral,rose,purple][i-1],alpha=.8,zorder=3)
        ax.set_title(f"({chr(97+k)}) {title}",loc="left",pad=15)
        ax.set_xticks(range(4),phase_names)
        ax.set_ylabel(title)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        axes(ax)
    figs.text(.07,.065,"巡航为最小值、均值和最大值；其余工况每组包含18个关键布局，散点为各布局数值。",fontsize=14,color="#555555")
    save(figs,"3.2.1","数据_02_飞行工况迎角与分析方法")


def candidates():
    d=pd.read_csv(src.CANDIDATES).dropna(subset=["J_energy_mj","J_stability","J_response_s"])
    d=d[(d.J_energy_mj>0)&(d.J_stability>0)&(d.J_response_s>=0)]
    fig,axs=plt.subplots(1,2,figsize=(17,7.4))
    fig.subplots_adjust(left=.075,right=.94,top=.85,bottom=.2,wspace=.35)
    x=d.J_energy_mj.to_numpy()/1000
    a=axs[0]
    hb=a.hexbin(x,d.J_stability.to_numpy(),gridsize=36,mincnt=1,bins="log",
                yscale="log",cmap="Blues",linewidths=0)
    cb=fig.colorbar(hb,ax=a,pad=.02,shrink=.9)
    cb.set_label("候选数量",fontsize=16); cb.ax.tick_params(labelsize=13)
    a.set_xlabel("任务能量 / GJ"); a.set_ylabel("轨迹跟踪指标（对数刻度）")
    a.set_title("(a) 能量与轨迹跟踪",loc="left",pad=14); axes(a)
    a=axs[1]
    edges=np.linspace(np.floor(x.min()),np.ceil(x.max()),19)
    v=d.J_response_s.to_numpy()
    bins=[v==4,v==6,(v!=4)&(v!=6)]
    hist=np.vstack([np.histogram(x[mask],bins=edges)[0] for mask in bins])
    im=a.imshow(hist,aspect="auto",origin="lower",interpolation="nearest",
                extent=(edges[0],edges[-1],-.5,2.5),cmap="Blues",
                norm=colors.LogNorm(vmin=1,vmax=max(2,hist.max())))
    a.set_yticks([0,1,2],["4 s","6 s","其他"])
    a.set_xlabel("任务能量 / GJ"); a.set_ylabel("响应指标")
    a.set_title("(b) 响应指标与能量分布",loc="left",pad=14); axes(a)
    cb=fig.colorbar(im,ax=a,pad=.02,shrink=.9); cb.set_label("候选数量",fontsize=16); cb.ax.tick_params(labelsize=13)
    fig.text(.075,.06,f"共 {len(d):,} 条候选记录。响应指标含恢复时间、模式切换时间及可能的惩罚项。",fontsize=14,color="#555555")
    save(fig,"4.2.1","数据_01_可行候选多目标分布")


def distribution():
    d=pd.read_csv(src.CANDIDATES)
    v=d.J_response_s.dropna().to_numpy()
    v=v[v>=0]
    edges=[0,4,6,10,30,100,1000,np.inf]
    labels=["0","(0, 4]","(4, 6]","(6, 10]","(10, 30]","(30, 100]","(100, 1000]",">1000"]
    counts=np.array([np.sum(v==0)]+[np.sum((v>a)&(v<=b)) for a,b in zip(edges[:-1],edges[1:])])
    fig,axs=plt.subplots(2,1,figsize=(15.5,10),gridspec_kw={"height_ratios":[1.2,1]})
    fig.subplots_adjust(left=.1,right=.96,top=.90,bottom=.11,hspace=.42)
    x=np.arange(len(labels)); a=axs[0]
    a.bar(x,counts,color=[blue,blue,coral,rose,purple,purple,purple,purple],width=.63,edgecolor="#555",linewidth=.4)
    a.set_ylim(0,counts.max()*1.20)
    for i,n in enumerate(counts):
        a.text(i,n+counts.max()*.025,f"{n:,}",ha="center",fontsize=14)
    a.set_xticks(x,labels); a.set_ylabel("候选数量"); a.set_title("(a) 响应指标分布",loc="left",pad=12)
    axes(a)
    a=axs[1]
    cumul=counts.cumsum()/counts.sum()*100
    a.plot(x,cumul,"-o",color=blue,lw=2,ms=7)
    a.set_xticks(x,labels); a.set_ylim(0,108); a.set_ylabel("累计比例 / %")
    a.set_xlabel("响应指标区间 / s"); a.set_title("(b) 候选累计比例",loc="left",pad=12)
    axes(a)
    save(fig,"4.2.1","数据_02_响应指标分布与长尾")


def surrogate():
    rows=src.TABLES[106][1:]
    raw=np.array([[float(r[1]),float(r[2])] for r in rows])
    fig,axs=plt.subplots(1,2,figsize=(15.5,7.2))
    fig.subplots_adjust(left=.08,right=.97,bottom=.2,top=.85,wspace=.3)
    x=np.arange(2); w=.32
    for ax,col,title in [(axs[0],1,"(a) 决定系数 $R^2$"),(axs[1],0,"(b) 均方根误差（×0.01）")]:
        old,new=raw[:2,col],raw[2:,col]
        ax.bar(x-w/2,old,w,color="#5d98b3",label="原始尺寸代理模型")
        ax.bar(x+w/2,new,w,color="#5eaa97",label="改进代理模型")
        for i in range(2):
            for pos,value in [(i-w/2,old[i]),(i+w/2,new[i])]:
                ax.text(pos,value+max(old.max(),new.max())*.025,f"{value:.2f}",ha="center",fontsize=16)
        ax.set_xticks(x,["升力斜率","尾迹效率因子"])
        ax.set_ylim(0,max(old.max(),new.max())*1.30)
        ax.set_title(title,loc="left",pad=14)
        axes(ax)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False,fontsize=16,
               bbox_to_anchor=(.5,.035))
    save(fig,"4.2.2","数据_01_代理模型独立测试对比")


if __name__ == "__main__":
    arrangement()
    phases()
    candidates()
    distribution()
    surrogate()
    (root / "图片来源与数据说明.txt").write_text(
        "本次图片调整的数据来源\n\n"
        "3.2.1 图一  总报告表21。机翼形状仅为安装位置示意，圆形是桨轴位置标记。\n"
        "3.2.1 图二  总报告表22、25、26、27。巡航仅有最小值、均值和最大值；"
        "爬升、降落、机动分别取18个报告列出的关键布局。\n"
        "4.2.1 两图  报告图117对应候选数据，共13971条候选记录，不代表独立重复试验。"
        "其中响应指标等于4秒的6401条，等于6秒的6882条。"
        "轨迹跟踪指标介于110至113的11513条，集中不等于数值完全相同。"
        "响应指标包含恢复、切换和可能的惩罚项；只凭表格无法将各项分离。\n"
        "4.2.2  总报告表106，配色参照总报告图114；数值对应总报告所述代理版本。\n",
        encoding="utf-8-sig")
    print("已更新5张图片及对应SVG。")
