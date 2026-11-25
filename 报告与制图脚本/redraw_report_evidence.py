from pathlib import Path
import hashlib
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageChops
from docx import Document

spec = importlib.util.spec_from_file_location("base", r"D:\LDTP\midterm_figures_v3.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = Path(r"C:\Users\pc\Desktop\PTDLfigure\总报告3.2与4.2_学术配色版\结构与指标重绘_复核版")
OUT.mkdir(parents=True, exist_ok=True)
DOC = Document(r"C:\Users\pc\Desktop\【V3】附件3-技术进展报告总_交叉引用宋体修正版.docx")
DATA = Path(r"D:\LDTP\LDTP\报告图文增强_20261014\绘图数据\四算法合并种子可行非支配方案.csv")
colors = ["#5c97b4", "#61a995", "#d68170", "#9482b3"]
plt.rcParams.update({
    "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
    "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 16,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "svg.fonttype": "none", "axes.linewidth": .8, "savefig.facecolor": "white",
})


def save(fig, folder, name):
    target=OUT / folder
    target.mkdir(exist_ok=True)
    fig.savefig(target / (name+".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(target / (name+".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(color="#e0e5e8", linewidth=.7)
    ax.tick_params(direction="out", length=3, colors="#343f49")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#77848b")


def origfig(number):
    p=DOC.paragraphs
    import re
    for i, paragraph in enumerate(p[:-1]):
        ids=paragraph._p.xpath(".//a:blip/@r:embed")
        if ids and re.match(fr"^图{number}(?!\d)\s",p[i+1].text.strip()):
            return DOC.part.related_parts[ids[0]].blob, p[i+1].text.strip()
    raise ValueError(f"cannot locate exact caption for figure {number}")


def cropwhite(image):
    image=image.convert("RGB")
    diff=ImageChops.difference(image,Image.new("RGB",image.size,"white"))
    box=diff.point(lambda x: 255 if x>8 else 0).getbbox()
    return image.crop((max(box[0]-15,0),max(box[1]-15,0),min(box[2]+15,image.width),min(box[3]+15,image.height)))


def research_layout():
    from io import BytesIO
    top=cropwhite(Image.open(BytesIO(origfig(20)[0])))
    bottom=cropwhite(Image.open(BytesIO(origfig(22)[0])))
    fig=plt.figure(figsize=(16,10.5))
    ax=fig.add_axes([.06,.55,.88,.35]); ax.imshow(top); ax.axis("off")
    fig.text(.06,.93,"(a) 典型翼桨布局的机翼表面压力分布",fontsize=17)
    ax=fig.add_axes([.06,.12,.88,.34]); ax.imshow(bottom); ax.axis("off")
    fig.text(.06,.49,"(b) 代表布局的局部速度场与压力场",fontsize=17)
    fig.text(.06,.045,"图 (a)、(b) 分别取自总报告图20、图22，保留原始计算图例与颜色。",fontsize=13,color="#58646b")
    save(fig,"3.2.1","01_翼桨布局及代表构型流场")


def phase_response():
    ts = [base.TABLES[i][1:] for i in [27,26,25]]
    by_phase=[{row[0]:np.array([float(z) for z in row[1:4]]) for row in rows} for rows in ts]
    keys=set(by_phase[0])
    assert len(keys)==18 and all(set(t)==keys for t in by_phase)
    fig,axs=plt.subplots(1,3,figsize=(16.5,6.4),sharex=True)
    fig.subplots_adjust(left=.065,right=.98,bottom=.24,top=.86,wspace=.31)
    labels=["(a) 升力系数 $C_L$","(b) 阻力系数 $C_D$","(c) 力矩系数 $C_M$"]
    for k,ax in enumerate(axs):
        for key in sorted(keys):
            c=colors[0] if "（2.00m,7.00m）" in key else colors[2]
            vals=[phase[key][k] for phase in by_phase]
            ax.plot([0,1,2],vals,color=c,alpha=.35,lw=1.0,marker="o",markersize=3)
        for cond,c in [("（2.00m,7.00m）",colors[0]),("（4.00m,7.00m）",colors[2])]:
            groups=[key for key in sorted(keys) if cond in key]
            mean=[np.mean([phase[key][k] for key in groups]) for phase in by_phase]
            ax.plot([0,1,2],mean,color=c,lw=2.2,marker="o",markersize=6)
        ax.set_xticks([0,1,2],["机动  −2°","降落  1°","爬升  6°"])
        ax.set_title(labels[k],loc="left",pad=13)
        style(ax)
    handles=[Line2D([0],[0],color=colors[0],lw=2,label="展向（2 m，7 m）"),
             Line2D([0],[0],color=colors[2],lw=2,label="展向（4 m，7 m）")]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.10),fontsize=14)
    fig.text(.065,.025,"细线连接同一安装布局在不同工况下的结果，粗线为各展向组合的九个布局均值。",fontsize=12.5,color="#606b70")
    save(fig,"3.2.1","02_关键布局跨工况气动响应")


def tradeoffs():
    d=pd.read_csv(DATA)
    names=["NSGA-II","MOEA/D","MOPSO","Hybrid NSGA-II/MOPSO"]
    display=["NSGA-II","MOEA/D","MOPSO","混合算法"]
    assert len(d)==277
    fig,axs=plt.subplots(1,2,figsize=(15.8,6.8))
    fig.subplots_adjust(left=.085,right=.97,bottom=.22,top=.88,wspace=.35)
    for alg,c,label in zip(names,colors,display):
        q=d[d.plot_algorithm.eq(alg)]
        axs[0].scatter(q.J_energy_mj/1000,q.J_stability,s=24,color=c,alpha=.65,label=label,
                       edgecolors="white",linewidths=.25)
        axs[1].scatter(q.diagnostic_fuel_used_kg,q.diagnostic_terminal_soc*100,s=24,color=c,alpha=.65,
                       edgecolors="white",linewidths=.25)
    axs[0].set_xlabel("任务能量 / GJ"); axs[0].set_ylabel("飞行跟踪指标")
    axs[0].set_title("(a) 任务能量与飞行跟踪",loc="left",pad=15)
    axs[0].set_yscale("log")
    axs[1].set_xlabel("燃油消耗 / kg"); axs[1].set_ylabel("末端剩余电量 / %")
    axs[1].set_title("(b) 燃油消耗与末端电量",loc="left",pad=15)
    for ax in axs: style(ax)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=13)
    fig.text(.08,.015,"各算法分别合并多次运行后筛选的非支配候选，共277条。左图纵轴为对数刻度。",fontsize=12,color="#616d72")
    save(fig,"4.2.1","01_能源与飞行指标关系")


def responses():
    d=pd.read_csv(DATA)
    vals=np.sort(d.J_response_s.unique())
    # At this version's candidates the three components sum to the plotted objective.
    comp=(d.diagnostic_response_recovery_time_s + d.diagnostic_response_mode_switch_time_s
          +d.diagnostic_response_failure_penalty_s)
    assert np.allclose(comp,d.J_response_s) and np.allclose(d.diagnostic_response_failure_penalty_s,0)
    fig,axs=plt.subplots(1,2,figsize=(16,7))
    fig.subplots_adjust(left=.075,right=.97,bottom=.21,top=.86,wspace=.32)
    ax=axs[0]
    freq=d.J_response_s.value_counts().reindex(vals,fill_value=0)
    xpos=np.arange(len(vals))
    ax.bar(xpos,freq,color=[colors[0] if x<=4 else colors[1] if x==6 else colors[2] if x==8 else colors[3] for x in vals],
           width=.7,edgecolor="white")
    for x,v in zip(xpos,freq):
        ax.annotate(str(v),(x,v),xytext=(0,4),textcoords="offset points",ha="center",fontsize=11)
    ax.set_xticks(xpos,[str(int(x)) for x in vals])
    ax.set_xlabel("响应时间取值 / s（分类排列）"); ax.set_ylabel("非支配候选数")
    ax.set_ylim(0,max(freq)*1.2); ax.set_title("(a) 响应时间的离散分布",loc="left",pad=15); style(ax)
    ax=axs[1]
    names=["NSGA-II","MOEA/D","MOPSO","Hybrid NSGA-II/MOPSO"]
    display=["NSGA-II","MOEA/D","MOPSO","混合算法"]
    accum=np.zeros(4)
    groups=[("≤4 s",lambda a:a<=4,colors[0]),("6 s",lambda a:a==6,colors[1]),
            ("8 s",lambda a:a==8,colors[2]),(">8 s",lambda a:a>8,colors[3])]
    y=np.arange(4)
    for label,fn,c in groups:
        rates=np.array([fn(d.loc[d.plot_algorithm.eq(a),"J_response_s"]).mean()*100 for a in names])
        ax.barh(y,rates,left=accum,color=c,height=.56,label=label,edgecolor="white",linewidth=.4)
        for k,r in enumerate(rates):
            if r>=10: ax.text(accum[k]+r/2,k,f"{r:.0f}%",ha="center",va="center",fontsize=12,color="white")
        accum+=rates
    ax.set_yticks(y,[f"{label}  (n={sum(d.plot_algorithm.eq(name))})" for label,name in zip(display,names)])
    ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel("候选比例 / %")
    ax.set_title("(b) 各算法候选的响应分布",loc="left",pad=15); style(ax)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,-.25),ncol=4,frameon=False,fontsize=11)
    fig.text(.075,.025,"候选来自各算法分别筛选的非支配解；本组数据的响应指标等于恢复时间与动力切换时间之和。",fontsize=12,color="#616d72")
    save(fig,"4.2.1","02_候选响应时间分布")


def corrected_originals():
    p=OUT/"报告原图图号校正"
    p.mkdir(exist_ok=True)
    for n in [20,24,25,26,27,28,30]:
        blob,caption=origfig(n)
        (p/f"总报告图{n:02d}.png").write_bytes(blob)
    return p


if __name__=="__main__":
    research_layout()
    phase_response()
    tradeoffs()
    responses()
    corrected_originals()
    (OUT/"来源与图号核对说明.txt").write_text(
        "3.2.1 图01：总报告图20和图22，分别展示典型布局压力分布与局部流场。"
        "当前未找到可确认与报告完全一致的完整三维几何模型。\n"
        "3.2.1 图02：总报告表25—27。每条细线连接同一布局在三个工况的18组气动结果，"
        "粗线为两类展向站位各9组结果的均值。工况之间连线仅表示配对关系，不表示中间迎角计算结果。\n"
        "4.2.1 图01、图02：四算法合并种子可行非支配方案.csv，共277条。"
        "每种算法各自合并多次运行后筛选非支配解，未进行跨算法联合支配筛选；候选数非独立运行次数。"
        "数据的响应指标等于恢复加切换，惩罚项为零。离散取值保留，不加随机噪声。\n"
        "图号校正：此前图集3.2.2、3.2.3中的图20、24、25、26、27、28、30文件名与内容错位。"
        "校正版按总报告中图片紧邻图题提取，原文件未覆盖。\n",
        encoding="utf-8-sig")
    print("完成4张新图和7张图号校正原图。")
