from pathlib import Path
import re
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

REPORT = Path(r"C:\Users\pc\Desktop\【V3】附件3-技术进展报告总_交叉引用宋体修正版.docx")
ROOT = Path(r"C:\Users\pc\Desktop\PTDLfigure\总报告3.2与4.2_修订版")
RESULTS = Path(r"D:\LDTP\LDTP\PTDL飞推控协同优化代码包\运行结果\H6四算法同预算_20代_8种子_131GP_20261011")
CANDIDATES = Path(r"D:\LDTP\LDTP\报告图文增强_20261014\绘图数据\相关性分析可行候选.csv")
SECTIONS = {
    "3.2.1": "3.2.1_研究对象与工况定义",
    "3.2.2": "3.2.2_巡航代理与局部流场",
    "3.2.3": "3.2.3_关键工况与气动动力学",
    "4.2.1": "4.2.1_优化指标与约束",
    "4.2.2": "4.2.2_优化算法与代理模型",
    "4.2.3": "4.2.3_仿真平台与算法评估",
}
font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "svg.fonttype": "none", "font.size": 12, "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, BLUE, TEAL, ORANGE, RED = "#263b49", "#3e6d98", "#0c817f", "#cd9440", "#bd665e"
ALL = []
DOC = Document(REPORT)
TABLES = {}
caption = ""
for block in DOC.iter_inner_content():
    if isinstance(block, Paragraph):
        text = block.text.strip()
        if re.match(r"^表\s*\d+", text):
            caption = text
    elif isinstance(block, Table) and caption:
        m = re.match(r"^表\s*(\d+)", caption)
        if m:
            TABLES[int(m.group(1))] = [[c.text.replace("\n", " ").strip() for c in r.cells] for r in block.rows]
        caption = ""


def save(fig, section, name, source, detail, derived=True):
    folder = ROOT / SECTIONS[section]
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    ALL.append((section, name + ".png", source, detail, "数据重绘" if derived else "方法示意"))


def header(fig, title, subtitle):
    fig.text(.055, .95, title, fontsize=23, weight="bold", color=INK, va="top")
    fig.text(.055, .89, subtitle, fontsize=12, color="#647783", va="top")


def footer(fig, text):
    fig.text(.055, .025, text, color="#647783", fontsize=10.3)


def style(ax, grid="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#e3e9eb", lw=.75)
    ax.tick_params(length=0, pad=5, labelsize=11, colors=INK)
    ax.spines["left"].set_color("#cbd5d9")
    ax.spines["bottom"].set_color("#cbd5d9")


def extract_report():
    ids = list(range(19, 36)) + [114, 116, 117]
    # 图片编号映射以正文中紧随图片的图题为准，而非文件内部 image 序号。
    paras = DOC.paragraphs
    for n in ids:
        found = []
        for idx, para in enumerate(paras):
            if re.match(fr"^图\s*{n}(?!\d)\s", para.text.strip()):
                for prev in paras[max(0,idx-4):idx][::-1]:
                    embeds = prev._p.xpath(".//a:blip/@r:embed")
                    if embeds:
                        found = embeds
                        break
                if found:
                    caption = para.text.strip()
                    break
        if len(found) != 1:
            raise RuntimeError(f"图{n} image match {found}")
        section = ("3.2.2" if 19 <= n <= 28 else
                   "3.2.3" if 29 <= n <= 35 else
                   "4.2.2" if n == 114 else "4.2.3")
        secdir = ROOT / SECTIONS[section]
        secdir.mkdir(parents=True, exist_ok=True)
        title = re.sub(r"[\\/:*?\"<>|（）()，, ]+", "_", caption).strip("_")
        filename = f"总报告原图_{title}.png"
        part = DOC.part.related_parts[found[0]]
        (secdir / filename).write_bytes(part.blob)
        ALL.append((section, filename, f"总报告图{n}", caption, "原图提取"))


def layout_and_cases():
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.7), gridspec_kw={"width_ratios":[1.12,1]})
    fig.subplots_adjust(left=.10,right=.94,bottom=.15,top=.78,wspace=.30)
    header(fig,"翼桨安装参数与典型工况","参数取值与21组展向站位组合  ·  坐标按报告定义")
    ax=axes[0]
    mat=np.full((7,7),np.nan)
    for inner in range(1,8):
        for outer in range(inner+1,8):
            mat[inner-1,outer-1]=1
    ax.imshow(mat, cmap=colors.ListedColormap(["#6ba8a7"]),vmin=0,vmax=1)
    ax.set_facecolor("#f0f3f4")
    for i,j in zip(*np.where(np.isfinite(mat))):
        ax.text(j,i,f"({i+1}, {j+1})",ha="center",va="center",fontsize=9,color=INK)
    ax.set_xticks(range(7),range(1,8)); ax.set_yticks(range(7),range(1,8))
    ax.set_xlabel("外侧桨展向站位 / m"); ax.set_ylabel("内侧桨展向站位 / m")
    ax.set_title("(a) 21组展向位置",loc="left",fontsize=15,weight="bold",pad=12,color=INK)
    ax.grid(False); ax.tick_params(length=0)
    ax=axes[1]; cases=["机动","巡航","降落","爬升"]; angles=[-2,1,1,6]
    ax.axvline(0,color="#9eb0b8",lw=1)
    for i,(case,a) in enumerate(zip(cases,angles)):
        ax.scatter(a,i,s=155,color=[RED,TEAL,BLUE,ORANGE][i],zorder=3)
        ax.text(a+(.22 if a>=0 else -.22),i,f"{a:+d}°",ha="left" if a>=0 else "right",va="center",fontsize=14,weight="bold",color=INK)
    ax.set_yticks(range(4),cases); ax.set_xlim(-3.5,7.5); ax.set_xlabel("迎角 / °")
    ax.set_title("(b) 工况与迎角",loc="left",fontsize=15,weight="bold",pad=12,color=INK); style(ax,"x")
    ax.text(-3.3,-1.0,"流向位置  0.10c / 0.50c / 0.90c\n垂直位置  −0.40R / 共面 / +0.40R",
            fontsize=11,color=INK,linespacing=1.7)
    footer(fig,"数据来源  总报告表20、表21。该图表示报告列出的参数与工况，不表示各组合均已完成计算。")
    save(fig,"3.2.1","数据_01_翼桨位置与典型工况","表20、表21","21组站位、代表迎角和安装位置参数")

    rows=TABLES[20][1:]
    fig, ax=plt.subplots(figsize=(14.8,8.3)); fig.subplots_adjust(left=.20,right=.90,bottom=.18,top=.76)
    header(fig,"研究工况与分析内容","迎角及相应气动研究方法  ·  总报告表20")
    y=np.arange(len(rows))
    vals=[float(r[1].replace("−","-").replace("°","")) for r in rows]
    ax.barh(y,vals,color=[TEAL,ORANGE,BLUE,RED],height=.56)
    ax.axvline(0,color="#83969e",lw=1)
    ax.set_yticks(y,[r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(-3.4,8.4)
    for yi,(row,val) in enumerate(zip(rows,vals)):
        ax.text(val + (.15 if val>=0 else -.15),yi,f"{val:+g}°",ha="left" if val>=0 else "right",va="center",fontsize=13,weight="bold",color=INK)
        ax.text(7.1,yi,row[2],va="center",fontsize=11,color=INK)
    ax.text(7.1,-.68,"分析方法",fontsize=12,weight="bold",color=INK)
    ax.set_xlabel("迎角 / °"); style(ax,"x")
    footer(fig,"来源  总报告表20。巡航采用代理模型，其他工况采用关键点CFD分析。")
    save(fig,"3.2.1","数据_02_飞行工况迎角与分析方法","表20","按工况展示迎角和计算方式")


def cruise():
    fig, axs=plt.subplots(1,2,figsize=(15.5,8.5),gridspec_kw={"width_ratios":[1.1,1]})
    fig.subplots_adjust(left=.10,right=.93,bottom=.19,top=.76,wspace=.32)
    header(fig,"巡航气动代理模型分组交叉验证","按展向站位组合分组，合并各折留出预测  ·  CL、CD、CM分别建模")
    data=TABLES[23][1:]
    keys=[r[0] for r in data]; x=np.arange(3); width=.32
    for k,(col,txt,c) in enumerate([(1,"训练 R²",BLUE),(2,"分组折外 R²",TEAL)]):
        val=np.array([float(r[col]) for r in data])
        axs[0].bar(x+(k-.5)*width,val,width,color=c,label=txt)
        for xx,v in zip(x+(k-.5)*width,val): axs[0].text(xx,v+.012,f"{v:.4f}",ha="center",fontsize=11,color=INK)
    axs[0].set_ylim(0,1.15); axs[0].set_xticks(x,keys); axs[0].set_title("(a) 决定系数",loc="left",weight="bold",fontsize=15)
    axs[0].legend(frameon=False,loc="upper right",fontsize=10); style(axs[0])
    rmse=np.array([float(r[3]) for r in data]); mae=np.array([float(r[4]) for r in data])
    axs[1].barh(x+.16,rmse,height=.29,color=BLUE,label="RMSE")
    axs[1].barh(x-.16,mae,height=.29,color=TEAL,label="MAE")
    axs[1].set_yticks(x,keys); axs[1].invert_yaxis()
    axs[1].set_xlim(0,.0041); axs[1].set_title("(b) 折外绝对误差",loc="left",weight="bold",fontsize=15)
    for i,(a,b) in enumerate(zip(rmse,mae)):
        axs[1].text(a+.00005,i+.16,f"{a:.6f}",va="center",fontsize=10,color=INK)
        axs[1].text(b+.00005,i-.16,f"{b:.6f}",va="center",fontsize=10,color=INK)
    axs[1].legend(frameon=False,loc="lower right",fontsize=10); style(axs[1],"x")
    footer(fig,"来源  总报告表23。R²为分组合并折外结果，RMSE和MAE使用各系数原有尺度。")
    save(fig,"3.2.2","数据_01_巡航代理分组交叉验证","表23","训练和分组折外R²、RMSE、MAE")

    data=TABLES[24][1:]; stations=np.array([float(r[0]) for r in data])
    fig,axs=plt.subplots(1,2,figsize=(15.5,8.5))
    fig.subplots_adjust(left=.09,right=.95,bottom=.19,top=.76,wspace=.25)
    header(fig,"螺旋桨垂直位置对截面压力的影响","展向（4 m, 7 m），流向0.50c  ·  共面与上移0.40R比较")
    for ax,cols,title in zip(axs,[(1,2),(3,4)],["(a) 上下表面压差峰值","(b) 归一化压差积分"]):
        a=np.array([float(r[cols[0]]) for r in data]); b=np.array([float(r[cols[1]]) for r in data])
        x=np.arange(3); w=.33
        ax.bar(x-w/2,a,w,color=BLUE,label="共面"); ax.bar(x+w/2,b,w,color=ORANGE,label="上移 0.40R")
        for i,(v1,v2) in enumerate(zip(a,b)):
            ax.text(i-w/2,v1+.08,f"{v1:.3f}",ha="center",fontsize=10)
            ax.text(i+w/2,v2+.08,f"{v2:.3f}",ha="center",fontsize=10)
        ax.set_ylim(0,max(max(a),max(b))*1.27); ax.set_xticks(x,[f"{z:.2f}" for z in stations])
        ax.set_xlabel("展向截面位置 / m"); ax.set_title(title,loc="left",fontsize=15,weight="bold")
        ax.legend(frameon=False,fontsize=10); style(ax)
    footer(fig,"来源  总报告表24。积分按截面弦长归一化；不同截面呈现载荷重新分配。")
    save(fig,"3.2.2","数据_02_垂直位置与截面压差","表24","对比三处展向截面的压差峰值和积分")


def case_heatmaps():
    titles={25:("爬升",6),26:("降落",1),27:("机动",-2)}
    for num,(phase,angle) in titles.items():
        rows=TABLES[num][1:]
        values=np.full((3,2,3,3),np.nan)
        for r in rows:
            name=r[0]
            span=0 if "（2.00m,7.00m）" in name else 1 if "（4.00m,7.00m）" in name else None
            match=re.search(r"流向(0\.\d+)c",name)
            if span is None or not match: raise ValueError(name)
            stream=[.10,.50,.90].index(round(float(match.group(1)),2))
            vert=0 if "下移" in name else 2 if "上移" in name else 1
            for k in range(3): values[k,span,vert,stream]=float(r[k+1])
        if np.isnan(values).any(): raise ValueError(f"missing values table {num}")
        fig,axs=plt.subplots(3,2,figsize=(15.5,10))
        fig.subplots_adjust(left=.11,right=.87,bottom=.13,top=.79,wspace=.18,hspace=.37)
        header(fig,f"{phase}工况翼桨布局的气动响应",f"迎角 {angle:+d}°  ·  两组展向站位  ·  色块为CFD关键点，无插值")
        for k,label in enumerate(["升力系数 $C_L$","阻力系数 $C_D$","力矩系数 $C_M$"]):
            vmin,vmax=values[k].min(),values[k].max()
            for span in range(2):
                ax=axs[k,span]; im=ax.imshow(values[k,span],cmap="YlGnBu",vmin=vmin,vmax=vmax,aspect="auto")
                for i in range(3):
                    for j in range(3):
                        v=values[k,span,i,j]; txt=f"{v:.4f}" if k==1 else f"{v:.3f}"
                        ax.text(j,i,txt,ha="center",va="center",fontsize=10,color="white" if v>vmin+.63*(vmax-vmin) else INK)
                ax.set_xticks(range(3),["0.10c","0.50c","0.90c"]); ax.set_yticks(range(3),["下移0.40R","共面","上移0.40R"] if span==0 else [])
                ax.tick_params(length=0,pad=5,labelsize=10)
                if k==0: ax.set_title(f"展向 ({2 if span==0 else 4} m, 7 m)",fontsize=13,weight="bold")
                if span==0: ax.set_ylabel(label,fontsize=12,weight="bold")
                if k==2: ax.set_xlabel("流向位置",fontsize=10)
            cb=fig.colorbar(im,ax=axs[k,:],fraction=.026,pad=.02)
            cb.ax.tick_params(labelsize=9,length=0)
        footer(fig,f"数据来源  总报告表{num}。每个单元对应一组展向、流向及垂直安装位置。")
        save(fig,"3.2.3",f"数据_0{num-24}_{phase}工况布局响应矩阵",f"表{num}","18个关键点的升力、阻力与力矩")

    fig,axs=plt.subplots(1,3,figsize=(15.5,8.4))
    fig.subplots_adjust(left=.08,right=.96,bottom=.18,top=.75,wspace=.32)
    header(fig,"不同工况下螺旋桨抬高的气动响应","展向（4 m, 7 m）、流向0.50c  ·  两个垂直位置配对比较")
    phases=["爬升","降落","机动"]
    for k,(ax,label) in enumerate(zip(axs,["升力系数 $C_L$","阻力系数 $C_D$","力矩系数 $C_M$"])):
        for i,num in enumerate((25,26,27)):
            rows=[r for r in TABLES[num][1:] if "（4.00m,7.00m）" in r[0] and "流向0.50c" in r[0]]
            baseline=float(next(r[k+1] for r in rows if "共面" in r[0]))
            raised=float(next(r[k+1] for r in rows if "上移" in r[0]))
            ax.plot([baseline,raised],[i,i],color="#b7c8cc",lw=3,zorder=1)
            ax.scatter([baseline],[i],color=BLUE,s=80,zorder=3)
            ax.scatter([raised],[i],color=ORANGE,s=80,zorder=3)
            digits=4 if k==1 else 3
            ax.annotate(f"{baseline:.{digits}f}",(baseline,i),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9,color=BLUE)
            ax.annotate(f"{raised:.{digits}f}",(raised,i),xytext=(0,-16),textcoords="offset points",ha="center",fontsize=9,color=ORANGE)
        ax.set_yticks(range(3),phases); ax.invert_yaxis(); ax.set_title(label,fontsize=14,weight="bold")
        style(ax,"x"); ax.margins(x=.35,y=.25)
    fig.text(.40,.11,"● 共面",color=BLUE,fontsize=12); fig.text(.53,.11,"● 上移0.40R",color=ORANGE,fontsize=12)
    footer(fig,"数据来源  总报告表25—表27。每个子图采用独立坐标尺度；连接线表示同工况配对。")
    save(fig,"3.2.3","数据_04_跨工况垂直位置效应","表25—表27","比较垂直位置调整在不同工况下的响应")


def candidate_figures():
    d=pd.read_csv(CANDIDATES)
    need=["J_energy_mj","J_stability","J_response_s","diagnostic_terminal_soc"]
    d=d[need].dropna()
    d=d[(d.J_energy_mj>0)&(d.J_stability>0)&(d.J_response_s>=0)]
    fig,axs=plt.subplots(1,2,figsize=(15.5,8.4))
    fig.subplots_adjust(left=.09,right=.93,bottom=.17,top=.76,wspace=.28)
    header(fig,"可行候选方案的多目标分布","报告图117对应的候选样本  ·  颜色表示任务末端电池剩余电量")
    a=axs[0].scatter(d.J_energy_mj/1000,d.J_stability,c=d.diagnostic_terminal_soc,
                     s=5,alpha=.45,cmap="viridis",rasterized=True)
    axs[0].set_yscale("log"); axs[0].set_xlabel("任务能量 / GJ"); axs[0].set_ylabel("飞行跟踪误差指标（对数坐标）")
    axs[0].set_title("(a) 能量与跟踪",loc="left",fontsize=15,weight="bold"); style(axs[0],"both")
    b=axs[1].scatter(d.J_energy_mj/1000,d.J_response_s,c=d.diagnostic_terminal_soc,
                     s=5,alpha=.4,cmap="viridis",rasterized=True)
    axs[1].set_yscale("symlog",linthresh=1); axs[1].set_xlabel("任务能量 / GJ")
    axs[1].set_ylabel("响应时间指标 / s（含0的对数坐标）")
    axs[1].set_title("(b) 能量与响应",loc="left",fontsize=15,weight="bold"); style(axs[1],"both")
    cb=fig.colorbar(b,ax=axs,location="right",fraction=.025,pad=.025); cb.set_label("末端剩余电量比例")
    footer(fig,f"来源  报告图117的可行候选数据，共 {len(d):,} 条。响应指标含恢复、切换及约束惩罚。")
    save(fig,"4.2.1","数据_01_可行候选多目标分布","报告图117对应候选数据","目标分布及末端电量；样本为候选而非独立重复试验")

    vals=d.J_response_s.to_numpy()
    bins=[-0.001,0,4,6,10,30,100,1000,np.inf]
    counts=np.histogram(vals,bins=bins)[0]
    labels=["0","(0,4]","(4,6]","(6,10]","(10,30]","(30,100]","(100,1000]",">1000"]
    # numpy's bins are left-closed; count ranges explicitly to match the labels.
    counts=[int(np.sum(vals==0))]
    for a,b in zip([0,4,6,10,30,100,1000],[4,6,10,30,100,1000,np.inf]):
        counts.append(int(np.sum((vals>a)&(vals<=b))))
    fig,axs=plt.subplots(1,2,figsize=(15.5,8.5))
    fig.subplots_adjust(left=.09,right=.95,bottom=.20,top=.75,wspace=.28)
    header(fig,"响应时间指标的候选分布","与图117相同的可行候选集合  ·  指标包含恢复、切换与约束惩罚")
    ax=axs[0]; x=np.arange(len(labels))
    ax.bar(x,counts,color=[TEAL,BLUE,BLUE,BLUE,ORANGE,ORANGE,RED,RED]); ax.set_yscale("symlog",linthresh=1)
    ax.set_xticks(x,labels,rotation=45,ha="right",fontsize=10); ax.set_ylabel("候选数（含零的对数坐标）")
    for i,v in enumerate(counts): ax.text(i,max(v,1)*1.2,str(v),ha="center",fontsize=9,color=INK)
    ax.set_ylim(0,max(counts)*4); ax.set_title("(a) 分区间样本数量",loc="left",fontsize=15,weight="bold"); style(ax)
    ax=axs[1]; thresholds=np.array([0,4,6,10,30,100,1000,10000,13001])
    percentages=[100*np.mean(vals<=v) for v in thresholds]
    ax.plot(thresholds,percentages,"o-",color=TEAL,lw=2,ms=6); ax.set_xscale("symlog",linthresh=1)
    ax.set_ylim(0,105); ax.set_xlabel("响应时间指标 / s（含0的对数坐标）"); ax.set_ylabel("累计候选比例 / %")
    ax.set_title("(b) 累积分布",loc="left",fontsize=15,weight="bold"); style(ax,"both")
    for threshold,perc in zip((4,6,30),[100*np.mean(vals<=v) for v in (4,6,30)]):
        ax.annotate(f"≤{threshold} s  {perc:.1f}%",(threshold,perc),xytext=(5,-21 if threshold==4 else 10),
                    textcoords="offset points",fontsize=10,color=INK)
    footer(fig,f"来源  报告图117对应候选数据，共 {len(vals):,} 条。长尾可能包含惩罚，不能等同于纯恢复时间。")
    save(fig,"4.2.1","数据_02_响应指标分布与长尾","报告图117对应候选数据","完整保留零值与长尾，避免把归一化值误读为恢复时间")


def algorithm_results():
    rows=TABLES[106][1:]
    raw=np.array([[float(r[1]),float(r[2])] for r in rows])
    fig,axs=plt.subplots(1,2,figsize=(15.5,8.4))
    fig.subplots_adjust(left=.10,right=.95,bottom=.17,top=.75,wspace=.32)
    header(fig,"气动代理模型的独立构型预测比较","同一独立构型集  ·  左图为决定系数，右图为均方根误差")
    x=np.arange(2); w=.32
    for ax,col,title in [(axs[0],1,"(a) 决定系数 R²"),(axs[1],0,"(b) RMSE（×0.01）")]:
        old=raw[:2,col]; new=raw[2:,col]
        ax.bar(x-w/2,old,w,color=BLUE,label="原始尺寸模型")
        ax.bar(x+w/2,new,w,color=TEAL,label="改进代理模型")
        for i in range(2):
            ax.text(i-w/2,old[i]+max(old.max(),new.max())*.025,f"{old[i]:.2f}",ha="center",fontsize=11)
            ax.text(i+w/2,new[i]+max(old.max(),new.max())*.025,f"{new[i]:.2f}",ha="center",fontsize=11)
        ax.set_xticks(x,["升力斜率","尾迹效率因子"]); ax.set_ylim(0,max(old.max(),new.max())*1.3)
        ax.set_title(title,loc="left",fontsize=15,weight="bold"); ax.legend(frameon=False,fontsize=10); style(ax)
    footer(fig,"数据来源  总报告表106。该结果对应报告所述代理版本，与3.2节的翼桨CFD代理不同。")
    save(fig,"4.2.2","数据_01_代理模型独立测试对比","表106","同一独立构型集上的R²和RMSE")

    metrics=pd.read_csv(RESULTS/"算法多随机种子统计.csv")
    mapping={"NSGA-II":"NSGA-II","MOEA/D":"MOEA/D","MOPSO":"MOPSO","Hybrid NSGA-II/MOPSO":"混合算法"}
    metrics["label"]=metrics.algorithm.map(mapping)
    metrics=metrics.set_index("label").loc[["NSGA-II","MOEA/D","MOPSO","混合算法"]]
    for row in TABLES[108][1:]:
        label = "混合算法" if "混合" in row[0] else row[0]
        mean, std = row[2].split("±")
        metrics.loc[label,"final_shared_hypervolume_mean"] = float(mean)
        metrics.loc[label,"final_shared_hypervolume_std"] = float(std)
        metrics.loc[label,"final_feasible_ratio_mean"] = float(row[1].rstrip("%")) / 100
        metrics.loc[label,"runtime_s_mean"] = float(row[3]) * 60
    fig,axs=plt.subplots(1,3,figsize=(16,8.5))
    fig.subplots_adjust(left=.11,right=.95,bottom=.19,top=.75,wspace=.42)
    header(fig,"四类多目标优化算法的共同预算比较","报告表108对应结果  ·  超体积使用相同归一化基准")
    colors_=[BLUE,TEAL,ORANGE,RED]; y=np.arange(4)
    specs=[("final_shared_hypervolume_mean","final_shared_hypervolume_std","超体积",1),
           ("final_feasible_ratio_mean",None,"终代可行比例 / %",100),
           ("runtime_s_mean","runtime_s_std","计算时间 / min",1/60)]
    for i,(ax,(col,err,title,scale)) in enumerate(zip(axs,specs)):
        vals=metrics[col].values*scale; errors=None if err is None else metrics[err].values*scale
        ax.barh(y,vals,color=colors_,height=.55,xerr=errors,error_kw={"elinewidth":1.3,"capsize":3,"ecolor":INK})
        ax.set_yticks(y,metrics.index if i==0 else []); ax.invert_yaxis(); ax.set_title(title,loc="left",fontsize=14,weight="bold")
        ax.set_xlim(0,(vals+(0 if errors is None else errors)).max()*1.32)
        for j,v in enumerate(vals):
            ax.text((v+(0 if errors is None else errors[j]))+max(vals)*.025,j,f"{v:.2f}" if i==0 else f"{v:.1f}",va="center",fontsize=10)
        style(ax,"x")
    footer(fig,"来源  总报告表108及其逐次运行统计。误差线表示标准差；可行比例显示均值。")
    save(fig,"4.2.3","数据_01_四算法质量可行率与耗时","表108及其统计CSV","展示均值、离散度、可行率与耗时")

    run=pd.read_csv(RESULTS/"多随机种子逐代Pareto超体积.csv")
    fig,axs=plt.subplots(1,2,figsize=(15.5,8.4))
    fig.subplots_adjust(left=.10,right=.95,bottom=.18,top=.76,wspace=.30)
    header(fig,"算法搜索过程与末代结果分布","可行历史Pareto档案  ·  实线为中位数，色带为四分位区间")
    for alg,color in zip(["NSGA-II","MOEA/D","MOPSO","Hybrid NSGA-II/MOPSO"],colors_):
        d=run[run.algorithm==alg]
        stats=d.groupby("generation").shared_normalized_hypervolume.agg(
            median="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75))
        axs[0].plot(stats.index,stats["median"],color=color,lw=2,label=mapping[alg])
        axs[0].fill_between(stats.index,stats.q25,stats.q75,color=color,alpha=.13)
        final=d[d.generation==d.generation.max()].shared_normalized_hypervolume
        i=["NSGA-II","MOEA/D","MOPSO","Hybrid NSGA-II/MOPSO"].index(alg)
        rng=np.random.default_rng(13+i)
        axs[1].scatter(np.repeat(i,len(final))+rng.uniform(-.12,.12,len(final)),final,s=38,color=color,alpha=.8)
        axs[1].plot([i-.2,i+.2],[final.median()]*2,lw=2.3,color=color)
    axs[0].set_xlabel("迭代代数"); axs[0].set_ylabel("历代可行档案超体积")
    axs[0].set_title("(a) 收敛过程",loc="left",fontsize=15,weight="bold"); axs[0].legend(frameon=False,fontsize=10); style(axs[0],"both")
    axs[1].set_xticks(range(4),["NSGA-II","MOEA/D","MOPSO","混合算法"],rotation=20)
    axs[1].set_ylabel("末代历代档案超体积"); axs[1].set_title("(b) 各次运行结果",loc="left",fontsize=15,weight="bold"); style(axs[1])
    footer(fig,"来源  报告图116对应逐代CSV。每组8次运行；末代统计的是跨代可行档案，非最终种群。")
    save(fig,"4.2.3","数据_02_四算法逐代收敛与单次分布","图116对应逐代CSV","中位数、四分位区间与每次运行结果")

    rows=TABLES[109][1:]
    labels=[r[0] for r in rows]
    feasible=np.array([float(r[1].rstrip("%")) for r in rows])
    gains=np.array([float(r[2]) for r in rows])
    fig,axs=plt.subplots(1,2,figsize=(15.5,8.4))
    fig.subplots_adjust(left=.12,right=.94,bottom=.17,top=.75,wspace=.35)
    header(fig,"六自由度约束下的算法结果","增加搜索预算后的可行方案保持与解集改进")
    y=np.arange(len(labels))
    for ax,values,title,unit,limit in [
        (axs[0],feasible,"(a) 终代可行方案比例","%",115),
        (axs[1],gains,"(b) 解集超体积平均增量","",max(gains)*1.3),
    ]:
        ax.barh(y,values,color=[BLUE,TEAL,ORANGE],height=.5)
        ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,limit)
        ax.set_title(title,loc="left",fontsize=15,weight="bold")
        for i,v in enumerate(values):
            ax.text(v+limit*.025,i,f"{v:.1f}{unit}" if unit else f"{v:.2f}",va="center",fontsize=12)
        style(ax,"x")
    footer(fig,"数据来源  总报告表109。图中增量为增加预算前后的平均差值，比例为终代可行方案比例。")
    save(fig,"4.2.3","数据_03_六自由度预算与可行率","表109","三类算法增加预算后的可行比例与超体积增量")


def index():
    lines=["总报告3.2与4.2中期汇报插图索引","","报告来源  "+str(REPORT),
           "统计依据  图表取自对应章节的图、表及与报告图116、图117对应的原始计算结果。",
           "图中数据应按其算例、模型和评价口径解读；流程与参数分布图不代表新的试验。",
           "3.2节翼桨CFD代理预测CL、CD、CM；4.2节涡格法气动代理预测升力斜率和尾迹效率，二者为不同模型。",
           "可行候选图中的13971条是候选样本数量，不是13971次独立重复试验。",
           "报告原图保留原文件；新绘数据图提供PNG及SVG。",""]
    for sec,folder in SECTIONS.items():
        lines.append(folder)
        for s,name,src,detail,kind in ALL:
            if s==sec: lines.append(f"  {name}\n    {kind}  来源  {src}\n    内容  {detail}")
        lines.append("")
    (ROOT/"插图索引与数据来源.txt").write_text("\n".join(lines),encoding="utf-8-sig")


if __name__ == "__main__":
    extract_report()
    layout_and_cases()
    cruise()
    case_heatmaps()
    candidate_figures()
    algorithm_results()
    index()
    for sec in SECTIONS:
        print(sec,sum(x[0]==sec for x in ALL))
    print("total",len(ALL))
