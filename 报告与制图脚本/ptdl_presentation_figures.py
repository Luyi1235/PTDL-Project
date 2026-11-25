from pathlib import Path
import math
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
from docx import Document


ROOT = Path(r"C:\Users\pc\Desktop\PTDLfigure")
REPORT = Path(r"C:\Users\pc\Desktop\PTDL 1.2 2.2节报告.docx")
SECTIONS = {
    "1.2.1": "1.2.1_协同作用模型",
    "1.2.2": "1.2.2_协同控制与智能调度",
    "1.2.3": "1.2.3_多工况验证与优化迭代",
    "2.2.1": "2.2.1_多目标指标体系",
    "2.2.2": "2.2.2_多目标优化算法",
    "2.2.3": "2.2.3_仿真分析平台",
}
TEAL = "#087f83"
BLUE = "#426d9b"
AMBER = "#c58b32"
CORAL = "#c96f61"
INK = "#263642"
GRAY = "#607580"
PALE = "#e9f3f2"
PALEBLUE = "#ebf1f8"
PALEAMBER = "#fcf4e5"
PALECORAL = "#faeeee"
FONT = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})
entries = []


def save(fig, sec, stem, desc, source, kind="方法示意"):
    out = ROOT / SECTIONS[sec]
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (stem + ".png"), dpi=180, facecolor="white")
    fig.savefig(out / (stem + ".svg"), facecolor="white")
    plt.close(fig)
    entries.append((sec, stem + ".png", kind, desc, source))


def diagram(title, subtitle):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(.7, 8.30, title, fontsize=25, fontweight="bold", color=INK, va="center")
    ax.text(.72, 7.68, subtitle, fontsize=13, color=GRAY, va="center")
    ax.plot([.72, 15.28], [7.30, 7.30], color="#dce8e9", lw=1.2)
    return fig, ax


def box(ax, x, y, w, h, title, sub="", color=TEAL, fill=PALE, fs=16, subfs=12):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12",
                           linewidth=1.35, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    if sub:
        ax.text(x+w/2, y+h*.69, title, ha="center", va="center", fontsize=fs,
                color=INK, fontweight="bold", linespacing=1.25)
        ax.text(x+w/2, y+h*.30, sub, ha="center", va="center", fontsize=subfs,
                color=GRAY, linespacing=1.4)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=fs,
                color=INK, fontweight="bold", linespacing=1.25)


def arrow(ax, a, b, color=GRAY, lw=1.8, style="-|>", rad=0, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, lw=lw,
                                color=color, linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}"))


def note(ax, text):
    ax.text(.72, .35, text, color=GRAY, fontsize=11, va="center")


def plot_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e4eaec", linewidth=.8)
    ax.tick_params(labelsize=12, colors=INK, length=0, pad=7)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c7d2d5")


def chart(title, subtitle):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .93, title, fontsize=25, fontweight="bold", color=INK)
    fig.text(.055, .87, subtitle, fontsize=13, color=GRAY)
    return fig


def original_images():
    doc = Document(REPORT)
    mapping = {
        1: ("1.2.1", "报告原图_推进系统与飞行控制状态反馈"),
        2: ("1.2.2", "报告原图_能量管理输入与执行"),
        3: ("1.2.3", "报告原图_五种配置能源使用对照"),
        4: ("1.2.3", "报告原图_长航时飞行状态与能源变化"),
        5: ("1.2.3", "报告原图_六阶段扰动响应"),
        6: ("2.2.2", "报告原图_多目标搜索与候选评价"),
        7: ("2.2.2", "报告原图_气动代理模型建立与应用"),
        8: ("2.2.2", "报告原图_气动代理独立验证结果"),
        9: ("2.2.3", "报告原图_平台模型与任务评价"),
        10: ("2.2.3", "报告原图_四算法收敛与重复运行"),
        11: ("2.2.3", "报告原图_设计参数与目标相关关系"),
        12: ("2.2.3", "报告原图_协同设计流程"),
    }
    media = {}
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            media[Path(str(rel.target_part.partname)).name] = rel.target_part.blob
    for num, (sec, stem) in mapping.items():
        out = ROOT / SECTIONS[sec]
        out.mkdir(parents=True, exist_ok=True)
        (out / (stem + ".png")).write_bytes(media[f"image{num}.png"])
        entries.append((sec, stem + ".png", "报告原图", f"报告图{num}，原样提取。",
                        f"参考报告图{num}"))


def draw_method_figures():
    fig, ax = diagram("混合动力系统的能量传递", "发动机、电池与电动机通过直流母线形成推进功率")
    box(ax, .7, 4.35, 2.25, 1.15, "燃油", "化学能", AMBER, PALEAMBER)
    box(ax, 3.45, 4.35, 2.40, 1.15, "活塞发动机", "机械功率与燃油消耗", AMBER, PALEAMBER)
    box(ax, 6.4, 4.35, 2.45, 1.15, "发电与整流", "机械能 → 电能", BLUE, PALEBLUE)
    box(ax, 9.4, 4.35, 2.0, 1.15, "直流母线", "电功率汇集", TEAL, PALE)
    box(ax, 12., 4.35, 2.8, 1.15, "电动机与螺旋桨", "轴功率 → 推力", TEAL, PALE)
    for start, end in [(2.95,3.45),(5.85,6.4),(8.85,9.4),(11.4,12.)]:
        arrow(ax,(start,4.92),(end,4.92))
    box(ax, 9.4, 2.1, 2.0, 1.15, "电池", "充电 / 放电", CORAL, PALECORAL)
    arrow(ax,(10.4,3.27),(10.4,4.31),style="<->",color=CORAL)
    note(ax, "方法示意  ·  实际功率受部件能力、响应动态及储能状态约束")
    save(fig,"1.2.1","新绘_01_混合动力能量传递","解释发动机、电池、电动机的能量传递。","报告1.2.1第（1）部分")

    fig, ax = diagram("推进与飞行控制的闭环关系", "任务指令驱动飞行控制，实际推力与飞行状态共同决定下一步功率需求")
    items = [(.8,"任务指令","航点 / 高度 / 速度", BLUE,PALEBLUE),
             (4.0,"飞行控制","轨迹与姿态控制",TEAL,PALE),
             (7.2,"能量管理","发动机与电池分配",AMBER,PALEAMBER),
             (10.4,"动力系统","部件动态与实际推力",CORAL,PALECORAL)]
    for x,t,s,c,f in items: box(ax,x,4.25,2.7,1.3,t,s,c,f)
    for x in (3.5,6.7,9.9): arrow(ax,(x,4.90),(x+.5,4.90))
    box(ax,5.65,1.75,4.75,1.2,"飞行动力学","气动力、推力与重力共同作用",BLUE,PALEBLUE)
    arrow(ax,(11.75,4.22),(10.4,2.75),rad=-.08)
    arrow(ax,(5.65,2.35),(5.3,4.22),rad=-.12,color=TEAL)
    arrow(ax,(8.05,2.98),(8.55,4.22),color=TEAL)
    ax.text(3.9,2.65,"位置 · 速度 · 姿态",fontsize=11,color=TEAL)
    note(ax, "状态反馈还包括发动机输出、燃油余量和电池剩余电量")
    save(fig,"1.2.1","新绘_02_推进飞行控制闭环","展示任务、控制、动力与飞行动力学之间的反馈。","报告1.2.1")

    fig, ax = diagram("六自由度飞行控制与执行", "任务航迹、姿态控制与推进系统通过飞行状态相互连接")
    box(ax,.8,4.55,3,1.25,"任务与航迹","航点 · 目标高度 · 目标速度",BLUE,PALEBLUE)
    box(ax,4.45,4.55,3,1.25,"轨迹控制","高度、速度与航向指令",TEAL,PALE)
    box(ax,8.10,4.55,3,1.25,"姿态控制","角度与角速度反馈",TEAL,PALE)
    box(ax,11.75,4.55,3,1.25,"舵机与舵面","动态响应与偏角限制",AMBER,PALEAMBER)
    for x in (3.8,7.45,11.1): arrow(ax,(x,5.18),(x+.65,5.18))
    box(ax,5.15,1.75,5.7,1.4,"六自由度飞行动力学","三个方向平动  ·  绕三个机体轴转动",BLUE,PALEBLUE)
    arrow(ax,(13.2,4.52),(10.85,2.7),rad=-.08)
    arrow(ax,(5.15,2.6),(5.95,4.52),rad=-.15,color=TEAL)
    box(ax,.85,1.75,2.8,1.4,"动力系统","实际推力",CORAL,PALECORAL)
    arrow(ax,(3.65,2.45),(5.15,2.45),color=CORAL)
    note(ax,"方法示意  ·  飞行状态与能源状态随任务阶段连续传递")
    save(fig,"1.2.1","新绘_03_六自由度与分层飞控","解释六自由度、外环轨迹与内环姿态控制。","报告1.2.1第（2）部分")

    fig, ax = diagram("能量管理的四种决策方式","各方法读取飞行与能源状态，输出发动机和电池的功率分配")
    boxes=[(.7,4.75,"预设规则","依据任务阶段\n功率需求与电池电量",BLUE,PALEBLUE),
           (4.45,4.75,"模糊逻辑","合成多条规则\n连续调整供电比例",TEAL,PALE),
           (8.2,4.75,"模型预测控制","预测未来状态\n滚动选择功率分配",AMBER,PALEAMBER),
           (11.95,4.75,"神经网络","学习模型预测控制\n给出的供电规律",CORAL,PALECORAL)]
    for x,y,t,s,c,f in boxes: box(ax,x,y,3.15,1.45,t,s,c,f,15,12)
    box(ax,4.1,1.65,7.8,1.35,"执行与闭环反馈","部件约束 → 实际供能 → 飞行与能源状态",BLUE,PALEBLUE)
    for x in (2.275,6.025,9.775,13.525):
        arrow(ax,(x,4.72),(min(max(x,4.6),11.4),3.03),color=GRAY)
    note(ax,"神经网络在报告中采用模型预测控制仿真数据进行模仿学习")
    save(fig,"1.2.2","新绘_01_四类能量管理方法","比较四类控制方法的决策机制与共同执行环节。","报告1.2.2第（1）（2）部分")

    fig, ax = diagram("模型预测控制的滚动决策","在当前状态下比较未来功率分配，只执行当前一步并重新预测")
    for x,t,s,c,f in [(1,"读取当前状态","飞行状态 / 电池电量",BLUE,PALEBLUE),
                      (4.55,"预测未来过程","需求 / 功率 / 能源",TEAL,PALE),
                      (8.1,"选择功率序列","燃油 / 电量 / 功率变化",AMBER,PALEAMBER),
                      (11.65,"执行当前动作","发动机与电池分配",CORAL,PALECORAL)]:
        box(ax,x,4.2,2.8,1.4,t,s,c,f,15,11)
    for x in (3.8,7.35,10.9): arrow(ax,(x,4.9),(x+.75,4.9))
    arrow(ax,(13.05,4.17),(2.4,4.17),rad=-.38,color=TEAL)
    ax.text(6.4,2.55,"新状态反馈后进入下一控制周期",fontsize=14,color=TEAL)
    note(ax,"预测同时考虑部件能力、电池状态和功率变化约束")
    save(fig,"1.2.2","新绘_02_模型预测滚动控制","说明预测、决策、执行与反馈的循环。","报告1.2.2第（1）部分")

    fig, ax = diagram("自适应调度与故障处理","依据可用能力和状态信息更新功率限制，再调整供能与飞行指令")
    for y,t,s in [(5.2,"发动机能力下降","修正持续供电上限"),(4.2,"电动机能力下降","修正可实现推进功率"),
                  (3.2,"电池支路变化","更新容量与等效内阻"),(2.2,"状态观测异常","使用有效信息或估计值")]:
        box(ax,1.0,y,4.45,.72,t,"",BLUE,PALEBLUE,14)
        box(ax,6.1,y,4.45,.72,s,"",TEAL,PALE,14)
        arrow(ax,(5.45,y+.36),(6.1,y+.36))
        arrow(ax,(10.55,y+.36),(11.35,3.83),color=GRAY)
    box(ax,11.35,3.05,3.7,1.55,"更新控制指令","功率分配 / 电池目标\n速度与高度要求",CORAL,PALECORAL)
    note(ax,"各类变化的处理关系依据报告表3整理")
    save(fig,"1.2.2","新绘_03_自适应与容错控制","展示四类能力或观测变化的控制处理。","报告表3")

    fig, ax = diagram("多工况任务剖面与控制重点","飞行状态、燃油与电池状态在各阶段连续传递")
    names=[("地面起飞","建立速度"),("爬升","推力与高度"),("巡航","持续供能"),
           ("下降与滑翔","空速与下降率"),("进近拉平","低速轨迹"),("着陆滑跑","减速停车")]
    for i,(t,s) in enumerate(names):
        x=.65+i*2.5
        box(ax,x,4.0,2.18,1.35,t,s,[BLUE,TEAL,TEAL,AMBER,CORAL,BLUE][i],
            [PALEBLUE,PALE,PALE,PALEAMBER,PALECORAL,PALEBLUE][i],15,11)
        if i<5: arrow(ax,(x+2.18,4.67),(x+2.5,4.67))
    ax.text(8,2.3,"功率响应  ·  能源消耗  ·  轨迹偏差  ·  恢复时间",ha="center",fontsize=17,color=INK)
    note(ax,"阶段安排与评价内容依据报告表4整理")
    save(fig,"1.2.3","新绘_01_任务阶段与控制重点","用于介绍完整任务及典型工况。","报告表4")

    fig, ax = diagram("多工况结果驱动的迭代","按部件响应、能源变化与飞行状态分析参数和控制动作")
    for x,t,s,c,f in [(1,"任务与扰动","任务阶段 / 风扰 / 能力变化",BLUE,PALEBLUE),
                      (4.6,"闭环仿真","推进系统与飞行控制",TEAL,PALE),
                      (8.2,"结果分析","功率 / 电量 / 轨迹 / 响应",AMBER,PALEAMBER),
                      (11.8,"参数与策略调整","部件参数 / 阶段控制",CORAL,PALECORAL)]:
        box(ax,x,4.2,2.8,1.35,t,s,c,f,15,10.5)
    for x in (3.8,7.4,11): arrow(ax,(x,4.87),(x+.8,4.87))
    arrow(ax,(13.2,4.17),(2.4,4.17),rad=-.36,color=TEAL)
    ax.text(6.45,2.6,"更新模型与任务控制配置",fontsize=14,color=TEAL)
    note(ax,"方法框架依据报告1.2.3节，图中未引入额外数值")
    save(fig,"1.2.3","新绘_02_多工况分析与迭代","说明仿真分析与模型控制参数调整关系。","报告1.2.3第（2）部分")

    fig, ax = diagram("多目标指标、辅助评价与约束","候选方案由三项目标描述性能取舍，并按能源与飞行条件筛选")
    box(ax,.8,4.5,4.4,1.8,"三个优化目标","任务能量消耗\n飞行跟踪误差\n响应时间指标",TEAL,PALE,17,13)
    box(ax,5.8,4.5,4.4,1.8,"辅助评价指标","燃油与电池损失\n电池转换效率及剩余电量\n阶段偏差与恢复情况",BLUE,PALEBLUE,17,13)
    box(ax,10.8,4.5,4.4,1.8,"任务与部件约束","功率 / 电流 / 能源储备\n速度 / 姿态 / 迎角\n离地 / 触地 / 停车",AMBER,PALEAMBER,17,13)
    for x in (3,8,13): arrow(ax,(x,4.48),(8,2.7),color=GRAY)
    box(ax,5.25,1.45,5.5,1.2,"候选方案综合评价","目标值 + 辅助指标 + 约束判定",CORAL,PALECORAL)
    note(ax,"响应时间指标由扰动恢复、动力切换与恢复失败惩罚构成")
    save(fig,"2.2.1","新绘_01_指标与约束体系","区分三个正式目标、辅助评价和约束。","报告表7及2.2.1节")

    fig, ax = diagram("任务能量与电池效率评价","燃油消耗和电池储能变化进入任务能量，转换损失单独记录")
    box(ax,.9,4.65,3.3,1.35,"燃油消耗","燃油质量 × 低热值",AMBER,PALEAMBER)
    box(ax,.9,2.6,3.3,1.35,"电池储能变化","任务初末储能差",CORAL,PALECORAL)
    box(ax,5.55,3.65,3.5,1.35,"任务能源核算","任务能量 = 燃油化学能 + 储能净减少",TEAL,PALE,16,10)
    arrow(ax,(4.2,5.3),(5.55,4.55)); arrow(ax,(4.2,3.27),(5.55,4.1))
    box(ax,10.2,4.65,4.75,1.35,"转换效率","有效输出或储能增加 / 输入能量",BLUE,PALEBLUE)
    box(ax,10.2,2.6,4.75,1.35,"电池内阻损失","按电流平方与内阻沿时间累计",CORAL,PALECORAL)
    arrow(ax,(9.05,4.7),(10.2,5.3)); arrow(ax,(9.05,4.0),(10.2,3.27))
    note(ax,"电池转换效率与内阻损失为辅助评价量；任务末端储备同时用于约束判定")
    save(fig,"2.2.1","新绘_02_任务能量与电池评价","解释能量目标和电池效率、损失的区别。","报告2.2.1第（1）部分及表8")

    fig, ax = diagram("飞行跟踪与响应指标","以参考轨迹为基准，分别记录跟踪偏差与扰动后的恢复过程")
    box(ax,.8,4.6,4.3,1.55,"飞行跟踪误差","高度 / 速度 / 俯仰角偏差\n角速度变化与局部峰值",BLUE,PALEBLUE)
    box(ax,5.85,4.6,4.3,1.55,"扰动恢复时间","扰动结束后进入允许范围\n并持续保持规定时间",TEAL,PALE)
    box(ax,10.9,4.6,4.3,1.55,"动力切换响应","实际功率跟随新指令\n记录切换建立过程",AMBER,PALEAMBER)
    for x in (2.95,8,13.05): arrow(ax,(x,4.57),(8,2.78))
    box(ax,5.1,1.45,5.8,1.25,"响应时间指标","恢复时间 + 切换时间 + 恢复失败惩罚",CORAL,PALECORAL)
    note(ax,"跟踪误差与响应时间分别反映持续偏差和受扰后的调整过程")
    save(fig,"2.2.1","新绘_03_跟踪误差与响应指标","说明响应指标的组成和恢复判据。","报告2.2.1第（2）部分")

    fig, ax = diagram("联合优化设计变量与约束","几何、动力和控制配置共同决定任务性能及可行性")
    for y,t,s,c,f in [(5.6,"气动几何","机翼与机身构型",BLUE,PALEBLUE),
                       (4.1,"动力配置","发动机 / 电池 / 电动机",AMBER,PALEAMBER),
                       (2.6,"控制参数","飞行控制与能量调度",TEAL,PALE)]:
        box(ax,.9,y,3.6,1.05,t,s,c,f,15,11)
        arrow(ax,(4.5,y+.525),(5.65,4.1),color=GRAY)
    box(ax,5.65,3.35,3.8,1.5,"协同任务仿真","飞行、动力、能源与控制",CORAL,PALECORAL)
    box(ax,10.75,4.55,4.1,1.35,"目标值","能量 / 跟踪 / 响应",TEAL,PALE)
    box(ax,10.75,2.4,4.1,1.35,"可行性","部件 / 能源 / 飞行 / 起降",BLUE,PALEBLUE)
    arrow(ax,(9.45,4.35),(10.75,5.2)); arrow(ax,(9.45,3.95),(10.75,3.1))
    note(ax,"目标与约束依据报告表7、表9绘制")
    save(fig,"2.2.1","新绘_04_设计变量与约束关系","展示设计变量、联合仿真与评价输出。","报告表7和表9")

    fig, ax = diagram("多目标优化算法的搜索机制","四类算法共享同一任务评价模型和目标约束")
    for x,t,s,c,f in [(0.8,"NSGA-II","非支配排序\n交叉与变异",BLUE,PALEBLUE),
                      (4.55,"MOEA/D","多目标分解\n邻域协同更新",TEAL,PALE),
                      (8.3,"MOPSO","个体历史位置\n群体引导搜索",AMBER,PALEAMBER),
                      (12.05,"混合算法","遗传重组\n粒子历史引导",CORAL,PALECORAL)]:
        box(ax,x,4.7,3.0,1.5,t,s,c,f,16,12)
        arrow(ax,(x+1.5,4.67),(8,2.95))
    box(ax,4.65,1.55,6.7,1.35,"共享候选评价","协同任务仿真 → 三目标与约束 → 方案更新",BLUE,PALEBLUE)
    note(ax,"混合算法名称与机制按参考报告2.2.2节表述")
    save(fig,"2.2.2","新绘_01_四类优化算法机制","简明展示四种算法的不同搜索方式。","报告2.2.2第（1）部分")

    fig, ax = diagram("气动代理建模与独立验证","利用参数化几何和气动计算样本建立快速预测关系")
    steps=[("参数化构型","机翼与机身几何"),("气动计算","OpenVSP / VSPAERO"),
           ("样本检查","极曲线与有效性"),("代理建模","分别预测两项气动量"),
           ("独立验证","误差与决定系数")]
    for i,(t,s) in enumerate(steps):
        x=.55+i*3.05
        box(ax,x,4.25,2.7,1.45,t,s,[BLUE,TEAL,AMBER,CORAL,BLUE][i],
            [PALEBLUE,PALE,PALEAMBER,PALECORAL,PALEBLUE][i],15,11)
        if i<4: arrow(ax,(x+2.7,4.98),(x+3.05,4.98))
    box(ax,4.25,1.65,7.5,1.2,"用于联合优化","升力斜率影响升力预测  ·  尾迹效率影响诱导阻力",TEAL,PALE,16,12)
    arrow(ax,(13.35,4.22),(11.75,2.85))
    note(ax,"方法流程依据报告2.2.2节；独立验证结果见同目录中的报告原图")
    save(fig,"2.2.2","新绘_02_气动代理训练验证流程","解释训练样本、气动量、模型验证和优化用途。","报告2.2.2第（2）部分")

    fig, ax = diagram("可行解保留与方案选择","通过约束筛选、非支配排序和权重扫描确定推荐方案")
    steps=[("候选方案","几何 / 动力 / 控制"),("可行性筛选","能源 / 速度 / 姿态等"),
           ("Pareto解集","保留非支配方案"),("决策与复算","权重扫描 / 裕度检查")]
    for i,(t,s) in enumerate(steps):
        x=.8+i*3.75
        box(ax,x,4.2,3.1,1.5,t,s,[BLUE,AMBER,TEAL,CORAL][i],
            [PALEBLUE,PALEAMBER,PALE,PALECORAL][i],15,11)
        if i<3: arrow(ax,(x+3.1,4.95),(x+3.75,4.95))
    box(ax,4.4,1.55,7.2,1.15,"推荐设计","输出设计参数、目标值与能源和速度裕度",BLUE,PALEBLUE)
    arrow(ax,(13.6,4.17),(11.6,2.7))
    note(ax,"跨代保留可行非支配解；最终种群可行率与历代解集分别统计")
    save(fig,"2.2.2","新绘_03_可行解与方案决策","介绍Pareto解集、权重扫描和安全裕度筛选。","报告2.2.2第（3）部分")

    fig, ax = diagram("多目标优化平台的数据流","通过配置、联合仿真、搜索决策和结果展示组织研究流程")
    steps=[("参数配置","几何 / 动力 / 控制"),("模型计算","气动 / 推进 / 飞行"),
           ("联合优化","搜索 / 候选评价"),("结果与决策","Pareto / 任务轨迹")]
    for i,(t,s) in enumerate(steps):
        x=.8+i*3.75
        box(ax,x,4.5,3.1,1.5,t,s,[BLUE,TEAL,AMBER,CORAL][i],
            [PALEBLUE,PALE,PALEAMBER,PALECORAL][i],16,11)
        if i<3: arrow(ax,(x+3.1,5.25),(x+3.75,5.25))
    arrow(ax,(13.6,4.47),(2.35,4.47),rad=-.35,color=TEAL)
    ax.text(6.1,2.85,"依据分析结果调整参数并重新计算",fontsize=14,color=TEAL)
    note(ax,"平台模块和输入输出依据报告表12整理")
    save(fig,"2.2.3","新绘_01_平台模块数据流","以适合PPT的形式说明平台计算流程。","报告表12")

    fig, ax = diagram("算法比较与配置选择","分别考虑可行性、解集质量与计算成本")
    box(ax,1,4.55,4.25,1.6,"约束满足情况","终代可行方案比例",BLUE,PALEBLUE)
    box(ax,5.9,4.55,4.25,1.6,"解集质量","可行非支配解集超体积",TEAL,PALE)
    box(ax,10.8,4.55,4.25,1.6,"计算成本","完成同等评价预算的耗时",AMBER,PALEAMBER)
    for x in (3.125,8.025,12.925): arrow(ax,(x,4.52),(8,2.8))
    box(ax,4.65,1.45,6.7,1.3,"配置选择","按任务约束、性能取舍与时间要求选择算法",CORAL,PALECORAL)
    note(ax,"超体积比较需采用相同归一化尺度和参考点")
    save(fig,"2.2.3","新绘_02_算法评价与配置选择","解释算法比较指标的含义与用途。","报告2.2.3第（2）（3）部分")


def draw_result_figures():
    labels=["预设规则","模糊逻辑","固定参数\n模型预测","分阶段参数\n模型预测","神经网络"]
    vals=[[28.50,28.95,29.92,28.78,28.52],[1.19,1.47,18.31,4.09,3.25],[32.0,58.5,20.,27.2,26.5]]
    titles=["燃油消耗 / kg","电池内阻损失 / MJ","任务最小剩余电量 / %"]
    fig=chart("五种能量管理配置的任务结果","同一任务  ·  三项指标分别衡量燃油支出、电池损失与电量储备")
    for i,(data,title) in enumerate(zip(vals,titles)):
        ax=fig.add_axes([.19+i*.265,.18,.20,.60])
        y=np.arange(5)
        bars=ax.barh(y,data,color=[BLUE,TEAL,"#a8b5bf",CORAL,AMBER],height=.58)
        ax.invert_yaxis(); plot_style(ax); ax.set_title(title,loc="left",fontsize=15,fontweight="bold",color=INK,pad=16)
        ax.set_yticks(y,labels if i==0 else [""]*5)
        ax.set_xlim(0,max(data)*1.3)
        for bar,v in zip(bars,data): ax.text(v+max(data)*.025,bar.get_y()+bar.get_height()/2,f"{v:.2f}" if i<2 else f"{v:.1f}%",va="center",fontsize=12,color=INK)
    fig.text(.055,.055,"数据来源  参考报告表5；不同指标的单位和评价含义不同。",fontsize=11,color=GRAY)
    save(fig,"1.2.3","新绘_03_控制方法能源结果对比","分项展示报告表5中的燃油、电池损失与最小电量。","报告表5","结果图")

    fig=chart("分阶段控制参数的算例表现","固定参数与分阶段参数的模型预测控制对比")
    ax1=fig.add_axes([.11,.22,.35,.55]); ax2=fig.add_axes([.57,.22,.35,.55])
    for ax,values,title,unit in [(ax1,[18.31,4.09],"电池内阻损失","MJ"),(ax2,[304,28],"进近阶段扰动恢复时间","s")]:
        ax.bar(["固定参数","分阶段参数"],values,color=[BLUE,TEAL],width=.5)
        ax.set_title(title,fontsize=17,loc="left",fontweight="bold",color=INK,pad=14)
        ax.set_ylabel(unit,fontsize=12,color=GRAY); ax.grid(axis="y",color="#e5ebed"); ax.set_axisbelow(True)
        ax.set_ylim(0,max(values)*1.25); ax.tick_params(labelsize=13,length=0,pad=8)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color("#c7d2d5")
        for i,v in enumerate(values): ax.text(i,v+max(values)*.035,f"{v:g}",ha="center",fontsize=15,color=INK,fontweight="bold")
    fig.text(.055,.055,"数据来源  报告表5及图5的文字说明；对应报告中的算例。",fontsize=11,color=GRAY)
    save(fig,"1.2.3","新绘_04_分阶段控制结果","突出报告中分阶段参数对电池损失和进近恢复的影响。","报告表5及图5说明","结果图")

    fig=chart("动力恢复与轨迹协调的终点高度偏差","两种恢复方式的报告结果范围  ·  高度偏差越小越接近目标")
    ax=fig.add_axes([.24,.23,.64,.51]); plot_style(ax)
    y=[1,0]; ranges=[(146,154),(23,25)]; colors=[BLUE,TEAL]
    for j,((lo,hi),c) in enumerate(zip(ranges,colors)):
        ax.plot([lo,hi],[y[j],y[j]],lw=18,color=c,solid_capstyle="round",alpha=.38)
        ax.scatter([(lo+hi)/2],[y[j]],s=135,color=c,zorder=3)
        ax.text(hi+8,y[j],f"约 {lo}–{hi} m",va="center",fontsize=16,color=INK)
    ax.set_yticks(y,["按预定时间恢复动力","按剩余高度恢复并调整轨迹"])
    ax.set_xlim(0,200); ax.set_ylim(-.6,1.6); ax.set_xlabel("终点高度偏差 / m",fontsize=14,color=INK)
    fig.text(.055,.055,"数据来源  参考报告表6；数值为报告给出的近似范围。",fontsize=11,color=GRAY)
    save(fig,"1.2.3","新绘_05_动力恢复终点偏差","可视化两种恢复方法的报告数值范围。","报告表6","结果图")

    fig=chart("气动代理模型独立验证结果","同一独立构型集  ·  参考报告中的原始尺寸模型与改进模型")
    fig.text(.055,.815,"报告版本的历史模型对比",fontsize=12,color=CORAL)
    a=fig.add_axes([.10,.22,.35,.48]); b=fig.add_axes([.58,.22,.35,.48])
    labels=["升力斜率","尾迹效率因子"]; x=np.arange(2); w=.32
    for ax,old,new,title in [(a,[.81,.71],[.88,.89],"决定系数 R²"),(b,[.0563,.0857],[.0488,.0635],"均方根误差 RMSE")]:
        ax.bar(x-w/2,old,w,label="原始尺寸模型",color=BLUE)
        ax.bar(x+w/2,new,w,label="改进模型",color=TEAL)
        ax.set_xticks(x,labels,fontsize=13); ax.grid(axis="y",color="#e5ebed"); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color("#c7d2d5"); ax.tick_params(length=0,pad=7)
        ax.set_title(title,loc="left",fontsize=16,fontweight="bold",color=INK,pad=12)
        ax.set_ylim(0,max(old+new)*1.28)
        for j,v in enumerate(old): ax.text(j-w/2,v+max(old+new)*.025,f"{v:.2f}" if ax is a else f"{v:.4f}",ha="center",fontsize=12,color=INK)
        for j,v in enumerate(new): ax.text(j+w/2,v+max(old+new)*.025,f"{v:.2f}" if ax is a else f"{v:.4f}",ha="center",fontsize=12,color=INK)
    handles, legend_labels = a.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(.055, .78),
               ncol=2, frameon=False, fontsize=11)
    fig.text(.055,.055,"数据来源  参考报告表11。该图展示报告中的历史模型结果，不代表后续更新模型。",fontsize=11,color=GRAY)
    save(fig,"2.2.2","新绘_04_报告代理模型历史结果","按报告表11重新绘制历史代理模型的R²与RMSE。","报告表11","结果图")

    algos=["NSGA-II","MOEA/D","MOPSO","遗传与粒子群混合算法"]
    fig=chart("四类多目标算法的性能比较","相同计算预算下分别展示解集质量、终代可行比例与平均耗时")
    data=[[.69,.78,.97,.89],[100.,100.,80.4,100.],[17.6,20.4,23.2,14.5]]
    titles=["Pareto解集超体积","终代可行方案比例 / %","平均计算时间 / min"]
    for i,(vals,title) in enumerate(zip(data,titles)):
        ax=fig.add_axes([.23+i*.245,.21,.18,.52]); y=np.arange(4)
        bars=ax.barh(y,vals,color=[BLUE,TEAL,AMBER,CORAL],height=.56)
        ax.invert_yaxis(); plot_style(ax); ax.set_yticks(y,algos if i==0 else [""]*4)
        ax.set_xlim(0,max(vals)*1.3); ax.set_title(title,fontsize=14,loc="left",fontweight="bold",color=INK,pad=14)
        for r,v in zip(bars,vals): ax.text(v+max(vals)*.025,r.get_y()+r.get_height()/2,f"{v:.2f}" if i==0 else f"{v:.1f}",va="center",fontsize=12,color=INK)
    fig.text(.055,.055,"数据来源  参考报告表13；各指标分别反映解集质量、可行性与计算成本。",fontsize=11,color=GRAY)
    save(fig,"2.2.3","新绘_03_四算法性能对比","将报告表13转为适合汇报的三指标并列图。","报告表13","结果图")


def write_index():
    lines=["PTDL中期汇报插图索引","",f"报告来源  {REPORT}","",
           "说明  报告原图按原文件提取。新绘方法示意图用于解释报告内容；新绘结果图的数据均来自报告中注明的表格或文字。",
           "报告中的代理模型结果为该版本报告的历史结果，不应与后续更新模型的结果混用。",
           "结果图反映报告所列算例与条件；方法图不代表新增实验或实测验证。",
           "新绘图同时提供PNG和可编辑SVG格式。原图保留报告中的PNG格式。",""]
    for sec, dirname in SECTIONS.items():
        lines += [dirname]
        for s, filename, kind, desc, src in entries:
            if s==sec: lines.append(f"  {filename}\n    类型  {kind}    来源  {src}\n    用途  {desc}")
        lines.append("")
    (ROOT/"插图索引与使用说明.txt").write_text("\n".join(lines),encoding="utf-8-sig")


if __name__ == "__main__":
    original_images()
    draw_method_figures()
    draw_result_figures()
    write_index()
    print(f"Created {len(entries)} figures in {ROOT}")
    for sec in SECTIONS:
        print(sec, sum(x[0]==sec for x in entries))
