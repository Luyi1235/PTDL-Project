from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

base = Path(r"D:\LDTP\power_diagnosis")
name = "JSBSim候选/candidate_0001/PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv"
data = {case: pd.read_csv(base / case / name) for case in ("period2", "gain2", "gain2_batt")}
plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 11,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
for case, label, color in (
    ("period2", "原参数 Kp=4", "#c65a48"),
    ("gain2", "调整参数 Kp=2", "#2576a3"),
):
    df = data[case]
    segment = df[(df.time_s >= 10800) & (df.time_s <= 11100)]
    x = segment.time_s - 10800
    axes[0, 0].plot(x, segment.vertical_speed_up_mps, color=color, label=label, lw=1.25)
    axes[0, 1].plot(x, segment.engine_power_kw, color=color, label=label, lw=1.25)
    segment = df[(df.time_s >= 7200) & (df.time_s <= 14000)]
    axes[1, 0].plot(segment.time_s / 3600, segment.altitude_m, color=color, label=label, lw=1.25)
axes[1, 0].axhline(2400, color="#666666", linestyle="--", lw=0.9, label="高度指令")
for case, label, color in (
    ("gain2", "巡航电池上限约 0.006 kW", "#2576a3"),
    ("gain2_batt", "巡航电池上限 2 kW", "#d07742"),
):
    segment = data[case]
    segment = segment[(segment.time_s >= 7200) & (segment.time_s <= 14000)]
    axes[1, 1].plot(segment.time_s / 3600, 100 * segment.soc, label=label, color=color, lw=1.5)

for ax, title, ylabel, xlabel in zip(
    axes.flat,
    ("（a）巡航垂直速度", "（b）发动机供电功率", "（c）巡航高度", "（d）电池剩余电量"),
    ("垂直速度 / m/s", "发动机供电功率 / kW", "高度 / m", "SOC / %"),
    ("相对时间 / s", "相对时间 / s", "任务时间 / h", "任务时间 / h"),
):
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, loc="best")

target = Path(r"C:\Users\pc\Desktop\巡航供电波动_控制参数诊断对比.png")
fig.savefig(target, dpi=250)
print(target)
