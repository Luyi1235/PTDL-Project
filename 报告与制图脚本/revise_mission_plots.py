from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

data_path = Path(r"D:\LDTP\LDTP\PTDL飞推控协同优化代码包\运行结果\JSBSim地面起降80h联算_20261012\JSBSim候选\candidate_0001\PTDL_JSBSim六阶段姿态轨迹航迹动力闭环.csv")
desktop = Path(r"C:\Users\pc\Desktop")
data = pd.read_csv(data_path)
assert len(data) == 493 and data.time_s.is_monotonic_increasing
t = data.time_s / 3600

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#68717A",
    "axes.linewidth": 0.7, "grid.color": "#E4E9EB", "grid.linewidth": 0.6,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
blue, green, red = "#4285A5", "#439A86", "#D47C69"
fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5))
ax = axes[0, 0]
ax.plot(t, data.altitude_m, color=blue, lw=1.5)
ax.set_ylabel("高度 / m")
ax.set_title("(a) 空中任务高度", loc="left", pad=10)
ax = axes[0, 1]
ax.plot(t, data.fuel_used_kg, color=red, lw=1.5)
ax.set_ylabel("累计燃油消耗 / kg")
ax.set_title("(b) 能源累积消耗", loc="left", pad=10)

ax = axes[1, 0]
engine, = ax.plot(t, data.engine_power_kw, color=blue, lw=1.2, label="发动机供电")
ax.set_ylabel("发动机供电 / kW", color=blue)
ax.tick_params(axis="y", colors=blue)
ax.set_ylim(0, 70)
ax.set_title("(c) 推进功率分配", loc="left", pad=10)
battery_ax = ax.twinx()
battery_ax.spines["right"].set_visible(True)
battery, = battery_ax.plot(t, data.motor_bus_power_kw, color=green, lw=1.3, label="电池辅助功率")
battery_ax.set_ylabel("电池功率 / kW", color=green, labelpad=5)
battery_ax.tick_params(axis="y", colors=green)
battery_ax.set_ylim(-3, 3)
battery_ax.set_yticks([-3, -1.5, 0, 1.5, 3])
ax.legend(handles=[engine, battery], frameon=False, loc="upper right")
power_inset = ax.inset_axes([0.10, 0.54, 0.39, 0.40])
early = t <= 2
power_inset.axhline(0, color="#888888", lw=0.6)
power_inset.plot(t[early], data.loc[early, "motor_bus_power_kw"], color=green,
                 lw=1.1, marker="o", markersize=2.0)
power_inset.set_xlim(0, 2)
power_inset.set_ylim(-2.7, 2.7)
power_inset.set_xticks([0, 1, 2])
power_inset.set_yticks([-2, 0, 2])
power_inset.tick_params(labelsize=7, length=2)
power_inset.grid(True, linewidth=0.4)
power_inset.set_title("电池功率（0–2 h）", fontsize=8, pad=2)
power_inset.set_xlabel("时间 / h", fontsize=7, labelpad=0)

ax = axes[1, 1]
ax.plot(t, data.soc * 100, color=green, lw=1.5)
ax.set_ylim(80, 100)
ax.set_yticks([80, 85, 90, 95, 100])
ax.set_ylabel("电池剩余电量比例 / %")
ax.set_title("(d) 储能状态传递", loc="left", pad=10)
inset = ax.inset_axes([0.45, 0.48, 0.51, 0.47])
inset.plot(t[early], data.loc[early, "soc"] * 100, color=green,
           lw=1.2, marker="o", markersize=2.2)
inset.set_xlim(0, 2)
inset.set_ylim(84, 97)
inset.set_xticks([0, 1, 2])
inset.set_yticks([85, 90, 95])
inset.tick_params(labelsize=7, length=2)
inset.grid(True, linewidth=0.45)
inset.set_title("起始阶段（0–2 h）", fontsize=8, pad=2)
inset.set_xlabel("时间 / h", fontsize=7, labelpad=0)
for ax in axes.flat:
    ax.set_xlim(0, 84)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("任务时间 / h")
    ax.set_axisbelow(True)
    ax.grid(True)
fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.1, wspace=0.63, hspace=0.55)
fig.savefig(desktop / "长航时任务与能源协同_坐标优化版.png", dpi=300, bbox_inches="tight", pad_inches=0.16)
plt.close(fig)

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "font.family": "sans-serif",
                     "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9})
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
ax = axes[0]
ax.plot(t, data.altitude_target_m, color="#cf826b", lw=1.4, ls="--", label="高度指令")
ax.plot(t, data.altitude_m, color="#2c668d", lw=1.35, label="实际高度")
ax.set_ylabel("高度 / m")
ax.legend(loc="lower left", bbox_to_anchor=(0.04, 0.08), ncol=2,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
ax = axes[1]
ax.plot(t, data.engine_power_kw, color="#2c668d", lw=1.35, label="发动机—发电机供电")
ax.set_ylabel("供电功率 / kW")
ax.legend(loc="upper right", frameon=False)
ax = axes[2]
ax.axhline(0, color="#555555", lw=0.7)
ax.plot(t, data.battery_terminal_power_kw, color="#d27662", lw=1.2, label="电池端功率")
ax.set_ylabel("电池功率 / kW")
ax.set_xlabel("空中任务时间 / h")
ax.legend(loc="upper right", frameon=False)
for ax in axes:
    ax.grid(axis="y", alpha=0.23)
    ax.set_xlim(0, 82)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.savefig(desktop / "任务剖面与供电变化_图例调整版.png", dpi=300, facecolor="white")
plt.close(fig)

print("Created two figures from", data_path)
print("Initial SOC samples:", data.soc.iloc[0] * 100, data.loc[data.time_s.eq(4800), "soc"].iloc[0] * 100)
