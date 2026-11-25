"""Export research figures from completed, versioned solver and optimizer results."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, LogLocator
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO气动代理模型 import load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign

RESULTS = ROOT / "运行结果"
OUT = RESULTS / "科研图集_20261015"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = ["#3487A3", "#55A693", "#D77A65", "#8D79AD", "#D3A348", "#727F88", "#9DB69C"]
ALGORITHMS = ["NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO",
              "PTDL-自适应分块", "PTDL-无分块", "PTDL-固定算子"]
NAMES = ["NSGA-II", "MOEA/D", "MOPSO", "NSGA-II 与 MOPSO 混合",
         "PTDL 自适应分块", "取消分块", "固定算子比例"]
POLICIES = dict(rule="规则能量管理", fuzzy="模糊能量管理", mpc="模型预测能量管理",
                scheduled="分阶段模型预测能量管理", neural="神经网络模仿能量管理")
LABELS = ["J_energy_mj", "J_stability", "J_response_s"]
manifest = []
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#87939B", "axes.labelcolor": "#36434C", "text.color": "#29353D",
    "xtick.color": "#59666E", "ytick.color": "#59666E", "grid.color": "#DCE3E7",
    "grid.linewidth": .7, "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def save(fig, stem, source, explanation):
    fig.savefig(OUT / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append(dict(figure=stem, source=source, interpretation=explanation))


def tidy(ax):
    ax.grid(axis="y", alpha=.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(LogLocator(base=10) if ax.get_yscale() == "log" else MaxNLocator(5))


def parity():
    folder = RESULTS / "物理特征代理_20261015"
    f = pd.read_csv(folder / "新独立验证_逐点.csv")
    old = load_vsp_surrogate(ROOT / "气动代理模型数据" / "气动代理模型_211构型_分目标GP_20261013.json")
    features = old.payload["feature_labels"]
    old_predictions = np.array([old.predict(AirframeDesign(**{k: float(r[k]) for k in features}))
                                for _, r in f.iterrows()])
    metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), layout="constrained")
    targets = ["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency"]
    titles = ["(a) 升力斜率", "(b) 尾迹效率因子 Ew"]
    for i, ax in enumerate(axes):
        good = f["scorable_"+targets[i]].fillna(False).to_numpy(bool, copy=True)
        y = f[targets[i]].to_numpy(float)
        good &= np.isfinite(y)
        y = y[good]
        predictions = [old_predictions[good, i], f[f"prediction_{i}"].to_numpy(float)[good]]
        lo = min(y.min(), *(p.min() for p in predictions))
        hi = max(y.max(), *(p.max() for p in predictions))
        pad = .08*(hi-lo)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="#6F7C84", lw=1, ls="--")
        for j, (name, pred) in enumerate(zip(["原分目标高斯过程", "物理特征高斯过程"], predictions)):
            r2 = float(r2_score(y, pred))
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            ax.scatter(y, pred, s=44, color=COLORS[j], marker=["o", "D"][j],
                       edgecolor="white", linewidth=.7, alpha=.85,
                       label=f"{name}   R²={r2:.2f}")
            metrics.append(dict(model=name, target=targets[i], count=len(y), r2=r2, rmse=rmse))
        ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), title=titles[i],
               xlabel="独立 VSPAERO 实算值", ylabel="代理预测值")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.text(.97, .04, f"有效构型 {len(y)} / {len(f)}", ha="right", transform=ax.transAxes)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        tidy(ax)
    pd.DataFrame(metrics).to_csv(OUT / "独立验证模型比较.csv", index=False)
    f["incumbent_prediction_lift"], f["incumbent_prediction_ew"] = old_predictions.T
    f.to_csv(OUT / "独立验证绘图数据.csv", index=False)
    save(fig, "01_代理独立验证", str(folder),
         "同一批预先锁定的24个新构型；无有效Ew的构型保留失败状态，Ew仅22个可评分。测试集未用于模型选择。")


def mesh():
    folder = RESULTS / "网格加密对照_20261015"
    f = pd.read_csv(folder / "网格加密比较.csv")
    f = f.drop_duplicates(["source_sample", "mesh_level"], keep="last")
    f = f[f.source_sample == "baseline_ultra_reference"].set_index("mesh_level").loc[
        ["legacy", "medium", "fine"]].reset_index()
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), layout="constrained")
    keys = ["vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency", "solver_wall_time_s"]
    titles = ["(a) 升力斜率", "(b) 尾迹效率因子 Ew", "(c) 五迎角极曲线求解时间"]
    for i, ax in enumerate(axes):
        y = f[keys[i]].to_numpy(float, copy=True)
        if i == 2:
            y /= 60
        ax.plot(x, y, "-o", color=COLORS[i], markersize=7, lw=2)
        for xx, yy in zip(x, y):
            ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 9), textcoords="offset points", ha="center")
        ax.set_xticks(x, ["原始网格", "中等加密", "细网格"])
        ax.set_title(titles[i])
        ax.set_ylabel(["每弧度", "Ew", "分钟"][i])
        ax.margins(y=.25, x=.1)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        tidy(ax)
    f.to_csv(OUT / "网格对照绘图数据.csv", index=False)
    costs = f.set_index("mesh_level").solver_wall_time_s
    (folder / "运行时间基准.json").write_text(json.dumps({
        "relative_cost": (costs/costs["legacy"]).to_dict(),
        "basis": "same geometry, five-angle polar, wake iterations 60",
        "limitation": "ratios are approximate for stepwise and other geometries",
    }, indent=2), encoding="utf8")
    save(fig, "02_相同构型网格加密对照", str(folder),
         "几何尺寸相同，尾迹迭代60次。Ew仍随网格明显变化，当前结果用于识别标签敏感性。")


def nondominated(frame):
    v = frame[LABELS].to_numpy(float)
    keep = np.ones(len(v), bool)
    for i in range(len(v)):
        keep[i] = not np.any(np.all(v <= v[i], axis=1) & np.any(v < v[i], axis=1))
    return frame.loc[keep].drop_duplicates(LABELS)


def algorithms(folder_name, prefix):
    folder = RESULTS / folder_name
    if not (folder / "多种子算法结果.csv").exists():
        raise RuntimeError(f"Incomplete algorithm comparison: {folder}")
    runs = pd.read_csv(folder / "多种子算法结果.csv")
    hv = pd.read_csv(folder / "共享基准逐代超体积.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), layout="constrained")
    for i, algorithm in enumerate(ALGORITHMS):
        h = hv[hv.algorithm == algorithm]
        q = h.groupby("generation").hypervolume.quantile([.25, .5, .75]).unstack()
        axes[0].plot(q.index, q[.5], color=COLORS[i], lw=1.8, label=NAMES[i])
        axes[0].fill_between(q.index, q[.25], q[.75], color=COLORS[i], alpha=.12)
        values = runs[runs.algorithm == algorithm].hypervolume.to_numpy()
        if len(values) > 1 and np.ptp(values) > 1e-8:
            violin = axes[1].violinplot(values, positions=[i], widths=.75, showextrema=False)
            violin["bodies"][0].set_facecolor(COLORS[i])
            violin["bodies"][0].set_alpha(.16)
        axes[1].boxplot([values], positions=[i], widths=.27, patch_artist=True,
                       showfliers=False, boxprops=dict(facecolor=COLORS[i], alpha=.55),
                       medianprops=dict(color="#263943"))
        axes[1].scatter(i + np.linspace(-.13, .13, len(values)), values, color=COLORS[i], s=20, zorder=4)
    axes[0].set(title="(a) 收敛过程与种子间差异", xlabel="迭代代数", ylabel="共享归一化基准下的超体积")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    axes[1].set(title="(b) 各随机种子的最终解集表现", ylabel="最终超体积")
    axes[1].set_xticks(range(7), ["NSGA-II", "MOEA/D", "MOPSO", "混合算法", "自适应\n分块", "取消\n分块", "固定\n算子"])
    for ax in axes:
        tidy(ax)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    save(fig, f"{prefix}_算法收敛与分布", str(folder),
         "相同种群、代数、随机种子和候选评价预算。曲线为中位数，色带为四分位范围。不同指标定义的实验分开绘图。")
    if "改进响应" not in folder_name:
        return
    fronts = pd.read_csv(folder / "逐代可行Pareto.csv")
    final = fronts[fronts.generation == runs.generations.max()]
    chosen = []
    fig = plt.figure(figsize=(16, 9.4), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.2, h_pad=.15, wspace=.18, hspace=.12)
    for i, algorithm in enumerate(ALGORITHMS[:5]):
        f = nondominated(final[final.algorithm == algorithm])
        chosen.append(f.assign(algorithm=algorithm))
        ax = fig.add_subplot(2, 3, i+1, projection="3d")
        ax.scatter(f.J_energy_mj/1000, f.J_stability, f.J_response_s,
                   color=COLORS[i], s=25, alpha=.8, edgecolor="white", linewidth=.35)
        ax.set_title(f"({chr(97+i)}) {NAMES[i]}\n{len(f)} 个可行非支配方案", fontsize=10, pad=8)
        ax.set_xlabel("任务能量 / GJ", labelpad=6)
        ax.set_ylabel("轨迹跟踪误差", labelpad=7)
        ax.set_zlabel("响应综合时间 / s", labelpad=9)
        ax.view_init(elev=23, azim=-55)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.pane.fill = False
        ax.tick_params(labelsize=8)
    merged = pd.concat(chosen)
    for ax in fig.axes:
        for setter, col, scale in [(ax.set_xlim, "J_energy_mj", 1000),
                                   (ax.set_ylim, "J_stability", 1),
                                   (ax.set_zlim, "J_response_s", 1)]:
            lo, hi = merged[col].min()/scale, merged[col].max()/scale
            pad = max(.02*(hi-lo), 1e-6)
            setter(lo-pad, hi+pad)
    ax = fig.add_subplot(2, 3, 6)
    for i, algorithm in enumerate(ALGORITHMS[:5]):
        f = merged[merged.algorithm == algorithm]
        ax.scatter(f.J_energy_mj/1000, f.J_response_s, color=COLORS[i], s=20, alpha=.7, label=NAMES[i])
    ax.set(title="(f) 能量与响应投影", xlabel="任务能量 / GJ", ylabel="响应综合时间 / s")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax)
    merged.to_csv(OUT / "改进响应Pareto绘图数据.csv", index=False)
    save(fig, "05_改进响应三目标Pareto", str(folder),
         "三个坐标使用实际目标值，各算法共用轴范围。前沿来自各算法所有种子的最终历代档案并集，再作非支配筛选。")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), layout="constrained")
    focus = runs[runs.algorithm == ALGORITHMS[4]].set_index("seed")
    for i, name in enumerate(ALGORITHMS[5:]):
        other = runs[runs.algorithm == name].set_index("seed")
        for seed in focus.index.intersection(other.index):
            axes[0].plot([2*i, 2*i+1], [other.loc[seed].hypervolume, focus.loc[seed].hypervolume],
                         color=COLORS[5+i], alpha=.65, marker="o", lw=1)
    axes[0].set_xticks(range(4), ["取消分块", "完整方法", "固定算子", "完整方法"])
    axes[0].set(title="(a) 相同种子的消融配对", ylabel="最终超体积")
    for i, a in enumerate(ALGORITHMS):
        r = runs[runs.algorithm == a]
        axes[1].scatter(r.runtime_s/60, r.feasible_ratio*100, color=COLORS[i], s=35, label=NAMES[i])
    axes[1].set(title="(b) 耗时与最终种群可行率", xlabel="单次优化耗时 / min", ylabel="可行率 / %")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        tidy(ax)
    save(fig, "06_改进算法消融与计算开销", str(folder),
         "每条连线对应相同随机种子。耗时受并行资源竞争影响，图中保留各次运行原始测量。")


def responses():
    folder = RESULTS / "响应指标与控制参数_可行基准_20261015"
    rows = pd.read_csv(folder / "响应与能耗汇总.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    for i, (policy, name) in enumerate(POLICIES.items()):
        sample = rows[(rows.policy == policy) & (rows.wind_mps == 8) &
                      (rows.engine_delay_s == 0) & (rows.response_step_s == 1)].iloc[0]
        f = pd.read_csv(Path(sample.folder) / "响应轨迹.csv")
        axes[0, 0].plot(f.time_from_event_start_s, f.delta_speed_mps, color=COLORS[i],
                        lw=1.6, ls=["-", "--", "-.", ":", "-"][i], label=name)
        axes[0, 1].plot(f.time_from_event_start_s, f.p_motor_kw, color=COLORS[i],
                        lw=1.6, ls=["-", "--", "-.", ":", "-"][i], label=name)
        s = rows[(rows.policy == policy) & (rows.engine_delay_s == 0) & (rows.response_step_s == 1)].sort_values("wind_mps")
        axes[1, 0].plot(s.wind_mps, s.J_response_s, "-o", color=COLORS[i], lw=1.8, label=name)
        for _, r in s.iterrows():
            if not r.feasible:
                axes[1, 0].scatter(r.wind_mps, r.J_response_s, s=80, marker="x", color="#303D45", zorder=5)
        axes[1, 1].scatter(s.J_response_s, s.fuel_kg, color=COLORS[i], s=45, label=name)
    axes[0, 0].axhspan(-.5, .5, color="#9AB3B0", alpha=.12)
    for ax in axes[0]:
        ax.axvspan(0, 61.7, color="#D3A348", alpha=.12)
        ax.set_xlim(-20, 600)
        ax.set_xlabel("距扰动开始的时间 / s")
    axes[0, 0].set(title="(a) 速度偏差与响应判定带", ylabel="受扰与名义轨迹速度差 / (m/s)")
    axes[0, 0].text(.98, .04, "相同飞行控制律下部分曲线重合", ha="right",
                    transform=axes[0, 0].transAxes, fontsize=8)
    axes[0, 1].set(title="(b) 电池充放电功率响应", ylabel="电池母线功率 / kW")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].set(title="(c) 响应综合时间随风扰强度变化", xlabel="等效风速阶跃 / (m/s)", ylabel="响应综合时间 / s")
    axes[1, 0].set_xticks([4, 8, 12])
    axes[1, 0].text(.02, .97, "× 表示存在设备或状态约束超限", va="top", transform=axes[1, 0].transAxes, fontsize=8)
    axes[1, 1].set(title="(d) 燃油消耗与响应权衡", xlabel="响应综合时间 / s", ylabel="任务燃油消耗 / kg")
    for ax in axes.flat:
        tidy(ax)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    save(fig, "07_能量管理与扰动响应", str(folder),
         "同一构型、功率配置、风扰事件。阴影为61.70秒扰动。约束超限的策略仍保留并标记，未作为可行推荐。")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    s = rows[(rows.policy == "rule") & (rows.wind_mps == 8) & (rows.engine_delay_s == 0)].sort_values("response_step_s")
    axes[0].plot(s.response_step_s, s.J_response_s, "-o", color=COLORS[0], lw=2)
    for _, r in s.iterrows():
        axes[0].annotate(f"{r.J_response_s:.2f}", (r.response_step_s, r.J_response_s),
                         textcoords="offset points", xytext=(0, 8), ha="center")
    axes[0].set(title="(a) 响应时间步敏感性", xlabel="响应区间积分步长 / s", ylabel="响应综合时间 / s")
    axes[0].margins(y=.25)
    s = rows[(rows.policy == "rule") & (rows.engine_delay_s == 0) & (rows.response_step_s == 1)].sort_values("wind_mps")
    x = np.arange(len(s))
    axes[1].bar(x-.25, s.paired_recovery_time_s, width=.25, color=COLORS[0], label="扰动结束后的恢复时间")
    axes[1].bar(x, s.mode_switch_time_s, width=.25, color=COLORS[2], label="模式切换时间")
    axes[1].bar(x+.25, s.response_normalized_error_integral_s, width=.25, color=COLORS[1], label="扰动误差归一化积分")
    axes[1].set_xticks(x, [f"{v:.0f} m/s" for v in s.wind_mps])
    axes[1].set(title="(b) 原响应目标各分量与新积分分量", ylabel="时间 / s")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        tidy(ax)
    save(fig, "08_响应指标定义与步长敏感性", str(folder),
         "恢复时间为零可由扰动结束时已进入判定带产生。新增积分分量保留扰动期间的偏差；积分步长仍有明显影响。")
    rows.to_csv(OUT / "控制器响应绘图数据.csv", index=False)


def flight_gains():
    folder = RESULTS / "响应指标与控制参数_可行基准_20261015"
    f = pd.read_csv(folder / "飞控参数联合搜索.csv")
    valid = f.feasible & (f.response_recovery_success > .5)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.18, h_pad=.12, wspace=.2)
    ax = axes[0]
    im = ax.scatter(f.loc[valid, "J_energy_mj"]/1000, f.loc[valid, "J_response_s"],
                    c=f.loc[valid, "speed_tracking_time_constant_s"], cmap="viridis",
                    s=60, edgecolor="white", linewidth=.7)
    ax.scatter(f.loc[~valid, "J_energy_mj"]/1000, f.loc[~valid, "J_response_s"],
               color="#ABB4B9", marker="x", s=22, label="约束超限或恢复未通过")
    ax.set(title="(a) 飞行控制参数进入联合搜索", xlabel="任务能量 / GJ", ylabel="响应综合时间 / s")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.colorbar(im, ax=ax, label="速度跟踪时间常数 / s", shrink=.8)
    accepted = f.loc[valid]
    im2 = axes[1].scatter(accepted.speed_tracking_time_constant_s, accepted.gamma_tracking_gain,
                         c=accepted.J_response_s, cmap="viridis_r", s=60,
                         edgecolor="white", linewidth=.7)
    axes[1].set(title="(b) 通过恢复判据的控制参数分布", xlabel="速度跟踪时间常数 / s",
                ylabel="航迹倾角反馈增益")
    fig.colorbar(im2, ax=axes[1], label="响应综合时间 / s", shrink=.8)
    for ax in axes:
        tidy(ax)
    f.to_csv(OUT / "飞控参数联合搜索绘图数据.csv", index=False)
    save(fig, "10_飞行控制参数联合优化", str(folder),
         "同时搜索几何、功率分配和三个低阶飞控参数，共60次评价。失败恢复与约束超限保留灰色标记。该图未采用JSBSim飞控增益搜索。")


def correlations():
    f = pd.read_csv(OUT / "改进响应Pareto绘图数据.csv").drop_duplicates(
        ["wing_area_m2", "aspect_ratio", "hybrid_system_mass_kg", "J_energy_mj", "J_response_s"])
    features = ["wing_area_m2", "aspect_ratio", "hybrid_system_mass_kg", "terminal_soc"]
    if not all(k in f for k in features):
        features[-1] = "speed_endurance_mps"
    features = [k for k in features if k in f]
    labels = {"wing_area_m2": "机翼面积", "aspect_ratio": "展弦比",
              "hybrid_system_mass_kg": "混动系统质量", "terminal_soc": "任务末端电量",
              "speed_endurance_mps": "续航速度"}
    corr = f[features+LABELS].corr(method="spearman").loc[LABELS, features].fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), layout="constrained")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("research", ["#6097AB", "#F6F6F2", "#D77A65"])
    image = axes[0].imshow(corr, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    axes[0].set_xticks(range(len(features)), [labels[k] for k in features])
    axes[0].set_yticks(range(3), ["任务能量", "轨迹跟踪误差", "响应综合时间"])
    for i in range(3):
        for j in range(len(features)):
            axes[0].text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
    axes[0].set_title("(a) 设计参数与目标的秩相关系数")
    fig.colorbar(image, ax=axes[0], orientation="horizontal", fraction=.07, pad=.14)
    ax = axes[1]
    n = len(features)+3
    angles = np.linspace(.4, 2*np.pi+.4, n, endpoint=False)
    positions = np.c_[np.cos(angles), np.sin(angles)]
    for i in range(3):
        for j in range(len(features)):
            value = corr.iloc[i, j]
            a, b = positions[j], positions[len(features)+i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COLORS[2] if value >= 0 else COLORS[0],
                    lw=.55+2*abs(value), alpha=.12+.78*abs(value), ls="-" if value >= 0 else "--")
    texts = [labels[k] for k in features]+["任务能量", "轨迹跟踪误差", "响应综合时间"]
    for i, ((x, y), label) in enumerate(zip(positions, texts)):
        ax.scatter(x, y, s=95, color=COLORS[1] if i < len(features) else COLORS[0],
                   edgecolor="white", zorder=3)
        ax.text(x*1.15, y*1.15, label, ha="left" if x > .1 else "right" if x < -.1 else "center",
                va="center", fontsize=9)
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.4, 1.4), title="(b) 全部参数与目标关联")
    ax.set_aspect("equal")
    ax.axis("off")
    corr.to_csv(OUT / "设计目标相关系数.csv")
    save(fig, "09_设计参数与目标关联", "改进响应Pareto绘图数据.csv",
         "全部参数与目标连线均保留，关联越弱线越淡。相关性描述当前可行前沿内的统计联系。")


if __name__ == "__main__":
    parity()
    mesh()
    algorithms("七算法H6多种子_20261015", "03_八种子历史响应")
    algorithms("七算法改进响应_20261015", "04_四种子改进响应")
    responses()
    correlations()
    flight_gains()
    (OUT / "图集说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    import html
    sections = []
    for item in manifest:
        stem = html.escape(item["figure"])
        sections.append(f'<figure><a href="{stem}.png"><img src="{stem}.png" alt="{stem}" loading="lazy"></a>'
                        f'<figcaption><h2>{stem.replace("_", " ")}</h2>'
                        f'<p>{html.escape(item["interpretation"])}</p>'
                        f'<a href="{stem}.pdf">PDF</a> <a href="{stem}.png">PNG</a></figcaption></figure>')
    index = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1"><title>PTDL 科研图集</title>'
             '<style>body{margin:0;background:#f7f9fa;color:#273b44;font:15px "Microsoft YaHei",sans-serif}'
             'main{max-width:1440px;margin:0 auto;padding:24px}h1{font-size:26px;margin:0 0 12px}'
             'h2{font-size:17px;margin:0}p{line-height:1.7}a{color:#287a91;margin-right:16px}'
             '.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}'
             'figure{margin:0;background:white;border:1px solid #dce4e8;border-radius:6px;overflow:hidden}'
             'img{width:100%;height:auto;display:block}figcaption{padding:16px}'
             '@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:14px}}</style>'
             '<main><h1>PTDL 科研图集</h1><p>2026 年 10 月 15 日计算结果</p>'
             '<div class="grid">'+''.join(sections)+'</div></main></html>')
    (OUT / "index.html").write_text(index, encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
