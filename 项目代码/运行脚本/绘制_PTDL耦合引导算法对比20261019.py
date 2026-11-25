"""Publication figures from the frozen, paired 16-seed benchmark."""

import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "运行结果" / "九算法独立十六种子_20261016"
OUT = ROOT / "运行结果" / "PTDL耦合引导算法对比图_20261019"
FOCUS = "PTDL-取消在线代理"
BASELINES = [
    "NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO", "PTDL-自适应分块",
]
ALGORITHMS = BASELINES + [FOCUS]
NAMES = {
    "NSGA-II": "NSGA-II",
    "MOEA/D": "MOEA/D",
    "MOPSO": "MOPSO",
    "Hybrid NSGA-II/MOPSO": "NSGA-II/MOPSO 混合",
    "PTDL-自适应分块": "PTDL 自适应分块",
    FOCUS: "PTDL 耦合引导",
}
COLORS = dict(zip(
    ALGORITHMS, ["#4B7FA5", "#7A8E59", "#D38760", "#9480AC", "#7A8991", "#007D79"],
))
INK = "#293C43"
NOTE = "低阶 80 小时任务  ·  冻结的 258 构型气动代理  ·  16 次独立重复  ·  每次 364 次任务评价"


def read_and_validate():
    runs = pd.read_csv(SOURCE / "多种子算法结果.csv")
    history = pd.read_csv(SOURCE / "共享基准逐代超体积.csv")
    settings = json.loads((SOURCE / "评价设置.json").read_text(encoding="utf-8"))
    confirmation = json.loads(
        (SOURCE / "原耦合引导独立确认验收.json").read_text(encoding="utf-8")
    )
    runs = runs[runs.algorithm.isin(ALGORITHMS)].copy()
    history = history[history.algorithm.isin(ALGORITHMS)].copy()
    seeds = settings["seeds"]
    expected = {(a, s) for a in ALGORITHMS for s in seeds}
    if runs.duplicated(["algorithm", "seed"]).any():
        raise ValueError("Repeated run in benchmark")
    if set(zip(runs.algorithm, runs.seed)) != expected:
        raise ValueError("Incomplete paired benchmark")
    size, generations = settings["population_size"], settings["generations"]
    budget = size * (generations + 1)
    if not runs.evaluations.eq(budget).all() or runs.model_sha256.nunique() != 1:
        raise ValueError("Unequal evaluation budgets or aerodynamic models")
    if history.duplicated(["algorithm", "seed", "generation"]).any():
        raise ValueError("Duplicate history point")
    if any(set(g.generation) != set(range(generations + 1))
           for _, g in history.groupby(["algorithm", "seed"])):
        raise ValueError("Incomplete convergence histories")
    if set(zip(history.algorithm, history.seed)) != expected:
        raise ValueError("Missing convergence run")
    if history.groupby(["algorithm", "seed"]).hypervolume.apply(
        lambda x: bool((np.diff(x) < -1e-9).any())
    ).any():
        raise ValueError("Historical archive HV must be nondecreasing")
    final = history[history.generation.eq(generations)].set_index(["algorithm", "seed"])
    for row in runs.itertuples():
        if not np.isclose(final.loc[(row.algorithm, row.seed), "hypervolume"],
                          row.hypervolume, atol=1e-10, rtol=1e-10):
            raise ValueError("Final history does not match run statistics")
    paired = pd.DataFrame(confirmation["comparisons"]).set_index("comparator")
    focus = runs[runs.algorithm.eq(FOCUS)].set_index("seed").loc[seeds]
    for name in BASELINES:
        other = runs[runs.algorithm.eq(name)].set_index("seed").loc[seeds]
        difference = focus.hypervolume - other.hypervolume
        row = paired.loc[name]
        if not np.isclose(difference.mean(), row.delta_mean):
            raise ValueError("Paired differences do not match frozen statistics")
        if int((difference > 1e-12).sum()) != int(row.wins):
            raise ValueError("Win counts do not match frozen statistics")
    history["evaluations"] = (history.generation + 1) * size
    return runs, history, paired, settings


def style():
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titlepad": 13,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": "#A1ADB0",
        "xtick.color": "#52666B",
        "ytick.color": "#52666B",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "grid.color": "#E4EBED",
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def convergence(ax, history, compact=False):
    for name in ALGORITHMS:
        group = history[history.algorithm.eq(name)].groupby("evaluations").hypervolume
        median, lower, upper = group.median(), group.quantile(.25), group.quantile(.75)
        focus = name == FOCUS
        ax.fill_between(median.index, lower, upper, color=COLORS[name],
                        alpha=.13 if focus else .055, linewidth=0)
        ax.plot(median.index, median, color=COLORS[name], lw=2.8 if focus else 1.6,
                marker="o" if focus else None, markersize=3.8,
                label=NAMES[name], zorder=5 if focus else 3)
    ax.set(xlabel="累计任务评价次数", ylabel="方案集超体积 HV",
           title="(a) 相同计算预算下的收敛过程" if compact else "相同计算预算下的收敛过程",
           xlim=(28, 371), ylim=(.48, 1.13))
    ax.set_xticks([28, 112, 196, 280, 364])
    ax.grid(axis="both", alpha=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8 if compact else 9, loc="lower right",
              ncol=2 if compact else 1)
    ax.text(.02, .97, "实线为中位数，色带为中间 50% 的结果",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#61777D")


def distribution(ax, runs, compact=False):
    rng = np.random.default_rng(20261019)
    for i, name in enumerate(ALGORITHMS):
        values = runs[runs.algorithm.eq(name)].sort_values("seed").hypervolume.to_numpy()
        violin = ax.violinplot([values], positions=[i], orientation="horizontal", widths=.76,
                              showextrema=False)
        body = violin["bodies"][0]
        vertices = body.get_paths()[0].vertices
        vertices[:, 1] = np.minimum(vertices[:, 1], i)
        body.set_facecolor(COLORS[name])
        body.set_edgecolor("none")
        body.set_alpha(.23)
        ax.boxplot([values], positions=[i], orientation="horizontal", widths=.13, showfliers=False,
                   patch_artist=True, manage_ticks=False,
                   boxprops={"facecolor": "white", "edgecolor": COLORS[name], "linewidth": 1.2},
                   medianprops={"color": INK, "linewidth": 1.3},
                   whiskerprops={"color": COLORS[name]}, capprops={"color": COLORS[name]})
        ax.scatter(values, i + rng.uniform(.13, .31, len(values)),
                   c=COLORS[name], s=20 if compact else 29,
                   alpha=.85, edgecolors="white", linewidths=.45, zorder=4)
        ax.scatter([values.mean()], [i], marker="D", s=35, c=COLORS[name],
                   edgecolors="white", linewidths=.6, zorder=6)
        ax.text(1.19, i, f"{values.mean():.2f}", ha="right", va="center",
                fontweight="bold" if name == FOCUS else "normal", color=COLORS[name])
    ax.set_yticks(range(len(ALGORITHMS)), [NAMES[a] for a in ALGORITHMS])
    ax.invert_yaxis()
    ax.set(xlabel="最终历代可行方案集超体积 HV", xlim=(.69, 1.21),
           title="(b) 16 次独立重复的结果分布" if compact else "16 次独立重复的结果分布")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.text(.98, 1.025, "平均值", transform=ax.transAxes, ha="right", fontsize=9)
    if not compact:
        ax.legend(handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor="#61777D",
                   label="单次运行", markersize=5),
            Line2D([], [], marker="D", color="none", markerfacecolor="#61777D",
                   label="平均值", markersize=5),
        ], frameon=False, loc="lower right", fontsize=9)


def effect(ax, paired, compact=False):
    for i, name in enumerate(BASELINES):
        row = paired.loc[name]
        ax.errorbar(row.delta_mean, i,
                    xerr=[[row.delta_mean-row.delta_ci_low], [row.delta_ci_high-row.delta_mean]],
                    fmt="o", color=COLORS[name], ecolor=COLORS[name], capsize=5,
                    markersize=7, elinewidth=2)
        ax.text(.245, i, f"+{row.relative_gain*100:.2f}%", va="center",
                ha="left", color=COLORS[name], fontweight="bold", fontsize=10)
    ax.axvline(0, color="#7F9197", lw=1, ls="--")
    ax.set_yticks(range(len(BASELINES)), [NAMES[a] for a in BASELINES])
    ax.invert_yaxis()
    ax.set(xlabel="耦合引导减去基线的平均 HV 差值", xlim=(-.015, .315),
           title="(c) 配对提升及 95% 置信区间" if compact else "配对提升及 95% 置信区间")
    ax.set_xticks([0, .05, .10, .15, .20])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.text(.97, 1.025, "平均 HV 相对提升", transform=ax.transAxes, ha="right", fontsize=9)
    if not compact:
        ax.text(.02, -.17, "置信区间采用配对自助重采样，各比较均通过 Holm 校正后的显著性检验",
                transform=ax.transAxes, fontsize=8.5, color="#61777D")


def feasibility(ax, runs, compact=False):
    summary = runs.groupby("algorithm")[["feasible_ratio", "evaluated_feasible_ratio"]].mean()
    y = np.arange(len(ALGORITHMS))
    for i, name in enumerate(ALGORITHMS):
        final = summary.loc[name, "feasible_ratio"] * 100
        all_calls = summary.loc[name, "evaluated_feasible_ratio"] * 100
        ax.barh(i-.16, final, height=.27, color=COLORS[name], alpha=.92)
        ax.barh(i+.16, all_calls, height=.27, color=COLORS[name], alpha=.33)
        for position, value in ((i-.16, final), (i+.16, all_calls)):
            ax.text(value+1, position, f"{value:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [NAMES[a] for a in ALGORITHMS])
    ax.invert_yaxis()
    ax.set(xlabel="平均可行率 / %", xlim=(0, 118),
           title="(d) 最终种群与全部评价的可行率" if compact else "最终种群与全部评价的可行率")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#61777D", alpha=.92, label="最终种群"),
        plt.Rectangle((0, 0), 1, 1, color="#61777D", alpha=.33, label="全部评价"),
    ], frameon=False, ncol=2, loc="lower right", fontsize=8)


def paired_points(ax, runs):
    seeds = sorted(runs.seed.unique())
    selected = runs[runs.algorithm.eq(FOCUS)].set_index("seed").loc[seeds].hypervolume
    other = runs[runs.algorithm.eq("MOPSO")].set_index("seed").loc[seeds].hypervolume
    delta = (selected-other).to_numpy()
    colors = np.where(delta >= 0, COLORS[FOCUS], "#C37667")
    ax.axvline(0, color="#87999F", lw=1)
    ax.hlines(np.arange(len(delta)), 0, delta, color=colors, lw=2.2, alpha=.65)
    ax.scatter(delta, np.arange(len(delta)), c=colors, s=42, edgecolors="white", zorder=3)
    for i, value in enumerate(delta):
        label = (("下降 <0.01" if value < 0 else "提升 <0.01")
                 if abs(value) < .005 else f"{value:+.2f}")
        ax.text(max(value, 0)+.008, i, label, va="center", fontsize=9)
    ax.set_yticks(range(len(delta)), [str(i+1) for i in range(len(delta))])
    ax.invert_yaxis()
    ax.set(xlabel="耦合引导减去 MOPSO 的 HV 差值", ylabel="独立重复序号",
           title="与最强既有基线的逐次配对比较",
           xlim=(min(-.055, float(delta.min())-.012), max(.19, float(delta.max())+.045)))
    ax.grid(axis="x")
    ax.text(.98, .99, f"{int((delta > 0).sum())} 次胜出 / {len(delta)} 次比较\n"
            f"平均差值 {delta.mean():.2f}\nHolm 校正后 p = 0.00525",
            transform=ax.transAxes, ha="right", va="top", linespacing=1.8, fontsize=9)


def save(fig, stem, title, caption, images):
    fig.text(.01, .012, NOTE, color="#62767C", fontsize=8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    images.append({"stem": stem, "title": title, "caption": caption})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, history, paired, settings = read_and_validate()
    style()
    images = []
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.6))
    fig.subplots_adjust(left=.12, right=.98, bottom=.09, top=.92, hspace=.43, wspace=.43)
    fig.suptitle("PTDL 耦合引导与五种基线算法的同预算比较", fontsize=19,
                 fontweight="bold", x=.12, ha="left", y=.985)
    convergence(axes[0, 0], history, compact=True)
    distribution(axes[0, 1], runs, compact=True)
    effect(axes[1, 0], paired, compact=True)
    feasibility(axes[1, 1], runs, compact=True)
    save(fig, "01_算法对比总览", "算法综合对比",
         "收敛、结果分布、配对提升和可行率共同展示算法表现。"
         "超体积采用共享归一化基准和参考点，统计对象为全部已计数可行评价形成的历代非支配档案。",
         images)
    for stem, title, draw, caption in [
        ("02_收敛曲线", "收敛曲线", lambda ax: convergence(ax, history),
         "曲线给出 16 次运行的中位数，色带表示第 25% 至第 75% 分位。横坐标包含初始种群评价。"),
        ("03_结果分布", "独立重复结果分布", lambda ax: distribution(ax, runs),
         "每个点对应一次完整运行。半小提琴表示分布，箱体表示中间 50% 的结果，菱形表示平均值。"),
        ("05_可行率", "两个统计口径的可行率", lambda ax: feasibility(ax, runs),
         "分别统计最终种群和全部已评价候选的可行比例。耦合引导最终种群可行率为 100%，"
         "MOEA/D 的全部评价可行率最高。"),
    ]:
        fig, ax = plt.subplots(figsize=(10.6, 6.4))
        fig.subplots_adjust(left=.24 if "收敛" not in title else .10,
                            right=.96, top=.90, bottom=.16)
        draw(ax)
        save(fig, stem, title, caption, images)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.8))
    fig.subplots_adjust(left=.14, right=.97, top=.88, bottom=.18, wspace=.36)
    effect(axes[0], paired)
    paired_points(axes[1], runs)
    save(fig, "04_配对提升与最强基线", "配对提升与最强基线",
         "左图给出五项配对差值及其 95% 置信区间，右图保留与 MOPSO 的全部 16 次比较，"
         "其中 13 次提升、3 次下降。显著性采用已有冻结统计中的双侧 Wilcoxon 检验与 Holm 校正。",
         images)
    images.sort(key=lambda item: item["stem"])
    summary = runs.groupby("algorithm").agg(
        repeats=("seed", "count"), hv_mean=("hypervolume", "mean"),
        hv_std=("hypervolume", "std"), final_feasible=("feasible_ratio", "mean"),
        all_feasible=("evaluated_feasible_ratio", "mean"),
    ).loc[ALGORITHMS]
    summary.insert(0, "display_name", [NAMES[a] for a in summary.index])
    for name in BASELINES:
        summary.loc[name, "focus_relative_gain_pct"] = paired.loc[name, "relative_gain"] * 100
        summary.loc[name, "focus_wins"] = paired.loc[name, "wins"]
        summary.loc[name, "p_holm"] = paired.loc[name, "p_holm"]
    summary.to_csv(OUT / "绘图汇总数据.csv", encoding="utf-8-sig")
    runs.to_csv(OUT / "逐次运行数据.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUT / "收敛绘图数据.csv", index=False, encoding="utf-8-sig")
    paired.loc[BASELINES].to_csv(OUT / "配对差值与区间.csv", encoding="utf-8-sig")
    files = ["多种子算法结果.csv", "共享基准逐代超体积.csv",
             "评价设置.json", "原耦合引导独立确认验收.json"]
    manifest = {
        "source_directory": str(SOURCE), "algorithms": ALGORITHMS,
        "name_mapping": NAMES, "runs": len(runs), "seeds": 16,
        "evaluations_per_run": 364, "task_evaluations_total": int(runs.evaluations.sum()),
        "source_sha256": {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                          for name in files},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization": settings["normalization"],
        "reference": settings["reference"], "figures": images,
        "inference_scope": "Frozen 258-configuration surrogate and low-order 80-hour task",
        "runtime_comparison": "Excluded because concurrent worker contention affects timings",
        "checks": ["paired seeds complete", "equal budgets and model hash",
                   "histories complete and nondecreasing", "final HV matches history",
                   "paired means and wins match frozen confirmation"],
    }
    (OUT / "图表数据说明.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    body = "".join(
        f"<figure id='f{i}'><h2>{html.escape(item['title'])}</h2>"
        f"<img src='{item['stem']}.png' alt='{html.escape(item['title'])}'>"
        f"<figcaption>{html.escape(item['caption'])}</figcaption>"
        f"<p><a href='{item['stem']}.png'>高清 PNG</a> · "
        f"<a href='{item['stem']}.pdf'>矢量 PDF</a> · "
        f"<a href='{item['stem']}.svg'>可编辑 SVG</a></p></figure>"
        for i, item in enumerate(images)
    )
    (OUT / "index.html").write_text(
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>PTDL 耦合引导算法对比</title><style>"
        "body{margin:0;background:#f4f7f7;color:#293c43;font-family:'Microsoft YaHei',sans-serif}"
        "main{max-width:1450px;margin:auto;padding:30px}h1{font-size:27px}"
        "h2{font-size:19px}p,figcaption{line-height:1.8}figure{margin:34px 0;background:white;"
        "padding:22px}img{width:100%;height:auto;display:block}a{color:#007d79}"
        "@media(max-width:600px){main{padding:14px}figure{padding:10px}}</style>"
        "<main><h1>PTDL 耦合引导算法对比</h1>"
        f"<p>{NOTE}</p><p>平均超体积相对 MOPSO 提升 5.10%，"
        "16 次配对比较中胜出 13 次。五项基线比较均通过 Holm 校正后的显著性检验。</p>"
        "<p>图中保留全部独立重复及两种可行率。结论适用于本组冻结模型与评价预算。</p>"
        + body + "</main></html>", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), "figures": len(images),
                      "runs": len(runs), "evaluations": int(runs.evaluations.sum())},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
