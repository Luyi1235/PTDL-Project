"""Small, exactly enumerated energy-dispatch study; no production changes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "运行结果" / "算法选题机制验证_20261024"
MAP = ROOT / "核心模型" / "动力模型" / "engine_map_genset" / "EngineMap_60Case.csv"
MODES = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
DURATION = np.array([0.20, 0.65, 0.15])
WEIGHTS = np.linspace(0.05, 0.95, 19)
OBJECTIVE_SCALE = np.array([25.0, 40.0])


def make_model(engine_scale: float, battery_scale: float, operating_map: np.ndarray):
    # Convex mixing of the five throttle points defines this reduced model.
    # Ramps, battery resistance and flight states are omitted.
    count = len(operating_map)
    size = 3 * count + 6 + 4
    discharge = np.arange(3 * count, 3 * count + 3)
    charge = discharge + 3
    energy = np.arange(size - 4, size)
    fuel = np.zeros(size)
    throughput = np.zeros(size)
    capacity = 299.7 * 55.65 / 1000.0 * battery_scale
    load = np.array([42.0, 24.0, 48.0]) * (
        1.0 + 0.035 * (engine_scale - 1) + 0.06 * (battery_scale - 1)
    )
    equality = []
    right = []
    for phase in range(3):
        row = np.zeros(size)
        row[phase * count:(phase + 1) * count] = operating_map[:, 0] * engine_scale
        row[discharge[phase]] = 1.0
        row[charge[phase]] = -1.0
        equality.append(row)
        right.append(load[phase])
        row = np.zeros(size)
        row[energy[phase + 1]] = 1.0
        row[energy[phase]] = -1.0
        row[discharge[phase]] = DURATION[phase]
        row[charge[phase]] = -DURATION[phase]
        equality.append(row)
        right.append(0.0)
        fuel[phase * count:(phase + 1) * count] = (
            DURATION[phase] * operating_map[:, 1] * engine_scale
        )
        throughput[discharge[phase]] = DURATION[phase]
        throughput[charge[phase]] = DURATION[phase]
    bounds = [(0.0, None)] * (3 * count)
    bounds += [(0.0, 30.0)] * 6
    bounds += [(0.2 * capacity, capacity)] * 4
    bounds[energy[0]] = (0.9 * capacity, 0.9 * capacity)
    bounds[energy[-1]] = (0.5 * capacity, capacity)
    return {
        "count": count, "size": size, "a_eq": np.array(equality),
        "b_eq": np.array(right), "bounds": bounds,
        "objectives": np.stack([fuel, throughput]),
        "capacity": capacity, "load": load,
        "generator_min": float(operating_map[:, 0].min() * engine_scale),
        "generator_max": float(operating_map[:, 0].max() * engine_scale),
    }


def reachable_modes(model):
    """Established interval reachability baseline for this ideal 1-D storage."""
    result = []
    for mode in MODES:
        low = high = 0.9 * model["capacity"]
        possible = True
        for phase, duration in enumerate(DURATION):
            generator_low = max(mode[phase] * model["generator_min"],
                                model["load"][phase] - 30.0)
            generator_high = min(mode[phase] * model["generator_max"],
                                 model["load"][phase] + 30.0)
            low = max(0.2 * model["capacity"],
                      low + (generator_low - model["load"][phase]) * duration)
            high = min(model["capacity"],
                       high + (generator_high - model["load"][phase]) * duration)
            if generator_low > generator_high + 1e-8 or low > high + 1e-8:
                possible = False
                break
        possible &= high >= 0.5 * model["capacity"] - 1e-8
        result.append(possible)
    return np.array(result)


def solve(model, weight, mode=None):
    objective = np.array([weight, 1 - weight]) / OBJECTIVE_SCALE
    c = objective @ model["objectives"]
    count, size = model["count"], model["size"]
    rows = np.zeros((3, size))
    for phase in range(3):
        rows[phase, phase * count:(phase + 1) * count] = 1.0
    if mode is None:
        # Sum of throttle weights in [0,1] relaxes binary engine status.
        result = linprog(c, A_ub=rows, b_ub=np.ones(3),
                         A_eq=model["a_eq"], b_eq=model["b_eq"],
                         bounds=model["bounds"], method="highs")
    else:
        result = linprog(c, A_eq=np.vstack([model["a_eq"], rows]),
                         b_eq=np.r_[model["b_eq"], mode],
                         bounds=model["bounds"], method="highs")
    if not result.success:
        if result.status == 2:
            return np.inf, None, None
        raise RuntimeError(f"Unresolved LP status {result.status}: {result.message}")
    point = model["objectives"] @ result.x
    return float(c @ result.x), point, rows @ result.x


def nondominated(points):
    points = np.asarray(points)
    return np.array([
        not np.any(np.all(points <= p + 1e-9, axis=1)
                   & np.any(points < p - 1e-9, axis=1)) for p in points
    ])


def simulate(values, lower, fractional, budget, seed, use_bounds, guided_modes,
             mode_feasibility=None):
    rng = np.random.default_rng(seed)
    designs, modes = values.shape
    orders = []
    for design in range(designs):
        tie = rng.random(modes)
        distance = np.sum((MODES - fractional[design]) ** 2, axis=1)
        order = np.lexsort((tie, distance)) if guided_modes else rng.permutation(modes)
        if mode_feasibility is not None:
            order = np.array([mode for mode in order if mode_feasibility[design, mode]])
        orders.append(order)
    positions = np.zeros(designs, dtype=int)
    limits = np.array([len(order) for order in orders])
    best = np.inf
    used = designs if use_bounds or guided_modes else 0
    while used < budget:
        available = positions < limits
        if use_bounds:
            # Elimination uses a lower bound and a feasible incumbent only.
            available &= lower < best - 1e-10
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        if use_bounds:
            design = int(candidates[np.argmin(lower[candidates])])
        else:
            count = positions[candidates]
            design = int(rng.choice(candidates[count == count.min()]))
        mode = orders[design][positions[design]]
        positions[design] += 1
        used += 1
        best = min(best, values[design, mode])
    return best, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("native-range", "broad-range"),
                        default="broad-range")
    args = parser.parse_args()
    global OUT
    OUT = OUT / args.scenario
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    with MAP.open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    rows = [row for row in source
            if float(row["Height (m)"]) == 0 and float(row["Engine Speed (rpm)"]) == 5000]
    operating_map = np.array([
        [float(row["Brake Power (kW)"]) * 0.96, float(row["Fuel Flow (kg/h)"])]
        for row in rows
    ])
    operating_map = operating_map[np.argsort(operating_map[:, 0])]
    assert operating_map.shape == (5, 2)
    engine_scales = ((0.70, 0.90, 1.10, 1.30) if args.scenario == "native-range"
                     else (0.35, 0.50, 0.65, 0.80))
    battery_scales = ((0.60, 1.00, 1.40, 1.80) if args.scenario == "native-range"
                      else (0.50, 1.00, 1.50, 2.00))
    settings = list(itertools.product(engine_scales, battery_scales))
    models = [make_model(*setting, operating_map) for setting in settings]
    reach_started = perf_counter()
    mode_feasibility = np.array([reachable_modes(model) for model in models])
    reach_elapsed = perf_counter() - reach_started
    values = np.full((len(WEIGHTS), len(models), len(MODES)), np.inf)
    points = np.full(values.shape + (2,), np.nan)
    lower = np.full(values.shape[:2], np.inf)
    fractional = np.zeros(values.shape[:2] + (3,))
    reference_rows = []
    for wi, weight in enumerate(WEIGHTS):
        for di, model in enumerate(models):
            bound, _, status = solve(model, weight)
            lower[wi, di] = bound
            fractional[wi, di] = status if status is not None else np.zeros(3)
            for mi, mode in enumerate(MODES):
                values[wi, di, mi], point, _ = solve(model, weight, mode)
                if point is not None:
                    points[wi, di, mi] = point
                    reference_rows.append({
                        "weight": float(weight), "design": di, "mode": mi,
                        "engine_scale": settings[di][0], "battery_scale": settings[di][1],
                        "fuel_kg": point[0], "battery_throughput_kwh": point[1],
                        "scalar_value": values[wi, di, mi],
                    })
        exact = np.min(values[wi], axis=1)
        assert np.all(lower[wi] <= exact + 1e-8)
        if wi % 6 == 0:
            print(f"Reference directions {wi + 1}/{len(WEIGHTS)}", flush=True)
    results = []
    assert np.array_equal(mode_feasibility, np.isfinite(values[0]))
    methods = (
        ("uniform", False, False),
        ("mode-guided", False, True),
        ("bound-only", True, False),
        ("bound-and-mode", True, True),
        ("resource-reachability", False, False),
    )
    for wi, weight in enumerate(WEIGHTS):
        optimum = float(np.min(values[wi]))
        assert np.isfinite(optimum) and optimum > 0
        for budget in (32, 64, 96):
            for seed in range(24):
                for name, bounded, guided in methods:
                    best, used = simulate(values[wi], lower[wi], fractional[wi],
                                          budget, seed, bounded, guided,
                                          mode_feasibility if name == "resource-reachability"
                                          else None)
                    results.append({
                        "weight": float(weight), "budget": budget, "seed": seed,
                        "method": name, "lp_calls": used, "success": np.isfinite(best),
                        "relative_regret": (best - optimum) / optimum
                        if np.isfinite(best) else None,
                        "exact": abs(best - optimum) < 1e-8,
                    })
    cloud = np.array([[r["fuel_kg"], r["battery_throughput_kwh"]]
                      for r in reference_rows])
    cloud = np.unique(np.round(cloud, 10), axis=0)
    front = cloud[nondominated(cloud)]
    fixed_wi = int(np.argmin(abs(WEIGHTS - 0.5)))
    fixed_indices = np.argmin(values[fixed_wi], axis=1)
    representatives = np.array([
        points[fixed_wi, di, mi] for di, mi in enumerate(fixed_indices)
        if np.isfinite(values[fixed_wi, di, mi])
    ])
    fixed_front = representatives[nondominated(representatives)]
    uncovered = np.array([
        not np.any(np.all(representatives <= point + 1e-8, axis=1)) for point in front
    ])
    summary = []
    for budget in (32, 64, 96):
        for name, _, _ in methods:
            subset = [r for r in results if r["budget"] == budget and r["method"] == name]
            regrets = [r["relative_regret"] for r in subset if r["success"]]
            summary.append({
                "budget": budget, "method": name, "trials": len(subset),
                "feasible_fraction": np.mean([r["success"] for r in subset]),
                "exact_fraction": np.mean([r["exact"] for r in subset]),
                "mean_relative_regret_success_only": float(np.mean(regrets)) if regrets else None,
                "mean_lp_calls": np.mean([r["lp_calls"] for r in subset]),
            })
    for name, data in (("完整方向与模式评价.csv", reference_rows),
                       ("有限预算逐项结果.csv", results), ("预算分配汇总.csv", summary)):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = {
        "scope": "Reduced three-stage energy LP with enumerated engine modes; not PTDL full mission.",
        "source_map": str(MAP), "source_sha256": hashlib.sha256(MAP.read_bytes()).hexdigest(),
        "source_operating_points_dc_kw_fuel_kgph": operating_map.tolist(),
        "designs": len(models), "modes_per_design": len(MODES), "directions": len(WEIGHTS),
        "scenario": args.scenario, "engine_scales": engine_scales,
        "battery_scales": battery_scales,
        "objective_scale": OBJECTIVE_SCALE.tolist(), "duration_h": DURATION.tolist(),
        "base_load_kw": [42, 24, 48], "battery_power_cap_kw": 30,
        "soc_initial": 0.9, "soc_min": 0.2, "soc_terminal_min": 0.5,
        "reference_lp_calls": len(models) * len(WEIGHTS) * (len(MODES) + 1),
        "budget_counts": "One fixed-mode LP or one relaxed LP equals one call; all methods that use relaxed values or modes pay for all sixteen relaxed solves; simulated runs replay frozen exact results.",
        "reference": "Best among all eight modes for each of nineteen fixed weighted sums; not an exhaustive continuous Pareto front.",
        "uniform_fixed_weight_representatives": len(representatives),
        "fixed_weight_nondominated_representatives": len(fixed_front),
        "sampled_reference_nondominated_points": len(front),
        "sampled_front_uncovered_by_fixed_weight": int(uncovered.sum()),
        "exactly_feasible_designs": int(np.isfinite(values[fixed_wi]).any(axis=1).sum()),
        "relaxed_feasible_but_all_engine_modes_infeasible": int(np.sum(
            np.isfinite(lower[fixed_wi]) & ~np.isfinite(values[fixed_wi]).any(axis=1)
        )),
        "resource_reachability_seconds": reach_elapsed,
        "resource_reachability_exact_feasibility_agreement": True,
        "resource_reachability_checks": int(mode_feasibility.size),
        "single_policy_infeasible_but_other_mode_feasible": int(sum(
            not np.isfinite(values[fixed_wi, di, 7]) and np.isfinite(values[fixed_wi, di]).any()
            for di in range(len(models))
        )),
        "single_policy_checked": "all-on engine mode",
        "limitations": [
            "Convex mixing of five source operating points is a reduced dispatch model.",
            "The two objectives are fuel and battery throughput, not the production three objectives.",
            "Bounds are rigorous for this LP relaxation, not JSBSim or the original nonlinear battery model.",
            "No claims of algorithm novelty, full-mission improvement, or global continuous Pareto completeness.",
            "Stored oracle values are used only to reproduce the cost-counted query experiment and measure regret.",
            "Ideal battery energy balance; positive throughput weight removes unnecessary simultaneous charging and discharging.",
            "No switching cost, ramp, battery resistance or thermal dynamics.",
            "Resource reachability is an established baseline, not a new algorithm.",
            "Reachability arithmetic time is recorded separately; LP counts alone are not equal wall-clock costs.",
        ],
        "elapsed_s": perf_counter() - started, "summary": summary,
    }
    (OUT / "机制验证汇总.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    axes[0].scatter(front[:, 0], front[:, 1], s=20, color="#268981", label="多方向及模式参考")
    axes[0].scatter(fixed_front[:, 0], fixed_front[:, 1], s=55, marker="x",
                    color="#bc594d", label="每个构型仅保留等权控制")
    axes[0].set_xlabel("燃油消耗 / kg", fontproperties=font)
    axes[0].set_ylabel("电池吞吐能量 / kWh", fontproperties=font)
    axes[0].legend(prop=font, frameon=False)
    labels = {"uniform": "均匀搜索", "mode-guided": "模式引导",
              "bound-only": "仅用上下界", "bound-and-mode": "上下界与模式引导",
              "resource-reachability": "普通资源可达性检查"}
    colors = ("#4477aa", "#cc8866", "#9988bb", "#228877", "#aa6688")
    for (method, _, _), color in zip(methods, colors):
        group = [r for r in summary if r["method"] == method]
        axes[1].plot([r["budget"] for r in group],
                     [100 * r["exact_fraction"] for r in group], "o-",
                     label=labels[method], color=color)
    axes[1].set_xlabel("总求解预算 / 次", fontproperties=font)
    axes[1].set_ylabel("找到该方向参考最优值的比例 / %", fontproperties=font)
    axes[1].legend(prop=font, frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.13)
    fig.savefig(OUT / "控制求解与设计评价_机制示意.png", dpi=200)
    fig.savefig(OUT / "控制求解与设计评价_机制示意.pdf")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
