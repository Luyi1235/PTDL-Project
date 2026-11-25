"""Evidence-aware profiles and summaries for the PTDL optimizer methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ALGORITHM_PROFILES: dict[str, dict[str, str]] = {
    "NSGA-II": {
        "method": "非支配排序、拥挤距离与精英保留",
        "recommended_use": "通用基准；需要稳定的 Pareto 多样性和便于解释的对照时",
        "caveat": "高维目标/决策下需检查多样性与计算预算",
    },
    "MOEA/D": {
        "method": "分解为一组权重子问题并进行邻域协同更新",
        "recommended_use": "偏好权重明确、希望检查不同目标折衷区域时",
        "caveat": "结果受权重方向、归一化基准和邻域设置影响",
    },
    "MOPSO": {
        "method": "粒子群位置更新与外部非支配档案",
        "recommended_use": "需要快速探索、作为群体智能对照时",
        "caveat": "当前为项目实现；需重点监测可行率、档案拥挤和早熟",
    },
    "Hybrid NSGA-II/MOPSO": {
        "method": "NSGA-II 精英保留与 MOPSO 粒子位置更新混合",
        "recommended_use": "希望兼顾精英保持与群体搜索、并与单算法作消融对照时",
        "caveat": "混合收益需在相同预算、多随机种子下持续复核",
    },
    "PTDL-自适应分块": {
        "method": "总体与调度变量分块、周期性联合更新、自适应 DE/PSO/遗传算子和可行精英档案",
        "recommended_use": "研究总体参数与动力调度之间的搜索耦合，并保留相同预算的基准与消融比较",
        "caveat": "当前组合的收益依赖任务和响应定义，尚未形成超过所有强基线的优势",
    },
    "PTDL-无分块": {
        "method": "保持算子信用更新和精英档案，取消变量分块",
        "recommended_use": "检验分块搜索的独立贡献",
        "caveat": "仅作为完整方法的配对消融配置",
    },
    "PTDL-固定算子": {
        "method": "保持变量分块和精英档案，各算子的选择概率固定为三分之一",
        "recommended_use": "检验自适应算子比例的独立贡献",
        "caveat": "仅作为完整方法的配对消融配置",
    },
    "PTDL-耦合代理辅助": {
        "method": "参考方向引导、总体与动力调度相关提案、在线代理预筛选及真实评价档案",
        "recommended_use": "昂贵的飞推控耦合候选评价，按同预算比较搜索质量与计算耗时",
        "caveat": "在线代理仅使用本次已计入预算的真实评价，收益由独立种子比较检验",
    },
    "PTDL-取消在线代理": {
        "method": "保持参考方向与相关提案，取消在线代理预筛选",
        "recommended_use": "检验在线代理筛选的独立贡献",
        "caveat": "仅作为完整方法的配对消融配置",
    },
    "PTDL-耦合引导": {
        "method": "参考方向引导、构型与动力调度相关提案及可行非支配档案",
        "recommended_use": "与独立八种子研究中的取消在线代理配置一致，作为较优的项目搜索配置",
        "caveat": "冻结代理与低阶任务的独立十六种子比较中优于五个既有基线，适用范围按该任务与预算说明",
    },
    "PTDL-去重贡献学习": {
        "method": "保留提案先验的贡献学习，对重复候选进行局部变量组调整",
        "recommended_use": "研究重复评价消除与提案学习的影响",
        "caveat": "独立比较未超过原耦合引导，保留为研究配置",
    },
    "PTDL-耦合贡献学习": {
        "method": "参考方向与耦合提案，按真实评价的改善率调整提案类型概率",
        "recommended_use": "检验构型、调度、边界调整对任务优化的不同贡献",
        "caveat": "开发种子选定配置后，使用独立种子验证收益",
    },
    "PTDL-耦合贡献学习保先验": {
        "method": "以耦合引导的提案比例为先验，再按真实改善率调整选择概率",
        "recommended_use": "在保留原搜索特性的基础上学习有效变量调整方式",
        "caveat": "与原耦合引导和均匀先验学习版本同时进行独立种子比较",
    },
    "PTDL-耦合局部灵敏度": {
        "method": "真实评价档案拟合局部灵敏度，交替进行局部提案筛选和随机探索",
        "recommended_use": "检验局部线性信息对任务耦合搜索的影响",
        "caveat": "局部模型依赖已评价区域，保留随机探索比例",
    },
    "PTDL-任务耦合自适应": {
        "method": "持续粒子状态、差分演化与分组局部调整，自适应更新算子与步长",
        "recommended_use": "研究任务变量联合搜索机制",
        "caveat": "当前开发比较表现弱于耦合引导，保留为研究对照",
    },
    "PTDL-任务耦合固定": {
        "method": "任务耦合提案采用固定算子比例和步长",
        "recommended_use": "检验任务耦合自适应更新的贡献",
        "caveat": "配对消融配置",
    },
}


def summarize_equal_budget_benchmark(
    statistics_csv: str | Path,
    run_metadata_json: str | Path,
) -> dict[str, Any]:
    """Build a method-selection record without asserting a universal winner."""
    stats_path = Path(statistics_csv)
    metadata_path = Path(run_metadata_json)
    stats = pd.read_csv(stats_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "algorithm",
        "seed_count",
        "final_feasible_ratio_mean",
        "runtime_s_mean",
        "final_shared_hypervolume_mean",
    }
    missing = required.difference(stats.columns)
    if missing:
        raise ValueError(f"Benchmark statistics missing columns: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for row in stats.to_dict(orient="records"):
        name = str(row["algorithm"])
        if name not in ALGORITHM_PROFILES:
            raise ValueError(f"No method profile is registered for {name!r}.")
        records.append(
            {
                **ALGORITHM_PROFILES[name],
                "algorithm": name,
                "seed_count": int(row["seed_count"]),
                "mean_final_feasible_ratio": float(row["final_feasible_ratio_mean"]),
                "mean_runtime_s": float(row["runtime_s_mean"]),
                "mean_shared_hypervolume": float(
                    row["final_shared_hypervolume_mean"]
                ),
                "hypervolume_std": _optional_float(
                    row.get("final_shared_hypervolume_std")
                ),
                "feasibility_rank": None,
                "runtime_rank": None,
                "hypervolume_rank": None,
            }
        )
    if not records:
        raise ValueError("Benchmark statistics contain no algorithms.")

    _assign_ranks(records, "mean_final_feasible_ratio", descending=True, output="feasibility_rank")
    _assign_ranks(records, "mean_runtime_s", descending=False, output="runtime_rank")
    _assign_ranks(records, "mean_shared_hypervolume", descending=True, output="hypervolume_rank")
    return {
        "schema": "ptdl_optimizer_method_catalog_v1",
        "selection_rule": (
            "No universal winner is declared. Compare feasibility, shared normalized "
            "hypervolume, and runtime ranks under the same model and budget; select "
            "according to the project priority."
        ),
        "benchmark_metadata": metadata,
        "benchmark_scope": {
            "algorithms": list(metadata.get("algorithms", [])),
            "seeds": list(metadata.get("seeds", [])),
            "population_size": metadata.get("population_size"),
            "resolution": metadata.get("resolution"),
            "generations": metadata.get("generations"),
            "aero_training_samples": metadata.get("aero_training_samples"),
            "note": (
                "Scope is taken from explicit run fields; the source study description "
                "string may be a generic or stale label."
            ),
        },
        "algorithm_profiles": records,
        "evidence_files": {
            "statistics": str(stats_path.resolve()),
            "run_metadata": str(metadata_path.resolve()),
        },
        "limitations": [
            "The catalog inherits the benchmark model, candidate anchor, and budget limits.",
            "A low-order or estimated-model benchmark is method evidence, not hardware validation.",
            "Hypervolume is comparable only under the recorded shared normalization and reference point.",
        ],
    }


def _assign_ranks(
    records: list[dict[str, Any]],
    key: str,
    *,
    descending: bool,
    output: str,
) -> None:
    ordered = sorted(
        records,
        key=lambda item: float(item[key]),
        reverse=descending,
    )
    current_rank = 0
    previous: float | None = None
    for index, item in enumerate(ordered, start=1):
        value = float(item[key])
        if previous is None or value != previous:
            current_rank = index
            previous = value
        item[output] = current_rank


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None
