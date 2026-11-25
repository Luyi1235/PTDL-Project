"""Archive the two completed studies without replacing historical evidence."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"运行结果"
PROXY=RESULTS/"分目标代理211点_20261013"
EXPANDED=RESULTS/"扩大预算_已完成12组归档_20261013"
ANALYSIS=RESULTS/"六自由度跨预算分析_20261013"
OUT=RESULTS/"两项推进归档_20261013"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gate=load(PROXY/"部署门槛决策.json")
    replay=load(PROXY/"代理闭环接入验收.json")
    tests=load(OUT/"回归测试.json")
    progress=load(EXPANDED/"进度.json")
    assert progress["completed"]==progress["total"]==12 and not progress["errors"]
    runs=pd.read_csv(EXPANDED/"多种子运行比较.csv")
    assert len(runs)==12 and (runs.evaluation_calls==120).all()
    assert (runs.population==15).all() and (runs.generations==7).all()
    assert (runs.sixdof_endurance_hours==80).all()
    assert tests["passed"]
    model=PROXY/"分目标候选代理.json"
    assert hashlib.sha256(model.read_bytes()).hexdigest()==gate["frozen_model_sha256"]==replay["surrogate_sha256"]
    assert replay["evaluation_calls"]==24 and replay["sixdof_candidate_calls"]==25
    if replay["optional_deployment_accepted"]:
        assert hashlib.sha256(Path(replay["deployed_model"]).read_bytes()).hexdigest()==gate["frozen_model_sha256"]
    stats=pd.read_csv(ANALYSIS/"跨预算算法统计.csv")
    expanded=stats[stats.budget=="H4G7"]
    entries=[]
    for row in expanded.itertuples():
        entries.append(dict(algorithm=row.algorithm,resolution=4,population=15,generations=7,
                            seeds=sorted(runs[runs.algorithm==row.algorithm].seed.tolist()),
                            evaluation_budget_per_run=120,mission_endurance_hours=80,
                            feasible_population_mean=float(row.feasibility_mean),
                            historical_hv_mean=float(row.hv_mean),historical_hv_std=float(row.hv_std),
                            tail_plateau_fraction=float(row.plateau_fraction),
                            role="feasible_population_configuration" if row.feasibility_mean>=.9 else "archive_exploration_configuration",
                            budget_status="observed_tail_plateau" if row.plateau_fraction==1 else "further_generations_required",
                            scope="Original 131-point proxy, rule dispatch, fixed estimated-aircraft scenario",
                            normalization=str(ANALYSIS/"分析口径.json")))
    catalog=load(RESULTS/"蓝色攻关_最终归档_20261013"/"协同设计方法配置库.json")
    catalog["optimization_budget_extension"]=entries
    catalog["updated_surrogate"]=dict(model=replay.get("deployed_model",str(model)),
                                     status=replay["usage"],validation=str(PROXY/"部署门槛决策.json"),
                                     replay=str(PROXY/"代理闭环接入验收.json"),default_replaced=False)
    (OUT/"协同设计方法配置库_预算扩展.json").write_text(
        json.dumps(catalog,ensure_ascii=False,indent=2),encoding="utf-8")
    audit=dict(checks_passed=True,core_tests=tests["tests_run"],surrogate_gate_passed=gate["candidate_accepted"],
               training_counts=gate["per_target_training_count"],test_counts=gate["scored_test_counts"],
               model_sha256=gate["frozen_model_sha256"],replay=replay,
               expanded_runs=len(runs),expanded_evaluations=int(runs.evaluation_calls.sum()),
               sixdof_check_records=int(runs.sixdof_candidate_calls.sum()),
               configurations=entries,pending_runs=progress["pending"],
               stopped_for_report=True,unfinished_convergence=[e["algorithm"] for e in entries
                   if e["budget_status"]=="further_generations_required"]+["Hybrid NSGA-II/MOPSO"])
    (OUT/"两项工作验收.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=[
        "# 气动代理与六自由度预算推进",
        "",
        "本记录更新第三版报告保留的两项工作。旧模型、旧结果及前三版报告保持原样。",
        "",
        "## 气动代理",
        "",
        f"- 211个训练构型，两项标签实际训练数分别为{gate['per_target_training_count']}。",
        f"- 预留24个验证构型，两项标签可评分数分别为{gate['scored_test_counts']}。",
        "- 整条极曲线质量检查、异常重算、无量纲特征、分目标选模和折外区间调整已实现。",
        "- 极曲线分层预测也完成比较，本轮选择两项无量纲Matérn 3/2标量高斯过程。",
        f"- 更新代理独立验证R²为{gate['metrics']['candidate']['r2']}，两项部署门槛通过状态为{gate['candidate_accepted']}。",
        f"- 24次候选评价、25次六自由度检查，终代可行率{replay['final_feasible_ratio']:.3f}。",
        "- 作为可选模型接入统一加载器。旧实验与工作台默认配置未自动替换。",
        "",
        "## 扩大预算",
        "",
        "- 原预算为6个体3代8种子，覆盖四种算法。新预算为15个体7代4种子，已完成NSGA-II、MOEA/D和MOPSO。",
        "- 单次评价从24次增至120次。新增12次完整优化和1440次候选评价。",
        "- 按用户要求停止混合算法大预算补跑并转入报告撰写，未完成运行不计入统计。",
        "- 统一跨预算归一化、历史档案超体积、终代可行率、配对种子收益和末两代平台判据。",
        "- 0.05 s积分步长、先80 h续航再下降着陆及原有约束保持不变。",
        "- 正式三目标仍按既有任务评价，完整六自由度用于候选约束。",
        f"- 尚未全部种子达到末段平台的算法为{audit['unfinished_convergence']}。",
        "",
        "## 结果入口",
        "",
        "- `运行结果/分目标代理211点_20261013/部署门槛决策.json`",
        "- `运行结果/分目标代理211点_20261013/代理闭环接入验收.json`",
        "- `气动代理模型数据/分目标代理部署配置_20261013.json`",
        "- `运行结果/六自由度跨预算分析_20261013/跨预算算法统计.csv`",
        "- `运行结果/六自由度跨预算分析_20261013/同种子配对预算收益.csv`",
        "- `运行结果/两项推进归档_20261013/协同设计方法配置库_预算扩展.json`",
        "",
        f"本轮回归测试{tests['tests_run']}项通过。区间为经验评价，算法平台判据仅描述已计算预算。",
    ]
    (ROOT/"说明文档"/"代理精度与扩大预算推进_20261013.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
