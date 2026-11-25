"""Compare five policies through the same candidate ground-air-ground interface."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"核心模型"))
from MOEAD多目标联合优化 import PTDLEnduranceJointProblem
from PTDL联合优化高保真接口 import PTDL6DoFFullMissionOptimizerValidator
from VSPAERO气动代理模型 import load_vsp_surrogate
from ptdl_hybrid_system_adapter import HybridSystemPowertrain
from 智能能量管理与容错调度 import MPCEnergyManagementCoordinator,FuzzyAdaptiveCoordinator,NeuralImitationCoordinator
from 规则混动协同控制 import RuleBasedHybridCoordinator
from 协同自适应方法 import PhaseScheduledMPC


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase-config",type=Path,required=True)
    p.add_argument("--neural",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.phase_config.read_text(encoding="utf-8"))
    model=load_vsp_surrogate(ROOT/"气动代理模型数据"/"气动代理模型_扩充DOE131点_ARD_RBF_GP_20261012.json")
    seed=json.loads((ROOT/"运行结果"/"H6_三项风险闭环复验"/"NSGA-II与MOEA-D择优可行候选.json").read_text(encoding="utf-8"))["decision"]
    factories={"rule":RuleBasedHybridCoordinator,"fuzzy":FuzzyAdaptiveCoordinator,
               "mpc":MPCEnergyManagementCoordinator,"scheduled":partial(PhaseScheduledMPC,config),
               "neural":partial(NeuralImitationCoordinator,args.neural)}
    rows=[]
    for name,factory in factories.items():
        validator=PTDL6DoFFullMissionOptimizerValidator(out/name,mission_hours=4,
            dispatch_policy_factory=factory,sample_period_s=10)
        problem=PTDLEnduranceJointProblem(
            aero_provider=lambda d,c:model.corrected_aero(d,c),
            powertrain_factory=HybridSystemPowertrain,dispatch_policy_factory=factory,
            sixdof_validator=validator)
        result=problem.evaluate(seed)
        rows.append({"policy":name,"overall_candidate_feasible":result.feasible,
                     **result.result.diagnostics,
                     **{"g_"+k:v for k,v in result.result.constraints.items()}})
        pd.DataFrame(rows).to_csv(out/"五策略地空地候选比较.csv",index=False)
        print(name,result.feasible,flush=True)


if __name__=="__main__":
    main()
