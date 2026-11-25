"""Replay the frozen candidate in full-mission constrained joint optimization."""

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from 运行_六自由度多种子闭环比较 import ROOT, run_one
from 训练部署VSPAERO_ARD_RBF代理 import DEFAULT_DATA


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--study",type=Path,required=True)
    args=p.parse_args()
    study=args.study
    gate=json.loads((study/"部署门槛决策.json").read_text(encoding="utf-8"))
    model=study/"分目标候选代理.json"
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    if digest!=gate["frozen_model_sha256"]:
        raise ValueError("Model changed after independent test")
    folder=study/"闭环回放H2G3"
    folder.mkdir(parents=True,exist_ok=True)
    stats=folder/"单次统计.json"
    if stats.exists():
        record=json.loads(stats.read_text(encoding="utf-8"))
        if record.get("surrogate_sha256")!=digest:
            raise ValueError("Existing replay used a different model")
    else:
        record=run_one("NSGA-II",20261013,folder,2,3,80,model,DEFAULT_DATA)
    accepted=bool(gate["candidate_accepted"] and record["final_feasible_ratio"]>0)
    record.update(accuracy_gate_passed=gate["candidate_accepted"],optional_deployment_accepted=accepted,
                  usage="optional_validated_surrogate" if accepted else "experimental_only",
                  comparison_basis="Original 131-DOE bounds, rule coordinator, H2G3 seed 20261013; only surrogate changed",
                  default_replaced=False)
    if accepted:
        destination=ROOT/"气动代理模型数据"/"气动代理模型_211构型_分目标GP_20261013.json"
        shutil.copy2(model,destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Deployment copy hash mismatch")
        record["deployed_model"]=str(destination)
        manifest=dict(model=str(destination.relative_to(ROOT)),sha256=digest,
                      domain_data=str(DEFAULT_DATA.relative_to(ROOT)),validation=str((study/"部署门槛决策.json").relative_to(ROOT)),
                      replay=str((study/"代理闭环接入验收.json").relative_to(ROOT)),
                      status="optional_validated_surrogate",default_replaced=False)
        (ROOT/"气动代理模型数据"/"分目标代理部署配置_20261013.json").write_text(
            json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (study/"代理闭环接入验收.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
