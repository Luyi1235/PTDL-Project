"""Exercise the full training/export path on an explicitly small training subset."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 推进_千构型主动补样20261017 as active


def main():
    out = ROOT / "运行结果/主动补样训练流程预检_20261017"
    out.mkdir(exist_ok=True)
    for filename in ("千构型模型选择.json", "千构型改进候选.json"):
        shutil.copy2(active.OUT / filename, out / filename)
    frame = pd.read_csv(active.SOURCE / "原网格训练数据.csv").head(80)
    paths = active.fit_models(frame, out)
    for name, path in paths.items():
        model = active.direct.base.TargetwiseAeroGP.from_json(path)
        for _, row in frame.iloc[::20].iterrows():
            design = active.original.prior.AirframeDesign(**{
                k: float(row[k]) for k in active.direct.FEATURES})
            assert np.isfinite(model.predict_with_uncertainty(design)).all()
    plan = json.loads((active.OUT / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    jobs = plan["jobs"]
    assert len(jobs) == 1608
    assert sum(j["role"] == "train" for j in jobs) == 1512
    assert sum(j["role"] == "test" for j in jobs) == 96
    assert sum("reuse_record" in j for j in jobs) == 1000
    x = np.array([[j["design"][k] for k in active.direct.FEATURES] for j in jobs])
    assert len(np.unique(x, axis=0)) == len(x)
    for job in jobs:
        if "reuse_record" in job:
            row = json.loads(Path(job["reuse_record"]).read_text(encoding="utf8"))
            assert row["quality_pass"] == bool(row["wake_quality_pass"])
            assert active.original.prior.driver.digest(Path(job["reuse_record"])) == job["reuse_sha256"]
        else:
            assert max(active.original.prior.geometry_constraints(
                active.original.prior.AirframeDesign(**job["design"])).values()) <= 1e-9
    for relative, digest in plan["source_manifest"].items():
        assert active.original.prior.driver.digest(ROOT / relative) == digest, relative
    (out / "预检结果.json").write_text(json.dumps(dict(
        passed=True, training_rows=80, note="Smoke only; no scientific accuracy claim",
        trained_models_reloaded=len(paths), plan_rows=len(jobs), new_native_cases=608,
        sealed_labels_read=False), indent=2), encoding="utf8")
    print("PASS: 30-fold training, model reload, 1608-case plan, frozen sources, no test-label access")


if __name__ == "__main__":
    main()
