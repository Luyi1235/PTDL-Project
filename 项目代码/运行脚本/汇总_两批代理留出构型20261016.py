"""Pool both solver batches without excluding valid high-error geometries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from 训练_主动增量混合代理20261016 import portable_target_prediction, metric, FEATURE_LABELS
from 验证_物理趋势新构型20261016 import MODELS, ROOT, TARGETS


def main():
    out = ROOT / "运行结果" / "主动增量代理_20261015"
    first = pd.read_csv(ROOT / "运行结果" / "主动气动补样_20261015" / "独立测试逐点.csv")
    second = pd.read_csv(ROOT / "运行结果" / "物理趋势残差代理_20261016" / "全新测试逐点.csv")
    columns = list(FEATURE_LABELS)+list(TARGETS)+["sample", "wake_quality_pass"]
    test = pd.concat([
        first[columns].assign(validation_batch="first_frozen_batch"),
        second[columns].assign(validation_batch="second_frozen_batch")], ignore_index=True)
    if test.duplicated(list(FEATURE_LABELS)).any():
        raise RuntimeError("Duplicate held-out geometry")
    train = pd.read_csv(out / "增量训练样本.csv")
    if len(pd.merge(test[list(FEATURE_LABELS)], train[list(FEATURE_LABELS)],
                    on=list(FEATURE_LABELS), how="inner")):
        raise RuntimeError("Test/training overlap")
    x = test[list(FEATURE_LABELS)].to_numpy(float)
    results = {}
    for name, path in MODELS.items():
        payload = json.loads(path.read_text(encoding="utf8"))
        results[name] = []
        for i, target in enumerate(TARGETS):
            y = test[target].to_numpy(float)
            valid = np.isfinite(y)
            if i:
                valid &= test.wake_quality_pass.to_numpy(bool)
            predictions = portable_target_prediction(payload["targets"][i], x)
            test[f"{name}_prediction_{i}"] = predictions
            results[name].append(dict(target=target, count=int(valid.sum()),
                                      **metric(y[valid], predictions[valid])))
    test.to_csv(out / "合并留出验证.csv", index=False)
    baseline = results["incrementalgp"]
    passes = {name: all(a["rmse"] < b["rmse"] for a, b in zip(values, baseline))
              for name, values in results.items() if name != "incrementalgp"}
    result = dict(
        metrics=results, promotion_gate=passes, validation_geometries=len(test),
        candidate_goal_pass={name: all(v["r2"] >= .9 for v in values) for name, values in results.items()},
        first_batch="16 new geometries; incumbent and active mixture frozen before collection",
        second_batch="16 further new geometries; all three models frozen before collection",
        physical_trend_note="First batch is retrospective for physicaltrendgp; second batch is its fresh blind test",
        exclusions="Only predefined solver/label-quality failures; all valid large errors retained",
        mesh_labels="Both test batches remain legacy-mesh labels; medium-mesh diagnostics kept separate",
    )
    (out / "合并留出评估.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
