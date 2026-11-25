"""Exercise the full new training hook on a small historical development subset."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 推进_千构型原定义代理20261017 as experiment
from VSPAERO气动代理模型 import load_vsp_surrogate


def main():
    frame = experiment.pd.read_csv(experiment.prior.LOW_DATA).iloc[:40].copy()
    frame["sample"] = [f"stable_train_{i:04d}" for i in range(len(frame))]
    out = ROOT / "运行结果" / "千构型训练流程预检_20261017"
    out.mkdir(exist_ok=True)
    paths = experiment.fit_models(frame, out)
    for path in paths.values():
        model = load_vsp_surrogate(path)
        for _, row in frame.iloc[:3].iterrows():
            design = experiment.prior.AirframeDesign(**{
                key: float(row[key]) for key in experiment.prior.FEATURE_LABELS})
            result = experiment.np.asarray(model.predict(design))
            if not experiment.np.isfinite(result).all():
                raise AssertionError("Exported model cannot predict a known geometry")
    payload = json.loads(paths["high_candidate"].read_text(encoding="utf8"))
    assert payload["wake_protocol"] == "free-wake-legacy-v1"
    assert payload["low_fidelity_used_by_target"] == [False, False]
    assert len(payload["targets"]) == 2
    result = dict(stage="passed", historical_training_geometries=40,
                  new_test_labels_used=False, models_loaded=len(paths),
                  selected_configuration_count=len(experiment.regularized.CONFIGS))
    (out / "训练流程预检结果.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
