"""Export selected models and verify exact inference before holdout evaluation."""

import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import 研究_密度均衡与残差代理20261017 as study
from 导出_几何交互代理20261017 import portable as portable_tree
from 分目标气动高斯过程 import TargetwiseAeroGP, target_mean


def portable(fitted, interval):
    config = fitted["config"]
    if config["method"] == "tree":
        return portable_tree(dict(
            config={**config, "method": "cat", "transform": "identity"},
            estimator=fitted["estimator"]), interval)
    gp = dict(config=dict(method="gp", mode=config["mode"], kernel=config["kernel"]),
              estimator=fitted["estimator"], center=fitted["center"], scale=fitted["scale"])
    correction = study.previous.prior.portable_scalar(gp, interval)
    scalar = correction["models"][0]
    if len(scalar["length"]) == 1:
        scalar["length"] = np.repeat(scalar["length"], len(scalar["feature_mean"])).tolist()
    return dict(kind="crossfit-residual", base=portable(fitted["base"], interval),
                correction=correction,
                shrinkage=config["shrinkage"], interval_std=interval)


def main():
    selected = json.loads((study.OUT / "模型选择.json").read_text(encoding="utf8"))
    payload = copy.deepcopy(json.loads(study.REFERENCE.read_text(encoding="utf8")))
    frame = pd.read_csv(study.previous.DATA)
    x = frame[study.FEATURES].to_numpy(float)[::24]
    probe = np.vstack((x, (x[1:]+x[:-1])/2))
    targets, errors = [], []
    for target_index, entry in enumerate(selected["selected"]):
        components, expected = [], np.zeros(len(probe))
        for spec in entry["components"]:
            path = Path(spec["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise RuntimeError("Selected model changed after training")
            if spec["family"] == "reference":
                component = json.loads(path.read_text(encoding="utf8"))["targets"][spec["target_index"]]
                prediction = np.array([target_mean(component, row[None, :]) for row in probe])
            else:
                fitted = joblib.load(path)
                component = portable(fitted, entry["interval_std"])
                prediction = study.predict(fitted, probe)
            actual = np.array([target_mean(component, row[None, :]) for row in probe])
            np.testing.assert_allclose(actual, prediction, rtol=1e-9, atol=1e-9)
            errors.append(float(abs(actual-prediction).max()))
            expected += spec["weight"]*prediction
            components.append(component)
        target = dict(kind="weighted-ensemble", models=components,
                      weights=[c["weight"] for c in entry["components"]],
                      interval_std=entry["interval_std"])
        np.testing.assert_allclose(
            [target_mean(target, row[None, :]) for row in probe], expected, rtol=1e-9, atol=1e-9)
        targets.append(target)
    payload.update(
        targets=targets, selection="Training-only density/robust/ordered/crossfit residual comparison",
        external_labels_used_for_selection=False, automatic_production_promotion=False,
        uncertainty="Fixed absolute error band from training OOF predictions")
    path = study.OUT / "密度均衡候选代理.json"
    TargetwiseAeroGP(payload).to_json(path)
    loaded = TargetwiseAeroGP.from_json(path)
    for row in probe[:3]:
        design = study.previous.prior.original.prior.AirframeDesign(**dict(zip(study.FEATURES, row)))
        assert np.isfinite(loaded.predict_with_uncertainty(design)).all()
    (study.OUT / "导出检查.json").write_text(json.dumps(dict(
        passed=True, probe_count=len(probe), max_difference=max(errors),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2), encoding="utf8")
    print(path)


if __name__ == "__main__":
    main()
