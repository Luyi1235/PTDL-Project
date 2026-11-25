"""Export selected learners without a training-library dependency at inference time."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

import 研究_几何交互代理20261017 as study

ROOT, OUT = study.ROOT, study.OUT


def portable(model, interval):
    config = model["config"]
    if "gp" in model:
        return study.prior.portable_scalar(model["gp"], interval)
    if config["method"] == "boost":
        target = study.prior.direct.base.boosting_payload(model["estimator"], config["mode"])
        return dict(kind="weighted-ensemble", models=[target], weights=[1.], interval_std=interval)
    with tempfile.TemporaryDirectory(prefix="ptdl_cat_") as folder:
        path = Path(folder) / "model.json"
        model["estimator"].save_model(str(path), format="json")
        source = json.loads(path.read_text(encoding="utf8"))
    trees = []
    for tree in source["oblivious_trees"]:
        splits = tree.get("splits") or []
        if any(s["split_type"] != "FloatFeature" for s in splits):
            raise ValueError("Only numerical geometry splits are supported")
        trees.append(dict(columns=[s["float_feature_index"] for s in splits],
                          thresholds=[s["border"] for s in splits], leaves=tree["leaf_values"]))
    scale, bias = source["scale_and_bias"]
    return dict(kind="oblivious-boosting", features=config["mode"], trees=trees,
                scale=scale, bias=bias[0], transform=config["transform"],
                float32_input=True, interval_std=interval)


def main():
    selection = json.loads((OUT / "模型选择.json").read_text(encoding="utf8"))
    base = study.prior.OUT / "多精度候选代理.json"
    payload = copy.deepcopy(json.loads(base.read_text(encoding="utf8")))
    frame = pd.read_csv(study.DATA)
    probe = frame[study.FEATURES].to_numpy(float)[::23].copy()
    # Include unseen points as well as exact training rows in the export comparison.
    probe = np.vstack((probe, (probe[:-1]+probe[1:])/2))
    targets = []
    for entry in selection["selected"]:
        components, expected = [], np.zeros(len(probe))
        for component in entry["components"]:
            source = Path(component["path"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != component["sha256"]:
                raise RuntimeError("Frozen training model changed")
            fitted = joblib.load(source)
            target = portable(fitted, entry["interval_std"])
            prediction = np.array([study.prior.direct.base.target_mean(target, row[None, :])
                                   for row in probe])
            np.testing.assert_allclose(prediction, study.predict(fitted, probe), rtol=1e-9, atol=1e-9)
            expected += component["weight"]*prediction
            components.append(target)
        ensemble = dict(kind="weighted-ensemble", models=components,
                        weights=[c["weight"] for c in entry["components"]],
                        interval_std=entry["interval_std"])
        targets.append(ensemble)
        np.testing.assert_allclose(
            [study.prior.direct.base.target_mean(ensemble, row[None, :]) for row in probe], expected,
            rtol=1e-9, atol=1e-9)
    payload.update(targets=targets, training_samples=len(frame),
                   selection="Geometry interactions, training-only random/spatial CV",
                   automatic_production_promotion=False,
                   uncertainty="Fixed empirical training OOF absolute error bands",
                   low_fidelity_used_by_target=[False, False])
    path = OUT / "几何交互候选代理.json"
    study.prior.direct.base.TargetwiseAeroGP(payload).to_json(path)
    restored = study.prior.direct.base.TargetwiseAeroGP.from_json(path)
    for row in probe[:5]:
        design = study.prior.original.prior.AirframeDesign(**dict(zip(study.FEATURES, row)))
        assert np.isfinite(restored.predict_with_uncertainty(design)).all()
    (OUT / "导出检查.json").write_text(json.dumps(dict(
        passed=True, probes=len(probe), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        external_labels_used=False), indent=2), encoding="utf8")
    print(path)


if __name__ == "__main__":
    main()
