"""Precompute comparable validation metrics for every deployed surrogate."""

import argparse
import hashlib
import importlib.util
import json
from html import escape
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from VSPAERO气动代理模型 import FEATURE_LABELS


def replace_generated_block(text, name, content):
    start = f"<!-- BEGIN GENERATED {name} -->"
    end = f"<!-- END GENERATED {name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected one generated {name} block")
    prefix, _, remainder = text.partition(start)
    _, separator, suffix = remainder.partition(end)
    if not separator:
        raise ValueError(f"Unclosed generated {name} block")
    return prefix + start + "\n" + content + "\n  " + end + suffix


def publish_inline_metrics(public_config, default_model):
    validation = public_config["model_configs"][default_model]["validation"]
    metrics = validation["metrics"]
    rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{metric['mae']:.3f}</td>"
        f"<td>{metric['r2']:.3f}</td></tr>"
        for label, metric in zip(("升力斜率 / rad⁻¹", "尾迹效率"), metrics)
    )
    table = (
        '<table class="validation-table"><thead><tr><th>独立测试</th>'
        '<th>MAE</th><th>R²</th></tr></thead><tbody>' + rows + "</tbody></table>"
    )
    data = json.dumps(public_config, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    document = ROOT / "前端" / "index.html"
    text = document.read_text(encoding="utf8")
    text = replace_generated_block(text, "MODEL VALIDATION", table)
    text = replace_generated_block(text, "MODEL DATA",
        f'<script type="application/json" id="modelData">{data}</script>')
    document.write_text(text, encoding="utf8")


def export(test_path=None):
    spec = importlib.util.spec_from_file_location("workbench", ROOT / "启动_PTDL前端.py")
    workbench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workbench)
    test_path = test_path or ROOT / "运行结果/扩样筛选网格代理_20261020/独立256构型/独立逐点.csv"
    test_path = Path(test_path)
    test = pd.read_csv(test_path)
    frozen_path = test_path.parent / "预测冻结.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf8"))
    for source, expected in frozen["hashes"].items():
        with Path(source).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError(f"Frozen validation source changed: {source}")
    if not frozen["locked_before_test_solver"] or frozen["test_labels_seen"]:
        raise RuntimeError("Validation predictions must precede test labels")
    frozen_predictions = pd.read_csv(test_path.parent / "锁定预测.csv").set_index("sample")
    plan = json.loads((test_path.parent.parent / "构型与方法冻结.json").read_text(encoding="utf8"))
    bases = {job["sample"]: job["basis"] for job in plan["jobs"]}
    if test["sample"].duplicated().any() or set(test["sample"]) != set(bases):
        raise RuntimeError("Validation geometries differ from the frozen sampling plan")
    if test["stratum"].tolist() != [bases[sample] for sample in test["sample"]]:
        raise RuntimeError("Validation strata differ from the frozen sampling plan")
    targets = ("vspaero_cl_alpha_per_rad", "vspaero_oswald_efficiency")
    models = {}
    for name, (model_path, doe_path) in workbench.MODEL_PATHS.items():
        development = pd.read_csv(doe_path)
        overlap = pd.merge(test[list(FEATURE_LABELS)], development[list(FEATURE_LABELS)],
                           on=list(FEATURE_LABELS), how="inner")
        if len(overlap):
            raise RuntimeError(f"Validation batch overlaps {name} training geometries")
        if frozen["hashes"].get(str(model_path)) != workbench.model_digest(model_path):
            raise RuntimeError(f"Saved predictions no longer match {name}")
        source_name = {"fine6144": "fine6144", "selectedmesh6144": "selected_mesh"}[name]
        prediction_columns = [f"{source_name}_prediction_{i}" for i in range(2)]
        predictions = frozen_predictions.loc[test["sample"], prediction_columns].to_numpy(float)
        np.testing.assert_allclose(test[prediction_columns], predictions, rtol=1e-12, atol=1e-14)
        _, bounds = workbench.read_doe(doe_path)
        inside = np.ones(len(test), dtype=bool)
        for k, (lo, hi) in bounds.items():
            inside &= test[k].between(lo, hi).to_numpy()
        cohorts = {}
        for cohort, mask in (
            ("uniform", test.stratum.eq("fresh_uniform_feasible").to_numpy()),
            ("difficult", test.stratum.eq("fresh_training_error_neighborhood").to_numpy()),
            ("all", np.ones(len(test), dtype=bool)),
        ):
            metrics = []
            for index, target in enumerate(targets):
                y = test[target].to_numpy(float)
                valid = mask & np.isfinite(y)
                if index == 1:
                    valid &= test.wake_quality_pass.fillna(False).to_numpy(bool)
                if not np.isfinite(predictions[valid, index]).all():
                    raise RuntimeError(f"Nonfinite validation prediction in {name}")
                error = predictions[valid, index] - y[valid]
                sst = np.sum((y[valid] - y[valid].mean()) ** 2)
                metrics.append(dict(
                    target=target, count=int(valid.sum()), mae=float(np.mean(abs(error))),
                    rmse=float(np.sqrt(np.mean(error ** 2))),
                    r2=float(1 - np.sum(error ** 2) / sst),
                    within_bounds_count=int((valid & inside).sum()),
                ))
            cohorts[cohort] = metrics
        uniform = test.stratum.eq("fresh_uniform_feasible").to_numpy()
        models[name] = dict(
            model_sha256=workbench.model_digest(model_path),
            metrics=cohorts["uniform"], difficult_metrics=cohorts["difficult"],
            combined_metrics=cohorts["all"],
            dataset_label=f"均匀独立测试 · {int(uniform.sum())} 构型",
            dataset_count=int(uniform.sum()), within_bounds_count=int((uniform & inside).sum()),
            validation_date="2026-10-20", training_count=len(development),
            prediction_source=source_name, requires_geometry=name == "selectedmesh6144",
            test_role="Predeclared uniform feasible holdout; difficult stratum reported separately",
        )
        print(name, json.dumps(cohorts["uniform"]), flush=True)
    payload = dict(
        schema="ptdl_surrogate_validation_registry_v1",
        dataset=str(test_path),
        dataset_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
        note="Same predeclared uniform cohort for both models; all valid labels retained. Difficult and combined metrics are also retained.",
        models=models,
    )
    path = ROOT / "气动代理模型数据" / "工作台代理精度.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    public_config = dict(model_configs={
        name: dict(name=name, validation=models[name],
                   doe_ranges=workbench.read_doe(paths[1])[1])
        for name, paths in workbench.MODEL_PATHS.items()
    })
    (ROOT / "前端" / "代理模型精度.json").write_text(
        json.dumps(public_config, ensure_ascii=False, indent=2), encoding="utf8")
    publish_inline_metrics(public_config, workbench.DEFAULT_PRECISION)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=Path)
    export(parser.parse_args().test_path)
