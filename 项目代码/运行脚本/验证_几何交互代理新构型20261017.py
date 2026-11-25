"""Evaluate frozen surrogates on new geometries generated after model selection."""

import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

import 研究_几何交互代理20261017 as study

ROOT = study.ROOT
OUT = ROOT / "运行结果/几何交互独立验证_20261017"
FEATURES = study.FEATURES
TARGETS = study.TARGETS
NTEST = 128


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    frame = pd.read_csv(study.DATA)
    x = frame[FEATURES].to_numpy(float)
    lo, hi = x.min(0), x.max(0)
    fixed = {k: float(frame.iloc[0][k]) for k in
             ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg")}
    forbidden = [x]
    for folder in (study.prior.SOURCE, study.prior.OUT):
        forbidden.append(pd.read_csv(folder / "中网格独立测试标签.csv")[FEATURES].to_numpy(float))
    forbidden.append(pd.read_csv(study.prior.original.prior.driver.LOW_DIR / "合并留出验证.csv")[
        FEATURES].to_numpy(float))
    for source in (study.prior.original.prior.FIRST, ROOT / "运行结果/多精度气动代理_20261016"):
        old = json.loads((source / "冻结采样与计算配置.json").read_text(encoding="utf8"))
        forbidden.append(np.asarray([[j["design"][k] for k in FEATURES] for j in old["jobs"]
                                     if j["role"] == "test"]))
    avoid = np.vstack(forbidden)
    rng = np.random.default_rng(2026101723)
    jobs = []
    while len(jobs) < NTEST:
        row = rng.uniform(lo, hi)
        design = {**fixed, **dict(zip(FEATURES, row))}
        constraint = study.prior.original.prior.geometry_constraints(
            study.prior.original.prior.AirframeDesign(**design))
        if max(constraint.values()) <= 1e-9 and np.linalg.norm((avoid-row)/(hi-lo), axis=1).min() > 1e-5:
            jobs.append(dict(sample=f"interaction_test_{len(jobs):04d}", design=design))
            avoid = np.vstack((avoid, row))
    models = dict(
        geometry_interaction=study.OUT / "几何交互候选代理.json",
        previous1512=study.prior.OUT / "多精度候选代理.json",
        workbench289=study.prior.original.prior.driver.LOW_MODEL)
    sources = study.prior.original.prior.driver.sources()
    for path in (Path(__file__), Path(study.__file__), ROOT / "核心模型/气动几何特征.py",
                 study.prior.original.WORKER, Path(study.prior.original.controlled.__file__)):
        sources[str(path.relative_to(ROOT))] = digest(path)
    return dict(jobs=jobs, models={k: str(v) for k, v in models.items()},
                model_hashes={k: digest(v) for k, v in models.items()},
                sources=sources, target_definition="Original median Ew and lift slope",
                model_selection_frozen_before_test=True, count=NTEST,
                protocol="free-wake-legacy-v1", automatic_production_promotion=False)


def main():
    OUT.mkdir(exist_ok=True)
    path = OUT / "冻结配置.json"
    if not path.exists():
        path.write_text(json.dumps(plan(), indent=2), encoding="utf8")
    frozen = json.loads(path.read_text(encoding="utf8"))
    def check():
        for relative, expected in frozen["sources"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"Frozen source changed: {relative}")
        for name, expected in frozen["model_hashes"].items():
            if digest(frozen["models"][name]) != expected:
                raise RuntimeError(f"Frozen model changed: {name}")
    started = time.monotonic()
    rows = []
    def progress(stage, error=None):
        (OUT / "进度.json").write_text(json.dumps(dict(
            pid=os.getpid(), stage=stage, completed=len(rows), requested=NTEST,
            elapsed_s=time.monotonic()-started, error=error,
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf8")
    try:
        check()
        progress("loading_models")
        study.prior.original.prior.solver.WORKER = study.prior.original.WORKER
        models = {name: study.prior.direct.base.TargetwiseAeroGP.from_json(path)
                  for name, path in frozen["models"].items()}
        progress("solver")
        for job in frozen["jobs"]:
            check()
            folder = OUT / "solver" / job["sample"]
            cache = folder / "record.json"
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf8"))
            else:
                folder.mkdir(parents=True)
                try:
                    row = study.prior.original.native(
                        study.prior.original.prior.solver.DEFAULT_PYTHON, job["design"], job["sample"],
                        folder / "case", 3600, "test", "fresh_sealed_uniform_feasible",
                        (0., float("inf")), 30, "legacy")
                    row.update(study.prior.original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
                    row["wake_quality_pass"] = bool(row["quality_pass"])
                except Exception as exc:
                    row = dict(sample=job["sample"], **job["design"], solver_failure=repr(exc),
                               wake_quality_pass=False, **{key: None for key in TARGETS})
                cache.write_text(json.dumps(row, indent=2), encoding="utf8")
            design = study.prior.original.prior.AirframeDesign(**job["design"])
            for name, model in models.items():
                for t, value in enumerate(model.predict(design)):
                    row[f"{name}_prediction_{t}"] = value
            rows.append(row)
            frame = pd.DataFrame(rows)
            frame.to_csv(OUT / "独立测试逐点.csv", index=False)
            progress("solver")
            print("Independent geometry", len(rows), "/", NTEST, flush=True)
        metrics = []
        for name in models:
            for t, target in enumerate(TARGETS):
                y = frame[target].to_numpy(float)
                mask = np.isfinite(y)
                if t:
                    mask &= frame.wake_quality_pass.to_numpy(bool)
                if mask.sum() < 3:
                    raise RuntimeError("Insufficient valid fresh test labels")
                prediction = frame[f"{name}_prediction_{t}"].to_numpy(float)
                metrics.append(dict(model=name, target=target, count=int(mask.sum()),
                                    **study.prior.direct.base.metric(y[mask], prediction[mask])))
        (OUT / "独立评价.json").write_text(json.dumps(dict(
            metrics=metrics, protocol=frozen["protocol"], goal_r2=.9,
            grid_convergence_claimed=False, training_labels_only_for_selection=True,
            model_hashes=frozen["model_hashes"]), indent=2), encoding="utf8")
        progress("completed")
    except Exception:
        progress("failed", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
