"""Check aerodynamic-label sensitivity to homothetic training geometry changes."""

import json
from pathlib import Path
import time

import pandas as pd

import 研究_尺度约束与总体误差20261017 as study
import 推进_千构型原定义代理20261017 as original

ROOT = study.ROOT
OUT = ROOT / "运行结果/等比例几何响应_20261017"


def main():
    OUT.mkdir(exist_ok=True)
    data = study.prior.prior.previous.DATA
    frame = pd.read_csv(data).set_index("sample")
    anchors = ["stable_train_0000", "stable_train_0821"]
    factors = [.98, 1.02]
    paths = [Path(__file__), data, original.WORKER]
    manifest = dict(anchors=anchors, factors=factors,
                    hashes={str(p): study.prior.digest(p) for p in paths},
                    training_geometries_only=True, production_labels_replaced=False,
                    purpose="Check canonical-scale feature assumption without adding synthetic labels",
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    frozen = OUT / "配置冻结.json"
    if frozen.exists():
        old = json.loads(frozen.read_text(encoding="utf8"))
        if old["hashes"] != manifest["hashes"]:
            raise RuntimeError("Homothetic response sources changed")
    else:
        frozen.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    original.controlled.original.WORKER = original.WORKER
    records, started = [], time.monotonic()
    for sample in anchors:
        anchor = frame.loc[sample]
        if not bool(anchor.wake_quality_pass):
            raise RuntimeError("Expected an existing valid training anchor")
        for factor in factors:
            design = {key: float(anchor[key]) for key in study.FEATURES}
            for key in ("propeller_diameter_m", "payload_mass_kg", "hybrid_system_mass_kg"):
                design[key] = float(anchor[key])
            design["wing_area_m2"] *= factor**2
            design["fuselage_length_m"] *= factor
            design["fuselage_diameter_m"] *= factor
            folder = OUT / "solver" / f"{sample}_scale{int(factor*100)}"
            folder.mkdir(parents=True, exist_ok=True)
            cache = folder / "record.json"
            if cache.exists():
                row = json.loads(cache.read_text(encoding="utf8"))
                if row["source_sample"] != sample or row["scale_factor"] != factor:
                    raise RuntimeError("Homothetic cache mismatch")
            else:
                try:
                    row = original.controlled.run_case(
                        original.prior.solver.DEFAULT_PYTHON, design, f"{sample}_scale{factor:g}",
                        folder / "case", 300, "diagnostic", "homothetic_training_geometry",
                        (0., float("inf")), 30, "legacy")
                    row.update(original.verify(folder / "case"))
                    row.update(original.prior.diagnostic_quality(folder / "case/vspaero_polar.csv"))
                    lines = (folder / "case/airframe.vspgeom").read_text().splitlines()
                    row.update(zip(("nodes", "loops", "surfaces"), map(int, lines[2].split())))
                except Exception as exc:
                    row = dict(solver_failure=repr(exc))
                row.update(source_sample=sample, scale_factor=factor,
                           baseline_lift=float(anchor[study.prior.prior.TARGETS[0]]),
                           baseline_wake=float(anchor[study.TARGET]))
                if "vspaero_oswald_efficiency" in row:
                    row["lift_change"] = row["vspaero_cl_alpha_per_rad"]-row["baseline_lift"]
                    row["wake_change"] = row["vspaero_oswald_efficiency"]-row["baseline_wake"]
                cache.write_text(json.dumps(row, indent=2), encoding="utf8")
            records.append(row)
            pd.DataFrame(records).to_csv(OUT / "等比例响应.csv", index=False)
            (OUT / "进度.json").write_text(json.dumps(dict(
                stage="completed" if len(records) == 4 else "solver", completed=len(records),
                requested=4, elapsed_s=time.monotonic()-started)), encoding="utf8")
            print("Homothetic response", len(records), "/ 4", flush=True)
    valid = [r for r in records if "solver_failure" not in r]
    summary = dict(
        successful=len(valid), requested=4,
        max_abs_lift_change=max((abs(r["lift_change"]) for r in valid), default=None),
        max_abs_wake_change=max((abs(r["wake_change"]) for r in valid), default=None),
        synthetic_training_labels_added=False,
        interpretation="Limited training-anchor diagnostic; model choice still uses held-out predictions")
    (OUT / "响应摘要.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
