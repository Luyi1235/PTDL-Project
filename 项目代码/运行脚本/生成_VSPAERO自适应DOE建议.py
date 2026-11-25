"""Generate reproducible maximin VSPAERO DOE points within the current data range."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from VSPAERO气动代理模型 import (
    FEATURE_LABELS,
    KrigingVSPAeroSurrogate,
)
from 机体参数化设计与气动阻力 import AirframeDesign


DEFAULT_DATA = ROOT / "气动代理模型数据" / "VSPAERO_DOE扩充样本_23点.csv"
DEFAULT_MODEL = (
    ROOT / "运行结果" / "Kriging气动代理留一比较_20261004" / "VSPAERO_Kriging模型.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "运行结果" / "VSPAERO自适应DOE建议_20261004",
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.count < 1 or args.pool_size < args.count:
        raise SystemExit("count must be positive and pool-size must be >= count.")

    data = pd.read_csv(args.data)
    missing = set(FEATURE_LABELS).difference(data.columns)
    if missing:
        raise ValueError(f"DOE data is missing features: {sorted(missing)}")
    observed = data.loc[:, FEATURE_LABELS].to_numpy(dtype=float)
    if len(observed) < 10 or not np.isfinite(observed).all():
        raise ValueError("At least ten finite DOE points are required.")

    lower = observed.min(axis=0)
    upper = observed.max(axis=0)
    span = np.maximum(upper - lower, 1e-12)
    rng = np.random.default_rng(args.seed)
    pool = rng.uniform(lower, upper, size=(args.pool_size, len(FEATURE_LABELS)))
    normalized_observed = (observed - lower) / span
    normalized_pool = (pool - lower) / span
    selected: list[np.ndarray] = []
    selected_distances: list[float] = []

    for _ in range(args.count):
        references = normalized_observed
        if selected:
            references = np.vstack(
                [references, (np.vstack(selected) - lower) / span]
            )
        distances = np.linalg.norm(
            normalized_pool[:, None, :] - references[None, :, :], axis=2
        )
        nearest = distances.min(axis=1)
        index = int(np.argmax(nearest))
        selected.append(pool[index].copy())
        selected_distances.append(float(nearest[index]))
        pool = np.delete(pool, index, axis=0)
        normalized_pool = np.delete(normalized_pool, index, axis=0)

    model = KrigingVSPAeroSurrogate.from_json(args.model)
    records: list[dict[str, object]] = []
    for index, values in enumerate(selected, start=1):
        design_values = dict(zip(FEATURE_LABELS, values.tolist()))
        design = AirframeDesign(**design_values)
        cl_alpha, oswald, cl_std, oswald_std = model.predict_with_uncertainty(design)
        record: dict[str, object] = {
            "suggestion": f"adaptive_doe_{index:02d}",
            **design_values,
            "nearest_normalized_distance_at_selection": selected_distances[index - 1],
            "kriging_predicted_cl_alpha_per_rad": cl_alpha,
            "kriging_predicted_oswald_efficiency": oswald,
            "kriging_std_cl_alpha_per_rad": cl_std,
            "kriging_std_oswald_efficiency": oswald_std,
            "vspaero_label_status": "pending_solver_run",
        }
        records.append(record)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        output_dir / "VSPAERO自适应DOE待计算点.csv", index=False
    )
    metadata = {
        "method": "greedy maximin in feature-wise normalized space",
        "seed": args.seed,
        "requested_points": args.count,
        "candidate_pool_size": args.pool_size,
        "observed_samples": len(observed),
        "feature_labels": list(FEATURE_LABELS),
        "featurewise_bounds": {
            label: [float(low), float(high)]
            for label, low, high in zip(FEATURE_LABELS, lower, upper)
        },
        "training_model_samples": model.training_samples,
        "status": "design suggestions only; no new VSPAERO responses were computed",
        "limitations": [
            "Maximin distance improves geometric coverage but does not guarantee lower cross-validation error.",
            "Kriging uncertainty is conditional on the current small DOE and is not calibrated confidence.",
            "All proposed geometries require OpenVSP/VSPAERO execution and quality checks before model refitting.",
        ],
    }
    (output_dir / "DOE选点方法与限制.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(records).to_string(index=False))
    print(f"\n待计算建议点：{output_dir}")


if __name__ == "__main__":
    main()
