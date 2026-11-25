"""Run PTDL 80 h MOEA/D with the supplied Hybrid_System component plant."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from ptdl_2_2_2_optimizer import run_ptdl_80h_joint_moead
from ptdl_hybrid_system_adapter import HybridSystemAdapterConfig, HybridSystemPowertrain


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PTDL 80 h MOEA/D candidates with the integrated component powertrain."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ptdl_hybrid_component_moead_results")
    parser.add_argument("--resolution", type=int, default=6, help="MOEA/D simplex resolution H")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--engine-map", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run H=1 for one generation")
    args = parser.parse_args()

    resolution = 1 if args.smoke else args.resolution
    generations = 1 if args.smoke else args.generations
    config = HybridSystemAdapterConfig(
        engine_map_path=args.engine_map,
    )
    factory = partial(HybridSystemPowertrain, config=config)
    run_ptdl_80h_joint_moead(
        args.output_dir,
        resolution=resolution,
        generations=generations,
        seed=args.seed,
        aero_model_label="PTDL low-order/VSPAERO provider",
        powertrain_factory=factory,
    )


if __name__ == "__main__":
    main()
