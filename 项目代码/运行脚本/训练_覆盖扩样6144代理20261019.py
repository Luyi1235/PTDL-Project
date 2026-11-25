"""Compare four previously competitive structures on an expanded training cohort."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from pathlib import Path
import traceback

import 研究_精细树与核代理20261018 as trainer

# Execute this assignment in spawned workers as well as the parent process.
ORIGINAL_INDICES = (3, 5, 9, 10)
trainer.CONFIGS = [trainer.CONFIGS[i] for i in ORIGINAL_INDICES]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        trainer.run(args.data.resolve(), args.output_dir.resolve(), workers=4)
    except Exception:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "训练失败.txt").write_text(traceback.format_exc(), encoding="utf8")
        raise
