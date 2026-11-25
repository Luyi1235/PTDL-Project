"""Extend regularization where training-only selection reached the search boundary."""

import argparse
from pathlib import Path

import 研究_代理正则化与嵌套验证20261016 as regularized

# This runs on worker import as well, keeping configuration indices identical.
regularized.CONFIGS = regularized.configurations()
for mode, ard, penalties in (
    ("dimensionless", True, (.1, .2, .4, .8)),
    ("dimensionless", False, (.2, .4, .8)),
    ("raw", False, (.1, .2, .4)),
    ("physics", False, (.1, .2, .4)),
):
    regularized.CONFIGS.extend(
        dict(method="gp", mode=mode, ard=ard, penalty=value) for value in penalties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=regularized.ROOT / "运行结果" / "强正则化与嵌套验证_20261016")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    regularized.train(args.output_dir, args.workers)
