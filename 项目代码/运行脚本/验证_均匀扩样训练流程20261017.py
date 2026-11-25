"""Exercise the complete parallel fitting and export flow using existing training rows."""

from pathlib import Path
import pandas as pd

import 训练_均匀覆盖代理20261017 as train


if __name__ == "__main__":
    folder = train.ROOT / "运行结果/均匀扩样训练流程预检_20261017"
    folder.mkdir(exist_ok=True)
    frame = pd.read_csv(train.balanced.previous.DATA).iloc[::15].head(96)
    path = folder / "预检训练标签.csv"
    frame.to_csv(path, index=False)
    train.main(path, folder, workers=8)
