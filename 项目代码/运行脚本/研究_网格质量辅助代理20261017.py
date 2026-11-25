"""Compare panel-quality descriptors using the same full-geometry folds."""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

import 研究_网格辅助代理20261017 as study
from 气动网格质量特征 import mesh_quality, FEATURE_NAMES

OUT = study.direct.ROOT / "运行结果/网格质量辅助代理_20261017"
MESH = OUT / "训练网格质量.csv"

study.OUT = OUT
study.MESH = MESH
study.COLS = FEATURE_NAMES


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if not MESH.exists():
        frame = pd.read_csv(study.direct.DATA)
        rows = []
        started = time.monotonic()
        for _, row in frame.iterrows():
            if not isinstance(row.geometry_file, str):
                continue
            values = mesh_quality(Path(row.geometry_file).with_suffix(".vspgeom"),
                                  row.wing_area_m2, row.aspect_ratio)
            rows.append(dict(sample=row["sample"], **dict(zip(FEATURE_NAMES, values))))
            if len(rows) % 100 == 0:
                print("Mesh quality", len(rows), "/", len(frame), flush=True)
                (OUT / "进度.json").write_text(json.dumps(dict(
                    stage="extracting_geometry", completed=len(rows), requested=1467,
                    elapsed_s=time.monotonic()-started)), encoding="utf8")
        pd.DataFrame(rows).to_csv(MESH, index=False)
    study.main()
