"""Load deployed workbench models without changing frozen experiment sources."""

import json
from pathlib import Path

from VSPAERO气动代理模型 import load_vsp_surrogate


def load_workbench_surrogate(path):
    if Path(path).suffix == ".joblib":
        from 工作台网格融合代理 import WorkbenchSelectedMeshAero

        return WorkbenchSelectedMeshAero.load(path)
    schema = json.loads(Path(path).read_text(encoding="utf-8")).get("schema")
    if schema == "ptdl-fine-aero-v1":
        from 精细气动代理 import FineAeroSurrogate

        return FineAeroSurrogate.from_json(path)
    return load_vsp_surrogate(path)
