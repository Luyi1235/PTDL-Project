"""Adapt the frozen mesh surrogate to the optimizer's aerodynamic interface."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import uuid

import numpy as np

from VSPAERO气动代理模型 import FEATURE_LABELS
from 机体参数化设计与气动阻力 import AirframeDesign, aero_from_design
from 尾迹射线网格特征 import FEATURE_NAMES, extract

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "运行脚本/运行_尾迹射线几何预处理20261018.py"
PYTHON = ROOT.parent.parent / "tools/Python311/python.exe"
CACHE = ROOT / "运行结果/工作台代理几何缓存_20261020"


class WorkbenchGeometryFeatures:
    def __init__(self, cache_dir=CACHE, python_executable=PYTHON):
        self.cache_dir = Path(cache_dir)
        self.python_executable = Path(python_executable)
        self.memory = {}
        self.lock = threading.Lock()
        self.generated = 0
        sources = [WORKER, ROOT / "核心模型/机体参数化设计与气动阻力.py",
                   ROOT / "核心模型/OpenVSP_VSPAERO接口.py",
                   ROOT / "核心模型/尾迹射线网格特征.py"]
        self.protocol = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def _generate(self, folder, key, design):
        if not self.python_executable.is_file():
            raise RuntimeError("网格融合代理需要安装 OpenVSP 的 Python 解释器。")
        request = dict(output_root=str(folder.parent),
                       jobs=[dict(sample=key, design=design)])
        print("正在生成代理模型的几何网格特征，气动力由已训练模型预测。", flush=True)
        result = subprocess.run(
            [str(self.python_executable), "-X", "utf8", str(WORKER)],
            input=json.dumps(request), capture_output=True, text=True, encoding="utf8",
            timeout=180, cwd=ROOT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise RuntimeError(f"代理几何预处理失败。{result.stderr[-1200:]}")
        self.generated += 1

    def __call__(self, design):
        # Mesh construction uses these seven dimensions; payload and energy sizing
        # do not change the airframe surface mesh.
        geometry = asdict(AirframeDesign(**{
            key: float(getattr(design, key)) for key in FEATURE_LABELS
        }))
        key = hashlib.sha256(json.dumps(
            dict(design=geometry, protocol=self.protocol),
            sort_keys=True, allow_nan=False).encode("utf8")).hexdigest()
        with self.lock:
            if key in self.memory:
                return self.memory[key].copy()
            folder = self.cache_dir / key
            record_path = folder / "geometry.json"
            if not record_path.is_file():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._generate(folder, key, geometry)
            record = json.loads(record_path.read_text(encoding="utf8"))
            mesh_path = folder / "airframe.vspgeom"
            mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            if (record["design"] != geometry or record["mesh_sha256"] != mesh_hash
                    or record.get("force_solver_called") is not False):
                raise RuntimeError("代理几何缓存与当前构型不一致。")
            cache = folder / "features.npz"
            if cache.is_file():
                with np.load(cache, allow_pickle=False) as saved:
                    if str(saved["mesh_sha256"]) != mesh_hash:
                        raise RuntimeError("代理特征缓存与几何网格不一致。")
                    values = saved["features"]
            else:
                try:
                    values = extract(mesh_path, design.wing_area_m2, design.aspect_ratio)
                except (ValueError, IndexError, StopIteration) as error:
                    raise RuntimeError("当前构型的几何网格无法生成融合代理特征。") from error
                temporary = folder / f"features_{uuid.uuid4().hex}.npz"
                np.savez_compressed(temporary, features=values, mesh_sha256=mesh_hash)
                temporary.replace(cache)
            if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
                raise RuntimeError("代理几何特征的维度或数值无效。")
            self.memory[key] = values.copy()
            return values.copy()


class WorkbenchSelectedMeshAero:
    def __init__(self, model, geometry_features=None):
        self.model = model
        self.geometry_features = (
            geometry_features if geometry_features is not None else WorkbenchGeometryFeatures())
        self.predictions = {}

    @classmethod
    def load(cls, path):
        from 筛选网格气动代理 import SelectedMeshAero

        return cls(SelectedMeshAero.load(path))

    @property
    def training_samples(self):
        return int(self.model.payload["training_samples"])

    def predict(self, design):
        x = tuple(float(getattr(design, key)) for key in FEATURE_LABELS)
        if x not in self.predictions:
            mesh = self.geometry_features(design) if self.model.requires_mesh else None
            p = self.model.predict_many([x], None if mesh is None else mesh[None])[0]
            if p.shape != (2,) or not np.isfinite(p).all():
                raise ValueError("融合代理预测无效。")
            self.predictions[x] = tuple(map(float, p))
        return self.predictions[x]

    def corrected_aero(self, design, calibration):
        low = aero_from_design(design, calibration)
        lift, wake = self.predict(design)
        lift0, wake0 = self.model.payload["baseline"]["baseline_targets"]
        return replace(
            low,
            cl_alpha_per_rad=float(np.clip(low.cl_alpha_per_rad * lift / lift0, 3.5, 8.5)),
            oswald_efficiency=float(np.clip(low.oswald_efficiency * wake / wake0, .45, 1.)),
        )
