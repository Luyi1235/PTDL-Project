"""Exercise frozen model inference, mesh caching and optimizer aero conversion."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "核心模型"))
from 工作台气动代理加载 import load_workbench_surrogate
from 工作台网格融合代理 import WorkbenchGeometryFeatures, WorkbenchSelectedMeshAero
from VSPAERO气动代理模型 import FEATURE_LABELS
from 机体参数化设计与气动阻力 import AirframeDesign
from 尾迹射线网格特征 import extract

STUDY = ROOT / "运行结果/扩样筛选网格代理_20261020"
FINE = ROOT / "运行结果/覆盖扩样6144代理_20261019/精细训练/精细气动候选代理.json"


class WorkbenchSurrogateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = pd.read_csv(STUDY / "独立256构型/独立逐点.csv").iloc[0]
        cls.design = AirframeDesign(**{key: float(cls.row[key]) for key in FEATURE_LABELS})
        cls.mesh_path = STUDY / "独立256构型/geometry" / cls.row["sample"] / "airframe.vspgeom"

    def test_geometry_model_reproduces_frozen_prediction(self):
        model = load_workbench_surrogate(FINE)
        self.assertEqual(model.training_samples, 6144)
        np.testing.assert_allclose(model.predict(self.design),
            [self.row[f"fine6144_prediction_{i}"] for i in range(2)], atol=1e-12, rtol=0)

    def test_mesh_model_reproduces_frozen_prediction_and_reuses_geometry(self):
        model = load_workbench_surrogate(STUDY / "筛选网格候选.joblib")
        self.assertIsInstance(model, WorkbenchSelectedMeshAero)
        self.assertEqual(model.training_samples, 6144)
        values = extract(self.mesh_path, self.design.wing_area_m2, self.design.aspect_ratio)
        with patch.object(model, "geometry_features", return_value=values) as geometry:
            first = model.predict(self.design)
            second = model.predict(replace(self.design, payload_mass_kg=150))
        geometry.assert_called_once()
        np.testing.assert_allclose(first,
            [self.row[f"selected_mesh_prediction_{i}"] for i in range(2)], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(first, second)

    def test_corrected_coefficients_use_fused_predictions(self):
        model = load_workbench_surrogate(STUDY / "筛选网格候选.joblib")
        from types import SimpleNamespace
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Aero:
            cl_alpha_per_rad: float
            oswald_efficiency: float

        baseline = model.model.payload["baseline"]["baseline_targets"]
        low = Aero(5.5, .8)
        with (patch.object(model, "predict", return_value=(baseline[0], .8*baseline[1])),
              patch("工作台网格融合代理.aero_from_design", return_value=low)):
            result = model.corrected_aero(self.design, SimpleNamespace())
        self.assertAlmostEqual(result.cl_alpha_per_rad, 5.5)
        self.assertAlmostEqual(result.oswald_efficiency, .64)

    def test_disk_cache_survives_restart_and_ignores_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = WorkbenchGeometryFeatures(directory)

            def generate(folder, key, design):
                folder.mkdir(parents=True)
                mesh = folder / "airframe.vspgeom"
                shutil.copyfile(self.mesh_path, mesh)
                (folder / "geometry.json").write_text(json.dumps(dict(
                    design=design, mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
                    force_solver_called=False)), encoding="utf8")

            with patch.object(provider, "_generate", side_effect=generate) as create:
                first = provider(self.design)
                provider(replace(self.design, hybrid_system_mass_kg=100))
            create.assert_called_once()
            restored = WorkbenchGeometryFeatures(directory)
            with (patch.object(restored, "_generate") as create,
                  patch("工作台网格融合代理.extract") as extractor):
                second = restored(self.design)
            create.assert_not_called()
            extractor.assert_not_called()
            np.testing.assert_array_equal(first, second)

    def test_changed_geometry_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = WorkbenchGeometryFeatures(directory)

            def bad_cache(folder, key, design):
                folder.mkdir(parents=True)
                (folder / "airframe.vspgeom").write_text("invalid", encoding="utf8")
                (folder / "geometry.json").write_text(json.dumps(dict(
                    design=design, mesh_sha256="wrong", force_solver_called=False)),
                    encoding="utf8")

            with patch.object(provider, "_generate", side_effect=bad_cache):
                with self.assertRaisesRegex(RuntimeError, "缓存"):
                    provider(self.design)

    def test_deployed_metrics_retain_difficult_results(self):
        registry = json.loads((ROOT / "气动代理模型数据/工作台代理精度.json").read_text(encoding="utf8"))
        self.assertEqual(set(registry["models"]), {"fine6144", "selectedmesh6144"})
        for model in registry["models"].values():
            self.assertEqual(model["dataset_count"], 128)
            self.assertEqual([m["count"] for m in model["metrics"]], [128, 122])
            self.assertTrue(all(m["r2"] > .9 for m in model["metrics"]))
            self.assertLess(model["difficult_metrics"][1]["r2"], .9)
            self.assertLess(model["combined_metrics"][1]["r2"], .9)


if __name__ == "__main__":
    unittest.main()
