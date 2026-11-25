"""Training-only selection, independent sampling and artifact inference checks."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 研究_扩样筛选网格代理20261020 as study
from 筛选网格气动代理 import SelectedMeshAero, predict_component, select_columns
from 精细气动代理 import FineAeroSurrogate


class ExpandedMeshTests(unittest.TestCase):
    def data(self):
        source = study.expansion.evaluation.study.OUT / "极曲线训练数据.npz"
        with np.load(source) as saved:
            return saved["x"][:96], saved["y"][:96], saved["mesh"][:96]

    def test_selection_keeps_geometry_and_limits_nonconstant_mesh(self):
        rng = np.random.default_rng(104)
        matrix = rng.normal(size=(120, 12))
        matrix[:, -2:] = 0.
        target = matrix[:, 7] + rng.normal(0, .01, 120)
        columns = select_columns(matrix, target, 3, 2, threads=1)
        np.testing.assert_array_equal(columns[:3], np.arange(3))
        self.assertEqual(len(columns), 5)
        self.assertIn(7, columns)
        self.assertTrue(set(columns).isdisjoint({10, 11}))
        with self.assertRaises(ValueError):
            select_columns(matrix, target, 3, -1)

    def test_validation_targets_do_not_reach_feature_selection(self):
        x, y, mesh = self.data()
        a, b = np.arange(64), np.arange(64, 96)
        config = dict(study.CONFIGS[1], iterations=8)
        seen = []
        original = study.select_columns

        def record(values, target, *args):
            seen.append(target.copy())
            return original(values, target, *args)

        with (patch.object(study, "CONFIGS", [config]),
              patch.object(study, "select_columns", side_effect=record)):
            result = study.task((0, "random", 0, a, b, x, y, mesh))
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], y[a])
        np.testing.assert_array_equal(result["actual"], y[b])
        np.testing.assert_array_equal(result["validation"], b)

    def test_export_preserves_predictions_and_lift(self):
        x, y, mesh = self.data()
        component = study.fit(dict(study.CONFIGS[1], iterations=8), x[:64], y[:64], mesh[:64])
        baseline = study.batch.previous.read(study.STUDY.BASELINE)
        payload = dict(schema="ptdl-selected-mesh-aero-v1", baseline=baseline,
            wake_components=[dict(component, weight=.75), dict(kind="reference", weight=.25)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.joblib"
            joblib.dump(payload, path)
            restored = SelectedMeshAero.load(path)
            result = restored.predict_many(x[64:], mesh[64:])
        expected = FineAeroSurrogate(baseline).predict_many(x[64:])
        np.testing.assert_allclose(result[:, 0], expected[:, 0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(result[:, 1],
            .75*predict_component(component, x[64:], mesh[64:]) + .25*expected[:, 1])
        self.assertTrue(restored.requires_mesh)

    def test_stratified_test_excludes_all_frozen_training_and_test_jobs(self):
        frame = pd.read_csv(study.STUDY.DATA)
        manifest = study.batch.previous.read(study.SOURCE / "配置冻结.json")
        forbidden = np.vstack((frame[study.FEATURES].to_numpy(float),
            [[j["design"][k] for k in study.FEATURES] for j in manifest["jobs"]]))
        centers = frame[study.FEATURES].to_numpy(float)[:24]
        jobs = study.holdout_jobs(frame, forbidden, centers, count_per_stratum=8)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(sum(j["basis"] == "fresh_uniform_feasible" for j in jobs), 8)
        self.assertTrue(all(j["role"] == "test" for j in jobs))
        span = np.ptp(frame[study.FEATURES].to_numpy(float), axis=0)
        x = np.array([[j["design"][k] for k in study.FEATURES] for j in jobs])
        self.assertGreater(np.linalg.norm((x[:, None]-forbidden[None])/span, axis=2).min(), .02)

    def test_force_labels_block_late_prediction_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "solver/test_0").mkdir(parents=True)
            runner = study.expansion.native.original.controlled
            with (patch.object(study, "OUT", out), patch.object(runner, "run_process") as run):
                with self.assertRaisesRegex(RuntimeError, "before prediction registration"):
                    study.lock_predictions(dict(jobs=[dict(sample="test_0")]), None)
                run.assert_not_called()

    def test_expanded_source_validation_reaches_cache_shape_guard(self):
        prior = pd.read_csv(study.STUDY.DATA)
        expanded = pd.concat((prior, prior.iloc[:2048]), ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            np.savez(out / "训练网格特征.npz",
                     features=np.zeros((1, 1)), done=np.zeros(1, bool))
            with (patch.object(study, "OUT", out),
                  patch.object(study.pd, "read_csv", return_value=prior)):
                with self.assertRaisesRegex(RuntimeError, "mesh cache shape"):
                    study.prepare_mesh(expanded)


if __name__ == "__main__":
    unittest.main()
