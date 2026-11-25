"""Training acquisition and independent testing remain separated during expansion."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 推进_覆盖扩样6144代理20261019 as pipeline


class CoverageExpansionTests(unittest.TestCase):
    def test_acquisition_counts_geometry_constraints_and_holdout_separation(self):
        manifest = pipeline.batch.previous.read(pipeline.OUT / "配置冻结.json")
        jobs = manifest["jobs"]
        training = [j for j in jobs if j["role"] == "train"]
        tests = [j for j in jobs if j["role"] == "test"]
        self.assertEqual(len(training), 2048)
        self.assertEqual(len(tests), 256)
        self.assertEqual(sum(j["basis"] == "uniform_feasible_coverage" for j in training), 1536)
        self.assertEqual(sum(j["basis"] == "training_oof_error_neighborhood" for j in training), 512)
        frame = pd.read_csv(pipeline.study.DATA)
        avoid = pipeline.batch.forbidden_geometries(frame)
        for path in (
            pipeline.batch.OUT / "配置冻结.json",
            pipeline.native.OUT / "构型与计算冻结.json",
            pipeline.evaluation.OUT / "构型与计算冻结.json",
        ):
            previous = pipeline.batch.previous.read(path)["jobs"]
            avoid = np.vstack((avoid, [[j["design"][k] for k in pipeline.FEATURES]
                                      for j in previous]))
        values = np.array([[j["design"][k] for k in pipeline.FEATURES] for j in jobs])
        self.assertEqual(len(np.unique(values, axis=0)), len(jobs))
        span = np.ptp(frame[pipeline.FEATURES].to_numpy(float), axis=0)
        self.assertTrue(all(np.linalg.norm((avoid-row)/span, axis=1).min() > 1e-5
                            for row in values))
        train_x, test_x = values[:2048], values[2048:]
        self.assertTrue(all(np.linalg.norm((train_x-row)/span, axis=1).min() > 1e-5
                            for row in test_x))
        self.assertTrue(all(max(pipeline.original.prior.geometry_constraints(
            pipeline.original.prior.AirframeDesign(**j["design"])).values()) <= 1e-9
            for j in jobs))

    def test_error_centers_are_training_geometries(self):
        frame = pd.read_csv(pipeline.study.DATA)
        with patch.object(pipeline.pd, "read_csv", side_effect=AssertionError("Unexpected CSV access")):
            centers = pipeline.training_error_centers(frame)
        valid = frame.wake_quality_pass.fillna(False) & frame[pipeline.study.TARGET].notna()
        rows = {tuple(row) for row in frame.loc[valid, pipeline.FEATURES].to_numpy(float)}
        self.assertTrue(all(tuple(row) in rows for row in centers))
        self.assertEqual(centers.shape, (256, 7))

    def test_failed_native_case_is_preserved_and_reused(self):
        job = pipeline.batch.previous.read(pipeline.OUT / "配置冻结.json")["jobs"][0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pipeline.original.controlled, "run_case",
                              side_effect=RuntimeError("native test failure")) as solve:
                row = pipeline.run_case(Path(directory), job)
                self.assertFalse(row["wake_quality_pass"])
                self.assertIsNone(row[pipeline.TARGETS[1]])
                self.assertIn("native test failure", row["solver_failure"])
                self.assertEqual(pipeline.run_case(Path(directory), job), row)
                solve.assert_called_once()

    def test_late_prediction_registration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "independent"
            out.mkdir()
            (root / "solver" / "test_0").mkdir(parents=True)
            with patch.object(pipeline.native, "prepare") as prepare:
                with self.assertRaisesRegex(RuntimeError, "before prediction registration"):
                    pipeline.lock(out, [dict(sample="test_0")], root / "unused.json")
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
