"""Independent geometry separation and protection of prediction registration."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 验证_极曲线结构代理20261019 as pipeline


class PolarPipelineTests(unittest.TestCase):
    def test_new_cohort_excludes_all_inspected_geometries(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = pipeline.plan(Path(directory))
        jobs = manifest["jobs"]
        self.assertEqual(len(jobs), 384)
        features = pipeline.study.FEATURES
        frame = pd.read_csv(pipeline.study.DATA)
        avoid = pipeline.batch.forbidden_geometries(frame)
        for path in (pipeline.batch.OUT / "配置冻结.json",
                     pipeline.native.OUT / "构型与计算冻结.json"):
            previous = pipeline.batch.previous.read(path)["jobs"]
            avoid = np.vstack((avoid, [[j["design"][k] for k in features] for j in previous]))
        values = np.array([[j["design"][k] for k in features] for j in jobs])
        span = np.ptp(frame[features].to_numpy(float), axis=0)
        self.assertEqual(len(np.unique(values, axis=0)), 384)
        self.assertTrue(all(np.linalg.norm((avoid-row)/span, axis=1).min() > 1e-5
                            for row in values))
        self.assertTrue(all(max(pipeline.original.prior.geometry_constraints(
            pipeline.original.prior.AirframeDesign(**j["design"])).values()) <= 1e-9
            for j in jobs))

    def test_solver_directory_prevents_late_prediction_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "solver" / "case_1").mkdir(parents=True)
            with patch.object(pipeline, "predictors") as predictors:
                with self.assertRaisesRegex(RuntimeError, "before prediction lock"):
                    pipeline.lock(out, dict(jobs=[dict(sample="case_1")]))
                predictors.assert_not_called()

    def test_preparation_has_no_aerodynamic_label_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with (patch.object(pipeline, "plan", return_value=dict(hashes={})),
                  patch.object(pipeline.native, "prepare") as prepare,
                  patch.object(pipeline, "diagnostic") as diagnostic,
                  patch.object(pipeline, "lock") as lock,
                  patch.object(pipeline.original.controlled, "run_case") as solve):
                pipeline.main(out, prepare_only=True)
                prepare.assert_called_once()
                diagnostic.assert_not_called()
                lock.assert_not_called()
                solve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
