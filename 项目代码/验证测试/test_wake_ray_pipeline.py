"""Fresh geometry separation and solver access after prediction freezing."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 验证_尾迹网格新256构型20261018 as pipeline


class WakePipelineTests(unittest.TestCase):
    def test_new_cohort_excludes_training_and_previous_test_geometries(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = pipeline.plan(Path(directory))
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), 256)
            x = np.array([[j["design"][k] for k in pipeline.FEATURES] for j in jobs])
            training = pd.read_csv(pipeline.study.DATA)
            old = pipeline.batch.previous.read(pipeline.batch.OUT / "配置冻结.json")["jobs"]
            avoid = np.vstack((training[pipeline.FEATURES].to_numpy(float),
                               [[j["design"][k] for k in pipeline.FEATURES] for j in old]))
            self.assertEqual(len(np.unique(x, axis=0)), 256)
            span = np.ptp(training[pipeline.FEATURES].to_numpy(float), axis=0)
            self.assertTrue(all(np.linalg.norm((avoid-row)/span, axis=1).min() > 1e-5 for row in x))
            self.assertTrue(all(max(pipeline.original.prior.geometry_constraints(
                pipeline.original.prior.AirframeDesign(**j["design"])).values()) <= 1e-9 for j in jobs))

    def test_prediction_registration_rejected_after_solving_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "solver" / "test_0").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "before prediction lock"):
                pipeline.lock(out, dict(jobs=[dict(sample="test_0")]))

    def test_solver_cannot_run_when_prediction_lock_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            out, training = Path(directory) / "evaluation", Path(directory) / "training"
            out.mkdir()
            training.mkdir()
            (out / "构型与计算冻结.json").write_text(
                json.dumps(dict(jobs=[], hashes={})), encoding="utf8")
            (training / "模型选择.json").write_text("{}", encoding="utf8")
            with (patch.object(pipeline.study, "OUT", training),
                  patch.object(pipeline, "prepare"),
                  patch.object(pipeline, "lock", side_effect=RuntimeError("unfrozen")),
                  patch.object(pipeline.original.controlled, "run_case") as solve):
                with self.assertRaisesRegex(RuntimeError, "unfrozen"):
                    pipeline.main(out, wait=False, prepare_only=False)
                solve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
