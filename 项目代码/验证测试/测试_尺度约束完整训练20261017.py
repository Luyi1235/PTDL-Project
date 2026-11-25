import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 推进_尺度约束完整训练20261017 as pipeline


class FullScalePipelineTests(unittest.TestCase):
    def test_training_geometry_cannot_be_registered_as_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            design = dict(zip(pipeline.FEATURES, [30., 24., .5, 3., -1., 8., .8]))
            data = folder / "training.csv"
            pd.DataFrame([design]).to_csv(data, index=False)
            with self.assertRaisesRegex(RuntimeError, "appears in training"):
                pipeline.register_predictions(folder / "test",
                    [dict(sample="duplicate", role="test", design=design)], {}, data)

    def test_record_geometry_and_role_must_match(self):
        job = dict(sample="a", role="test",
                   design=dict(zip(pipeline.FEATURES, [30., 24., .5, 3., -1., 8., .8])))
        row = dict(sample="a", role="test", **job["design"])
        pipeline.validate_record(job, row)
        with self.assertRaises(RuntimeError):
            pipeline.validate_record(job, dict(row, role="train"))
        with self.assertRaises(AssertionError):
            pipeline.validate_record(job, dict(row, wing_area_m2=31.))

    def test_evaluation_retains_failures_and_uses_common_quality_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            evaluation, native = folder / "evaluation", folder / "native"
            evaluation.mkdir()
            jobs, predictions = [], []
            for i in range(5):
                design = dict(zip(pipeline.FEATURES, [30.+i, 24., .5, 3., -1., 8., .8]))
                job = dict(sample=f"case{i}", role="test", design=design)
                jobs.append(job)
                predictions.append(dict(sample=job["sample"],
                    scale_candidate_prediction_0=5.+i, same_data_control_prediction_0=5.1+i,
                    scale_candidate_prediction_1=.6+i*.05,
                    same_data_control_prediction_1=.62+i*.05))
            pipeline.save(evaluation / "预测冻结.json",
                          dict(jobs=jobs, models=["scale_candidate", "same_data_control"], hashes={}))
            pd.DataFrame(predictions).to_csv(evaluation / "锁定预测.csv", index=False)
            self.assertFalse(pipeline.evaluate(evaluation, native, "diagnostic"))
            for i, job in enumerate(jobs):
                case = native / job["sample"]
                case.mkdir(parents=True)
                row = dict(sample=job["sample"], role="test", **job["design"],
                           wake_quality_pass=i < 3)
                row[pipeline.TARGETS[0]] = 5.+i if i < 4 else None
                row[pipeline.TARGETS[1]] = .6+i*.05 if i < 4 else None
                if i == 4:
                    row["solver_failure"] = "test failure"
                pipeline.save(case / "record.json", row)
            self.assertTrue(pipeline.evaluate(evaluation, native, "diagnostic"))
            result = pipeline.read(evaluation / "diagnostic.json")
            self.assertEqual(result["attempted"], 5)
            for record in result["metrics"]:
                expected = 4 if record["target"] == pipeline.TARGETS[0] else 3
                self.assertEqual(record["count"], expected)
            candidate = [r for r in result["metrics"] if r["model"] == "scale_candidate"]
            self.assertTrue(all(np.isclose(r["r2"], 1.) for r in candidate))
            retained = pd.read_csv(evaluation / "diagnostic逐点.csv")
            self.assertEqual(len(retained), 5)
            self.assertEqual(retained.solver_failure.notna().sum(), 1)


if __name__ == "__main__":
    unittest.main()
