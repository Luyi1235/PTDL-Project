from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 评价_面元代理新192构型20261018 as evaluation


class IndependentMeshPredictionTests(unittest.TestCase):
    def test_prediction_never_reads_the_native_label_record(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            output = folder / "prediction"
            output.mkdir()
            job = dict(sample="heldout",
                       design=dict(zip(evaluation.batch.FEATURES, [30., 24., .5, 3., -1., 8., .8])))
            native = folder / "solver/heldout"
            native.mkdir(parents=True)
            (native / "record.json").write_text("labels must not be parsed during prediction")
            features = dict(zip(evaluation.panel.FEATURE_NAMES,
                                np.ones(len(evaluation.panel.FEATURE_NAMES))))
            features["extraction_error"] = None
            selection = dict(selected=[dict(components=[dict(weight=1., path="unused")])
                                       for _ in range(2)])
            with patch.object(evaluation.batch, "OUT", folder), \
                 patch.object(evaluation.panel, "extract", return_value=features), \
                 patch.object(evaluation.joblib, "load", return_value=None), \
                 patch.object(evaluation.panel, "predict", return_value=np.array([.7])), \
                 patch.object(evaluation.batch.previous, "read",
                              side_effect=AssertionError("Unexpected label access")):
                result = evaluation.predict_case(job, selection, output)
            self.assertAlmostEqual(result["panel_prediction_0"], .7)
            self.assertAlmostEqual(result["panel_prediction_1"], .7)
            self.assertTrue((output / "heldout.json").exists())


if __name__ == "__main__":
    unittest.main()
