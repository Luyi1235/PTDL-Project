import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 评价_代理候选队列20261017 as queue


class RegisteredEvaluationTests(unittest.TestCase):
    def test_paired_bootstrap_detects_known_gain(self):
        y = np.linspace(.4, .95, 30)
        result = queue.evaluator.paired_bootstrap(y, y+.01, y+.1, repetitions=100)
        self.assertGreater(result["lower95"], 0)

    def test_missing_labels_and_geometry_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "out"
            out.mkdir()
            rows = []
            for i in range(4):
                row = dict(sample=f"test_{i}", role="test",
                           **dict(zip(queue.FEATURES, [30.+i, 24., .5, 3., -1., 8., .8])))
                for name in ("local_geometry", "linear_leaf", "balanced1512"):
                    for t in range(2):
                        row[f"{name}_prediction_{t}"] = (6.+i*.1) if not t else (.7+i*.05)
                rows.append(row)
            pd.DataFrame(rows).to_csv(out / "预先锁定预测.csv", index=False)
            models = ["local_geometry", "linear_leaf", "balanced1512"]
            with patch.object(queue, "BATCH", root):
                self.assertFalse(queue.evaluate("test", out, dict(models=models)))
                for i, row in enumerate(rows):
                    case = root / "solver" / row["sample"]
                    case.mkdir(parents=True)
                    record = dict(row, **dict(zip(queue.TARGETS, [6.+i*.1, .7+i*.05])),
                                  wake_quality_pass=True)
                    (case / "record.json").write_text(json.dumps(record), encoding="utf8")
                self.assertTrue(queue.evaluate("test", out, dict(models=models)))
                result = json.loads((out / "独立测试.json").read_text(encoding="utf8"))
                self.assertTrue(result["independent_test"])
                self.assertEqual(len(result["metrics"]), 6)
                path = root / "solver/test_0/record.json"
                record = json.loads(path.read_text(encoding="utf8"))
                record[queue.FEATURES[0]] += 1
                path.write_text(json.dumps(record), encoding="utf8")
                with self.assertRaises(AssertionError):
                    queue.evaluate("test", out, dict(models=models))


if __name__ == "__main__":
    unittest.main()
