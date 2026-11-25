import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "核心模型"), str(ROOT / "运行脚本")]
import 推进_首批均匀增量代理20261017 as study


class UniformIncrementTests(unittest.TestCase):
    def test_snapshot_preserves_failures_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "original.csv"
            base = dict(zip(study.FEATURES, [30., 24., .5, 3., -1., 8., .8]))
            pd.DataFrame([dict(sample="original", role="train", **base)]).to_csv(data, index=False)
            jobs = []
            for i in range(128):
                design = dict(base)
                design[study.FEATURES[0]] += (i+1)/1000
                sample = f"new_{i}"
                jobs.append(dict(sample=sample, role="train", design=design))
                case = root / "solver" / sample
                case.mkdir(parents=True)
                row = dict(sample=sample, role="train", **design, wake_quality_pass=i > 0,
                           **dict(zip(study.TARGETS, [6., .8 if i else None])))
                if not i:
                    row["solver_failure"] = "deliberate fixture failure"
                (case / "record.json").write_text(json.dumps(row), encoding="utf8")
            (root / "冻结配置.json").write_text(json.dumps(dict(source_data=str(data), jobs=jobs)),
                                               encoding="utf8")
            out = root / "snapshot"
            with patch.object(study, "BATCH", root):
                study.prepare_training(out)
                snapshot = pd.read_csv(out / "训练标签冻结.csv")
                self.assertEqual(len(snapshot), 129)
                self.assertEqual(snapshot.solver_failure.notna().sum(), 1)
                study.prepare_training(out)
                path = root / "solver/new_0/record.json"
                row = json.loads(path.read_text(encoding="utf8"))
                row[study.TARGETS[1]] = .9
                path.write_text(json.dumps(row), encoding="utf8")
                with self.assertRaises(RuntimeError):
                    study.prepare_training(out)


if __name__ == "__main__":
    unittest.main()
