import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 推进_千构型原定义代理20261017 as experiment


class OriginalDefinitionTests(unittest.TestCase):
    def test_fresh_training_and_test_are_disjoint_and_keep_existing_geometry(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "运行结果") as directory, patch.object(
                experiment.prior.driver, "LOW_DATA", experiment.prior.LOW_DATA):
            result = experiment.plan(Path(directory), 1000, 96)
        self.assertEqual(result["old_geometry_count"], 289)
        self.assertEqual(result["new_interior_geometry_count"], 711)
        self.assertFalse(result["old_labels_reused"])
        jobs = result["jobs"]
        self.assertEqual(len(jobs), 1096)
        self.assertEqual(sum(j["role"] == "test" for j in jobs), 96)
        self.assertTrue(all(j["mesh"] == "legacy" and "reuse_record" not in j for j in jobs))
        x = np.array([[j["design"][k] for k in experiment.prior.FEATURE_LABELS] for j in jobs])
        old = experiment.pd.read_csv(experiment.prior.LOW_DATA)[
            list(experiment.prior.FEATURE_LABELS)].to_numpy(float)
        np.testing.assert_array_equal(x[:289], old)
        self.assertEqual(len(np.unique(x, axis=0)), len(x))
        previous = experiment.pd.read_csv(experiment.prior.driver.LOW_DIR / "合并留出验证.csv")[
            list(experiment.prior.FEATURE_LABELS)].to_numpy(float)
        self.assertGreater(np.linalg.norm(x[289:, None]-previous[None], axis=2).min(), 1e-5)
        for j in jobs[289:]:
            values = experiment.prior.geometry_constraints(experiment.prior.AirframeDesign(**j["design"]))
            self.assertTrue(all(v <= 1e-9 for v in values.values()))

    def test_fixed_wake_or_changed_resolution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            applied = dict(FixedWakeFlag=0, NumWakeNodes=8, NCPU=8, FarDistToggle=0, WakeNumIter=30)
            receipt = folder / "冻结求解设置.json"
            receipt.write_text(json.dumps(dict(wake_protocol="free-wake-legacy-v1",
                                               applied=applied)), encoding="utf8")
            (folder / "airframe.vspaero").write_text(
                "NumWakeNodes = 8\nWakeIters = 30\nFarDist = -1\n", encoding="utf8")
            (folder / "worker.log").write_text("vspaero.exe -omp 8 model\n", encoding="utf8")
            self.assertEqual(experiment.verify(folder)["applied_wake_nodes"], 8)
            applied["FixedWakeFlag"] = 1
            receipt.write_text(json.dumps(dict(wake_protocol="free-wake-legacy-v1",
                                               applied=applied)), encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
                experiment.verify(folder)


if __name__ == "__main__":
    unittest.main()
