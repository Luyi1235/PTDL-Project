import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 推进_尾迹分辨率与扩样20261016 as experiment
from VSPAERO气动代理模型 import FEATURE_LABELS


class WakeSamplingTests(unittest.TestCase):
    def test_fine_grid_training_keeps_baseline_and_protocol_metadata(self):
        frame = pd.DataFrame([dict(sample="fine64_train_000")])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)

            def fake_fit(internal, output):
                self.assertEqual(internal.iloc[0]["sample"], "medium_train_000")
                result = {}
                for name in ("high_candidate", "direct_high"):
                    path = output / (name+".json")
                    path.write_text(json.dumps(dict(mesh_reference="medium")), encoding="utf8")
                    result[name] = path
                return result

            def fake_native(*args):
                return {}

            with patch.object(experiment.driver, "fit_models", fake_fit), patch.object(
                    experiment.driver, "_run_case", fake_native):
                experiment.install_expansion_hooks()
                paths = experiment.driver.fit_models(frame, folder)
                for path in paths.values():
                    payload = json.loads(path.read_text(encoding="utf8"))
                    self.assertEqual(payload["mesh_reference"], "fine")
                    self.assertEqual(payload["wake_nodes"], 64)
                self.assertEqual(frame.iloc[0]["sample"], "fine64_train_000")
                (folder / "airframe.vspaero").write_text("NumWakeNodes = 32\n", encoding="utf8")
                with self.assertRaisesRegex(RuntimeError, "64 wake"):
                    experiment.driver._run_case(None, None, None, folder)
                (folder / "airframe.vspaero").write_text("NumWakeNodes = 64\n", encoding="utf8")
                result = experiment.driver._run_case(None, None, None, folder)
                self.assertEqual(result["applied_wake_nodes"], 64)

    def test_lift_discrepancy_is_diagnostic_only(self):
        path = (experiment.FIRST / "solver" / "medium_train_034" / "case" /
                "vspaero_polar.csv")
        # This record is reused from the first stage, so its raw polar remains there.
        if not path.exists():
            path = (ROOT / "运行结果" / "多精度气动代理_20261016" / "solver" /
                    "medium_train_034" / "case" / "vspaero_polar.csv")
        result = experiment.diagnostic_quality(path)
        self.assertFalse(result["force_consistency_pass"])
        self.assertTrue(result["quality_pass"])
        self.assertFalse(result["force_discrepancy_is_acceptance_rule"])

    def test_expansion_gate_requires_both_targets_and_all_training_geometries(self):
        rows = []
        for sample in ("medium_train_000", "medium_train_033", "medium_train_066"):
            for mesh, nodes in (("fine", 32), ("fine", 64), ("verification", 64)):
                rows.append(dict(source_sample=sample, mesh=mesh, nodes=nodes, repeat=0,
                                 quality_pass=True, vspaero_cl_alpha_per_rad=6.,
                                 vspaero_oswald_efficiency=1.05))
        self.assertTrue(experiment.convergence(rows)["passed"])
        rows[-1]["vspaero_oswald_efficiency"] = .8
        self.assertFalse(experiment.convergence(rows)["passed"])
        rows[-1]["vspaero_oswald_efficiency"] = 1.05
        rows[1]["quality_pass"] = False
        self.assertFalse(experiment.convergence(rows)["passed"])

    def test_new_protocol_recomputes_training_and_excludes_old_tests(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "运行结果") as directory:
            folder = Path(directory)
            (folder / "冻结探针计划.json").write_text("{}", encoding="utf8")
            (folder / "收敛检查.json").write_text("{}", encoding="utf8")
            plan = experiment.expansion_plan(folder, 102, 8, {})
            training = [j for j in plan["jobs"] if j["role"] == "train"]
            test = [j for j in plan["jobs"] if j["role"] == "test"]
            self.assertEqual(len(training), 102)
            self.assertEqual(len(test), 8)
            self.assertTrue(all(j["mesh"] == "fine" and "reuse_record" not in j for j in training))
            self.assertFalse(plan["old_labels_reused_as_high_fidelity"])
            prior = json.loads((experiment.FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
            old = np.array([[j["design"][k] for k in FEATURE_LABELS]
                            for j in prior["jobs"] if j["role"] == "test"])
            a = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in training])
            b = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in test])
            self.assertGreater(np.linalg.norm(a[:, None]-old[None, :], axis=2).min(), 1e-5)
            self.assertGreater(np.linalg.norm(b[:, None]-old[None, :], axis=2).min(), 1e-5)
            self.assertGreater(np.linalg.norm(a[:, None]-b[None, :], axis=2).min(), 1e-5)


if __name__ == "__main__":
    unittest.main()
