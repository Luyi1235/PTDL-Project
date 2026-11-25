import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 续跑_稳定标签与均衡采样20261016 as continuation
import 运行_VSPAERO自适应分辨率20261016 as worker
from 分目标气动高斯过程 import target_mean
from VSPAERO气动代理模型 import FEATURE_LABELS

regularized = continuation.strong.regularized


class RegularizedSurrogateTests(unittest.TestCase):
    def setUp(self):
        frame = continuation.pd.read_csv(continuation.LOW_DATA)
        self.x = frame[list(FEATURE_LABELS)].to_numpy(float)[:40]
        self.y = 5.8 + .2 * np.sin(self.x[:, 0]) + .01 * self.x[:, 1]

    def test_portable_models_match_sklearn_on_unseen_geometries(self):
        a, b = self.x[:30], self.x[30:]
        for config in (
            dict(method="gp", mode="raw", ard=False, penalty=.1),
            dict(method="ridge", mode="dimensionless", degree=2, penalty=10.),
            dict(method="boost", mode="physics", penalty=5.),
        ):
            with self.subTest(method=config["method"]):
                fitted = regularized.fit(config, a, self.y[:30])
                payload = regularized.serialize(fitted, a, self.y[:30], .1)
                portable = np.array([target_mean(payload, row[None]) for row in b])
                np.testing.assert_allclose(portable, regularized.predict(fitted, b),
                                           rtol=1e-10, atol=1e-10)

    def test_selection_prefers_simplicity_only_within_error_band(self):
        configs = [
            dict(method="gp", mode="raw", ard=True, penalty=.01),
            dict(method="ridge", mode="dimensionless", degree=1, penalty=10.),
            dict(method="ridge", mode="dimensionless", degree=1, penalty=100.),
        ]
        scores = [
            dict(config=0, mean_rmse=.10, rmse_se=.02),
            dict(config=1, mean_rmse=.11, rmse_se=.01),
            dict(config=2, mean_rmse=.13, rmse_se=.01),
        ]
        with patch.object(regularized, "CONFIGS", configs):
            selected, threshold = regularized.one_se_choice(scores)
        self.assertEqual(selected["config"], 1)
        self.assertAlmostEqual(threshold, .12)

    def test_spatial_folds_are_disjoint_and_scaling_uses_training_only(self):
        groups = regularized.geometry_groups(self.x)
        splits = regularized.selection_splits(self.x, groups)
        for a, b in splits:
            self.assertFalse(set(a) & set(b))
        for a, b in splits[3:]:
            self.assertFalse(set(groups[a]) & set(groups[b]))
        a, b = splits[0]
        fitted = regularized.fit(dict(method="gp", mode="raw", ard=False, penalty=.1),
                                self.x[a], self.y[a])
        expected = regularized.feature_map(self.x[a], "raw").mean(0)
        np.testing.assert_allclose(fitted["center"], expected)
        shifted = self.x[b].copy()
        shifted[:, 0] += 100.
        regularized.predict(fitted, shifted)
        np.testing.assert_allclose(fitted["center"], expected)


class StableSamplingTests(unittest.TestCase):
    def test_balanced_plan_recomputes_anchors_and_seals_new_test(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "运行结果") as directory:
            output = Path(directory)
            (output / "离散精度选择.json").write_text("{}", encoding="utf8")
            with patch.object(continuation.driver, "LOW_DATA", continuation.LOW_DATA):
                plan = continuation.balanced_plan(output, 300, 48, "fine", 128)
        jobs = plan["jobs"]
        training = [j for j in jobs if j["role"] == "train"]
        testing = [j for j in jobs if j["role"] == "test"]
        counts = {basis: sum(j["basis"] == basis for j in training)
                  for basis in {j["basis"] for j in training}}
        self.assertEqual(counts, dict(same_training_geometry_new_protocol=100,
                                     uniform_feasible_training=150,
                                     global_maximin_training=50))
        self.assertEqual(len(testing), 48)
        self.assertTrue(all(j["mesh"] == "fine" and "reuse_record" not in j for j in jobs))
        self.assertFalse(plan["old_high_labels_reused"])
        x = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in jobs])
        self.assertEqual(len(np.unique(x, axis=0)), len(jobs))
        for j in jobs:
            constraints = continuation.geometry_constraints(continuation.AirframeDesign(**j["design"]))
            self.assertTrue(all(v <= 1e-9 for v in constraints.values()))
        old_tests = []
        for folder in (continuation.FIRST, ROOT / "运行结果" / "多精度气动代理_20261016"):
            prior = json.loads((folder / "冻结采样与计算配置.json").read_text(encoding="utf8"))
            old_tests.extend([[j["design"][k] for k in FEATURE_LABELS]
                              for j in prior["jobs"] if j["role"] == "test"])
        fresh = x[100:]
        self.assertGreater(np.linalg.norm(fresh[:, None]-np.array(old_tests)[None], axis=2).min(),
                           1e-5)

    def test_solver_timeout_and_receipt_are_verified(self):
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            receipt = output / "自适应求解设置.json"
            receipt.write_text(json.dumps(dict(wake_nodes=128, mesh_level="fine")), encoding="utf8")
            (output / "airframe.vspaero").write_text("NumWakeNodes = 128\n", encoding="utf8")
            (output / "worker.log").write_text(
                "vspaero.exe -omp 8 model\nNumberOfWakeTrailingNodes_: 128\n", encoding="utf8")

            def fake_native(*args, **kwargs):
                captured["timeout"] = args[4]
                return {}

            with patch.object(continuation.driver, "_run_case", fake_native), patch.object(
                    continuation.driver, "fit_models", lambda *_: None):
                continuation.install_training("fine", 128)
                row = continuation.driver._run_case(None, {}, "sample", output, 1800)
                self.assertEqual(captured["timeout"], 7200)
                self.assertEqual(row["applied_wake_nodes"], 128)
                receipt.write_text(json.dumps(dict(wake_nodes=64, mesh_level="fine")), encoding="utf8")
                with self.assertRaisesRegex(RuntimeError, "protocol changed"):
                    continuation.driver._run_case(None, {}, "sample", output, 1800)
                receipt.write_text(json.dumps(dict(wake_nodes=128, mesh_level="fine")), encoding="utf8")
                (output / "worker.log").write_text(
                    "vspaero.exe -omp 1 model\nNumberOfWakeTrailingNodes_: 128\n", encoding="utf8")
                with self.assertRaisesRegex(RuntimeError, "protocol changed"):
                    continuation.driver._run_case(None, {}, "sample", output, 1800)

    def test_grid_gate_checks_both_quantities(self):
        a = dict(quality_pass=True, **dict(zip(continuation.driver.TARGETS, (6., .5))))
        b = dict(a)
        self.assertTrue(continuation.compare(a, b)["passed"])
        b[continuation.driver.TARGETS[1]] = .6
        self.assertFalse(continuation.compare(a, b)["passed"])
        b = dict(a, quality_pass=False)
        self.assertFalse(continuation.compare(a, b)["passed"])

    def test_worker_applies_native_controls_and_restores_bindings(self):
        calls = []
        api = types.ModuleType("openvsp")
        original_analysis = lambda name: "result"
        original_mesh = lambda *args: None
        api.ExecAnalysis = original_analysis
        api.SetIntAnalysisInput = lambda *args: calls.append(args)
        adapter = types.ModuleType("OpenVSP_VSPAERO接口")
        adapter.OSWALD_EFFICIENCY_DEFINITION = "test original Ew"
        geometry = types.ModuleType("机体参数化设计与气动阻力")
        geometry.configure_geometry_mesh = original_mesh
        geometry.AirframeDesign = lambda **kwargs: kwargs
        polar = types.SimpleNamespace(cl_alpha_per_rad=6., oswald_efficiency=.5)
        with tempfile.TemporaryDirectory() as directory:
            def run(*args, **kwargs):
                api.ExecAnalysis("VSPAEROSweep")
                return polar
            adapter.run_vspaero_polar = run
            request = dict(design={}, output_dir=directory, speed_mps=36.,
                           altitude_m=3000., wake_iterations=30, mesh_level="fine")
            with patch.dict(sys.modules, {"openvsp": api, "OpenVSP_VSPAERO接口": adapter,
                                         "机体参数化设计与气动阻力": geometry}), patch.dict(
                    worker.os.environ, {"PTDL_ADAPTIVE_WAKE_NODES": "128",
                                        "PTDL_ADAPTIVE_NATIVE_CORES": "8"}), patch.object(
                    worker, "native_solver_lock", contextlib.nullcontext), patch.object(
                    sys, "stdin", io.StringIO(json.dumps(request))), contextlib.redirect_stdout(io.StringIO()):
                worker.main()
            receipt = json.loads((Path(directory) / "自适应求解设置.json").read_text(encoding="utf8"))
        self.assertIn(("VSPAEROSweep", "NumWakeNodes", [128]), calls)
        self.assertIn(("VSPAEROSweep", "NCPU", [8]), calls)
        self.assertIs(api.ExecAnalysis, original_analysis)
        self.assertIs(geometry.configure_geometry_mesh, original_mesh)
        self.assertEqual(receipt["wake_nodes"], 128)


if __name__ == "__main__":
    unittest.main()
