import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 研究_联合迎角代理20261017 as joint
import VSPAERO受控子进程20261017 as controlled
import 推进_固定尾迹与均衡代理20261017 as fixed


class JointAngleTests(unittest.TestCase):
    def test_portable_gp_matches_library_on_unseen_inputs(self):
        rng = np.random.default_rng(17)
        train, test = rng.normal(size=(24, 3)), rng.normal(size=(8, 3))
        for outputs in (1, 2):
            y = np.column_stack((np.sin(train[:, 0]), train[:, 1]**2))[:, :outputs]
            captured, factory = [], joint.GaussianProcessRegressor

            def create(*args, **kwargs):
                estimator = factory(*args, **kwargs)
                captured.append(estimator)
                return estimator

            with patch.object(joint, "GaussianProcessRegressor", create):
                model = joint.fit_gp(train, y[:, 0] if outputs == 1 else y, .1)
            expected = captured[0].predict((test-model["center"])/model["scale"]).reshape(len(test), -1)
            np.testing.assert_allclose(joint.gp_predict(model, test), expected, rtol=1e-9, atol=1e-9)

    def test_angle_reconstruction_preserves_efficiency_above_one(self):
        x = np.array([[26., 24., .45, 1., 0., 8.3, .75]])
        model = dict(config=dict(method="joint-angle", mode="raw", transform="log"),
                     direct="direct", lift="lift", wake="wake")

        def fake(gp, inputs):
            if gp == "direct":
                return np.array([[6., .4]])
            if gp == "lift":
                return np.column_stack((6*np.deg2rad(joint.ANGLES), np.ones(5)*.3))
            return np.ones((5, 1))*np.log(1.2)

        with patch.object(joint, "gp_predict", fake):
            result = joint.predict(model, x)
        np.testing.assert_allclose(result, [[6., 1.2]], atol=1e-12)

    def test_missing_predicted_working_angles_uses_direct_model(self):
        x = np.array([[26., 24., .45, 1., 0., 8.3, .75]])
        model = dict(config=dict(method="joint-angle", mode="raw", transform="identity"),
                     direct="direct", lift="lift", wake="wake")

        def fake(gp, inputs):
            if gp == "direct":
                return np.array([[6., .4]])
            if gp == "lift":
                return np.column_stack((6*np.deg2rad(joint.ANGLES), np.zeros(5)))
            return np.ones((5, 1))*.8

        with patch.object(joint, "gp_predict", fake):
            result = joint.predict(model, x)
        np.testing.assert_allclose(result, [[6., .4]], atol=1e-12)

    def test_training_polars_reproduce_existing_target(self):
        frame, polar, sources = joint.read_training(joint.DATA / "中网格训练实算标签.csv")
        self.assertEqual(polar.shape, (len(frame), 5, 3))
        self.assertEqual(len(sources), len(frame))
        self.assertGreater(len(frame), 90)
        groups = joint.geometry_groups(frame[list(joint.FEATURE_LABELS)].to_numpy(float))
        for a, b in joint.selection_splits(frame, groups)[3:]:
            self.assertFalse(set(groups[a]) & set(groups[b]))


class ControlledSolverTests(unittest.TestCase):
    def test_success_and_failure_logs_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            text = controlled.run_process(
                [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                {"value": 17}, folder, 10)
            self.assertEqual(json.loads(text), {"value": 17})
            with self.assertRaisesRegex(RuntimeError, "exited 3"):
                controlled.run_process([sys.executable, "-c",
                    "import sys; print('diagnostic',file=sys.stderr); sys.exit(3)"], {}, folder, 10)
            self.assertIn("diagnostic", (folder / "worker.log").read_text(encoding="utf8"))

    def test_timeout_stops_descendant(self):
        program = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(p.pid,flush=True); time.sleep(60)"
        )
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                controlled.run_process([sys.executable, "-c", program], {}, folder, .7)
            child = int((folder / "worker.stdout.log").read_text(encoding="utf8").strip())
            if sys.platform == "win32":
                import ctypes
                api = ctypes.WinDLL("kernel32", use_last_error=True)
                api.OpenProcess.restype = ctypes.c_void_p
                handle = api.OpenProcess(0x100000, False, child)
                if handle:
                    api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                    self.assertEqual(api.WaitForSingleObject(handle, 0), 0)
                    api.CloseHandle.argtypes = [ctypes.c_void_p]
                    api.CloseHandle(handle)

    def test_fixed_protocol_requires_zero_wake_iterations_and_matching_far_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            receipt = dict(wake_nodes=64, mesh_level="medium", wake_protocol="fixed-wake-v1",
                           far_distance_spans=20., applied=dict(FixedWakeFlag=1, FarDistToggle=1, FarDist=500.))
            (folder / "自适应求解设置.json").write_text(json.dumps(receipt), encoding="utf8")
            (folder / "worker.log").write_text(
                "vspaero.exe -omp 8 model\nNumberOfWakeTrailingNodes_: 64\n", encoding="utf8")
            settings = folder / "airframe.vspaero"
            settings.write_text("NumWakeNodes = 64\nWakeIters = 0\nFarDist = 500\n", encoding="utf8")
            self.assertEqual(fixed.fixed_receipt(folder, "medium", 64)["wake_protocol"], "fixed-wake-v1")
            settings.write_text("NumWakeNodes = 64\nWakeIters = 30\nFarDist = 500\n", encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
                fixed.fixed_receipt(folder, "medium", 64)


if __name__ == "__main__":
    unittest.main()
