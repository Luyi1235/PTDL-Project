import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "运行脚本"), str(ROOT / "核心模型")]
import 研究_分组多尺度Kriging20261016 as grouped
import 续跑_百构型加密代理20261016 as expansion
from 检查_VSPAERO残差收敛20261016 import audit
from VSPAERO气动代理模型 import FEATURE_LABELS
from 检查_极曲线力一致性20261016 import checked_row, polar_quality


class GroupedProxyTests(unittest.TestCase):
    def test_force_check_rejects_inconsistent_lift_without_modifying_raw_polar(self):
        source = ROOT / "运行结果" / "多精度气动代理_20261016" / "solver" / "medium_train_029"
        polar = source / "case" / "vspaero_polar.csv"
        before = polar.read_bytes()
        row = json.loads((source / "record.json").read_text(encoding="utf8"))
        result = checked_row(row)
        self.assertFalse(result["force_consistency_pass"])
        self.assertIsNone(result["vspaero_cl_alpha_per_rad"])
        self.assertIsNone(result["vspaero_oswald_efficiency"])
        self.assertGreater(result["raw_vspaero_oswald_efficiency"], 2.)
        self.assertEqual(polar.read_bytes(), before)

    def test_force_check_preserves_lift_when_wake_drag_is_invalid(self):
        source = ROOT / "运行结果" / "多精度气动代理_20261016" / "solver" / "medium_train_026"
        row = json.loads((source / "record.json").read_text(encoding="utf8"))
        result = checked_row(row)
        self.assertTrue(result["lift_quality_pass"])
        self.assertFalse(result["wake_quality_pass"])
        self.assertEqual(result["vspaero_cl_alpha_per_rad"], row["vspaero_cl_alpha_per_rad"])
        self.assertIsNone(result["vspaero_oswald_efficiency"])

    def test_force_check_has_no_arbitrary_efficiency_upper_clip(self):
        alpha = np.array([-3., -1., 1., 3., 5.])
        lift = np.array([-.4, -.2, .0, .2, .4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "polar.csv"
            pd.DataFrame(dict(alpha_deg=alpha, cl=lift, cl_wake=lift,
                              cd_induced_wake=np.full(5, .01),
                              e_wake=np.full(5, 1.05))).to_csv(path, index=False)
            self.assertTrue(polar_quality(path)["quality_pass"])

    def test_residual_association_keeps_repeated_sample_names_separate(self):
        history = (ROOT / "运行结果" / "多精度气动代理_20261016" / "solver" /
                   "medium_train_000" / "case" / "airframe.history").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for name in ("geometry_a", "geometry_b"):
                folder = root / name
                folder.mkdir()
                (folder / "airframe.history").write_bytes(history)
                rows.append(dict(sample="repeated_name", geometry_file=str(folder / "airframe.vsp3")))
            source = root / "data.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            result = audit(source, root / "audit")
            self.assertEqual(result["histories_available"], 2)
            self.assertEqual(result["points"], 10)

    def test_portable_covariance_and_optimization_gradient(self):
        rng = np.random.default_rng(43)
        a, b = rng.normal(size=(4, 9)), rng.normal(size=(3, 9))
        for name in ("full", "grouped", "grouped-interaction", "multiscale"):
            kernel = grouped.make_kernel(name)
            np.testing.assert_allclose(
                grouped.cross_covariance(grouped.encode_kernel(kernel), a, b),
                kernel(a, b), rtol=1e-12, atol=1e-12)
        kernel = grouped.make_kernel("grouped-interaction")
        _, gradient = kernel(a, eval_gradient=True)
        for i, step in enumerate(np.eye(len(kernel.theta))*1e-6):
            finite_difference = (kernel.clone_with_theta(kernel.theta+step)(a)
                                 -kernel.clone_with_theta(kernel.theta-step)(a))/(2e-6)
            np.testing.assert_allclose(gradient[:, :, i], finite_difference, atol=1e-7, rtol=1e-5)

    def test_expansion_reuses_training_and_excludes_all_earlier_test_geometries(self):
        # Temporary fixtures exercise planning only; no aerodynamic solve is performed.
        original = ROOT / "运行结果" / "多精度气动代理_20261016"
        plan = json.loads((original / "冻结采样与计算配置.json").read_text(encoding="utf8"))
        training = [j for j in plan["jobs"] if j["role"] == "train"][:3]
        testing = [j for j in plan["jobs"] if j["role"] == "test"]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory)
            (first / "冻结采样与计算配置.json").write_text(
                json.dumps(dict(jobs=training+testing)), encoding="utf8")
            frame = []
            for job in training:
                source = original / "solver" / job["sample"] / "record.json"
                row = json.loads(source.read_text(encoding="utf8"))
                folder = first / "solver" / job["sample"]
                folder.mkdir(parents=True)
                (folder / "record.json").write_text(json.dumps(row), encoding="utf8")
                frame.append(row)
            pd.DataFrame(frame).to_csv(first / "中网格训练实算标签.csv", index=False)
            model = ROOT / "运行结果" / "主动增量代理_20261015" / "增量构型气动代理.json"
            (first / "多精度候选代理.json").write_bytes(model.read_bytes())
            future = expansion.plan_expansion(first, total_train=5, test_count=4, fine_count=0)
            train = [j for j in future["jobs"] if j["role"] == "train"]
            test = [j for j in future["jobs"] if j["role"] == "test"]
            self.assertEqual(len(train), 5)
            self.assertEqual(len(test), 4)
            self.assertTrue(all("reuse_record" in j for j in train[:3]))
            self.assertFalse(future["old_test_labels_used_for_training"])
            old = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in testing])
            a = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in train])
            b = np.array([[j["design"][k] for k in FEATURE_LABELS] for j in test])
            self.assertGreater(np.linalg.norm(a[:, None]-old[None, :], axis=2).min(), 1e-5)
            self.assertGreater(np.linalg.norm(a[:, None]-b[None, :], axis=2).min(), 1e-5)


if __name__ == "__main__":
    unittest.main()
