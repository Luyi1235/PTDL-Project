"""Holdout-preserving batching, cache resume and late-registration guards."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 续跑_稳定网格独立验证20261019 as resume


class HoldoutResumeTests(unittest.TestCase):
    def test_batches_preserve_all_jobs_and_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "registered"
            out.mkdir()
            cache = out / "geometry" / "test_00" / "geometry.json"
            cache.parent.mkdir(parents=True)
            cache.write_text("existing", encoding="utf-8")
            jobs = [dict(sample=f"test_{i:02d}", design=dict(index=i)) for i in range(19)]
            runner = resume.evaluation.expansion.native.original.controlled
            with (patch.object(resume.evaluation.expansion, "OUT", root / "expanded"),
                  patch.object(runner, "run_process") as run):
                resume.prepare_batched(out, dict(jobs=jobs))
            requests = [call.args[1] for call in run.call_args_list]
            self.assertEqual([job for request in requests for job in request["jobs"]], jobs)
            self.assertEqual([len(r["jobs"]) for r in requests], [8, 8, 3])
            self.assertTrue(all(call.args[3] == 1800 for call in run.call_args_list))
            self.assertTrue(all(r["output_root"] == str(out / "geometry") for r in requests))
            self.assertEqual(cache.read_text(encoding="utf-8"), "existing")

    def test_late_force_labels_rejected_before_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "expanded/solver/test_0").mkdir(parents=True)
            runner = resume.evaluation.expansion.native.original.controlled
            with (patch.object(resume.evaluation.expansion, "OUT", root / "expanded"),
                  patch.object(runner, "run_process") as run):
                with self.assertRaisesRegex(RuntimeError, "before prediction"):
                    resume.prepare_batched(root / "registered", dict(jobs=[dict(sample="test_0")]))
                run.assert_not_called()

    def test_timeout_retries_the_same_input_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "registered"
            out.mkdir()
            jobs = [dict(sample="test_0")]
            runner = resume.evaluation.expansion.native.original.controlled
            with (patch.object(resume.evaluation.expansion, "OUT", root / "expanded"),
                  patch.object(runner, "run_process",
                               side_effect=[subprocess.TimeoutExpired("worker", 1800), "ok"]) as run):
                resume.prepare_batched(out, dict(jobs=jobs))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[1], run.call_args_list[1].args[1])

    def test_geometry_errors_are_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "registered"
            out.mkdir()
            runner = resume.evaluation.expansion.native.original.controlled
            with (patch.object(resume.evaluation.expansion, "OUT", root / "expanded"),
                  patch.object(runner, "run_process", side_effect=RuntimeError("Mesh changed")) as run):
                with self.assertRaisesRegex(RuntimeError, "Mesh changed"):
                    resume.prepare_batched(out, dict(jobs=[dict(sample="test_0")]))
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
