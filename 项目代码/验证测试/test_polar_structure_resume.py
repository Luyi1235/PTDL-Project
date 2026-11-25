"""JSON configuration recovery must retain mode order and reject real changes."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "运行脚本"))
import 完成_极曲线训练与复评20261019 as recovery


class RecoveryTests(unittest.TestCase):
    def test_mode_tuple_survives_json_roundtrip(self):
        modes = ("direct", "profile", "half")
        frozen = json.loads(json.dumps(dict(modes=modes)))["modes"]
        self.assertEqual(recovery.normalized_modes(modes, frozen), frozen)

    def test_reordered_or_missing_modes_are_rejected(self):
        for modes in (["half", "direct", "profile"], ["direct", "half"]):
            with self.assertRaises(RuntimeError):
                recovery.normalized_modes(("direct", "profile", "half"), modes)


if __name__ == "__main__":
    unittest.main()
