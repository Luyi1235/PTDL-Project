from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from OpenVSP_VSPAERO接口 import (  # noqa: E402
    OSWALD_EFFICIENCY_DEFINITION,
    configure_vspaero_geometry_sets,
    load_vspaero_polar,
)
from 机体参数化设计与气动阻力 import AirframeDesign  # noqa: E402


class FakeVSP:
    SET_FIRST_USER = 5

    def __init__(self) -> None:
        self.names = {
            "wing": "Main_Wing",
            "tail_h": "Horizontal_Tail",
            "tail_v": "Vertical_Tail",
            "body": "Fuselage",
        }
        self.flags: dict[tuple[str, int], bool] = {}
        self.set_names: dict[int, str] = {}

    def SetSetName(self, set_id: int, name: str) -> None:
        self.set_names[set_id] = name

    def FindGeoms(self) -> list[str]:
        return list(self.names)

    def GetGeomName(self, geometry_id: str) -> str:
        return self.names[geometry_id]

    def SetSetFlag(self, geometry_id: str, set_id: int, enabled: bool) -> None:
        self.flags[(geometry_id, set_id)] = enabled


class VSPAEROGeometryAndEwTests(unittest.TestCase):
    def test_geometry_sets_separate_lifting_surfaces_and_fuselage(self) -> None:
        vsp = FakeVSP()
        thick_set, thin_set = configure_vspaero_geometry_sets(vsp)

        self.assertEqual(thick_set, 6)
        self.assertEqual(thin_set, 5)
        self.assertEqual(
            {
                vsp.names[geometry_id]
                for geometry_id, set_id in vsp.flags
                if set_id == thin_set and vsp.flags[(geometry_id, set_id)]
            },
            {"Main_Wing", "Horizontal_Tail", "Vertical_Tail"},
        )
        self.assertEqual(
            {
                vsp.names[geometry_id]
                for geometry_id, set_id in vsp.flags
                if set_id == thick_set and vsp.flags[(geometry_id, set_id)]
            },
            {"Fuselage"},
        )

    def test_polar_uses_wake_ew_not_surface_e(self) -> None:
        polar_text = (
            " AoA CLtot CDi E CLiw CDiw Ew\n"
            " -3.0 -0.3 0.01 0.11 -0.30 0.01 0.60\n"
            " -1.0 -0.1 0.01 0.12 -0.10 0.01 0.99\n"
            "  1.0  0.1 0.01 0.13  0.10 0.01 0.98\n"
            "  3.0  0.3 0.01 0.14  0.30 0.01 0.80\n"
            "  5.0  0.5 0.01 0.15  0.50 0.01 0.70\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            polar_path = Path(temp_dir) / "test.polar"
            output_path = Path(temp_dir) / "polar.csv"
            polar_path.write_text(polar_text, encoding="utf-8")
            result = load_vspaero_polar(
                polar_path, AirframeDesign(), Path(temp_dir) / "airframe.vsp3", output_path
            )
            saved = pd.read_csv(output_path)

        self.assertAlmostEqual(result.oswald_efficiency, 0.70)
        self.assertEqual(OSWALD_EFFICIENCY_DEFINITION,
                         "median VSPAERO Ew for polar points with abs(CLiw) >= 0.15")
        self.assertIn("e_surface", saved.columns)
        self.assertIn("e_wake", saved.columns)
        self.assertNotAlmostEqual(result.oswald_efficiency, float(saved["e_surface"].median()))


if __name__ == "__main__":
    unittest.main()
