from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"运行脚本"),str(ROOT/"核心模型")]
from 分析_六自由度跨预算收敛 import tail_assessment
from 检查_尾迹标签质量 import polar_quality


class BudgetAndPolarTests(unittest.TestCase):
    def test_zero_archive_is_not_converged(self):
        self.assertFalse(tail_assessment([0,0,0])["tail_plateau"])

    def test_significant_decline_is_not_plateau(self):
        self.assertFalse(tail_assessment([1,.9,.8])["tail_plateau"])

    def test_plateau_and_gain(self):
        self.assertTrue(tail_assessment([.2,.4,.4001,.4002])["tail_plateau"])
        self.assertFalse(tail_assessment([.2,.3,.4])["tail_plateau"])

    def test_single_positive_wake_point_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"polar.csv"
            f=pd.DataFrame(dict(alpha_deg=[-3,-1,1,3,5],cl_wake=[-.4,-.2,0,.2,.4],
                                cd_induced_wake=[-.002,-.001,-.001,-.001,.002],
                                e_wake=[-1,-1,0,-1,.8]))
            f.to_csv(path,index=False)
            result=polar_quality(path)
            self.assertFalse(result["quality_pass"])
            self.assertEqual(result["valid_points"],1)
            f["cd_induced_wake"]=[.002,.001,.0001,.001,.002]
            f["e_wake"]=[.8,.8,.001,.8,.8]
            f.to_csv(path,index=False)
            self.assertTrue(polar_quality(path)["quality_pass"])


if __name__=="__main__":
    unittest.main()
