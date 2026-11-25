"""运行整理版代码包的核心闭环与智能调度回归测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "核心模型"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

TEST_FILES = (
    ROOT / "验证测试" / "测试_飞推控闭环物理模型.py",
    ROOT / "验证测试" / "测试_智能能量管理与容错调度.py",
    ROOT / "验证测试" / "测试_MOEAD多目标联合优化.py",
    ROOT / "验证测试" / "测试_Hybrid_System动力接口.py",
    ROOT / "验证测试" / "测试_扩展验证.py",
    ROOT / "验证测试" / "测试_高保真支撑接口.py",
    ROOT / "验证测试" / "测试_方法体系补齐.py",
    ROOT / "验证测试" / "测试_电池可用容量闭环.py",
    ROOT / "验证测试" / "测试_协同自适应方法.py",
    ROOT / "验证测试" / "测试_20261015算法与响应.py",
)


def load_test_module(path: Path, index: int):
    specification = importlib.util.spec_from_file_location(f"ptdl_test_{index}", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"无法加载测试文件：{path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    suites = [unittest.defaultTestLoader.loadTestsFromModule(load_test_module(path, index)) for index, path in enumerate(TEST_FILES)]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
