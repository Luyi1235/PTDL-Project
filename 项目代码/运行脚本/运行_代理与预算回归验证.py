"""Run the Unicode-named test modules and save a reproducible test summary."""

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    folder=ROOT/"验证测试"
    sys.path.insert(0,str(folder))
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(f.stem)
                             for f in sorted(folder.glob("测试_*.py")))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","scikit-learn","jsbsim")}
    payload=dict(tests_run=result.testsRun,passed=result.wasSuccessful(),
                 failures=[dict(test=str(test),detail=detail) for test,detail in result.failures],
                 errors=[dict(test=str(test),detail=detail) for test,detail in result.errors],
                 skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],
                 python=sys.version,dependencies=versions)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    sys.exit(not result.wasSuccessful())


if __name__=="__main__":
    main()
