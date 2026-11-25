from pathlib import Path
import sys

original = Path(
    r"D:\LDTP\LDTP\PTDL飞推控协同优化代码包\运行脚本\运行_JSBSim地面起降联合闭环验证.py"
)
text = original.read_text(encoding="utf-8")
needle = "sample_period_s=600.0,"
assert text.count(needle) == 1
text = text.replace(needle, "sample_period_s=2.0,")
sys.argv = [
    str(original),
    "--output-dir",
    r"D:\LDTP\power_diagnosis\baseline_2s",
]
namespace = {"__name__": "__main__", "__file__": str(original)}
exec(compile(text, str(original), "exec"), namespace)
