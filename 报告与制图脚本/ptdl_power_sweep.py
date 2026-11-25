from pathlib import Path
import os
import sys

original = Path(
    r"D:\LDTP\LDTP\PTDL飞推控协同优化代码包\运行脚本\运行_JSBSim地面起降联合闭环验证.py"
)
case = os.environ["PTDL_DIAGNOSTIC_CASE"]
period, kp, kd = {
    "period2": (2.0, 4.0, 4.0),
    "period1": (1.0, 4.0, 4.0),
    "period05": (0.5, 4.0, 4.0),
    "gain2": (2.0, 2.0, 4.0),
    "gain1": (2.0, 1.0, 4.0),
    "gain2_full": (2.0, 2.0, 4.0),
    "gain2_batt": (2.0, 2.0, 4.0),
    "gain2_batt_rate": (2.0, 2.0, 4.0),
}[case]
text = original.read_text(encoding="utf-8")
assert text.count("sample_period_s=600.0,") == 1
text = text.replace(
    "sample_period_s=600.0,",
    f"sample_period_s={period}, pitch_kp={kp}, pitch_kd={kd},",
)
if case in ("gain2_batt", "gain2_batt_rate"):
    needle = "motor_assist = dict(zip(PHASES, map(float, motor_values)))"
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '\n    motor_assist["endurance_loiter"] = 2.0')
if case == "gain2_batt_rate":
    needle = "dispatch_policy_factory=controller_factory,"
    assert text.count(needle) == 1
    text = text.replace(
        needle, needle + '\n        powertrain_config={"engine_power_rate_limit_kwps": 0.05},'
    )
sys.argv = [
    str(original), "--output-dir", rf"D:\LDTP\power_diagnosis\{case}",
    "--mission-hours", "80" if case == "gain2_full" else "4",
    "--control-period-s", str(period),
]
exec(compile(text, str(original), "exec"), {"__name__": "__main__", "__file__": str(original)})
