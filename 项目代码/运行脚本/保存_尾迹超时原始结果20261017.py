"""Preserve the completed native polar left by a timed-out Python worker."""

import json
from pathlib import Path
import shutil

import 续跑_稳定标签与均衡采样20261016 as prior
from OpenVSP_VSPAERO接口 import load_vspaero_polar
from 机体参数化设计与气动阻力 import AirframeDesign


def main():
    source = Path(r"C:\Users\pc\AppData\Local\Temp\ptdl_vspaero_6qqjr_kb")
    out = prior.ROOT / "运行结果" / "自由尾迹超时恢复_20261017"
    out.mkdir(exist_ok=True)
    target = out / "原始求解文件"
    if not target.exists():
        shutil.copytree(source, target)
    plan = json.loads((prior.FIRST / "冻结采样与计算配置.json").read_text(encoding="utf8"))
    design = next(j["design"] for j in plan["jobs"] if j["sample"] == "medium_train_000")
    polar = load_vspaero_polar(target / "airframe.polar", AirframeDesign(**design),
                              target / "airframe.vsp3", out / "恢复极曲线.csv")
    if len(polar.data) != 5:
        raise RuntimeError("Recovered polar is incomplete")
    record = dict(sample="medium_train_000", mesh="verification", wake_nodes=256,
                  cl_alpha=polar.cl_alpha_per_rad, ew=polar.oswald_efficiency,
                  points=len(polar.data), original_worker_timed_out=True,
                  native_solver_completed=True, native_stdout_unavailable=True,
                  use="Resolution diagnostic only; excluded from training",
                  retired_controller_pid=25784,
                  retirement_reason="Neither 256-node grid can pass the existing baseline wake consistency gate",
                  original_source=str(source),
                  recovered_sources={p.name: prior.driver.digest(p) for p in target.iterdir() if p.is_file()})
    (out / "恢复与任务替代说明.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(dict(cl_alpha=polar.cl_alpha_per_rad, ew=polar.oswald_efficiency,
                          points=len(polar.data))), flush=True)


if __name__ == "__main__":
    main()
