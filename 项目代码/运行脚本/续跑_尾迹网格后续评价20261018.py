"""Complete diagnostic evaluation and summarize the separate fresh test cohort."""

from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

import 评价_尾迹网格192诊断20261018 as diagnostic

study = diagnostic.study
OUT = study.OUT
TEST = OUT / "新256构型独立验证"


def progress(stage, error=None):
    study.utility.save(OUT / "后续汇总进度.json", dict(
        stage=stage, error=error, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))


def wait_for(path, phase):
    started = time.monotonic()
    while not path.exists():
        progress(phase)
        if (OUT / "失败.json").exists():
            raise RuntimeError("Training failure recorded")
        if phase == "waiting_independent_test" and (TEST / "进度.json").exists():
            if study.batch.previous.read(TEST / "进度.json")["stage"] == "failed":
                raise RuntimeError("Independent evaluation failure recorded")
        if time.monotonic()-started > 12*3600:
            raise TimeoutError(phase)
        time.sleep(30)


def summarize():
    report = study.batch.previous.read(TEST / "独立评价.json")
    frozen = study.batch.previous.read(TEST / "预测冻结.json")
    study.utility.check_hashes(frozen["hashes"])
    frame = pd.read_csv(TEST / "独立逐点.csv")
    intervals = []
    for index, target in enumerate(study.batch.TARGETS):
        y = frame[target].to_numpy(float)
        valid = np.isfinite(y)
        if index:
            valid &= frame.wake_quality_pass.fillna(False).to_numpy(bool)
        for name in ("uniform_risk", "all_risk"):
            for reference in ("fine3072", "fine4096"):
                intervals.append(dict(model=name, reference=reference, target=target,
                    count=int(valid.sum()),
                    **study.batch.previous.evaluator.paired_bootstrap(
                        y[valid], frame[f"{name}_prediction_{index}"].to_numpy(float)[valid],
                        frame[f"{reference}_prediction_{index}"].to_numpy(float)[valid])))
    paths = [Path(__file__), TEST / "独立评价.json", TEST / "独立逐点.csv",
             TEST / "预测冻结.json"]
    study.utility.save(TEST / "配对误差比较.json", dict(
        comparisons=intervals, hashes={str(p): study.utility.digest(p) for p in paths},
        previously_inspected_192_excluded=True))
    names = dict(fine3072="现用3072构型模型", fine4096="4096构型精细模型",
                 local3840="3840构型局部树组合",
                 uniform_risk="均匀区域选型组合", all_risk="全部区域选型组合")
    lines = [
        "# 新256构型代理精度评价", "",
        f"评价完成时间为{time.strftime('%Y-%m-%d %H:%M:%S')}。全部预测在气动力求解前固定。",
        "升力和尾迹分别采用各自相同的有效样本，保留原始效率定义及质量规则。", "",
        "| 模型 | 预测量 | 有效构型数 | R² | MAE | RMSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["metrics"]:
        target = "升力斜率" if row["target"] == study.batch.TARGETS[0] else "尾迹效率"
        lines.append(
            f"| {names[row['model']]} | {target} | {row['count']} | "
            f"{row['r2']:.4f} | {row['mae']:.5f} | {row['rmse']:.5f} |")
    lines.extend(["", "| 新模型 | 对照 | 预测量 | R²差值的95%区间 |",
                  "| --- | --- | --- | --- |"])
    for row in intervals:
        target = "升力斜率" if row["target"] == study.batch.TARGETS[0] else "尾迹效率"
        lines.append(f"| {names[row['model']]} | {names[row['reference']]} | {target} | "
                     f"{row['lower95']:.4f} 至 {row['upper95']:.4f} |")
    lines.extend(["",
        f"几何预处理网格与实算网格不一致的记录数为{report['geometry_mismatch_count']}。",
        "192构型诊断复评另行保存，未合并到本组独立测试结果。", "",
        "新增模型作为研究候选保留，工作台默认代理的替换单独依据完整评价结果进行。",
    ])
    (TEST / "代理精度更新结论.md").write_text("\n".join(lines)+"\n", encoding="utf8")


if __name__ == "__main__":
    try:
        wait_for(OUT / "模型选择.json", "waiting_training")
        progress("diagnostic_192")
        diagnostic.main()
        wait_for(TEST / "独立评价.json", "waiting_independent_test")
        progress("summarizing")
        summarize()
        progress("completed")
    except Exception:
        progress("failed", traceback.format_exc())
        raise
