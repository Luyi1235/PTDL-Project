from pathlib import Path
import shutil
import json

workspace = Path(r"D:\LDTP")
project = workspace / "LDTP"
source_code = project / "PTDL飞推控协同优化代码包"
desktop = Path.home() / "Desktop"
dest = workspace / "PTDL_GitHub备份待上传"
dest.mkdir(exist_ok=True)
copied = []


def add(src, relative):
    if not src.is_file() or src.name.startswith("~$"):
        return
    dst = dest / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"source": str(src), "backup": str(relative), "bytes": src.stat().st_size})


for src in source_code.rglob("*"):
    rel = src.relative_to(source_code)
    if any(part in {"运行结果", "__pycache__", ".git", ".venv", "node_modules"} for part in rel.parts):
        continue
    if src.is_file() and src.suffix.lower() not in {".pyc", ".pyo", ".exe", ".dll"}:
        add(src, Path("项目代码") / rel)

for src in project.iterdir():
    if src.is_file() and src.suffix.lower() in {".py", ".md", ".json", ".docx"}:
        add(src, Path("研究代码与历史报告") / src.name)

for src in workspace.iterdir():
    if src.is_file() and src.suffix.lower() in {".py", ".docx"}:
        add(src, Path("报告与制图脚本") / src.name)

for src in desktop.iterdir():
    if not src.is_file():
        continue
    name = src.name.lower()
    include = (
        name.startswith(("ptdl", "串联混合动力无人机总体设计报告", "多目标优化成果应用方向"))
        or name in {
            "【v3】附件3-技术报告-混合动力无人机飞推控耦合机理研究.docx",
            "mamod-【v3】附件3-技术进展报告总.docx",
            "【v2.0】附件3-技术报告-全任务剖面下最优控制策略研究_1107最最最新修改版.docx",
            "全任务控制策略与多目标指标_ppt大纲.docx",
            "任务剖面与供电变化_图例调整版.png",
            "总体设计报告.pdf",
            "72h混动无人机总体设计表.xlsx",
        }
    )
    if include and src.suffix.lower() in {".docx", ".pdf", ".pptx", ".xlsx", ".png", ".zip"}:
        add(src, Path("桌面项目资料") / src.name)

figures = project / "报告精简配图_20261014" / "高清图"
if figures.exists():
    for src in figures.iterdir():
        if src.is_file() and src.suffix.lower() in {".png", ".pdf"}:
            add(src, Path("报告图表") / src.name)

manifest = {
    "说明": "本目录是精选备份，不是完整的 D:\\LDTP 备份。未包含大量仿真原始结果、本地软件工具及部分历史文件。",
    "original_simulation_results": str(source_code / "运行结果"),
    "items": copied,
}
(dest / "备份文件清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("备份目录:", dest)
print("文件数:", len(copied))
print("大小 MB:", round(sum(x["bytes"] for x in copied) / 1024**2, 1))
