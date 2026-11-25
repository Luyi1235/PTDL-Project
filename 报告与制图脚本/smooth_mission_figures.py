from pathlib import Path

source = Path(r"D:\LDTP\revise_mission_plots.py").read_text(encoding="utf-8")
source = source.replace(
    't = data.time_s / 3600\n',
    '''t = data.time_s / 3600
cruise = data.phase.eq("endurance_loiter")
smoothed = data.loc[cruise, "engine_power_kw"].rolling(
    window=19, center=True, min_periods=1
).mean()
data.loc[cruise, "engine_power_kw"] = smoothed
''',
)
source = source.replace('label="发动机供电"', 'label="发动机供电（平滑）"')
source = source.replace('label="发动机—发电机供电"', 'label="发动机—发电机供电（平滑）"')
source = source.replace("长航时任务与能源协同_坐标优化版.png", "长航时任务与能源协同_巡航功率平滑效果.png")
source = source.replace("任务剖面与供电变化_图例调整版.png", "任务剖面与供电变化_巡航功率平滑效果.png")
exec(compile(source, "revise_mission_plots.py", "exec"))
