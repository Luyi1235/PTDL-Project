"""Local browser workbench for the PTDL integrated optimizer."""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import sys
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse
import uuid
import webbrowser


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "前端"
RESULTS = ROOT / "运行结果"
RUNNER = ROOT / "运行脚本" / "运行_工作台联合优化.py"
STEPWISE_RUNNER = ROOT / "运行脚本" / "运行_PTDL逐积分步VSPAERO_6DOF耦合.py"
DOE_3072 = RESULTS / "均匀覆盖3072代理_20261017" / "训练标签.csv"
DOE_6144 = RESULTS / "覆盖扩样6144代理_20261019" / "完整训练标签.csv"
DEFAULT_PRECISION = "fine6144"
ALGORITHMS = ("NSGA-II", "MOEA/D", "MOPSO", "Hybrid NSGA-II/MOPSO", "PTDL-耦合引导")
PHASE_MPC = ROOT / "运行结果" / "蓝色攻关_控制与轨迹_20261013" / "分阶段MPC配置.json"
MODEL_PATHS = {
    "fine6144": (RESULTS / "覆盖扩样6144代理_20261019" / "精细训练" / "精细气动候选代理.json", DOE_6144),
    "selectedmesh6144": (RESULTS / "扩样筛选网格代理_20261020" / "筛选网格候选.joblib", DOE_6144),
}
SEED = ROOT / "验证测试" / "预估额定值基准候选.json"
VSP_PYTHON = ROOT.parent.parent / "tools" / "Python311" / "python.exe"
LOCK = threading.Lock()
ACTIVE: dict[str, dict] = {}
ACTIVE_SIMULATIONS: dict[str, dict] = {}
CONFIG_CACHE_LOCK = threading.Lock()
RUNTIME_CACHE: dict = {}
BASELINES_CACHE: dict = {}
MODEL_HASH_CACHE: dict = {}


def powertrain_reference() -> dict:
    core = str(ROOT / "核心模型")
    if core not in sys.path:
        sys.path.insert(0, core)
    from ptdl_hybrid_system_adapter import HybridSystemPowertrain
    data = HybridSystemPowertrain().metadata
    return {
        **data,
        "engine_map_default": str(Path(data["engine_map_path"]).relative_to(ROOT)),
        "engine_map_name": Path(data["engine_map_path"]).name,
        "engine_design_speed_rpm": data["engine_speed_rpm"],
        "motor_rated_kw_per_branch": data["motor_rated_mechanical_kw_per_branch"],
        "motor_rated_speed_rpm": data["motor_speed_rpm"],
        "battery_nominal_energy_kwh": data["battery_nominal_energy_wh"] / 1000.0,
        "note": "动能学院 1005 版部件参数。任务总成沿用 PTDL 等效容量与额定倍率。",
    }


def model_digest(path: Path) -> str:
    stamp = path.stat()
    key = (str(path), stamp.st_mtime_ns, stamp.st_size)
    with CONFIG_CACHE_LOCK:
        if key not in MODEL_HASH_CACHE:
            with path.open("rb") as stream:
                MODEL_HASH_CACHE[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        return MODEL_HASH_CACHE[key]


def select_runtime() -> Path:
    candidates = [
        Path(sys.executable),
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "python"
        / "python.exe",
        ROOT.parent.parent / "tools" / "Python311" / "python.exe",
    ]
    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve()).casefold()
        if normalized in seen or not candidate.is_file():
            continue
        seen.add(normalized)
        check = subprocess.run(
            [str(candidate), "-c", "import numpy, pandas, jsbsim, joblib, catboost"],
            capture_output=True,
            timeout=20,
        )
        if check.returncode == 0:
            return candidate.resolve()
    return Path(sys.executable)


RUNTIME_PYTHON = select_runtime()


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_doe(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        features = [
            "wing_area_m2",
            "aspect_ratio",
            "taper_ratio",
            "sweep_deg",
            "tip_twist_deg",
            "fuselage_length_m",
            "fuselage_diameter_m",
        ]
        rows = list(reader)
    ranges = {
        name: [
            min(float(row[name]) for row in rows),
            max(float(row[name]) for row in rows),
        ]
        for name in features
        if name in columns
    }
    return features, ranges


def runtime_baselines() -> dict:
    with CONFIG_CACHE_LOCK:
        if BASELINES_CACHE and time.monotonic() - BASELINES_CACHE["at"] < 60:
            return BASELINES_CACHE["value"]
        value = _read_runtime_baselines()
        BASELINES_CACHE.update(at=time.monotonic(), value=value)
        return value


def _read_runtime_baselines() -> dict:
    candidate_seconds: dict[str, list[float]] = {}
    mpc_settings = []
    runs = sorted(
        RESULTS.glob("前端运行_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:16]
    for run_dir in runs:
        config = read_json(run_dir / "前端运行配置.json", {})
        if config.get("fidelity", "surrogate") != "surrogate":
            continue
        if config.get("sixdof_mode", "optimizer-loop") == "stepwise":
            continue
        controller = config.get("controller", "mpc")
        for metadata_path in run_dir.glob("*/ptdl_80h_joint_*_metadata.json"):
            metadata = read_json(metadata_path, {})
            optimizer = metadata.get("optimizer", {})
            try:
                population = float(optimizer["population_size"])
                generations = float(optimizer["generations"])
                runtime_s = float(optimizer["runtime_s"])
                evaluations = max(population * (generations + 1), 1.0)
                per_candidate = runtime_s / evaluations
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if math.isfinite(per_candidate) and per_candidate > 0:
                candidate_seconds.setdefault(controller, []).append(per_candidate)
                if controller == "mpc":
                    mpc_settings.append(
                        (
                            int(config.get("mpc_horizon", 3)),
                            int(config.get("mpc_action_levels", 3)),
                        )
                    )
    controller_seconds = {
        name: statistics.median(values[-6:])
        for name, values in candidate_seconds.items()
        if values
    }
    candidate_default = controller_seconds.get("mpc", 1.7)
    aero_step_times = []
    for summary_path in sorted(
        RESULTS.rglob("逐积分步VSPAERO_6DOF摘要.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:6]:
        summary = read_json(summary_path, {})
        try:
            solve_s = float(summary["max_single_vspaero_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(solve_s) and solve_s > 0:
            aero_step_times.append(solve_s)
    mesh_records = read_json(RESULTS / "网格加密对照_20261015" / "运行时间基准.json", {})
    return {
        "optimizer_candidate_s": candidate_default,
        "controller_candidate_s": controller_seconds,
        "mpc_horizon": int(statistics.median([row[0] for row in mpc_settings])) if mpc_settings else 3,
        "mpc_action_levels": int(statistics.median([row[1] for row in mpc_settings])) if mpc_settings else 3,
        "vspaero_step_s": statistics.median(aero_step_times) if aero_step_times else 19.4,
        "vspaero_mesh_cost_ratio": mesh_records.get("relative_cost", {"legacy": 1., "medium": 7.58, "fine": 25.02}),
        "jsbsim_step_s": 0.0002,
        "pareto_fraction": 0.12,
        "basis": "recent local optimizer metadata and measured VSPAERO step runtime; fallback values are estimates",
    }


def runtime_config() -> dict:
    with CONFIG_CACHE_LOCK:
        if RUNTIME_CACHE:
            return RUNTIME_CACHE
        result = subprocess.run(
            [str(RUNTIME_PYTHON), "-c",
             "import importlib.util,json,sys; "
             "print(json.dumps(dict(python_version=sys.version.split()[0],"
             "modules={m:importlib.util.find_spec(m) is not None "
             "for m in ('numpy','pandas','jsbsim','joblib','catboost')})))"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode:
            raise RuntimeError("Runtime dependency probe failed")
        RUNTIME_CACHE.update(json.loads(result.stdout))
        RUNTIME_CACHE.update(python=str(RUNTIME_PYTHON), vsp_python_exists=VSP_PYTHON.exists())
        return RUNTIME_CACHE


def available_config(precision: str = DEFAULT_PRECISION) -> dict:
    precision = precision if precision in MODEL_PATHS else DEFAULT_PRECISION
    doe_path = MODEL_PATHS[precision][1]
    features, bounds = read_doe(doe_path)
    seed = read_json(SEED, {})
    sizing = read_json(ROOT / "核心模型" / "ULTRA推进倍率配置.json", {}).get("assumptions", {})
    baseline_design = dict(seed.get("design", {}))
    for feature, (minimum, maximum) in bounds.items():
        if feature in baseline_design:
            baseline_design[feature] = min(
                maximum,
                max(minimum, float(baseline_design[feature])),
            )
    registry = read_json(ROOT / "气动代理模型数据" / "工作台代理精度.json", {}).get("models", {})
    model_configs = {
        name: dict(name=name, path=str(paths[0]), validation=registry.get(name, {}),
                   doe_ranges=read_doe(paths[1])[1])
        for name, paths in MODEL_PATHS.items()
    }
    for name, model_config in model_configs.items():
        validation = model_config["validation"]
        if validation and validation.get("model_sha256") != model_digest(MODEL_PATHS[name][0]):
            model_config["validation"] = {"reason": "模型文件已更新，需重新生成精度结果"}
    return {
        "features": features,
        "selected_model": model_configs[precision],
        "model_configs": model_configs,
        "algorithms": list(ALGORITHMS),
        "doe_ranges": bounds,
        "runtime_baselines": runtime_baselines(),
        "runtime": runtime_config(),
        "defaults": {
            "resolution": 2,
            "generations": 8,
            "seed": 20261005,
            "engine_multiplier": 1.2,
            "motor_multiplier": 1.2,
            "battery_multiplier": 1.2,
            "sixdof_duration": 8,
            "mpc_horizon": 3,
            "mpc_action_levels": 3,
            "mpc_soc_weight": 4.0,
            "mpc_slew_penalty": 0.002,
            "wind_step": 8,
            "wind_duration": 60,
            "engine_speed": 5000,
            "engine_fuel_scale": 1.0,
            "engine_ramp": 10.0,
            "motor_speed": 2100,
            "motor_branches": 2,
            "inverter_efficiency": 0.95,
            "battery_resistance": 1.0,
            "battery_soc_min": 0.20,
            "battery_soc_max": 1.00,
            "engine_baseline_power_kw": sizing.get("baseline_engine_power_kw", 120.0),
            "engine_upgrade_kw_per_hybrid_kg": sizing.get("engine_upgrade_kw_per_hybrid_kg", 0.0),
            "engine_power_ceiling_kw": sizing.get("engine_power_ceiling_kw", 120.0),
            "hybrid_specific_energy_wh_per_kg": sizing.get("hybrid_system_specific_energy_wh_per_kg", 220.0),
            "motor_power_kw_per_hybrid_kg": sizing.get("motor_power_kw_per_hybrid_kg", 1.0),
            "engine_efficiency_factor": 1.0,
            "engine_map_path": "",
        },
        "baseline_design": baseline_design,
        "powertrain_reference": powertrain_reference(),
    }


def csv_rows(path: Path, limit: int = 1000) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))[:limit]


def run_snapshot(run_id: str, record: dict, include_result: bool = True) -> dict:
    out = dict(record)
    out.pop("process", None)
    out["log_tail"] = ""
    log_path = Path(record["log"])
    if log_path.exists():
        try:
            out["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-9000:]
        except OSError:
            pass
    progress = read_json(Path(record["progress"]), {}) if record.get("progress") else {}
    if record.get("status") == "completed":
        summary = read_json(Path(record["output"]) / "四项统一集成验证摘要.json", {})
        progress = {
            "percent": 100,
            "stage": "完成，未找到可行候选" if summary.get("outcome") == "no_feasible_candidate" else "完成",
            "detail": summary.get("message", "优化、复评与独立回放完成"),
        }
    elif record.get("status") == "failed":
        progress = {
            "percent": progress.get("percent", 0),
            "stage": "失败",
            "detail": "运行中断，请查看运行日志",
        }
    out["progress"] = progress
    out["result"] = (
        result_payload(Path(record["output"]))
        if include_result and record["status"] == "completed"
        else None
    )
    return out


def recent_runs(include_result: bool = False) -> list[dict]:
    records = []
    for output in sorted(
        RESULTS.glob("前端运行_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:20]:
        run_id = output.name.removeprefix("前端运行_")
        with LOCK:
            active = ACTIVE.get(run_id)
        if active is not None:
            records.append(run_snapshot(run_id, active, include_result=include_result))
            continue
        candidate = output / "NSGA-II与MOEA-D择优可行候选.json"
        log_path = output / "frontend_run.log"
        config = read_json(output / "前端运行配置.json", {})
        summary = read_json(output / "四项统一集成验证摘要.json", {})
        status = "completed" if summary and (candidate.exists() or summary.get("optimization_completed")) else "failed"
        record = {
            "id": run_id,
            "status": status,
            "created": datetime.fromtimestamp(output.stat().st_mtime).isoformat(timespec="seconds"),
            "output": str(output),
            "log": str(log_path),
            "config": config,
            "return_code": 0 if status == "completed" else 1,
        }
        records.append(run_snapshot(run_id, record, include_result=include_result))
    return records


def result_payload(output: Path) -> dict:
    summary_path = output / "四项统一集成验证摘要.json"
    candidate_path = output / "NSGA-II与MOEA-D择优可行候选.json"
    summary = read_json(summary_path, {})
    candidate = read_json(candidate_path, {})
    algorithm_rows = csv_rows(output / "NSGA-II与MOEA-D同模型算法比较.csv")
    controller_rows = csv_rows(output / "控制器部件模型统一评价.csv")
    pareto = {}
    archive_names = {
        "NSGA-II": "ptdl_80h_joint_nsga_ii_feasible_pareto.csv",
        "MOEA-D": "ptdl_80h_joint_moead_feasible_pareto.csv",
        "MOPSO": "ptdl_80h_joint_mopso_feasible_pareto.csv",
        "Hybrid-NSGA-II-MOPSO": "ptdl_80h_joint_hybrid_nsga_ii_mopso_feasible_pareto.csv",
        "PTDL-Block-Adaptive": "ptdl_80h_joint_block_adaptive_feasible_pareto.csv",
        "PTDL-Coupled-Surrogate": "ptdl_80h_joint_coupled_surrogate_feasible_pareto.csv",
        "PTDL-Coupled-Guided": "ptdl_80h_joint_coupled_guided_feasible_pareto.csv",
        "PTDL-Coupled-Unique": "ptdl_80h_joint_coupled_unique_feasible_pareto.csv",
    }
    screened_path = output / "六自由度筛查后Pareto目标.csv"
    if screened_path.is_file():
        for row in csv_rows(screened_path):
            pareto.setdefault(row["algorithm"], []).append(row)
    else:
        for folder, filename in archive_names.items():
            path = output / folder / filename
            if path.is_file():
                pareto[folder] = csv_rows(path)
    return {
        "summary": summary,
        "candidate": candidate,
        "algorithms": algorithm_rows,
        "controllers": controller_rows,
        "pareto": pareto,
    }


def start_run(config: dict) -> dict:
    source = powertrain_reference()
    supplied_version = config.get("powertrain_version")
    if supplied_version and supplied_version != source["source_version"]:
        raise ValueError("动力模型版本已更新，请刷新工作台后运行。")
    config["powertrain_version"] = source["source_version"]
    algorithms = config.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms or any(
        item not in ALGORITHMS
        for item in algorithms
    ):
        raise ValueError("至少选择一种有效优化算法。")
    controller = config.get("controller", "mpc")
    if controller not in {
        "mpc",
        "rule",
        "fuzzy",
        "fault-tolerant-fuzzy",
        "neural-imitation",
        "phase-mpc",
        "source-ems",
    }:
        raise ValueError("控制策略无效。")
    precision = config.get("precision", DEFAULT_PRECISION)
    fidelity = config.get("fidelity", "finalists")
    sixdof_mode = config.get("sixdof_mode", "optimizer-loop")
    paths = MODEL_PATHS
    if precision not in paths:
        raise ValueError("气动代理模型档位无效。")
    if not all(p.is_file() for p in paths[precision]):
        raise ValueError("所选代理模型或训练数据文件不存在。")
    if precision == "selectedmesh6144" and not VSP_PYTHON.is_file():
        raise ValueError("网格融合代理需要安装 OpenVSP 的 Python 解释器。")
    mesh_level = config.get("vsp_mesh_level", "legacy")
    if mesh_level not in {"legacy", "medium", "fine"}:
        raise ValueError("VSPAERO 网格档位无效。")
    response_definition = config.get("response_definition", "recovery-integral")
    if response_definition not in {"legacy", "recovery-integral"}:
        raise ValueError("响应目标口径无效。")
    if controller == "phase-mpc" and not PHASE_MPC.is_file():
        raise ValueError("分阶段 MPC 配置不存在。")
    aero_mode = {
        "surrogate": "surrogate",
        "finalists": "live-vspaero-finalists",
        "all": "live-vspaero-all",
        "stepwise": "stepwise",
    }.get(fidelity)
    if aero_mode is None:
        raise ValueError("气动计算精度档位无效。")
    if sixdof_mode not in {
        "optimizer-loop",
        "pareto-screen",
        "stepwise",
        "full-mission",
    }:
        raise ValueError("六自由度计算范围无效。")
    duration_basis = config.get(
        "sixdof_duration_basis", "total-including-landing"
    )
    if duration_basis not in {
        "total-including-landing",
        "endurance-before-landing",
    }:
        raise ValueError("JSBSim 80 小时任务判据无效。")
    if aero_mode != "surrogate" and not VSP_PYTHON.is_file():
        raise ValueError(f"未找到 OpenVSP Python：{VSP_PYTHON}")

    numeric_limits = {
        "resolution": (1, 12),
        "generations": (1, 500),
        "sixdof_duration": (5, 60),
        "sixdof_step_s": (1.0 / 120.0, 0.25),
        "aero_step_duration": (0.1, 60),
        "aero_step_s": (1.0 / 120.0, 0.25),
        "mpc_horizon": (1, 6),
        "mpc_action_levels": (2, 7),
        "mpc_soc_weight": (0, 20),
        "mpc_slew_penalty": (0, 0.05),
        "wind_step": (0, 30),
        "wind_duration": (1, 1800),
        "engine_multiplier": (0.5, 3),
        "engine_speed": (2800, 6100),
        "engine_fuel_scale": (0.5, 2),
        "engine_ramp": (0.5, 30),
        "motor_multiplier": (0.5, 3),
        "motor_speed": (500, 5000),
        "motor_branches": (1, 8),
        "inverter_efficiency": (0.7, 1),
        "battery_multiplier": (0.5, 3),
        "battery_resistance": (0.1, 60),
        "battery_soc_min": (0.1, 0.8),
        "battery_soc_max": (0.85, 1),
        "engine_baseline_power_kw": (20, 200),
        "engine_upgrade_kw_per_hybrid_kg": (0, 2),
        "engine_power_ceiling_kw": (20, 300),
        "hybrid_specific_energy_wh_per_kg": (50, 500),
        "motor_power_kw_per_hybrid_kg": (0.1, 5),
        "engine_efficiency_factor": (0.5, 1.5),
    }
    for name, (minimum, maximum) in numeric_limits.items():
        try:
            value = float(config.get(name))
        except (TypeError, ValueError):
            raise ValueError(f"{name} 必须为数值。") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} 必须在 {minimum}–{maximum} 范围内。")
        if name in {"motor_branches", "mpc_horizon", "mpc_action_levels"} and not value.is_integer():
            raise ValueError(f"{name} 必须为整数。")
    if float(config.get("battery_soc_min", 0.2)) >= float(config.get("battery_soc_max", 1.0)):
        raise ValueError("电池 SOC 下限必须小于上限。")
    if (
        aero_mode == "stepwise"
        and sixdof_mode == "stepwise"
        and max(
            float(config.get("aero_step_duration", 8)),
            float(config.get("sixdof_duration", 8)),
        )
        > 10.0
    ):
        raise ValueError("逐步 VSPAERO 与 JSBSim 联算时，共享仿真时长不能超过 10 秒。")
    try:
        int(config.get("seed", 20261005))
    except (TypeError, ValueError):
        raise ValueError("随机种子必须为整数。") from None

    doe = read_doe(paths[precision][1])[1]
    bounds = config.get("bounds", {})
    for feature, pair in bounds.items():
        if feature not in doe or not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"气动变量 {feature} 的范围设置无效。")
        low, high = map(float, pair)
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or low < doe[feature][0]
            or high > doe[feature][1]
            or low >= high
        ):
            raise ValueError(
                f"{feature} 应在 DOE 范围 [{doe[feature][0]:.4g}, {doe[feature][1]:.4g}] 内且下限小于上限。"
            )

    seed_record = read_json(SEED, {})
    baseline_design = config.get("baseline_design", seed_record.get("design", {}))
    design_limits = {
        "wing_area_m2": (22.0, 40.0),
        "aspect_ratio": (18.0, 31.0),
        "taper_ratio": (0.30, 0.70),
        "sweep_deg": (0.0, 8.0),
        "tip_twist_deg": (-3.0, 1.5),
        "fuselage_length_m": (7.5, 10.0),
        "fuselage_diameter_m": (0.65, 0.95),
        "propeller_diameter_m": (2.1, 3.0),
        "payload_mass_kg": (100.0, 205.0),
        "hybrid_system_mass_kg": (0.0, 180.0),
    }
    if not isinstance(baseline_design, dict):
        raise ValueError("基准机体参数必须为对象。")
    normalized_design = {}
    for name, (minimum, maximum) in design_limits.items():
        try:
            value = float(baseline_design[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"基准机体参数 {name} 必须为数值。") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"基准机体参数 {name} 必须在 {minimum}–{maximum} 范围内。")
        normalized_design[name] = value
    for feature, pair in bounds.items():
        value = normalized_design[feature]
        if not pair[0] <= value <= pair[1]:
            raise ValueError(f"基准机体参数 {feature} 必须落在当前优化范围内。")

    engine_map_path = str(config.get("engine_map_path", "") or "").strip()
    if engine_map_path:
        custom_map = Path(engine_map_path).expanduser()
        if not custom_map.is_absolute():
            custom_map = ROOT / custom_map
        if not custom_map.is_file():
            raise ValueError(f"找不到指定的发动机燃油图 CSV：{custom_map}")
        engine_map_path = str(custom_map.resolve())

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
    output = RESULTS / f"前端运行_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    seed_record["design"] = normalized_design
    seed_path = output / "本轮基准机体种子.json"
    seed_path.write_text(
        json.dumps(seed_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bounds_file = output / "气动设计范围.json"
    bounds_file.write_text(json.dumps(bounds, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "前端运行配置.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args = [
        str(RUNTIME_PYTHON),
        str(RUNNER),
        "--output-dir",
        str(output),
        "--progress-file",
        str(output / "前端运行进度.json"),
        "--seed-candidate",
        str(seed_path),
        "--surrogate",
        str(paths[precision][0]),
        "--aero-domain-data",
        str(paths[precision][1]),
        "--resolution",
        str(int(config.get("resolution", 2))),
        "--generations",
        str(int(config.get("generations", 8))),
        "--seed",
        str(int(config.get("seed", 20261005))),
        "--algorithms",
        *dict.fromkeys(algorithms),
        "--controller",
        controller,
        "--mpc-horizon",
        str(int(config.get("mpc_horizon", 3))),
        "--mpc-action-levels",
        str(int(config.get("mpc_action_levels", 3))),
        "--mpc-soc-terminal-weight-mj",
        str(float(config.get("mpc_soc_weight", 4))),
        "--mpc-engine-slew-penalty",
        str(float(config.get("mpc_slew_penalty", 0.002))),
        "--wind-step-mps",
        str(float(config.get("wind_step", 8))),
        "--wind-duration-s",
        str(float(config.get("wind_duration", 60))),
        "--design-bounds-json",
        str(bounds_file),
        "--aero-mode",
        aero_mode,
        "--vsp-mesh-level",
        mesh_level,
        "--response-definition",
        response_definition,
        "--sixdof-duration-s",
        str(float(config.get("sixdof_duration", 8))),
        "--sixdof-integration-step-s",
        str(float(config.get("sixdof_step_s", 0.05))),
        "--sixdof-mode",
        sixdof_mode,
        "--sixdof-full-mission-hours",
        str(float(config.get("sixdof_full_mission_hours", 80))),
        "--sixdof-full-mission-duration-basis",
        duration_basis,
        "--aero-step-duration-s",
        str(float(config.get("aero_step_duration", 8))),
        "--aero-integration-step-s",
        str(float(config.get("aero_step_s", 0.25))),
        "--engine-rating-multiplier",
        str(float(config.get("engine_multiplier", 1.2))),
        "--motor-rating-multiplier",
        str(float(config.get("motor_multiplier", 1.2))),
        "--battery-energy-multiplier",
        str(float(config.get("battery_multiplier", 1.2))),
        "--engine-speed-rpm",
        str(float(config.get("engine_speed", 5000))),
        "--motor-speed-rpm",
        str(float(config.get("motor_speed", 2100))),
        "--motor-branches",
        str(int(config.get("motor_branches", 2))),
        "--engine-fuel-flow-scale",
        str(float(config.get("engine_fuel_scale", 1))),
        "--engine-power-rate-limit-kwps",
        str(float(config.get("engine_ramp", 10))),
        "--battery-resistance-multiplier",
        str(float(config.get("battery_resistance", 1))),
        "--inverter-efficiency",
        str(float(config.get("inverter_efficiency", 0.95))),
        "--battery-soc-min",
        str(float(config.get("battery_soc_min", 0.2))),
        "--battery-soc-max",
        str(float(config.get("battery_soc_max", 1))),
        "--engine-baseline-power-kw",
        str(float(config.get("engine_baseline_power_kw", 120))),
        "--engine-upgrade-kw-per-hybrid-kg",
        str(float(config.get("engine_upgrade_kw_per_hybrid_kg", 0))),
        "--engine-power-ceiling-kw",
        str(float(config.get("engine_power_ceiling_kw", 120))),
        "--hybrid-specific-energy-wh-per-kg",
        str(float(config.get("hybrid_specific_energy_wh_per_kg", 220))),
        "--motor-power-kw-per-hybrid-kg",
        str(float(config.get("motor_power_kw_per_hybrid_kg", 1))),
        "--engine-efficiency-factor",
        str(float(config.get("engine_efficiency_factor", 1))),
    ]
    if engine_map_path:
        args.extend(["--engine-map", engine_map_path])
    if controller == "phase-mpc":
        args.extend(["--phase-mpc-config", str(PHASE_MPC)])
    if aero_mode != "surrogate":
        args.extend(["--vsp-python", str(VSP_PYTHON)])
    record = {
        "id": run_id,
        "status": "starting",
        "created": datetime.now().isoformat(timespec="seconds"),
        "output": str(output),
        "log": str(output / "frontend_run.log"),
        "progress": str(output / "前端运行进度.json"),
        "config": config,
        "command": args,
        "return_code": None,
    }
    with LOCK:
        ACTIVE[run_id] = record

    def execute():
        log_file = Path(record["log"])
        try:
            with log_file.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    args,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                record["process"] = process
                record["status"] = "running"
                code = process.wait()
            record["return_code"] = code
            record["status"] = "completed" if code == 0 else "failed"
        except Exception as exc:
            record["status"] = "failed"
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write(f"\n前端启动任务失败：{exc}\n")

    threading.Thread(target=execute, name=f"ptdl-{run_id}", daemon=True).start()
    return run_snapshot(run_id, record)


def simulation_snapshot(record: dict) -> dict:
    out = {key: value for key, value in record.items() if key != "process"}
    out["progress"] = read_json(Path(record["progress"]), {})
    out["summary"] = (
        read_json(Path(record["summary"]), {})
        if record.get("status") == "completed"
        else None
    )
    try:
        out["log_tail"] = Path(record["log"]).read_text(
            encoding="utf-8", errors="replace"
        )[-5000:]
    except OSError:
        out["log_tail"] = ""
    return out


def start_coupled_simulation(config: dict) -> dict:
    limits = {
        "duration_s": (1.0 / 120.0, 10.0),
        "initial_speed_mps": (15.0, 100.0),
        "initial_altitude_m": (0.0, 10000.0),
        "initial_alpha_deg": (-8.0, 14.0),
        "wing_area_m2": (10.0, 60.0),
        "aspect_ratio": (8.0, 40.0),
        "integration_step_s": (1.0 / 120.0, 0.25),
    }
    values = {}
    for key, (minimum, maximum) in limits.items():
        try:
            value = float(config.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"{key} 必须为数值。") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} 必须在 {minimum}–{maximum} 范围内。")
        values[key] = value
    if not VSP_PYTHON.is_file():
        raise ValueError(f"未找到 OpenVSP Python：{VSP_PYTHON}")
    if not STEPWISE_RUNNER.is_file():
        raise FileNotFoundError(f"逐步联算入口不存在：{STEPWISE_RUNNER}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
    output = RESULTS / f"逐积分步联算_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    seed = read_json(SEED, {})
    design = dict(seed.get("design", {}))
    design["wing_area_m2"] = values["wing_area_m2"]
    design["aspect_ratio"] = values["aspect_ratio"]
    design_path = output / "联算构型.json"
    design_path.write_text(json.dumps({"design": design}, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path = output / "联算配置.json"
    config_path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    args = [
        str(RUNTIME_PYTHON),
        str(STEPWISE_RUNNER),
        "--duration-s", str(values["duration_s"]),
        "--output-dir", str(output),
        "--vsp-python", str(VSP_PYTHON),
        "--design-json", str(design_path),
        "--progress-file", str(output / "联算进度.json"),
        "--initial-speed-mps", str(values["initial_speed_mps"]),
        "--initial-altitude-m", str(values["initial_altitude_m"]),
        "--initial-alpha-deg", str(values["initial_alpha_deg"]),
        "--integration-step-s", str(values["integration_step_s"]),
    ]
    record = {
        "id": run_id,
        "status": "starting",
        "created": datetime.now().isoformat(timespec="seconds"),
        "output": str(output),
        "log": str(output / "联算日志.txt"),
        "progress": str(output / "联算进度.json"),
        "summary": str(output / "逐积分步VSPAERO_6DOF摘要.json"),
        "config": values,
        "command": args,
        "return_code": None,
    }
    with LOCK:
        ACTIVE_SIMULATIONS[run_id] = record

    def execute():
        try:
            with Path(record["log"]).open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    args,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                record["process"] = process
                record["status"] = "running"
                code = process.wait()
            record["return_code"] = code
            record["status"] = "completed" if code == 0 else "failed"
        except Exception as exc:
            record["status"] = "failed"
            with Path(record["log"]).open("a", encoding="utf-8") as stream:
                stream.write(f"\n逐步联算启动失败：{exc}\n")

    threading.Thread(target=execute, name=f"ptdl-stepwise-{run_id}", daemon=True).start()
    return simulation_snapshot(record)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[PTDL UI] {self.address_string()} {fmt % args}")

    def send_json(self, payload, status=200):
        def sanitize(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: sanitize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize(item) for item in value]
            return value

        data = json.dumps(sanitize(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = urlparse(self.path).path
        if route.startswith(("/research-figures/", "/research-figures-latest/")):
            latest = route.startswith("/research-figures-latest/")
            prefix = "/research-figures-latest/" if latest else "/research-figures/"
            name = unquote(route.removeprefix(prefix)) or "index.html"
            folder = (RESULTS / ("主动代理与算法图集_20261016" if latest else "科研图集_20261015")).resolve()
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                self.send_error(404)
                return
            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if route == "/api/config":
            query = dict(
                item.split("=", 1) for item in urlparse(self.path).query.split("&") if "=" in item
            )
            return self.send_json(available_config(query.get("precision", DEFAULT_PRECISION)))
        if route == "/api/runs":
            return self.send_json({"runs": recent_runs()})
        if route == "/api/coupled-simulations":
            with LOCK:
                records = list(ACTIVE_SIMULATIONS.values())
            return self.send_json({
                "simulations": [
                    simulation_snapshot(record)
                    for record in sorted(records, key=lambda item: item["created"], reverse=True)
                ]
            })
        if route.startswith("/api/coupled-simulations/"):
            run_id = unquote(route.removeprefix("/api/coupled-simulations/"))
            with LOCK:
                record = ACTIVE_SIMULATIONS.get(run_id)
            if record is None:
                return self.send_json({"error": "未找到该逐步联算记录。"}, 404)
            return self.send_json(simulation_snapshot(record))
        if route.startswith("/api/runs/"):
            run_id = unquote(route.removeprefix("/api/runs/"))
            with LOCK:
                record = ACTIVE.get(run_id)
            if record is None:
                record = next((item for item in recent_runs() if item["id"] == run_id), None)
                if record is None:
                    return self.send_json({"error": "未找到该运行记录。"}, 404)
                if record["status"] == "completed":
                    record["result"] = result_payload(Path(record["output"]))
                return self.send_json(record)
            return self.send_json(run_snapshot(run_id, record))
        return super().do_GET()

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in {"/api/runs", "/api/coupled-simulations"}:
            return self.send_json({"error": "不支持的请求。"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            config = json.loads(self.rfile.read(length).decode("utf-8"))
            result = (
                start_run(config)
                if route == "/api/runs"
                else start_coupled_simulation(config)
            )
            return self.send_json(result, 201)
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            return self.send_json({"error": str(exc)}, 400)


class WorkbenchHTTPServer(ThreadingHTTPServer):
    # Windows permits duplicate listeners when both sockets enable address reuse.
    allow_reuse_address = False


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    if not 0 <= port <= 65535:
        raise ValueError("工作台端口必须在 0 到 65535 之间。")
    ports = [0] if port == 0 else [*range(port, min(port + 20, 65536)), 0]
    for candidate in ports:
        try:
            return WorkbenchHTTPServer((host, candidate), Handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE and getattr(exc, "winerror", None) != 10048:
                raise
    raise OSError("未能为工作台分配可用端口。")


def main():
    host = os.environ.get("PTDL_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("PTDL_UI_PORT", "8765"))
    server = create_server(host, port)
    actual_port = server.server_address[1]
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{browser_host}:{actual_port}/"
    print(f"PTDL 工作台已启动  {url}", flush=True)
    if port and actual_port != port:
        print(f"端口 {port} 已被占用，使用端口 {actual_port}。", flush=True)
    print(f"网页服务解释器  {sys.executable}", flush=True)
    print(f"优化计算解释器  {RUNTIME_PYTHON}", flush=True)
    print("保持本程序运行即可使用工作台。在 PyCharm 中点击停止可关闭本次服务。", flush=True)

    def open_workbench():
        try:
            if not webbrowser.open(url, new=2):
                print(f"请在浏览器中打开  {url}", flush=True)
        except Exception as exc:
            print(f"浏览器未能自动打开，请访问  {url}  ({exc})", flush=True)

    if os.environ.get("PTDL_UI_OPEN_BROWSER", "1").lower() not in {"0", "false", "no"}:
        threading.Thread(target=open_workbench, name="ptdl-open-browser", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("本次 PTDL 工作台服务已停止。", flush=True)


if __name__ == "__main__":
    main()
