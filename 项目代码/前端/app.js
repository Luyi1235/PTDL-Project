const $ = (selector) => document.querySelector(selector);
const form = $("#configForm");
let activeRunId = null;
let doeRanges = {};
let baselineDesignLoaded = false;
let runtimeBaselines = {
  optimizer_candidate_s: 1.7,
  vspaero_step_s: 19.4,
  jsbsim_step_s: 0.0002,
  jsbsim_full_mission_step_s: 0.00001,
  pareto_fraction: 0.12,
};
let pollTimer = null;
let paretoGroups = {};
let plotYaw = -0.55;
let plotPitch = 0.48;
let dragPoint = null;
let chartPoints = [];
let powertrainVersion = null;

const featureNames = {
  wing_area_m2: ["机翼面积", "m²"],
  aspect_ratio: ["展弦比", "—"],
  taper_ratio: ["梢根比", "—"],
  sweep_deg: ["后掠角", "deg"],
  tip_twist_deg: ["翼尖扭转角", "deg"],
  fuselage_length_m: ["机身长度", "m"],
  fuselage_diameter_m: ["机身直径", "m"],
};

const selectHelp = {
  precision: {
    fine6144: "使用 6144 个训练构型，由七项几何参数预测升力斜率和尾迹效率，可直接参与联合优化。右侧显示预先计算的 128 构型均匀独立测试精度。",
    selectedmesh6144: "使用 6144 个训练构型，融合筛选后的网格特征与几何参数预测。新构型先生成原始几何网格，特征缓存后复用，耗时高于几何参数模型。右侧为同组 128 构型独立测试精度。",
  },
  vsp_mesh_level: {
    legacy: "与既有训练标签相同的原始网格。该选择只影响实际 VSPAERO 求解。",
    medium: "主翼翼段展向离散参数 13、弦向离散参数 33，同时加密尾翼与机身。计算量增加，网格加密对照仍需结合系数变化判断。",
    fine: "主翼翼段展向离散参数 25、弦向离散参数 49，同时加密尾翼与机身。用于重点构型复算，运行时间明显增加。",
  },
  response_definition: {
    "recovery-integral": "响应目标由恢复时间与速度、高度偏差的归一化积分相加，单位为秒。模式切换时间单独记录；任务能量与轨迹跟踪两个目标保留。",
    legacy: "保留旧实验的恢复时间加模式切换时间口径。恢复时间从扰动结束计时，结束时已在判定带内可为零。",
  },
  fidelity: {
    surrogate: "优化评价只调用气动代理模型，速度较快；最终结果精度受代理模型误差影响。",
    all: "优化器每评价一个候选就调用一次 VSPAERO；计算量会显著增加。",
    finalists: "搜索阶段用代理模型，搜索后再对 Pareto 候选调用 VSPAERO 复评；复评结果不参与此前的搜索决策。",
    stepwise: "每个候选的短时低阶纵向轨迹上，每个积分步都调用 VSPAERO；逐步系数汇总后用于该候选的 80 小时低阶任务评价。下方可设时长和步长，不等同于全任务高保真联算。",
  },
  controller: {
    mpc: "模型预测控制：在有限预测窗口内滚动选择动力分配，择优候选还会用该控制器独立回放。",
    "phase-mpc": "按当前飞行阶段切换经过仿真整定的 MPC 预测窗口与代价参数，仍执行功率、SOC 和变化率约束。",
    rule: "规则控制：按预设条件切换发动机与电池功率分配，结构直接、计算开销较小。",
    "source-ems": "采用动能学院 1005 版三层调度，按飞行阶段、SOC 和母线功率缺口切换发动机。起降与爬升优先供能，下降阶段按源模型逻辑停机或补能，功率指令进入同一部件闭环。",
    fuzzy: "模糊自适应控制：根据状态隶属度和规则调整功率分配，择优候选会进行独立回放。",
    "fault-tolerant-fuzzy": "故障容错模糊控制：在模糊调度基础上考虑部件能力退化情景，择优候选会进行独立回放。",
    "neural-imitation": "神经网络 MPC 模仿策略：由离线 MPC 仿真样本训练，在线输出经功率边界、SOC 充电许可和发动机变化率约束投影；属于估计模型下的策略验证。",
  },
  sixdof_mode: {
    "optimizer-loop": "每个候选都运行一次 JSBSim 短时六自由度响应检查，检查约束直接参与优化；使用 JSBSim 模型默认积分步长。",
    "pareto-screen": "优化搜索完成后，只对 Pareto 候选运行 JSBSim 短时检查；检查结果不参与此前的搜索决策。",
    stepwise: "每个候选都由 JSBSim 逐积分步推进短时六自由度状态，约束参与优化；时长和积分步长均可设置。",
    "full-mission": "每个优化候选都运行 JSBSim 六阶段任务；可选择 80 小时按含着陆总时长计算，或要求先飞满 80 小时再下降着陆。候选约束回传优化器，计算量远高于短时检查。",
  },
};

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("zh-CN", { maximumFractionDigits: digits }) : "—";
}

function updateSelectHelp(name) {
  const select = form.elements.namedItem(name);
  const help = $(`#${name === "sixdof_mode" ? "sixdof" : name}Help`);
  if (!select || !help) return;
  let text = selectHelp[name]?.[select.value] || "";
  if (
    name === "fidelity"
    && select.value === "stepwise"
    && form.elements.namedItem("sixdof_mode").value === "stepwise"
  ) {
    text += " 同时启用 03 时将与 JSBSim 逐步联算，共用较长的设置时长和较小的积分步长。";
  }
  if (
    name === "sixdof_mode"
    && select.value === "stepwise"
    && form.elements.namedItem("fidelity").value === "stepwise"
  ) {
    text += " 与 02 逐步 VSPAERO 联算时，共用较长的设置时长和较小的积分步长。";
  }
  help.textContent = text;
}

function updateAllSelectHelp() {
  for (const name of Object.keys(selectHelp)) updateSelectHelp(name);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "无法估算";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} 秒`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} 分钟`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} 小时`;
  return `${(seconds / 86400).toFixed(1)} 天`;
}

function updateModeControls() {
  $("#aeroStepControls").hidden = form.elements.namedItem("fidelity").value !== "stepwise";
  $("#sixdofStepControls").hidden = form.elements.namedItem("sixdof_mode").value !== "stepwise";
  $("#sixdofMissionControls").hidden = form.elements.namedItem("sixdof_mode").value !== "full-mission";
  const durationBasis = form.elements.namedItem("sixdof_duration_basis");
  $("#sixdofDurationBasisHelp").textContent =
    durationBasis.value === "endurance-before-landing"
      ? "JSBSim 从空中初始状态计时，满 80 小时后才开始下降和着陆；地面起飞滑跑不计入这 80 小时。"
      : "爬升、续航、下降和着陆共用 80 小时任务时长；提前触地不代表完成了 80 小时续航。";
  updateSelectHelp("fidelity");
  updateSelectHelp("sixdof_mode");
}

function updateRuntimeEstimate() {
  const config = getConfig();
  const algorithms = config.algorithms.length;
  const population = (config.resolution + 1) * (config.resolution + 2) / 2;
  const evaluations = Math.max(0, population * (config.generations + 1) * algorithms);
  const paretoCandidates = Math.max(
    algorithms,
    Math.ceil(evaluations * (Number(runtimeBaselines.pareto_fraction) || 0.12)),
  );
  const nativeJsbsimStep = 1 / 120;
  const jsbsimStepCost = Number(runtimeBaselines.jsbsim_step_s) || 0.0002;
  const defaultSixdofCost = (config.sixdof_duration / nativeJsbsimStep) * jsbsimStepCost;
  let controllerScale = 1;
  let sharedStepSixdofSeconds = 0;
  if (config.controller === "mpc") {
    const referenceCombinations = Math.pow(
      Number(runtimeBaselines.mpc_action_levels) || 3,
      Number(runtimeBaselines.mpc_horizon) || 3,
    );
    controllerScale = Math.pow(config.mpc_action_levels, config.mpc_horizon) / referenceCombinations;
  } else if (config.fidelity === "stepwise" && config.sixdof_mode === "stepwise") {
    const sharedDuration = Math.max(config.aero_step_duration, config.sixdof_duration);
    const sharedStep = Math.min(config.aero_step_s, config.sixdof_step_s);
    sharedStepSixdofSeconds = evaluations * (sharedDuration / sharedStep) * jsbsimStepCost;
  } else {
    const observed = runtimeBaselines.controller_candidate_s || {};
    const reference = Number(runtimeBaselines.optimizer_candidate_s) || 1.7;
    controllerScale = Math.max(0.25, (Number(observed[config.controller]) || reference * 0.55) / reference);
  }
  let seconds = evaluations * Math.max(
    0.15,
    (Number(runtimeBaselines.optimizer_candidate_s) || 1.7) * controllerScale - defaultSixdofCost,
  );
  seconds += sharedStepSixdofSeconds;
  if (config.sixdof_mode === "optimizer-loop") {
    seconds += evaluations * defaultSixdofCost;
  } else if (config.sixdof_mode === "pareto-screen") {
    seconds += paretoCandidates * defaultSixdofCost;
  } else if (config.sixdof_mode === "full-mission") {
    const postEnduranceLandingSeconds =
      config.sixdof_duration_basis === "endurance-before-landing" ? 8100 : 0;
    const fullMissionSteps =
      (config.sixdof_full_mission_hours * 3600 + postEnduranceLandingSeconds + 7200)
      / config.sixdof_step_s;
    const fullMissionStepCost = Number(runtimeBaselines.jsbsim_full_mission_step_s) || 0.00001;
    seconds += evaluations * fullMissionSteps * fullMissionStepCost;
  } else {
    seconds += evaluations
      * (config.sixdof_duration / config.sixdof_step_s)
      * jsbsimStepCost;
  }
  const meshRatio = Number(runtimeBaselines.vspaero_mesh_cost_ratio?.[config.vsp_mesh_level]) || 1;
  const vspStepCost = (Number(runtimeBaselines.vspaero_step_s) || 19.4) * meshRatio;
  if (config.fidelity === "all") {
    seconds += (evaluations + algorithms) * vspStepCost;
  } else if (config.fidelity === "finalists") {
    seconds += paretoCandidates * vspStepCost;
  } else if (config.fidelity === "stepwise") {
    const shared = config.sixdof_mode === "stepwise";
    const duration = shared
      ? Math.max(config.aero_step_duration, config.sixdof_duration)
      : config.aero_step_duration;
    const step = shared
      ? Math.min(config.aero_step_s, config.sixdof_step_s)
      : config.aero_step_s;
    const steps = Math.max(1, Math.ceil(duration / step));
    const extraParetoPass = config.sixdof_mode === "pareto-screen" ? paretoCandidates * steps : 0;
    seconds += (evaluations * steps + extraParetoPass) * vspStepCost;
  }
  const geometryPreprocessing = config.precision === "selectedmesh6144"
    && ["surrogate", "finalists"].includes(config.fidelity);
  if (geometryPreprocessing) {
    seconds += evaluations * (Number(runtimeBaselines.surrogate_geometry_s) || 4);
  }
  seconds *= 1.18;
  const low = formatDuration(seconds * 0.75);
  const high = formatDuration(seconds * 1.35);
  const estimate = formatDuration(seconds);
  const vspSteps = config.fidelity === "stepwise"
    ? `，逐步 VSPAERO 约 ${Math.ceil(
      (config.sixdof_mode === "stepwise"
        ? Math.max(config.aero_step_duration, config.sixdof_duration)
        : config.aero_step_duration)
      / (config.sixdof_mode === "stepwise"
        ? Math.min(config.aero_step_s, config.sixdof_step_s)
        : config.aero_step_s),
    )} 次/候选`
    : "";
  const geometryNote = geometryPreprocessing ? "，含新构型几何特征预处理，缓存可缩短耗时" : "";
  $("#runtimeEstimate").textContent = `预计运行时间：约 ${estimate}（参考范围 ${low}–${high}，粗略估算${vspSteps}${geometryNote}）`;
}

function setBounds(ranges, reset = false) {
  doeRanges = ranges;
  const current = readBounds();
  $("#boundsGrid").replaceChildren();
  for (const [key, pair] of Object.entries(ranges)) {
    const [label, unit] = featureNames[key] || [key, ""];
    let saved = reset ? pair : (current[key] || pair);
    saved = [
      Math.max(pair[0], Math.min(pair[1], saved[0])),
      Math.max(pair[0], Math.min(pair[1], saved[1])),
    ];
    if (saved[0] >= saved[1]) saved = pair;
    const row = document.createElement("div");
    row.className = "bound-field";
    row.innerHTML = `<label>${label}<br><small>${unit} · DOE ${formatNumber(pair[0], 3)}–${formatNumber(pair[1], 3)}</small></label>
      <input type="number" step="any" aria-label="${label}下限" data-feature="${key}" data-side="0" value="${saved[0]}" min="${pair[0]}" max="${pair[1]}">
      <input type="number" step="any" aria-label="${label}上限" data-feature="${key}" data-side="1" value="${saved[1]}" min="${pair[0]}" max="${pair[1]}">`;
    $("#boundsGrid").append(row);
  }
}

function readBounds() {
  const result = {};
  document.querySelectorAll(".bound-field input").forEach((input) => {
    const key = input.dataset.feature;
    result[key] ||= [];
    result[key][Number(input.dataset.side)] = Number(input.value);
  });
  return result;
}

function getConfig() {
  const values = new FormData(form);
  const algorithms = [...form.querySelectorAll('input[name="algorithm"]:checked')].map((item) => item.value);
  const baselineDesign = Object.fromEntries([
    "wing_area_m2",
    "aspect_ratio",
    "taper_ratio",
    "sweep_deg",
    "tip_twist_deg",
    "fuselage_length_m",
    "fuselage_diameter_m",
    "propeller_diameter_m",
    "payload_mass_kg",
    "hybrid_system_mass_kg",
  ].map((key) => [key, Number(values.get(`baseline_${key}`))]));
  return {
    algorithms,
    powertrain_version: powertrainVersion,
    controller: values.get("controller"),
    precision: values.get("precision"),
    fidelity: values.get("fidelity"),
    vsp_mesh_level: values.get("vsp_mesh_level"),
    response_definition: values.get("response_definition"),
    resolution: Number(values.get("resolution")),
    generations: Number(values.get("generations")),
    seed: Number(values.get("seed")),
    sixdof_duration: Number(values.get("sixdof_duration")),
    sixdof_mode: values.get("sixdof_mode"),
    sixdof_step_s: Number(values.get("sixdof_step_s")),
    sixdof_full_mission_hours: 80,
    sixdof_duration_basis: values.get("sixdof_duration_basis"),
    aero_step_duration: Number(values.get("aero_step_duration")),
    aero_step_s: Number(values.get("aero_step_s")),
    mpc_horizon: Number(values.get("mpc_horizon")),
    mpc_action_levels: Number(values.get("mpc_action_levels")),
    mpc_soc_weight: Number(values.get("mpc_soc_weight")),
    mpc_slew_penalty: Number(values.get("mpc_slew_penalty")),
    wind_step: Number(values.get("wind_step")),
    wind_duration: Number(values.get("wind_duration")),
    engine_multiplier: Number(values.get("engine_multiplier")),
    engine_speed: Number(values.get("engine_speed")),
    engine_fuel_scale: Number(values.get("engine_fuel_scale")),
    engine_ramp: Number(values.get("engine_ramp")),
    motor_multiplier: Number(values.get("motor_multiplier")),
    motor_speed: Number(values.get("motor_speed")),
    motor_branches: Number(values.get("motor_branches")),
    inverter_efficiency: Number(values.get("inverter_efficiency")),
    battery_multiplier: Number(values.get("battery_multiplier")),
    battery_resistance: Number(values.get("battery_resistance")),
    battery_soc_min: Number(values.get("battery_soc_min")),
    battery_soc_max: Number(values.get("battery_soc_max")),
    baseline_design: baselineDesign,
    engine_baseline_power_kw: Number(values.get("engine_baseline_power_kw")),
    engine_upgrade_kw_per_hybrid_kg: Number(values.get("engine_upgrade_kw_per_hybrid_kg")),
    engine_power_ceiling_kw: Number(values.get("engine_power_ceiling_kw")),
    hybrid_specific_energy_wh_per_kg: Number(values.get("hybrid_specific_energy_wh_per_kg")),
    motor_power_kw_per_hybrid_kg: Number(values.get("motor_power_kw_per_hybrid_kg")),
    engine_efficiency_factor: Number(values.get("engine_efficiency_factor")),
    engine_map_path: String(values.get("engine_map_path") || "").trim(),
    bounds: readBounds(),
  };
}

function renderConfig(config) {
  if (!config || !Object.keys(config).length) {
    $("#runConfig").textContent = "该历史任务未保存前端参数快照。";
    return;
  }
  const labels = {
    powertrain_version: "动力系统版本",
    algorithms: "优化算法",
    controller: "控制策略",
    precision: "气动代理训练集",
    fidelity: "02 气动计算范围",
    resolution: "优化分辨率 H",
    generations: "迭代代数",
    seed: "随机种子",
    bounds: "气动几何边界",
    sixdof_duration: "六自由度候选筛查时长 (s)",
    sixdof_mode: "03 六自由度计算范围",
    sixdof_step_s: "JSBSim 积分步长 (s)",
    sixdof_full_mission_hours: "JSBSim 完整任务时长 (h)",
    sixdof_duration_basis: "JSBSim 80 小时任务判据",
    aero_step_duration: "VSPAERO 候选轨迹时长 (s)",
    aero_step_s: "VSPAERO 积分步长 (s)",
    mpc_horizon: "MPC 预测步数",
    mpc_action_levels: "发动机动作档位",
    mpc_soc_weight: "MPC 末端 SOC 权重 (MJ)",
    mpc_slew_penalty: "发动机功率平滑惩罚 (MJ/kW)",
    wind_step: "迎风阶跃 (m/s)",
    wind_duration: "迎风持续时间 (s)",
    engine_multiplier: "发动机额定倍率",
    engine_speed: "发动机转速 (rpm)",
    engine_fuel_scale: "燃油流量倍率",
    engine_ramp: "发动机功率变化率 (kW/s)",
    motor_multiplier: "电机额定倍率",
    motor_speed: "电机转速 (rpm)",
    motor_branches: "电机并联支路数",
    inverter_efficiency: "逆变器效率",
    battery_multiplier: "电池能量倍率",
    battery_resistance: "电池内阻倍率",
    battery_soc_min: "电池 SOC 下限",
    battery_soc_max: "电池 SOC 上限",
    baseline_design: "基准机体种子参数",
    engine_baseline_power_kw: "基础发动机功率 (kW)",
    engine_upgrade_kw_per_hybrid_kg: "混动质量功率增量 (kW/kg)",
    engine_power_ceiling_kw: "发动机功率上限 (kW)",
    hybrid_specific_energy_wh_per_kg: "混动系统比能量 (Wh/kg)",
    motor_power_kw_per_hybrid_kg: "电机比功率 (kW/kg)",
    engine_efficiency_factor: "发动机效率曲线倍率",
    engine_map_path: "自定义发动机燃油图 CSV",
  };
  const view = Object.fromEntries(Object.entries(config).map(([key, value]) => [labels[key] || key, value]));
  $("#runConfig").textContent = JSON.stringify(view, null, 2);
}

function validate(config) {
  if (!config.algorithms.length) return "请至少选择一种优化算法。";
  if (config.aero_step_duration < 0.1 || config.aero_step_duration > 60) return "VSPAERO 候选轨迹时长应在 0.1–60 秒之间。";
  if (
    config.fidelity === "stepwise"
    && config.sixdof_mode === "stepwise"
    && Math.max(config.aero_step_duration, config.sixdof_duration) > 10
  ) return "逐步 VSPAERO 与 JSBSim 联算时，两项设置时长的较大值不能超过 10 秒。";
  if (config.aero_step_s < 1 / 120 || config.aero_step_s > 0.25) return "VSPAERO 积分步长应在 0.008333–0.25 秒之间。";
  if (config.sixdof_step_s < 1 / 120 || config.sixdof_step_s > 0.25) return "JSBSim 积分步长应在 0.008333–0.25 秒之间。";
  for (const [key, range] of Object.entries(doeRanges)) {
    const [low, high] = config.bounds[key] || [];
    if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) return `${featureNames[key][0]}范围应为有效数值，且下限小于上限。`;
    if (low < range[0] || high > range[1]) return `${featureNames[key][0]}超出 DOE 数据范围。`;
  }
  const designLimits = {
    wing_area_m2: [22, 40],
    aspect_ratio: [18, 31],
    taper_ratio: [0.3, 0.7],
    sweep_deg: [0, 8],
    tip_twist_deg: [-3, 1.5],
    fuselage_length_m: [7.5, 10],
    fuselage_diameter_m: [0.65, 0.95],
    propeller_diameter_m: [2.1, 3],
    payload_mass_kg: [100, 205],
    hybrid_system_mass_kg: [0, 180],
  };
  for (const [key, [minimum, maximum]] of Object.entries(designLimits)) {
    const value = config.baseline_design[key];
    if (!Number.isFinite(value) || value < minimum || value > maximum) return `基准机体参数 ${key} 超出允许范围。`;
    if (doeRanges[key] && (value < doeRanges[key][0] || value > doeRanges[key][1])) return `基准机体参数 ${featureNames[key][0]} 超出当前 DOE 数据范围。`;
    if (config.bounds[key] && (value < config.bounds[key][0] || value > config.bounds[key][1])) return `基准机体参数 ${featureNames[key][0]} 应位于当前优化范围内。`;
  }
  if (config.resolution < 1 || config.resolution > 12) return "分辨率 H 应在 1–12 之间。";
  if (config.generations < 1 || config.generations > 500) return "迭代代数应在 1–500 之间。";
  if (config.battery_soc_min >= config.battery_soc_max) return "电池 SOC 下限必须小于上限。";
  return "";
}

function renderAircraftDerived() {
  const values = getConfig().baseline_design;
  const { wing_area_m2: area, aspect_ratio: ar, taper_ratio: taper } = values;
  const span = Math.sqrt(area * ar);
  const rootChord = 2 * area / (span * (1 + taper));
  const tipChord = rootChord * taper;
  const mac = (2 / 3) * rootChord * (1 + taper + taper ** 2) / (1 + taper);
  const tailArm = 0.52 * values.fuselage_length_m;
  const horizontalTail = 0.65 * area * mac / tailArm;
  const verticalTail = 0.075 * area;
  const mtowKg = 3000 * 0.45359237;
  const baselinePayloadKg = 400 * 0.45359237;
  const referenceFuelKg = 162 * 3.785411784 * 0.8;
  const planformFactor = (area / 26) ** 0.9 * (ar / 24.02) ** 0.4;
  const fuselageFactor = (
    values.fuselage_length_m * values.fuselage_diameter_m / (8.38 * 0.75)
  ) ** 0.85;
  const propellerFactor = (values.propeller_diameter_m / 2.4) ** 1.7;
  const massDelta = 150 * (planformFactor - 1)
    + 80 * (fuselageFactor - 1)
    + 15 * (propellerFactor - 1);
  const conventionalMass = mtowKg - referenceFuelKg - baselinePayloadKg + massDelta;
  const estimatedNonfuel = conventionalMass
    + values.payload_mass_kg
    + values.hybrid_system_mass_kg;
  const estimatedFuel = mtowKg - estimatedNonfuel;
  const clMax = Math.max(
    1.05,
    Math.min(
      1.55,
      1.42 - 0.012 * Math.abs(values.tip_twist_deg)
        - 0.001 * Math.max(values.sweep_deg - 5, 0),
    ),
  );
  const metrics = [
    ["翼展", span, "m"],
    ["翼根弦长", rootChord, "m"],
    ["翼尖弦长", tipChord, "m"],
    ["平均气动弦长", mac, "m"],
    ["平尾面积估算", horizontalTail, "m²"],
    ["垂尾面积估算", verticalTail, "m²"],
    ["机身长细比", values.fuselage_length_m / values.fuselage_diameter_m, "—"],
    ["机翼载荷（按 3000 lb）", mtowKg * 9.80665 / area, "N/m²"],
    ["失速速度估算", Math.sqrt(2 * mtowKg * 9.80665 / (1.225 * area * clMax)), "m/s"],
    ["非燃油质量估算", estimatedNonfuel, "kg"],
    ["剩余燃油估算", estimatedFuel, "kg"],
    ["电池能量估算", values.hybrid_system_mass_kg * 0.792 / 3.6, "kWh"],
  ];
  $("#aircraftDerived").replaceChildren(...metrics.map(([label, value, unit]) => {
    const item = document.createElement("div");
    item.className = "derived-item";
    item.append(document.createTextNode(label));
    const strong = document.createElement("strong");
    strong.textContent = `${formatNumber(value, 3)} ${unit}`;
    item.append(strong);
    return item;
  }));
}

function renderPowertrainReference(reference = {}) {
  powertrainVersion = reference.source_version || null;
  const items = [
    ["动力系统版本", reference.source_version || "", "动能学院 1005"],
    ["发动机额定机械功率", `${formatNumber(reference.engine_rated_mechanical_kw, 2)} kW`, "实际油耗由高度、转速及节气门查表"],
    ["发电机额定交流功率", `${formatNumber(reference.generator_rated_ac_kw, 2)} kW`, `发动机设计转速 ${formatNumber(reference.engine_design_speed_rpm, 0)} rpm`],
    ["电机单支路连续功率", `${formatNumber(reference.motor_rated_kw_per_branch, 2)} kW`, `${reference.motor_product || ""} · 输入上限 ${formatNumber(reference.motor_dc_input_max_kw_per_branch, 0)} kW`],
    ["电池包原始参数", `${formatNumber(reference.battery_nominal_energy_kwh, 2)} kWh`, `${formatNumber(reference.battery_series_cells, 0)} 串 ${formatNumber(reference.battery_parallel_cells, 0)} 并 · ${formatNumber(reference.battery_capacity_ah, 2)} Ah · ${formatNumber(reference.battery_nominal_voltage_v, 1)} V`],
    ["默认发动机图谱", reference.engine_map_name || "", `${formatNumber(reference.engine_map_total_points, 0)} 个工况 · 高度 0 至 5000 m`],
  ];
  $("#powertrainReference").replaceChildren(...items.map(([label, value, note]) => {
    const item = document.createElement("div");
    item.className = "reference-item";
    item.append(document.createTextNode(label));
    const strong = document.createElement("strong");
    strong.textContent = value;
    const small = document.createElement("small");
    small.textContent = note;
    item.append(strong, small);
    return item;
  }));
}

let cachedModelConfigs = JSON.parse($("#modelData").textContent).model_configs;

function selectModelConfig(precision) {
  const model = cachedModelConfigs[precision];
  if (!model) {
    $("#modelValidation").textContent = "所选模型的精度结果尚未发布。";
    return;
  }
  setBounds(model.doe_ranges);
  const validation = model.validation || {};
  const container = $("#modelValidation");
  container.replaceChildren();
  const table = document.createElement("table");
  table.className = "validation-table";
  const header = table.createTHead().insertRow();
  for (const title of ["独立测试", "MAE", "R²"]) {
    const cell = document.createElement("th");
    cell.textContent = title;
    header.append(cell);
  }
  const body = table.createTBody();
  for (const [index, label] of ["升力斜率 / rad⁻¹", "尾迹效率"].entries()) {
    const metric = validation.metrics?.[index];
    const row = body.insertRow();
    for (const value of [label, metric ? Number(metric.mae).toFixed(3) : "待评估",
      metric ? Number(metric.r2).toFixed(3) : "待评估"]) {
      row.insertCell().textContent = value;
    }
  }
  container.append(table);
  updateRuntimeEstimate();
}

async function loadConfig(precision = form.elements.namedItem("precision").value) {
  $("#today").textContent = new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium" }).format(new Date());
  try {
    const response = await fetch(`/api/config?precision=${encodeURIComponent(precision)}`);
    const data = await response.json();
    runtimeBaselines = data.runtime_baselines || runtimeBaselines;
    cachedModelConfigs = data.model_configs || {};
    selectModelConfig(form.elements.namedItem("precision").value);
    if (!baselineDesignLoaded) {
      for (const [key, value] of Object.entries(data.baseline_design || {})) {
        const input = form.elements.namedItem(`baseline_${key}`);
        if (input) input.value = value;
      }
      for (const [key, value] of Object.entries(data.defaults || {})) {
        const input = form.elements.namedItem(key);
        if (input && value !== undefined) input.value = value;
      }
      baselineDesignLoaded = true;
    }
    renderAircraftDerived();
    renderPowertrainReference(data.powertrain_reference);
    const missing = Object.entries(data.runtime.modules).filter(([, present]) => !present).map(([name]) => name);
    const ready = missing.length === 0;
    $("#runtimeDot").className = `status-dot ${ready ? "ok" : "bad"}`;
    $("#runtimeStatus").textContent = `${data.runtime.python_version} · ${ready ? "数值与 JSBSim 依赖就绪" : `缺少运行依赖：${missing.join(", ")}`} · OpenVSP ${data.runtime.vsp_python_exists ? "可用" : "未检测到"}`;
    $("#runEstimate").textContent = ready ? "运行环境就绪" : "请先检查运行环境";
    updateRuntimeEstimate();
    await loadRuns();
  } catch (error) {
    $("#runtimeDot").className = "status-dot bad";
    $("#runtimeStatus").textContent = `本地服务连接失败：${error.message}`;
    if (!cachedModelConfigs[form.elements.namedItem("precision").value]) {
      $("#modelValidation").textContent = "模型精度数据读取失败，请刷新页面重试。";
    }
  }
}

function setRunState(status, message, id = "", progress = {}) {
  const pill = $("#runStatusPill");
  pill.textContent = { idle: "空闲", starting: "启动中", running: "运行中", completed: "已完成", failed: "失败" }[status] || status;
  pill.className = `status-pill ${status}`;
  $("#runId").textContent = id || "尚未提交任务";
  const detail = progress.detail || message;
  $("#statusMessage").textContent = progress.stage && ["starting", "running"].includes(status)
    ? `${progress.stage} · ${detail}` : detail;
  const bar = $("#progressBar");
  const percent = status === "completed" ? 100 : Math.max(0, Math.min(100, Number(progress.percent) || 0));
  bar.className = status === "starting" ? "indeterminate" : "";
  bar.style.width = `${percent}%`;
  bar.style.background = status === "failed" ? "#bb514b" : "";
  $("#progressPercent").textContent = `${percent}%`;
  $(".progress-track").setAttribute("aria-valuenow", String(percent));
}

async function submitRun(event) {
  event.preventDefault();
  $("#formError").textContent = "";
  const config = getConfig();
  const error = validate(config);
  if (error) {
    $("#formError").textContent = error;
    return;
  }
  renderConfig(config);
  $("#runButton").disabled = true;
  setRunState("starting", "正在提交配置并启动计算任务…");
  $("#metricFeasible").textContent = "—";
  $("#metricEndurance").textContent = "—";
  $("#metricFuel").textContent = "—";
  $("#metricSoc").textContent = "—";
  $("#algorithmTable").innerHTML = '<tr><td colspan="4" class="empty-cell">计算中…</td></tr>';
  $("#runLog").textContent = "等待优化器输出…";
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "任务提交失败");
    activeRunId = data.id;
    await loadRuns();
    $("#runHistory").value = activeRunId;
    setRunState(data.status, "任务已创建，正在启动求解器。", activeRunId, data.progress);
    await refreshRun();
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(refreshRun, 1800);
  } catch (error) {
    setRunState("failed", error.message);
    $("#formError").textContent = error.message;
    $("#runButton").disabled = false;
  }
}

async function refreshRun() {
  if (!activeRunId) return;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(activeRunId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    const algorithmText = (data.config?.algorithms || [data.result?.candidate?.optimizer || ""]).join(" + ");
    const controller = (data.config?.controller || data.result?.candidate?.controller || "mpc").toUpperCase();
    const fidelity = data.config?.fidelity || data.result?.summary?.openvsp_vspaero_candidate_aerodynamics?.mode || "已保存结果";
    renderConfig(data.config || {});
    const stateText = data.status === "running" ? `正在运行：${algorithmText} · ${controller} · ${fidelity}` :
      data.status === "completed" ? (data.result?.summary?.message || "优化、Pareto 复评与择优候选独立回放已完成。") :
      data.status === "failed" ? `运行失败（退出码 ${data.return_code ?? "未知"}），请查看运行日志。` : "正在准备运行环境…";
    setRunState(data.status, stateText, data.id, data.progress);
    $("#runLog").textContent = data.log_tail || "等待优化器输出…";
    $("#runButton").disabled = ["starting", "running"].includes(data.status);
    if (data.status === "completed" && data.result) renderResult(data.result);
    if (data.status === "failed" || data.status === "completed") {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (error) {
    setRunState("failed", error.message, activeRunId);
    $("#runButton").disabled = false;
  }
}

async function loadRuns() {
  try {
    const response = await fetch("/api/runs");
    const data = await response.json();
    const selector = $("#runHistory");
    const selected = selector.value;
    selector.replaceChildren(new Option("当前会话", ""));
    for (const run of data.runs || []) {
      const date = run.created?.replace("T", " ") || "";
      const state = { completed: "完成", running: "运行中", starting: "启动中", failed: "失败" }[run.status] || run.status;
      selector.add(new Option(`${date} · ${state} · ${run.id.slice(-5)}`, run.id));
    }
    if (selected && [...selector.options].some((option) => option.value === selected)) selector.value = selected;
  } catch {
    // The run form remains usable if history cannot be loaded.
  }
}

async function selectRun(runId) {
  if (!runId) {
    activeRunId = null;
    setRunState("idle", "配置参数后启动优化。运行日志和结果将在此处更新。");
    return;
  }
  activeRunId = runId;
  await refreshRun();
}

function renderResult(result) {
  const candidate = result.candidate || {};
  const replay = candidate.independent_replay || {};
  $("#metricFeasible").textContent = result.summary?.outcome === "no_feasible_candidate"
    ? "无可行候选" : replay.feasible ? "通过" : "未通过";
  $("#metricFeasible").style.color = replay.feasible ? "var(--green)" : "var(--red)";
  $("#metricAlgorithm").textContent = `${candidate.optimizer || "—"} · ${candidate.controller || "—"}`;
  $("#metricEndurance").textContent = `${formatNumber(replay.achieved_total_duration_h, 1)} h`;
  $("#metricFuel").textContent = `${formatNumber(replay.fuel_used_kg, 1)} kg`;
  $("#metricSoc").textContent = `${formatNumber(Number(replay.terminal_soc) * 100, 1)}%`;
  const responseDefinition = result.summary?.response_disturbance?.objective_definition || "legacy";
  $("#resultMetricBasis").textContent = responseDefinition === "recovery-integral"
    ? "本次结果的响应目标采用恢复时间与扰动误差积分之和，各算法共用坐标范围。"
    : "本次结果采用历史响应定义，包含恢复时间与模式切换时间，各算法共用坐标范围。";
  $("#resultMetricBasis").textContent += ` 动力模型 ${result.summary?.powertrain_source?.source_version || "历史版本"}。`;
  renderAlgorithms(result.algorithms || []);
  renderPareto(result.pareto || {});
}

function renderAlgorithms(rows) {
  const body = $("#algorithmTable");
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="4" class="empty-cell">结果摘要中没有算法记录</td></tr>';
    return;
  }
  body.innerHTML = rows.map((row) => `<tr>
    <td>${row.algorithm || "—"}</td>
    <td>${row.final_population_feasible_count ?? "—"} / ${row.population_size ?? "—"}</td>
    <td>${row.historical_feasible_pareto_count ?? "—"}</td>
    <td>${formatNumber(row.runtime_s, 1)} s</td>
  </tr>`).join("");
}

function renderPareto(groups) {
  paretoGroups = groups;
  const selector = $("#paretoAlgorithm");
  const previous = selector.value;
  selector.replaceChildren();
  for (const [algorithm, rows] of Object.entries(groups)) {
    if (rows.length) selector.add(new Option(
      algorithm === "PTDL-Block-Adaptive" ? "PTDL 自适应分块" : algorithm, algorithm));
  }
  if ([...selector.options].some((option) => option.value === previous)) selector.value = previous;
  drawPareto();
}

function drawPareto() {
  const canvas = $("#paretoChart");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width, height = rect.height;
  context.clearRect(0, 0, width, height);
  const algorithm = $("#paretoAlgorithm").value;
  const rows = (paretoGroups[algorithm] || []).map((row) => ({
    row,
    values: [Number(row.J_energy_mj), Number(row.J_stability), Number(row.J_response_s)],
  })).filter((item) => item.values.every(Number.isFinite));
  chartPoints = [];
  if (!rows.length) {
    context.fillStyle = "#829095";
    context.font = '12px "Microsoft YaHei UI", sans-serif';
    context.textAlign = "center";
    context.fillText("暂无可绘制的三目标可行 Pareto 数据", width / 2, height / 2);
    return;
  }
  const shared = Object.values(paretoGroups).flat().map((row) =>
    [Number(row.J_energy_mj), Number(row.J_stability), Number(row.J_response_s)]
  ).filter((values) => values.every(Number.isFinite));
  const minima = [0, 1, 2].map((axis) => Math.min(...shared.map((values) => values[axis])));
  const maxima = [0, 1, 2].map((axis) => Math.max(...shared.map((values) => values[axis])));
  const spans = minima.map((min, axis) => maxima[axis] - min || 1);
  const center = { x: width * 0.52, y: height * 0.53 };
  const scale = Math.min(width * 0.31, height * 0.35);
  const rotate = ([x, y, z]) => {
    const cy = Math.cos(plotYaw), sy = Math.sin(plotYaw);
    const cp = Math.cos(plotPitch), sp = Math.sin(plotPitch);
    const x1 = x * cy - z * sy;
    const z1 = x * sy + z * cy;
    return [x1, y * cp - z1 * sp, y * sp + z1 * cp];
  };
  const project = (point) => {
    const [x, y, depth] = rotate(point);
    return { x: center.x + scale * x, y: center.y - scale * y, depth };
  };
  const corners = [
    [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
  ].map(project);
  const edgePairs = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]];
  context.strokeStyle = "#dce4e4";
  context.lineWidth = 1;
  for (const [a, b] of edgePairs) {
    context.beginPath();
    context.moveTo(corners[a].x, corners[a].y);
    context.lineTo(corners[b].x, corners[b].y);
    context.stroke();
  }
  context.font = '9px "Microsoft YaHei UI", sans-serif';
  context.fillStyle = "#78868b";
  context.textAlign = "center";
  const axisLabels = ["能量 (MJ)", "跟踪误差", "响应 (s)"];
  for (let axis = 0; axis < 3; axis += 1) {
    const endVector = [-1, -1, -1];
    endVector[axis] = 1;
    const end = project(endVector);
    context.fillStyle = "#819095";
    const offsets = [[24, 30], [0, -18], [-16, 35]];
    const labelHalfWidth = context.measureText(axisLabels[axis]).width / 2;
    context.fillText(axisLabels[axis],
      Math.max(labelHalfWidth + 4, Math.min(width - labelHalfWidth - 4, end.x + offsets[axis][0])),
      Math.max(12, Math.min(height - 4, end.y + offsets[axis][1])));
    const ticks = maxima[axis] === minima[axis] ? [2] : [0, 2, 4];
    for (const tick of ticks) {
      const value = maxima[axis] === minima[axis]
        ? minima[axis]
        : minima[axis] + spans[axis] * tick / 4;
      const position = [-1, -1, -1];
      position[axis] = -1 + tick / 2;
      const p = project(position);
      context.fillStyle = "#8a969a";
      context.fillText(formatNumber(value, axis === 0 ? 0 : 2), p.x, p.y + (axis === 1 ? -5 : 13));
    }
  }
  const plotted = rows.map(({ row, values }) => {
    const normalized = values.map((value, axis) => maxima[axis] === minima[axis]
      ? 0
      : -1 + 2 * (value - minima[axis]) / spans[axis]);
    return { row, values, point: project(normalized) };
  }).sort((a, b) => a.point.depth - b.point.depth);
  for (const item of plotted) {
    const radius = 3.8 + (item.point.depth + 1) * 0.65;
    context.beginPath();
    context.arc(item.point.x, item.point.y, radius, 0, Math.PI * 2);
    context.fillStyle = "#198b7e";
    context.globalAlpha = 0.82;
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle = "#fff";
    context.lineWidth = 1;
    context.stroke();
    chartPoints.push({ ...item, radius });
  }
  canvas.setAttribute("aria-label", `${algorithm} 三目标 Pareto 分布，${rows.length} 个可行解`);
}

form.addEventListener("submit", submitRun);
$("#resetBounds").addEventListener("click", () => setBounds(doeRanges, true));
$("#refreshButton").addEventListener("click", refreshRun);
$("#runHistory").addEventListener("change", (event) => selectRun(event.target.value));
$("#paretoAlgorithm").addEventListener("change", drawPareto);
$("#paretoChart").addEventListener("pointerdown", (event) => {
  dragPoint = { x: event.clientX, y: event.clientY };
  event.currentTarget.setPointerCapture(event.pointerId);
});
$("#paretoChart").addEventListener("pointermove", (event) => {
  const canvas = event.currentTarget;
  if (dragPoint) {
    const dx = event.clientX - dragPoint.x, dy = event.clientY - dragPoint.y;
    plotYaw += dx * 0.008;
    plotPitch = Math.max(-1.35, Math.min(1.35, plotPitch + dy * 0.008));
    dragPoint = { x: event.clientX, y: event.clientY };
    drawPareto();
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  const hit = [...chartPoints].reverse().find((item) => Math.hypot(item.point.x - x, item.point.y - y) <= item.radius + 5);
  canvas.title = hit
    ? `任务能量 ${formatNumber(hit.values[0], 2)} MJ · 轨迹跟踪误差 ${formatNumber(hit.values[1], 3)} · 响应目标 ${formatNumber(hit.values[2], 2)} s`
    : "拖动旋转视角；悬停查看目标值";
});
$("#paretoChart").addEventListener("pointerup", () => { dragPoint = null; });
$("#paretoChart").addEventListener("pointercancel", () => { dragPoint = null; });
window.addEventListener("resize", drawPareto);
  form.addEventListener("change", () => {
  const config = getConfig();
  $("#runEstimate").textContent = `${config.algorithms.length} 种算法 · H=${config.resolution} · ${config.generations} 代`;
  updateModeControls();
  updateRuntimeEstimate();
});
for (const name of Object.keys(selectHelp)) {
  form.elements.namedItem(name).addEventListener("change", () => {
    updateSelectHelp(name);
    updateModeControls();
    updateRuntimeEstimate();
  });
}
form.addEventListener("input", (event) => {
  updateRuntimeEstimate();
  if (event.target.name?.startsWith("baseline_")) renderAircraftDerived();
});
form.querySelector('[name="precision"]').addEventListener("change", (event) => selectModelConfig(event.target.value));
updateAllSelectHelp();
updateModeControls();
updateRuntimeEstimate();
selectModelConfig(form.elements.namedItem("precision").value);
loadConfig();
