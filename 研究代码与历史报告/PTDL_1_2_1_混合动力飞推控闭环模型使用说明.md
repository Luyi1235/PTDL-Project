# PTDL 1.2.1 混合动力系统与飞行控制策略协同作用模型

## 1 模型定位

该模型是 PTDL 任务剖面和 MOEA/D 外层优化之间的内层仿真对象。其目标不是替代动能学院的部件模型，而是提供稳定的统一接口，使发动机、电机、电池和螺旋桨模型能够进入同一条“功率到推力到飞行状态到控制误差”的闭环链路。

当前默认模型为可运行的参考总成，发动机采用一阶响应和效率图占位模型，电池采用 SOC 能量积分模型，螺旋桨采用动量理论。默认数值不能作为 PTDL 最终动力系统设计参数。

## 2 闭环计算顺序

每个积分步执行以下顺序：

1. 任务阶段给出目标高度、目标速度、功率上限和最大航迹角。
2. 纵向控制器根据实际高度和速度误差生成航迹角、迎角和所需推进功率。
3. 协同调度器根据 SOC、误差和功率能力生成发动机目标电功率。
4. 混合动力总成计算发动机实际输出、电池充放电功率、SOC 和燃油流量。
5. 螺旋桨模型将实际可用功率转换为实际推力。
6. 点质量纵向动力学积分实际速度、航迹角、高度和航程。
7. 新的高度误差、速度误差和 SOC 在下一步反馈给控制与调度器。

因此，发动机或电机能力不足不再只记录为 `power_shortfall_kw`，而会造成实际速度或爬升能力下降；当速度已降至安全下限且仍存在功率缺额时，仿真以 `propulsion_power_shortfall` 终止。

## 3 动能学院发动机模型替换接口

仅替换发动机发电机模型时，实现 `EngineGeneratorModel` 的两个方法：

```python
class DynamicsSchoolEngine:
    def initial_state(self, aircraft):
        # 返回发动机内部初始状态，可包含转速、扭矩、温度和启停状态
        return engine_state

    def step(self, state, request, dt_s, aircraft):
        # request.target_electrical_power_kw 为控制器要求的电功率
        # request.power_cap_kw 为任务阶段与总成共同给出的功率上界
        # 返回实际发电功率和对应燃油流量
        return EngineGeneratorOutput(
            state=next_state,
            electrical_power_kw=actual_generator_power_kw,
            fuel_flow_kgps=fuel_flow_kgps,
        )
```

接入时调用：

```python
simulation, result = evaluate_coupled_mission(
    mission,
    aircraft,
    dispatch_policy=controller,
    engine_model=DynamicsSchoolEngine(...),
)
```

发动机模型需要至少给出由目标指令、边界和内部状态共同决定的实际电功率，以及对应的燃油流量。建议其内部保留转速、扭矩、节气门、BSFC 或效率图、发电机效率、功率爬升率、启停状态和热限制；这些内部状态不需要暴露给飞行仿真器。

## 4 完整混动总成替换接口

若动能学院后续提供串联、并联或功率分流构型的完整发动机、电机、电池、逆变器和传动模型，则实现 `HybridPowertrainModel`。其输出必须包含实际发动机功率、实际电池贡献功率、进入螺旋桨模型的可用功率、燃油流量、SOC 和电池电流。这样可替换默认等效直流母线模型，而无需改动任务剖面、飞行动力学、2.2.1 评价函数或 MOEA/D。

## 5 现有文件与职责

- `ptdl_flight_propulsion_control.py`：纵向动力学、发动机与完整总成接口、默认参考总成、飞行控制和仿真主循环。
- `ptdl_hybrid_coordination.py`：规则型 SOC 反馈功率分配控制器。
- `ptdl_intelligent_energy_management.py`：模糊自适应调度器、神经网络 SOC 参考接口和健康状态监督接口。
- `ptdl_80h_endurance_profile.py`：PTDL 80 h 任务剖面适配器，已支持传入 `engine_model` 或 `powertrain`。
- `ptdl_full_task_profile.py`：原始任务 JSON 适配器，已支持传入 `engine_model` 或 `powertrain`。
- `test_ptdl_hybrid_flight_control_dynamics.py`：验证替换为零输出发动机后，推进功率缺额会改变实际飞行状态。

## 6 当前验证与边界

以既有 H=6、VSPAERO 修正后的 MOEA/D 候选构型重跑 80 h 任务，阶段定值、规则 SOC 反馈和模糊自适应策略均完成 80 h。该结果证明闭环软件链路和约束评价链路可运行，不证明默认发动机、电池或螺旋桨参数已经代表最终 PTDL 构型。

当前默认模型已经包含实际功率到推力到纵向状态的反馈，但仍未包含完整的横滚、偏航、舵面执行机构、发动机真实启停过程、电池热与极化、螺旋桨桨距和推进器气动图。上述内容应由后续总体控制与动能模型逐步替换或扩展。
