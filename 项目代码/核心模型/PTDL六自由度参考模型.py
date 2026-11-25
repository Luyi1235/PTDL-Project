"""PTDL preliminary 6-DoF parameter set based on public ULTRA scale anchors.

All values not explicitly identified as public geometry/mass anchors are
engineering assumptions for software integration and sensitivity studies.
They are not measured, calibrated, certified, or flight-validated values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PTDL6DoFParameters:
    # 2024 public ULTRA scale anchors; wing area and operating point are assumptions.
    wingspan_m: float = 24.9936
    mtow_kg: float = 1360.78
    fuel_capacity_l: float = 613.2367
    reference_wing_area_m2: float = 26.0
    fuselage_length_m: float = 8.38
    cruise_speed_mps: float = 36.011
    reference_altitude_m: float = 3000.0

    # Preliminary mass and inertia allocation.
    fuel_density_kg_l: float = 0.80
    empty_mass_including_payload_kg: float = 870.19
    ixx_kg_m2: float = 42520.0
    iyy_kg_m2: float = 9710.0
    izz_kg_m2: float = 52230.0

    # Estimated low-speed aerodynamic coefficients (per radian where relevant).
    cd0: float = 0.022
    oswald_efficiency: float = 0.84
    cl0: float = 0.25
    cl_alpha_per_rad: float = 5.20
    cy_beta_per_rad: float = -0.55
    cl_beta_per_rad: float = -0.08
    cl_p: float = -0.45
    cl_r: float = 0.08
    cl_aileron_per_rad: float = 0.12
    cm0: float = 0.0954
    cm_alpha_per_rad: float = -0.80
    cm_q: float = -8.0
    cm_elevator_per_rad: float = -1.10
    cn_beta_per_rad: float = 0.12
    cn_r: float = -0.18
    cn_rudder_per_rad: float = -0.10
    cy_rudder_per_rad: float = 0.18

    # Estimated actuator and propulsion interface assumptions.
    elevator_limit_deg: float = 20.0
    aileron_limit_deg: float = 18.0
    rudder_limit_deg: float = 22.0
    elevator_time_constant_s: float = 0.25
    aileron_time_constant_s: float = 0.20
    rudder_time_constant_s: float = 0.25
    propulsive_efficiency: float = 0.86
    hybrid_shaft_power_max_kw: float = 80.0
    propulsion_time_constant_s: float = 2.5

    @property
    def fuel_mass_kg(self) -> float:
        return self.fuel_capacity_l * self.fuel_density_kg_l

    @property
    def calculated_takeoff_mass_kg(self) -> float:
        return self.empty_mass_including_payload_kg + self.fuel_mass_kg

    @property
    def mass_closure_error_kg(self) -> float:
        return self.calculated_takeoff_mass_kg - self.mtow_kg


def write_parameter_manifest(path: str | Path) -> Path:
    """Write a JSON manifest with explicit provenance labels and mass closure."""
    params = PTDL6DoFParameters()
    payload = {
        "model_id": "PTDL-EST-6DOF-ULTRA-SCALE",
        "status": "assumption_only_not_measured_not_calibrated_not_flight_validated",
        "public_scale_anchors": {
            "wingspan_m": params.wingspan_m,
            "mtow_kg": params.mtow_kg,
            "fuel_capacity_l": params.fuel_capacity_l,
        },
        "assumption_parameters": asdict(params),
        "derived_checks": {
            "fuel_mass_kg": params.fuel_mass_kg,
            "calculated_takeoff_mass_kg": params.calculated_takeoff_mass_kg,
            "mass_closure_error_kg": params.mass_closure_error_kg,
            "inertia_uncertainty_sensitivity_fraction": 0.30,
            "aerodynamic_derivative_sensitivity_fraction": 0.30,
            "actuator_time_constant_sensitivity_fraction": 0.50,
        },
        "source_notes": [
            "DZYNE 2024 ULTRA产品卡用于翼展、最大起飞重量和油量的量级锚定。",
            "36.011 m/s（70 kt）仅为本项目选定的巡航参考工况，不是公开资料给出的ULTRA巡航速度；DZYNE 2026产品目录列出的ULTRA最大速度为96 kt。",
            "AFRL公开页面给出ULTRA续航超过80 h、载荷超过400 lb；DZYNE 2026产品目录给出ULTRA 70+ h和450 lb。不同资料版本/构型的任务指标不直接当作PTDL验收值。",
            "翼面积、空机/载荷质量分配、惯量、三轴气动导数和舵机动态均为PTDL暂估。",
            "功率到推力通过螺旋桨效率及空速换算；80 kW仅为项目现有预估接口值。",
            "此数据集只用于软件接入、仿真流程和敏感性分析，不能作为真实性能或安全证据。",
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "运行结果" / "PTDL六自由度估计验证" / "PTDL_六自由度参数清单.json"
    write_parameter_manifest(destination)
    print(destination)
