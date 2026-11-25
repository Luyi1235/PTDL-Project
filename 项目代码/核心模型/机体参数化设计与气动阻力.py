"""Dimension-driven airframe model for the PTDL ULTRA benchmark.

The module keeps geometry variables physical.  Drag polar coefficients and
Oswald efficiency are *outputs* of geometry, Reynolds number, and calibrated
low-order aero relations; they are never optimization variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path

import numpy as np

from 飞推控闭环物理模型 import AircraftParameters, isa_density, turbulent_skin_friction_coefficient
from ULTRA基准标定 import UltraCalibration, UltraOpenData


G = 9.80665
AIR_DYNAMIC_VISCOSITY = 1.789e-5


@dataclass(frozen=True)
class AirframeDesign:
    """Independent dimensions and fixed-MTOW allocation inputs.

    ``fuel_mass_kg`` is deliberately not an independent variable.  It is the
    residual mass after the dimension-derived conventional mass, payload, and
    hybrid-system mass are assigned at the declared MTOW.  This prevents the
    optimizer from reporting an apparently good endurance result by silently
    operating below the takeoff-mass limit.
    """

    wing_area_m2: float = 26.0
    aspect_ratio: float = 24.02
    taper_ratio: float = 0.45
    sweep_deg: float = 1.0
    tip_twist_deg: float = 0.0
    fuselage_length_m: float = 8.38
    fuselage_diameter_m: float = 0.75
    propeller_diameter_m: float = 2.40
    payload_mass_kg: float = 181.44
    hybrid_system_mass_kg: float = 0.0


@dataclass(frozen=True)
class AirframeMetrics:
    wing_span_m: float
    root_chord_m: float
    tip_chord_m: float
    mean_aerodynamic_chord_m: float
    horizontal_tail_area_m2: float
    vertical_tail_area_m2: float
    structural_and_conventional_mass_kg: float
    payload_mass_kg: float
    hybrid_system_mass_kg: float
    estimated_nonfuel_mass_kg: float
    fuel_mass_kg: float
    battery_capacity_mj: float
    wing_loading_npm2: float
    stall_speed_mps: float
    fuel_volume_margin_l: float
    mtow_margin_kg: float


@dataclass(frozen=True)
class AeroCoefficients:
    cd0: float
    oswald_efficiency: float
    cl_alpha_per_rad: float
    cl_max: float
    propulsive_efficiency: float
    reference_reynolds: float


BASELINE = AirframeDesign()


def design_metrics(design: AirframeDesign, reference: UltraOpenData = UltraOpenData()) -> AirframeMetrics:
    _validate_design(design)
    span = sqrt(design.wing_area_m2 * design.aspect_ratio)
    root_chord = 2.0 * design.wing_area_m2 / (span * (1.0 + design.taper_ratio))
    tip_chord = root_chord * design.taper_ratio
    mac = (2.0 / 3.0) * root_chord * (1.0 + design.taper_ratio + design.taper_ratio**2) / (1.0 + design.taper_ratio)

    # Tail dimensions are derived through conventional tail-volume assumptions.
    tail_arm = 0.52 * design.fuselage_length_m
    htail_area = 0.65 * design.wing_area_m2 * mac / tail_arm
    vtail_area = 0.075 * design.wing_area_m2

    # Geometry-dependent mass adjustment.  The baseline conventional mass
    # excludes the declared payload and the incremental hybrid system.  The
    # scaling is a transparent preliminary estimate, to be replaced by a
    # structural sizing model when spar, skin, and landing-gear data exist.
    planform_factor = (design.wing_area_m2 / BASELINE.wing_area_m2) ** 0.90 * (design.aspect_ratio / BASELINE.aspect_ratio) ** 0.40
    fuselage_factor = (design.fuselage_length_m * design.fuselage_diameter_m / (BASELINE.fuselage_length_m * BASELINE.fuselage_diameter_m)) ** 0.85
    prop_factor = (design.propeller_diameter_m / BASELINE.propeller_diameter_m) ** 1.70
    mass_delta = 150.0 * (planform_factor - 1.0) + 80.0 * (fuselage_factor - 1.0) + 15.0 * (prop_factor - 1.0)
    baseline_conventional_mass = reference.nonfuel_mass_kg - BASELINE.payload_mass_kg
    conventional_mass = baseline_conventional_mass + mass_delta
    nonfuel_mass = conventional_mass + design.payload_mass_kg + design.hybrid_system_mass_kg
    fuel_mass = reference.mtow_kg - nonfuel_mass
    total_takeoff_mass = nonfuel_mass + fuel_mass
    mtow_margin = reference.mtow_kg - total_takeoff_mass

    cl_max = _cl_max(design)
    stall_speed = sqrt(2.0 * total_takeoff_mass * G / (1.225 * design.wing_area_m2 * cl_max))
    wing_loading = total_takeoff_mass * G / design.wing_area_m2

    # Approximate usable internal wing volume.  It is a packaging check, not a
    # replacement for a real tank-layout model.
    mean_thickness_ratio = 0.12
    available_fuel_l = 1000.0 * 0.50 * design.wing_area_m2 * mac * mean_thickness_ratio * 0.58
    required_fuel_l = fuel_mass / reference.jet_a_density_kg_per_l
    return AirframeMetrics(
        wing_span_m=span,
        root_chord_m=root_chord,
        tip_chord_m=tip_chord,
        mean_aerodynamic_chord_m=mac,
        horizontal_tail_area_m2=htail_area,
        vertical_tail_area_m2=vtail_area,
        structural_and_conventional_mass_kg=conventional_mass,
        payload_mass_kg=design.payload_mass_kg,
        hybrid_system_mass_kg=design.hybrid_system_mass_kg,
        estimated_nonfuel_mass_kg=nonfuel_mass,
        fuel_mass_kg=fuel_mass,
        # System-level specific energy is provisional.  It represents battery
        # cells plus packaging and electrical integration, not cell-only data.
        battery_capacity_mj=max(1.0, design.hybrid_system_mass_kg * 0.792),
        wing_loading_npm2=wing_loading,
        stall_speed_mps=stall_speed,
        fuel_volume_margin_l=available_fuel_l - required_fuel_l,
        mtow_margin_kg=mtow_margin,
    )


def aero_from_design(
    design: AirframeDesign,
    calibration: UltraCalibration,
    speed_mps: float | None = None,
    altitude_m: float | None = None,
) -> AeroCoefficients:
    """Derive a calibrated drag polar from dimensions instead of free CD0/e."""
    reference = calibration.reference
    speed = reference.cruise_speed_mps if speed_mps is None else speed_mps
    altitude = reference.calibration_altitude_m if altitude_m is None else altitude_m
    metrics = design_metrics(design, reference)
    rho = isa_density(altitude)
    reynolds = rho * speed * metrics.mean_aerodynamic_chord_m / AIR_DYNAMIC_VISCOSITY

    raw_cd0 = _raw_cd0(design, metrics, rho, speed)
    baseline_metrics = design_metrics(BASELINE, reference)
    baseline_raw_cd0 = _raw_cd0(BASELINE, baseline_metrics, rho, reference.cruise_speed_mps)
    calibration_scale = calibration.cd0 / baseline_raw_cd0
    cd0 = raw_cd0 * calibration_scale

    e = _oswald_efficiency(design)
    cl_alpha_2d = 2.0 * pi
    cl_alpha = cl_alpha_2d / (1.0 + cl_alpha_2d / (pi * e * design.aspect_ratio))
    prop_eta = float(np.clip(0.86 - 0.035 * (design.propeller_diameter_m / BASELINE.propeller_diameter_m - 1.0) ** 2, 0.76, 0.90))
    return AeroCoefficients(
        cd0=float(cd0),
        oswald_efficiency=float(e),
        cl_alpha_per_rad=float(cl_alpha),
        cl_max=_cl_max(design),
        propulsive_efficiency=prop_eta,
        reference_reynolds=float(reynolds),
    )


def aircraft_from_design(
    design: AirframeDesign,
    calibration: UltraCalibration,
    aero_override: AeroCoefficients | None = None,
) -> AircraftParameters:
    """Create a coupled-model aircraft; optional VSPAERO trends may override the polar."""
    metrics = design_metrics(design, calibration.reference)
    aero = aero_override or aero_from_design(design, calibration)
    reference = calibration.reference
    return AircraftParameters(
        empty_mass_kg=metrics.estimated_nonfuel_mass_kg,
        initial_fuel_kg=max(0.0, metrics.fuel_mass_kg),
        wing_area_m2=design.wing_area_m2,
        wing_span_m=metrics.wing_span_m,
        cd0=aero.cd0,
        reference_chord_m=metrics.mean_aerodynamic_chord_m,
        parasite_drag_reference_reynolds=aero.reference_reynolds,
        oswald_efficiency=aero.oswald_efficiency,
        cl_alpha_per_rad=aero.cl_alpha_per_rad,
        cl_max=aero.cl_max,
        prop_disk_area_m2=pi * (design.propeller_diameter_m / 2.0) ** 2,
        propulsive_efficiency=aero.propulsive_efficiency,
        engine_electrical_power_max_kw=reference.assumed_engine_electrical_power_max_kw,
        engine_generator_efficiency=reference.assumed_engine_generator_efficiency,
        # The hybrid-system mass is mapped to preliminary power and energy
        # capability.  These two sizing correlations are placeholders for the
        # motor/inverter/battery supplier data required in the next model tier.
        motor_power_min_kw=-0.50 * design.hybrid_system_mass_kg,
        motor_power_max_kw=1.00 * design.hybrid_system_mass_kg,
        battery_capacity_mj=metrics.battery_capacity_mj,
        initial_soc=0.85 if design.hybrid_system_mass_kg > 0.0 else 0.50,
        soc_safe=0.15 if design.hybrid_system_mass_kg > 0.0 else 0.0,
        soc_reserve=0.20 if design.hybrid_system_mass_kg > 0.0 else 0.0,
    )


def geometry_constraints(design: AirframeDesign, reference: UltraOpenData = UltraOpenData()) -> dict[str, float]:
    """g <= 0 constraints for the geometry-level optimization problem."""
    metrics = design_metrics(design, reference)
    return {
        # 32 m is a declared early-study handling/infrastructure bound, not a
        # public ULTRA limit.  It keeps the expanded-wing study physically
        # finite until a transport, launch, and recovery concept is selected.
        "span_limit": metrics.wing_span_m - 32.0,
        "stall_speed": metrics.stall_speed_mps - 25.0,
        "fuel_volume": -metrics.fuel_volume_margin_l,
        "fuel_mass_min": 300.0 - metrics.fuel_mass_kg,
        "payload_min": 100.0 - metrics.payload_mass_kg,
        "hybrid_mass_max": metrics.hybrid_system_mass_kg - 220.0,
        "mtow_closure": abs(metrics.mtow_margin_kg) - 1e-6,
        "fuselage_fineness_min": 8.0 - design.fuselage_length_m / design.fuselage_diameter_m,
    }


def write_openvsp_geometry(
    design: AirframeDesign, path: str | Path, mesh_level: str = "legacy",
) -> AirframeMetrics:
    """Write a clean OpenVSP geometry file for the current dimensional design."""
    import openvsp as vsp

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = design_metrics(design)
    vsp.ClearVSPModel()
    try:
        fuselage_id = vsp.AddGeom("FUSELAGE")
        vsp.SetGeomName(fuselage_id, "Fuselage")
        vsp.SetParmVal(fuselage_id, "Length", "Design", design.fuselage_length_m)
        for index, location in enumerate((0.12, 0.35, 0.78), start=1):
            vsp.SetParmVal(fuselage_id, "XLocPercent", f"XSec_{index}", location)
            vsp.SetParmVal(fuselage_id, "Ellipse_Width", f"XSecCurve_{index}", design.fuselage_diameter_m)
            vsp.SetParmVal(fuselage_id, "Ellipse_Height", f"XSecCurve_{index}", design.fuselage_diameter_m)

        main_wing = _add_wing(
            vsp,
            "Main_Wing",
            metrics.wing_span_m,
            metrics.root_chord_m,
            metrics.tip_chord_m,
            design.sweep_deg,
            3.0,
            design.tip_twist_deg,
            0.32 * design.fuselage_length_m,
            0.15 * design.fuselage_diameter_m,
            symmetric=True,
        )
        htail_ar = 4.0
        htail_span = sqrt(htail_ar * metrics.horizontal_tail_area_m2)
        htail_root = 2.0 * metrics.horizontal_tail_area_m2 / (htail_span * 1.5)
        _add_wing(
            vsp,
            "Horizontal_Tail",
            htail_span,
            htail_root,
            0.5 * htail_root,
            8.0,
            0.0,
            0.0,
            0.80 * design.fuselage_length_m,
            0.35 * design.fuselage_diameter_m,
            symmetric=True,
        )
        vtail_ar = 1.5
        vtail_span = sqrt(vtail_ar * metrics.vertical_tail_area_m2)
        vtail_root = 2.0 * metrics.vertical_tail_area_m2 / (vtail_span * 1.6)
        _add_wing(
            vsp,
            "Vertical_Tail",
            vtail_span,
            vtail_root,
            0.6 * vtail_root,
            25.0,
            0.0,
            0.0,
            0.80 * design.fuselage_length_m,
            0.35 * design.fuselage_diameter_m,
            symmetric=False,
            vertical=True,
        )
        vsp.SetVSPAERORefWingID(main_wing)
        configure_geometry_mesh(vsp, mesh_level)
        vsp.Update()
        vsp.WriteVSPFile(str(path))
    finally:
        vsp.ClearVSPModel()
    return metrics


def configure_geometry_mesh(vsp: object, mesh_level: str) -> None:
    """Use explicit span/chord and body resolution without changing geometry."""
    levels = {
        "legacy": None,
        "medium": (13, 33, 25, 8),
        "fine": (25, 49, 33, 10),
        "verification": (37, 65, 41, 12),
    }
    if mesh_level not in levels:
        raise ValueError(f"Unknown geometry mesh level: {mesh_level}")
    settings = levels[mesh_level]
    if settings is None:
        return
    span, chord, body_circumference, body_section = settings
    for geom in vsp.FindGeoms():
        name = str(vsp.GetGeomName(geom))
        if name == "Fuselage":
            vsp.SetParmVal(geom, "Tess_W", "Shape", body_circumference)
            for section in range(5):
                vsp.SetParmVal(geom, "SectTess_U", f"XSec_{section}", body_section)
        else:
            vsp.SetParmVal(geom, "Tess_W", "Shape", chord)
            vsp.SetParmVal(geom, "SectTess_U", "XSec_1",
                           span if name == "Main_Wing" else max(9, (span + 1) // 2))


def _add_wing(
    vsp: object,
    name: str,
    span_m: float,
    root_chord_m: float,
    tip_chord_m: float,
    sweep_deg: float,
    dihedral_deg: float,
    twist_deg: float,
    x_location_m: float,
    z_location_m: float,
    symmetric: bool,
    vertical: bool = False,
) -> str:
    wing_id = vsp.AddGeom("WING")
    vsp.SetGeomName(wing_id, name)
    vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", root_chord_m)
    vsp.SetParmVal(wing_id, "Span", "XSec_1", span_m / 2.0 if symmetric else span_m)
    vsp.SetParmVal(wing_id, "Tip_Chord", "XSec_1", tip_chord_m)
    vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep_deg)
    vsp.SetParmVal(wing_id, "Dihedral", "XSec_1", dihedral_deg)
    vsp.SetParmVal(wing_id, "Twist", "XSec_1", twist_deg)
    vsp.SetParmVal(wing_id, "X_Rel_Location", "XForm", x_location_m)
    vsp.SetParmVal(wing_id, "Z_Rel_Location", "XForm", z_location_m)
    if vertical:
        vsp.SetParmVal(wing_id, "X_Rel_Rotation", "XForm", 90.0)
    if not symmetric:
        vsp.SetParmVal(wing_id, "Sym_Planar_Flag", "Sym", 0)
    return wing_id


def _raw_cd0(design: AirframeDesign, metrics: AirframeMetrics, rho_kgpm3: float, speed_mps: float) -> float:
    """Return a preliminary component-build-up parasite-drag coefficient.

    The build-up separates wing, fuselage, empennage and an equivalent
    propulsor-installation contribution. It is calibrated at the ULTRA
    baseline afterwards, so it supplies geometry and Reynolds-number trends
    rather than claiming an unvalidated absolute drag prediction.
    """
    speed = max(float(speed_mps), 1.0)
    reynolds_wing = rho_kgpm3 * speed * metrics.mean_aerodynamic_chord_m / AIR_DYNAMIC_VISCOSITY
    reynolds_fuselage = rho_kgpm3 * speed * design.fuselage_length_m / AIR_DYNAMIC_VISCOSITY
    tail_chord = max(0.45, 0.60 * metrics.mean_aerodynamic_chord_m)
    reynolds_tail = rho_kgpm3 * speed * tail_chord / AIR_DYNAMIC_VISCOSITY

    thickness_ratio = 0.12
    sweep_rad = np.deg2rad(design.sweep_deg)
    wing_wetted = 2.0 * design.wing_area_m2 * (1.0 + 0.25 * thickness_ratio)
    wing_form_factor = (1.0 + 2.7 * thickness_ratio + 100.0 * thickness_ratio**4) * max(np.cos(sweep_rad), 0.75) ** -0.28

    fineness = design.fuselage_length_m / design.fuselage_diameter_m
    fuselage_wetted = pi * design.fuselage_diameter_m * design.fuselage_length_m * (1.0 - 2.0 / fineness) ** (2.0 / 3.0) * (1.0 + 1.0 / fineness**2)
    fuselage_form_factor = 1.0 + 60.0 / fineness**3 + 0.0025 * fineness

    tail_wetted = 2.0 * (metrics.horizontal_tail_area_m2 + metrics.vertical_tail_area_m2) * (1.0 + 0.25 * thickness_ratio)
    tail_form_factor = 1.0 + 2.7 * thickness_ratio + 100.0 * thickness_ratio**4

    # Equivalent hub, pylon and cooling-installation area. The coefficient is
    # deliberately modest; the final value must be replaced by installation
    # geometry/CFD or test data when the propulsion architecture is fixed.
    propulsor_installation_area = 0.018 * design.propeller_diameter_m**2
    interference_factor = 1.05
    drag_area = interference_factor * (
        turbulent_skin_friction_coefficient(reynolds_wing) * wing_form_factor * wing_wetted
        + turbulent_skin_friction_coefficient(reynolds_fuselage) * fuselage_form_factor * fuselage_wetted
        + turbulent_skin_friction_coefficient(reynolds_tail) * tail_form_factor * tail_wetted
        + turbulent_skin_friction_coefficient(reynolds_wing) * 1.35 * propulsor_installation_area
    )
    miscellaneous_cd0 = 0.0008
    return float(drag_area / design.wing_area_m2 + miscellaneous_cd0)


def _oswald_efficiency(design: AirframeDesign) -> float:
    taper_penalty = 0.10 * (design.taper_ratio - 0.45) ** 2
    sweep_penalty = 0.0006 * design.sweep_deg**2
    twist_penalty = 0.0015 * design.tip_twist_deg**2
    ar_benefit = 0.008 * np.log(design.aspect_ratio / BASELINE.aspect_ratio)
    return float(np.clip(0.86 + ar_benefit - taper_penalty - sweep_penalty - twist_penalty, 0.65, 0.92))


def _cl_max(design: AirframeDesign) -> float:
    return float(np.clip(1.42 - 0.012 * abs(design.tip_twist_deg) - 0.001 * max(design.sweep_deg - 5.0, 0.0), 1.05, 1.55))


def _validate_design(design: AirframeDesign) -> None:
    if not 20.0 <= design.wing_area_m2 <= 42.0:
        raise ValueError("wing_area_m2 must stay within the declared study range.")
    if not 16.0 <= design.aspect_ratio <= 32.0:
        raise ValueError("aspect_ratio must stay within the declared study range.")
    if not 0.25 <= design.taper_ratio <= 0.80:
        raise ValueError("taper_ratio must stay within the declared study range.")
    if not 0.0 <= design.sweep_deg <= 12.0:
        raise ValueError("sweep_deg must stay within the declared study range.")
    if not -4.0 <= design.tip_twist_deg <= 3.0:
        raise ValueError("tip_twist_deg must stay within the declared study range.")
    if not 7.0 <= design.fuselage_length_m <= 10.5:
        raise ValueError("fuselage_length_m must stay within the declared study range.")
    if not 0.60 <= design.fuselage_diameter_m <= 1.05:
        raise ValueError("fuselage_diameter_m must stay within the declared study range.")
    if not 2.0 <= design.propeller_diameter_m <= 3.2:
        raise ValueError("propeller_diameter_m must stay within the declared study range.")
    if not 80.0 <= design.payload_mass_kg <= 205.0:
        raise ValueError("payload_mass_kg must stay within the declared study range.")
    if not 0.0 <= design.hybrid_system_mass_kg <= 220.0:
        raise ValueError("hybrid_system_mass_kg must stay within the declared study range.")
