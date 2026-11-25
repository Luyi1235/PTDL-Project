"""Matched-trajectory response metrics shared by optimization and validation."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

from 飞推控闭环物理模型 import LongitudinalDragDisturbance

RESPONSE_SPEED_TOLERANCE_MPS = 0.5
RESPONSE_ALTITUDE_TOLERANCE_FRACTION = 0.002


def zero_amplitude_disturbances(
    disturbances: Sequence[LongitudinalDragDisturbance],
) -> tuple[LongitudinalDragDisturbance, ...]:
    """Keep event timing and sampling while removing its physical input."""
    return tuple(
        replace(
            event,
            additional_drag_n=0.0,
            headwind_step_mps=0.0,
            headwind_gust_amplitude_mps=0.0,
        )
        for event in disturbances
    )


def project_response_stress_scenario(
    headwind_step_mps: float = 8.0,
) -> tuple[LongitudinalDragDisturbance, ...]:
    """Return the shared low-order wind-step screen used for PTDL comparisons."""
    if headwind_step_mps < 0.0:
        raise ValueError("headwind_step_mps must be non-negative.")
    return (
        LongitudinalDragDisturbance(
            phase_name="endurance_loiter",
            start_time_s=3600.0,
            duration_s=60.0,
            additional_drag_n=0.0,
            response_step_s=2.0,
            recovery_window_s=900.0,
            recovery_dwell_s=30.0,
            headwind_step_mps=float(headwind_step_mps),
        ),
    )


def paired_response_metrics(
    disturbed: pd.DataFrame,
    nominal: pd.DataFrame,
    phase_name: str,
    event_start_s: float,
    event_duration_s: float,
    recovery_window_s: float,
    recovery_dwell_s: float,
    altitude_tolerance_m: float,
    speed_tolerance_mps: float = RESPONSE_SPEED_TOLERANCE_MPS,
) -> dict[str, float | bool]:
    """Compare the disturbed trajectory with a matched nominal trajectory."""
    phase_rows = disturbed.index[disturbed["phase"] == phase_name]
    if len(phase_rows):
        first_index = int(phase_rows[0])
        phase_entry_s = 0.0 if first_index == 0 else float(disturbed.loc[first_index - 1, "time_s"])
    else:
        phase_entry_s = float("nan")

    event_start = phase_entry_s + event_start_s
    event_end = event_start + event_duration_s
    mission_end_s = min(float(disturbed["time_s"].max()), float(nominal["time_s"].max()))
    response_end = min(mission_end_s, event_end + recovery_window_s)
    terminal_phase_limited = response_end < event_end + recovery_window_s - 1e-9
    response = disturbed.loc[
        (disturbed["time_s"] >= event_start) & (disturbed["time_s"] <= response_end)
    ]
    if response.empty or not np.isfinite(phase_entry_s):
        return {
            "paired_recovery_time_s": recovery_window_s,
            "paired_recovery_success": False,
            "paired_peak_speed_deviation_mps": float("nan"),
            "paired_peak_altitude_deviation_m": float("nan"),
            "response_metric_valid": False,
            "terminal_phase_limited": terminal_phase_limited,
            "response_observation_window_s": 0.0,
            "response_excitation_exceeds_band": False,
        }

    times = response["time_s"].to_numpy(float)
    nominal_times = nominal["time_s"].to_numpy(float)
    delta_speed = response["v_mps"].to_numpy(float) - np.interp(
        times, nominal_times, nominal["v_mps"].to_numpy(float)
    )
    delta_altitude = response["h_m"].to_numpy(float) - np.interp(
        times, nominal_times, nominal["h_m"].to_numpy(float)
    )
    peak_speed = float(np.max(np.abs(delta_speed)))
    peak_altitude = float(np.max(np.abs(delta_altitude)))
    equivalent_error = np.sqrt(
        (delta_speed / speed_tolerance_mps)**2
        + (delta_altitude / altitude_tolerance_m)**2
    )
    error_integral = float(np.trapezoid(equivalent_error, times))

    recovery = response.loc[
        (response["time_s"] >= event_end) & (response["time_s"] <= response_end)
    ]
    recovery_times = recovery["time_s"].to_numpy(float)
    nominal_h = np.interp(recovery_times, nominal_times, nominal["h_m"].to_numpy(float))
    nominal_v = np.interp(recovery_times, nominal_times, nominal["v_mps"].to_numpy(float))
    in_band = (
        np.abs(recovery["h_m"].to_numpy(float) - nominal_h) <= altitude_tolerance_m
    ) & (
        np.abs(recovery["v_mps"].to_numpy(float) - nominal_v) <= speed_tolerance_mps
    )
    recovered_at = _first_dwell_start(recovery_times, in_band, recovery_dwell_s)
    return {
        "paired_recovery_time_s": (
            recovered_at - event_end if recovered_at is not None else recovery_window_s
        ),
        "paired_recovery_success": recovered_at is not None,
        "paired_peak_speed_deviation_mps": peak_speed,
        "paired_peak_altitude_deviation_m": peak_altitude,
        "response_metric_valid": bool(
            response_end >= event_end + recovery_dwell_s - 1e-9
        ),
        "terminal_phase_limited": terminal_phase_limited,
        "response_observation_window_s": max(0.0, response_end - event_end),
        "response_excitation_exceeds_band": bool(
            peak_speed > speed_tolerance_mps or peak_altitude > altitude_tolerance_m
        ),
        "response_normalized_error_integral_s": error_integral,
        "response_speed_tolerance_mps": speed_tolerance_mps,
        "response_altitude_tolerance_m": altitude_tolerance_m,
    }


def apply_paired_response_metrics(
    disturbed_evaluation: object,
    nominal_evaluation: object,
    disturbances: Sequence[LongitudinalDragDisturbance],
    definition: str = "legacy",
) -> dict[str, float | bool]:
    """Replace pre-event response scores with matched-baseline measurements."""
    if definition not in {"legacy", "recovery-integral"}:
        raise ValueError("Unsupported response objective definition")
    simulation = disturbed_evaluation.simulation
    nominal_profile = nominal_evaluation.simulation.profile
    phase_by_name = {phase.name: phase for phase in disturbed_evaluation.mission.phases}
    per_event: list[dict[str, float | bool]] = []
    for event in disturbances:
        phase = phase_by_name.get(event.phase_name)
        if phase is None:
            per_event.append(
                {
                    "paired_recovery_time_s": event.recovery_window_s,
                    "paired_recovery_success": False,
                    "paired_peak_speed_deviation_mps": float("nan"),
                    "paired_peak_altitude_deviation_m": float("nan"),
                    "response_metric_valid": False,
                    "terminal_phase_limited": True,
                    "response_observation_window_s": 0.0,
                    "response_excitation_exceeds_band": False,
                }
            )
            continue
        per_event.append(
            paired_response_metrics(
                simulation.profile,
                nominal_profile,
                event.phase_name,
                event.start_time_s,
                event.duration_s,
                event.recovery_window_s,
                event.recovery_dwell_s,
                max(
                    10.0,
                    RESPONSE_ALTITUDE_TOLERANCE_FRACTION
                    * max(abs(phase.altitude_target_m), 100.0),
                ),
            )
        )

    recovery_s = max(
        (float(item["paired_recovery_time_s"]) for item in per_event),
        default=0.0,
    )
    valid = all(bool(item["response_metric_valid"]) for item in per_event)
    recovery_success = valid and all(bool(item["paired_recovery_success"]) for item in per_event)
    peak_speed = max(
        (float(item["paired_peak_speed_deviation_mps"]) for item in per_event if np.isfinite(item["paired_peak_speed_deviation_mps"])),
        default=0.0,
    )
    peak_altitude = max(
        (float(item["paired_peak_altitude_deviation_m"]) for item in per_event if np.isfinite(item["paired_peak_altitude_deviation_m"])),
        default=0.0,
    )
    switching_s = float(simulation.response_metrics.get("mode_switch_time_s", 0.0))
    failure_penalty_s = (
        1.1 * sum(phase.duration_s for phase in disturbed_evaluation.mission.phases if phase.name != "endurance_loiter")
        + 2.0 * max((event.recovery_window_s for event in disturbances), default=0.0)
        if per_event and not recovery_success
        else 0.0
    )
    combined = {
        "recovery_time_s": recovery_s,
        "paired_recovery_time_s": recovery_s,
        "recovery_success": float(recovery_success),
        "paired_recovery_success": recovery_success,
        "failure_penalty_s": failure_penalty_s,
        "mode_switch_time_s": switching_s,
        "peak_speed_deviation_mps": peak_speed,
        "paired_peak_speed_deviation_mps": peak_speed,
        "peak_altitude_deviation_m": peak_altitude,
        "paired_peak_altitude_deviation_m": peak_altitude,
        "paired_response_metric_valid": float(valid),
        "response_metric_valid": bool(valid),
        "response_excitation_exceeds_band": float(
            any(bool(item["response_excitation_exceeds_band"]) for item in per_event)
        ),
        "paired_event_count": float(len(per_event)),
        "response_normalized_error_integral_s": sum(
            float(item.get("response_normalized_error_integral_s", 0.)) for item in per_event),
    }
    simulation.response_metrics.update(combined)
    result = disturbed_evaluation.result
    integral = combined["response_normalized_error_integral_s"]
    result.objectives["J_response_s"] = recovery_s + failure_penalty_s + (
        integral if definition == "recovery-integral" else switching_s)
    result.diagnostics.update(
        {
            "response_recovery_time_s": recovery_s,
            "response_recovery_success": float(recovery_success),
            "response_failure_penalty_s": failure_penalty_s,
            "response_mode_switch_time_s": switching_s,
            "response_paired_metric_valid": float(valid),
            "response_excitation_exceeds_band": combined["response_excitation_exceeds_band"],
            "response_peak_speed_deviation_mps": peak_speed,
            "response_peak_altitude_deviation_m": peak_altitude,
            "response_normalized_error_integral_s": combined["response_normalized_error_integral_s"],
            "response_definition_integral": float(definition == "recovery-integral"),
        }
    )
    return combined


def _first_dwell_start(
    times: np.ndarray,
    in_band: np.ndarray,
    dwell_s: float,
) -> float | None:
    start: int | None = None
    for index, inside in enumerate(in_band):
        if inside and start is None:
            start = index
        if start is not None and (not inside or index == len(in_band) - 1):
            end = index if inside and index == len(in_band) - 1 else index - 1
            if end > start and times[end] - times[start] >= dwell_s:
                return float(times[start])
            start = None
    return None
