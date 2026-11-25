"""Hardware-in-the-loop message validation primitives; no hardware driver is included."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class ActuatorLimits:
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    motor_power_min_kw: float = -30.0
    motor_power_max_kw: float = 80.0
    elevator_min_deg: float = -25.0
    elevator_max_deg: float = 25.0
    aileron_min_deg: float = -20.0
    aileron_max_deg: float = 20.0
    rudder_min_deg: float = -25.0
    rudder_max_deg: float = 25.0


@dataclass(frozen=True)
class ControlCommand:
    sequence: int
    timestamp_s: float
    throttle: float
    motor_power_kw: float
    elevator_deg: float
    aileron_deg: float
    rudder_deg: float


def validate_command(
    command: ControlCommand,
    limits: ActuatorLimits = ActuatorLimits(),
    expected_sequence: int | None = None,
) -> list[str]:
    errors: list[str] = []
    values = (
        command.timestamp_s,
        command.throttle,
        command.motor_power_kw,
        command.elevator_deg,
        command.aileron_deg,
        command.rudder_deg,
    )
    if not all(math.isfinite(value) for value in values):
        errors.append("报文包含非有限数值")
    if command.sequence < 0 or (expected_sequence is not None and command.sequence != expected_sequence):
        errors.append("序号无效或不连续")
    bounds = (
        ("throttle", command.throttle, limits.throttle_min, limits.throttle_max),
        ("motor_power_kw", command.motor_power_kw, limits.motor_power_min_kw, limits.motor_power_max_kw),
        ("elevator_deg", command.elevator_deg, limits.elevator_min_deg, limits.elevator_max_deg),
        ("aileron_deg", command.aileron_deg, limits.aileron_min_deg, limits.aileron_max_deg),
        ("rudder_deg", command.rudder_deg, limits.rudder_min_deg, limits.rudder_max_deg),
    )
    for name, value, lower, upper in bounds:
        if not lower <= value <= upper:
            errors.append(f"{name} 超出限制 [{lower}, {upper}]")
    return errors


class CommandWatchdog:
    """Reject stale commands and return neutral outputs after a timeout."""

    def __init__(self, timeout_s: float = 0.1, clock=time.monotonic) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s 必须为正数")
        self.timeout_s = timeout_s
        self._clock = clock
        self._last_valid_at: float | None = None

    def accept(self) -> None:
        self._last_valid_at = self._clock()

    def is_timed_out(self) -> bool:
        return self._last_valid_at is None or self._clock() - self._last_valid_at > self.timeout_s

    def failsafe_command(self, sequence: int, timestamp_s: float) -> ControlCommand | None:
        if not self.is_timed_out():
            return None
        return ControlCommand(sequence, timestamp_s, 0.0, 0.0, 0.0, 0.0, 0.0)
