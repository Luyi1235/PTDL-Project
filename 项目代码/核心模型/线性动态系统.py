"""Reusable continuous-time state-space and transfer-function response blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def first_order_transfer_step(
    state: float,
    command: float,
    dt_s: float,
    time_constant_s: float,
    gain: float = 1.0,
) -> float:
    """Exact zero-order-hold step for ``G(s)=gain/(tau*s+1)``."""
    if not np.isfinite([state, command, dt_s, time_constant_s, gain]).all():
        raise ValueError("Transfer-function inputs must be finite.")
    if dt_s < 0.0 or time_constant_s <= 0.0:
        raise ValueError("dt_s must be nonnegative and time_constant_s positive.")
    decay = np.exp(-dt_s / time_constant_s)
    return float(state * decay + gain * command * (1.0 - decay))


def first_order_state_space_step(
    state: float,
    command: float,
    dt_s: float,
    time_constant_s: float,
    gain: float = 1.0,
) -> float:
    """Exact zero-order-hold step for ``x_dot=(-x+gain*u)/tau, y=x``."""
    if not np.isfinite([state, command, dt_s, time_constant_s, gain]).all():
        raise ValueError("State-space inputs must be finite.")
    if dt_s < 0.0 or time_constant_s <= 0.0:
        raise ValueError("dt_s must be nonnegative and time_constant_s positive.")
    decay = np.exp(-dt_s / time_constant_s)
    return float(decay * state + gain * (1.0 - decay) * command)


@dataclass(frozen=True)
class ContinuousStateSpace:
    """Linear continuous-time model ``x_dot=A*x+B*u, y=C*x+D*u``."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.A, dtype=float)
        b = np.asarray(self.B, dtype=float)
        c = np.asarray(self.C, dtype=float)
        d = np.asarray(self.D, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("A must be a square matrix.")
        if b.ndim == 1:
            b = b[:, None]
        if c.ndim == 1:
            c = c[None, :]
        if d.ndim == 0:
            d = d.reshape(1, 1)
        n = a.shape[0]
        if b.ndim != 2 or b.shape[0] != n:
            raise ValueError("B must have one row per state.")
        if c.ndim != 2 or c.shape[1] != n:
            raise ValueError("C must have one column per state.")
        if d.shape != (c.shape[0], b.shape[1]):
            raise ValueError("D dimensions must match outputs and inputs.")
        if not all(np.isfinite(m).all() for m in (a, b, c, d)):
            raise ValueError("State-space matrices must be finite.")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)
        object.__setattr__(self, "C", c)
        object.__setattr__(self, "D", d)

    def step(
        self, state: np.ndarray, command: np.ndarray, dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one zero-order-held input step with RK4."""
        if not np.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be finite and nonnegative.")
        x = np.asarray(state, dtype=float).reshape(-1)
        u = np.asarray(command, dtype=float).reshape(-1)
        if x.size != self.A.shape[0] or u.size != self.B.shape[1]:
            raise ValueError("State or input vector has an incompatible size.")
        if not np.isfinite(x).all() or not np.isfinite(u).all():
            raise ValueError("State and input vectors must be finite.")

        def derivative(value: np.ndarray) -> np.ndarray:
            return self.A @ value + self.B @ u

        k1 = derivative(x)
        k2 = derivative(x + 0.5 * dt_s * k1)
        k3 = derivative(x + 0.5 * dt_s * k2)
        k4 = derivative(x + dt_s * k3)
        next_state = x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state, self.C @ next_state + self.D @ u
