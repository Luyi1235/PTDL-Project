"""Control-interval battery energy accounting, independent of output sampling."""

from dataclasses import dataclass
from math import isfinite


@dataclass
class BatteryEnergyLedger:
    discharge_terminal_mj: float = 0.0
    discharge_chemical_mj: float = 0.0
    charge_terminal_mj: float = 0.0
    charge_chemical_mj: float = 0.0
    internal_loss_mj: float = 0.0
    duration_s: float = 0.0
    intervals: int = 0

    def update(self, terminal_kw: float, chemical_kw: float, dt_s: float) -> None:
        values = (terminal_kw, chemical_kw, dt_s)
        if not all(isfinite(v) for v in values) or dt_s <= 0.0:
            raise ValueError("Energy accounting requires finite powers and positive duration")
        if terminal_kw * chemical_kw < -1e-9 or chemical_kw < terminal_kw - 1e-8:
            raise ValueError("Battery chemical and terminal power signs/losses are inconsistent")
        scale = dt_s / 1000.0
        self.discharge_terminal_mj += max(terminal_kw, 0.0) * scale
        self.discharge_chemical_mj += max(chemical_kw, 0.0) * scale
        self.charge_terminal_mj += max(-terminal_kw, 0.0) * scale
        self.charge_chemical_mj += max(-chemical_kw, 0.0) * scale
        self.internal_loss_mj += max(chemical_kw - terminal_kw, 0.0) * scale
        self.duration_s += dt_s
        self.intervals += 1

    def summary(self) -> dict:
        discharge = self.discharge_chemical_mj
        charge = self.charge_terminal_mj
        throughput = discharge + charge
        loss_residual = (
            discharge - self.discharge_terminal_mj
            + charge - self.charge_chemical_mj - self.internal_loss_mj
        )
        return {
            **vars(self),
            "net_chemical_withdrawal_mj": discharge - self.charge_chemical_mj,
            "discharge_efficiency": self.discharge_terminal_mj / discharge if discharge > 1e-12 else None,
            "charge_efficiency": self.charge_chemical_mj / charge if charge > 1e-12 else None,
            "conversion_efficiency": 1.0 - self.internal_loss_mj / throughput if throughput > 1e-12 else None,
            "flow_balance_residual_mj": loss_residual,
            "definition": "separate terminal/chemical charge-discharge flows; not a closed-cycle round-trip test",
        }
