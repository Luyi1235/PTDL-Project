"""Compatibility entry point for the current packaged PTDL powertrain."""

from pathlib import Path
import sys

CORE = Path(__file__).resolve().parent / "PTDL飞推控协同优化代码包" / "核心模型"
if str(CORE) not in sys.path:
    sys.path.append(str(CORE))

from PTDL飞推控协同优化代码包.核心模型.ptdl_hybrid_system_adapter import (
    POWERTRAIN_VERSION,
    HybridSystemAdapterConfig,
    HybridSystemPowertrain,
    HybridSystemPowertrainOutput,
    HybridSystemPowertrainState,
)

__all__ = [
    "POWERTRAIN_VERSION",
    "HybridSystemAdapterConfig",
    "HybridSystemPowertrain",
    "HybridSystemPowertrainOutput",
    "HybridSystemPowertrainState",
]
