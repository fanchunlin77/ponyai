"""Platform-wide configuration helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PlatformConfig:
    """Top-level configuration for a PonyAI platform instance.

    All settings are plain data so the config is trivially serialisable –
    this is key to the reproducibility guarantee.

    Parameters
    ----------
    initial_capital:
        Default starting capital for backtests.
    commission_rate:
        Default commission rate (fraction of trade value).
    slippage_pct:
        Default simulated slippage.
    seed:
        Global random seed.
    canary_initial_weight:
        Default initial canary traffic weight.
    canary_step_size:
        Default step size for weight increases.
    extra:
        Any additional key/value pairs for project-specific settings.
    """

    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.001
    slippage_pct: float = 0.0005
    seed: int = 42
    canary_initial_weight: float = 0.10
    canary_step_size: float = 0.10
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "PlatformConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        init_kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        instance = cls(**init_kwargs)
        instance.extra.update(extra)
        return instance

    @classmethod
    def from_json(cls, text: str) -> "PlatformConfig":
        return cls.from_dict(json.loads(text))

    def copy(self, **overrides: Any) -> "PlatformConfig":
        """Return a shallow copy with optional field overrides."""
        d = self.to_dict()
        d.update(overrides)
        return self.from_dict(d)
