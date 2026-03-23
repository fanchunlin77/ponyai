"""Canary / grey-release deployment (可灰度上线).

This module allows a new strategy version to be rolled out gradually
alongside the incumbent strategy.  A configurable fraction of
instruments (or random draws) are routed to the challenger while the
rest continue to use the champion.

Typical lifecycle
-----------------
1. Register the champion strategy with weight 1.0.
2. Register a challenger strategy with weight 0.0.
3. Gradually increase the challenger weight (and decrease the champion
   weight) as confidence grows.
4. When the challenger weight reaches 1.0 it becomes the new champion.

Both strategies continue to receive bars for observation; only the
**active** strategy's signals are acted upon by the broker.

Reproducibility
---------------
:class:`CanaryRouter` accepts an optional ``seed`` so that the routing
decisions are deterministic – the same seed and weights will always
route the same instruments / ticks to the same strategy.
"""

from __future__ import annotations

import zlib
import random
from dataclasses import dataclass, field
from typing import Any

from ponyai.data.feed import BarData
from ponyai.strategy.base import BaseStrategy, Signal


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class _StrategyEntry:
    strategy: BaseStrategy
    weight: float  # 0.0 – 1.0; fraction of traffic directed here


class StrategyRegistry:
    """Holds named strategy instances with associated traffic weights.

    Weights do **not** need to sum to 1; the router normalises them
    internally.

    Example
    -------
    >>> registry = StrategyRegistry()
    >>> registry.register("champion", champion_strategy, weight=0.8)
    >>> registry.register("challenger", challenger_strategy, weight=0.2)
    """

    def __init__(self) -> None:
        self._entries: dict[str, _StrategyEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self, name: str, strategy: BaseStrategy, weight: float = 1.0
    ) -> None:
        """Add or update a strategy in the registry.

        Parameters
        ----------
        name:
            Unique identifier for this strategy slot.
        strategy:
            :class:`~ponyai.strategy.base.BaseStrategy` instance.
        weight:
            Relative traffic weight in the range [0, 1].
        """
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        self._entries[name] = _StrategyEntry(strategy=strategy, weight=weight)

    def update_weight(self, name: str, weight: float) -> None:
        """Change the traffic weight for an existing entry."""
        if name not in self._entries:
            raise KeyError(f"Strategy '{name}' is not registered.")
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        self._entries[name].weight = weight

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def get_strategy(self, name: str) -> BaseStrategy:
        return self._entries[name].strategy

    def get_weight(self, name: str) -> float:
        return self._entries[name].weight

    def describe(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable description of all entries."""
        return [
            {
                "name": name,
                "weight": e.weight,
                **e.strategy.describe(),
            }
            for name, e in self._entries.items()
        ]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class CanaryRouter:
    """Routes incoming bars to strategies according to their weights.

    Parameters
    ----------
    registry:
        A :class:`StrategyRegistry` with at least one entry.
    seed:
        Optional integer seed for the internal RNG.  When provided the
        routing is fully deterministic (同一 seed 下路由结果一致).
    """

    def __init__(self, registry: StrategyRegistry, seed: int | None = None) -> None:
        self._registry = registry
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _pick_name(self, bar: BarData) -> str:
        """Choose which strategy should *act* on this bar.

        The selection uses a stable CRC32 hash of ``(symbol, dt)``
        combined with the current weights, so the same bar always goes
        to the same strategy for a given weight configuration.
        """
        names = self._registry.names()
        if not names:
            raise RuntimeError("StrategyRegistry is empty.")
        weights = [self._registry.get_weight(n) for n in names]
        total = sum(weights)
        if total == 0:
            raise RuntimeError("All strategy weights are 0.")
        # CRC32 is fast, deterministic, and non-cryptographic – suitable
        # for traffic-splitting purposes only.
        key = f"{bar.symbol}:{bar.dt.isoformat()}"
        digest = zlib.crc32(key.encode()) & 0xFFFFFFFF  # unsigned 32-bit
        threshold = digest / 0xFFFFFFFF  # value in [0, 1]
        cumulative = 0.0
        for name, w in zip(names, weights):
            cumulative += w / total
            if threshold < cumulative:
                return name
        return names[-1]  # fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, bar: BarData) -> tuple[str, list[Signal]]:
        """Deliver *bar* to **all** strategies; return signals from the
        **active** (winning) strategy only.

        All registered strategies receive the bar for observation, but
        only the chosen strategy's signals are returned to the caller.

        Returns
        -------
        tuple[str, list[Signal]]
            ``(active_strategy_name, signals)``
        """
        active_name = self._pick_name(bar)
        signals: list[Signal] = []
        for name in self._registry.names():
            strategy = self._registry.get_strategy(name)
            result = strategy.on_bar(bar)
            if name == active_name:
                signals = result
        return active_name, signals



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class _StrategyEntry:
    strategy: BaseStrategy
    weight: float  # 0.0 – 1.0; fraction of traffic directed here


class StrategyRegistry:
    """Holds named strategy instances with associated traffic weights.

    Weights do **not** need to sum to 1; the router normalises them
    internally.

    Example
    -------
    >>> registry = StrategyRegistry()
    >>> registry.register("champion", champion_strategy, weight=0.8)
    >>> registry.register("challenger", challenger_strategy, weight=0.2)
    """

    def __init__(self) -> None:
        self._entries: dict[str, _StrategyEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self, name: str, strategy: BaseStrategy, weight: float = 1.0
    ) -> None:
        """Add or update a strategy in the registry.

        Parameters
        ----------
        name:
            Unique identifier for this strategy slot.
        strategy:
            :class:`~ponyai.strategy.base.BaseStrategy` instance.
        weight:
            Relative traffic weight in the range [0, 1].
        """
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        self._entries[name] = _StrategyEntry(strategy=strategy, weight=weight)

    def update_weight(self, name: str, weight: float) -> None:
        """Change the traffic weight for an existing entry."""
        if name not in self._entries:
            raise KeyError(f"Strategy '{name}' is not registered.")
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        self._entries[name].weight = weight

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def get_strategy(self, name: str) -> BaseStrategy:
        return self._entries[name].strategy

    def get_weight(self, name: str) -> float:
        return self._entries[name].weight

    def describe(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable description of all entries."""
        return [
            {
                "name": name,
                "weight": e.weight,
                **e.strategy.describe(),
            }
            for name, e in self._entries.items()
        ]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class CanaryRouter:
    """Routes incoming bars to strategies according to their weights.

    Parameters
    ----------
    registry:
        A :class:`StrategyRegistry` with at least one entry.
    seed:
        Optional integer seed for the internal RNG.  When provided the
        routing is fully deterministic (同一 seed 下路由结果一致).
    """

    def __init__(self, registry: StrategyRegistry, seed: int | None = None) -> None:
        self._registry = registry
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _pick_name(self, bar: BarData) -> str:
        """Choose which strategy should *act* on this bar.

        The selection uses a stable CRC32 hash of ``(symbol, dt)``
        combined with the current weights, so the same bar always goes
        to the same strategy for a given weight configuration.
        """
        names = self._registry.names()
        if not names:
            raise RuntimeError("StrategyRegistry is empty.")
        weights = [self._registry.get_weight(n) for n in names]
        total = sum(weights)
        if total == 0:
            raise RuntimeError("All strategy weights are 0.")
        # CRC32 is fast, deterministic, and non-cryptographic – suitable
        # for traffic-splitting purposes only.
        key = f"{bar.symbol}:{bar.dt.isoformat()}"
        digest = zlib.crc32(key.encode()) & 0xFFFFFFFF  # unsigned 32-bit
        threshold = digest / 0xFFFFFFFF  # value in [0, 1]
        cumulative = 0.0
        for name, w in zip(names, weights):
            cumulative += w / total
            if threshold < cumulative:
                return name
        return names[-1]  # fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, bar: BarData) -> tuple[str, list[Signal]]:
        """Deliver *bar* to **all** strategies; return signals from the
        **active** (winning) strategy only.

        All registered strategies receive the bar for observation, but
        only the chosen strategy's signals are returned to the caller.

        Returns
        -------
        tuple[str, list[Signal]]
            ``(active_strategy_name, signals)``
        """
        active_name = self._pick_name(bar)
        signals: list[Signal] = []
        for name in self._registry.names():
            strategy = self._registry.get_strategy(name)
            result = strategy.on_bar(bar)
            if name == active_name:
                signals = result
        return active_name, signals
