"""Live trading engine with grayscale (canary) deployment support.

Grayscale deployment
--------------------
The :class:`GrayscaleRouter` assigns each incoming symbol (or account) to
one of several registered strategy *slots* based on a configurable weight
distribution.  This mirrors the classic A/B or canary-release pattern:

* **Control slot** – the current production strategy receives the majority
  of traffic.
* **Canary slot(s)** – one or more new strategies receive a small fraction
  of traffic, letting operators validate performance before a full roll-out.

Weights are normalised automatically so they always sum to 1.  The routing
decision is deterministic for a given ``(symbol, seed)`` pair, which makes
it **reproducible** across engine restarts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ponyai.data.feed import Bar
from ponyai.strategy.base import BaseStrategy, Order, StrategyContext


# ---------------------------------------------------------------------------
# Grayscale router
# ---------------------------------------------------------------------------


@dataclass
class StrategySlot:
    """A named strategy slot with an associated traffic weight."""

    name: str
    strategy: BaseStrategy
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


class GrayscaleRouter:
    """Routes live bars to strategies based on configurable weights.

    Parameters
    ----------
    slots:
        Ordered list of :class:`StrategySlot` instances.
    seed:
        Integer seed used when hashing routing keys so the assignment is
        deterministic and reproducible.

    Example
    -------
    >>> from ponyai.live.engine import GrayscaleRouter, StrategySlot
    >>> router = GrayscaleRouter(slots=[
    ...     StrategySlot("production", prod_strategy, weight=0.9),
    ...     StrategySlot("canary",     new_strategy,  weight=0.1),
    ... ])
    >>> slot = router.route("AAPL")
    >>> slot.name
    'production'
    """

    def __init__(self, slots: List[StrategySlot], seed: int = 0) -> None:
        if not slots:
            raise ValueError("At least one StrategySlot is required")
        total = sum(s.weight for s in slots)
        if total == 0:
            raise ValueError("Total weight must be > 0")
        self._slots = slots
        self._total_weight = total
        self._seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def slots(self) -> List[StrategySlot]:
        return list(self._slots)

    def route(self, routing_key: str) -> StrategySlot:
        """Return the :class:`StrategySlot` for *routing_key*.

        The assignment is stable: the same key always maps to the same slot
        as long as the slot weights remain unchanged.
        """
        if self._total_weight == 0:
            raise ValueError("Total weight is 0 – cannot route")
        bucket = self._hash_to_bucket(routing_key)
        cumulative = 0.0
        for slot in self._slots:
            cumulative += slot.weight / self._total_weight
            if bucket < cumulative:
                return slot
        return self._slots[-1]

    def update_weight(self, slot_name: str, new_weight: float) -> None:
        """Adjust the traffic weight for *slot_name*.

        This enables gradual ramp-up: start canary at 0.01, then increase
        to 0.1, 0.5, 1.0 over successive deployments.
        """
        if new_weight < 0:
            raise ValueError(f"weight must be non-negative, got {new_weight}")
        for slot in self._slots:
            if slot.name == slot_name:
                slot.weight = new_weight
                self._total_weight = sum(s.weight for s in self._slots)
                return
        raise KeyError(f"No slot named {slot_name!r}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_to_bucket(self, key: str) -> float:
        """Map *key* to a float in ``[0, 1)`` deterministically."""
        digest = hashlib.md5(f"{self._seed}:{key}".encode()).hexdigest()
        return int(digest, 16) / (16 ** len(digest))


# ---------------------------------------------------------------------------
# Live engine
# ---------------------------------------------------------------------------


@dataclass
class LiveEngine:
    """Simulated live-trading engine.

    In production this would connect to a broker API.  In tests and
    staging it drives strategies with in-memory bar data, making the
    behaviour **reproducible**.

    Parameters
    ----------
    router:
        A :class:`GrayscaleRouter` that decides which strategy receives
        each incoming bar.
    initial_cash:
        Starting capital per strategy context.
    """

    router: GrayscaleRouter
    initial_cash: float = 1_000_000.0

    def __post_init__(self) -> None:
        # One StrategyContext per slot so positions are isolated.
        self._contexts: Dict[str, StrategyContext] = {
            slot.name: StrategyContext(cash=self.initial_cash)
            for slot in self.router.slots
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Notify all strategies that the engine has started."""
        for slot in self.router.slots:
            ctx = self._contexts[slot.name]
            slot.strategy.on_start(ctx)

    def stop(self) -> None:
        """Notify all strategies that the engine is stopping."""
        for slot in self.router.slots:
            ctx = self._contexts[slot.name]
            slot.strategy.on_end(ctx)

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> Tuple[str, List[Order]]:
        """Process a live bar.

        The bar is routed to exactly one strategy based on the symbol.
        Returns the name of the selected slot and the list of orders
        emitted by the strategy.
        """
        slot = self.router.route(bar.symbol)
        ctx = self._contexts[slot.name]
        ctx.current_bar = bar
        ctx.pending_orders.clear()

        slot.strategy.on_bar(bar, ctx)

        orders = list(ctx.pending_orders)
        return slot.name, orders

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_context(self, slot_name: str) -> StrategyContext:
        """Return the :class:`StrategyContext` for *slot_name*."""
        return self._contexts[slot_name]
