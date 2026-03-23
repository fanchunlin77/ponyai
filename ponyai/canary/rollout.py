"""Canary (灰度) deployment support for quantitative strategies.

The :class:`CanaryManager` manages a **stable** version and a **canary**
(candidate) version of a strategy.  Traffic (capital allocation) is split
according to a :class:`CanaryPolicy`.

Workflow
--------
1. Deploy a new strategy version as canary with a low traffic weight (e.g. 10 %).
2. Monitor both versions side-by-side using the live signals they produce.
3. Gradually promote the canary by increasing its weight (via
   :meth:`CanaryManager.increase_canary_weight`).
4. Call :meth:`CanaryManager.promote_canary` when confidence is high enough –
   this makes the canary the new stable version.
5. Call :meth:`CanaryManager.rollback` at any time to discard the canary and
   restore full traffic to the stable version.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ponyai.core.strategy import Order, Signal, Strategy


class RolloutState(Enum):
    """Life-cycle state of a canary deployment."""

    STABLE_ONLY = "STABLE_ONLY"      # no canary running
    CANARY_ACTIVE = "CANARY_ACTIVE"  # both stable and canary active
    PROMOTING = "PROMOTING"          # canary weight increasing towards 100 %
    PROMOTED = "PROMOTED"            # canary is now fully stable (alias of STABLE_ONLY after swap)
    ROLLED_BACK = "ROLLED_BACK"      # canary was discarded


@dataclass
class StrategyVersion:
    """Metadata wrapper around a :class:`Strategy` instance."""

    version_id: str
    strategy: Strategy
    deployed_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    @staticmethod
    def create(strategy: Strategy, metadata: Optional[Dict] = None) -> "StrategyVersion":
        return StrategyVersion(
            version_id=str(uuid.uuid4())[:8],
            strategy=strategy,
            metadata=metadata or {},
        )


@dataclass
class CanaryPolicy:
    """Policy that governs canary traffic split and auto-promotion rules.

    Parameters
    ----------
    initial_canary_weight:
        Fraction of capital routed to the canary strategy on first deployment
        (0.0 – 1.0).
    max_canary_weight:
        Upper bound for the canary weight during gradual increase.
    step_size:
        Weight increment per :meth:`CanaryManager.increase_canary_weight` call.
    auto_promote_threshold:
        If the canary weight reaches this value it is automatically promoted to
        stable on the next call to :meth:`CanaryManager.route_signals`.
    """

    initial_canary_weight: float = 0.10
    max_canary_weight: float = 1.0
    step_size: float = 0.10
    auto_promote_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.initial_canary_weight <= 1.0):
            raise ValueError("initial_canary_weight must be in (0, 1]")
        if not (0.0 < self.step_size <= 1.0):
            raise ValueError("step_size must be in (0, 1]")
        if self.max_canary_weight > 1.0 or self.max_canary_weight <= 0.0:
            raise ValueError("max_canary_weight must be in (0, 1]")


class CanaryManager:
    """Manages side-by-side routing of stable vs canary strategy versions.

    Parameters
    ----------
    stable_version:
        The currently-deployed (stable) strategy version.
    policy:
        Rollout policy controlling traffic weights.

    Example
    -------
    >>> manager = CanaryManager(stable_version)
    >>> manager.deploy_canary(new_strategy_version, policy)
    >>> for bar in live_bars:
    ...     stable_orders, canary_orders = manager.route_signals(bar, capital)
    ...     # execute stable_orders with full capital, canary_orders with canary capital
    >>> manager.promote_canary()
    """

    def __init__(
        self,
        stable_version: StrategyVersion,
        policy: Optional[CanaryPolicy] = None,
    ) -> None:
        self._stable: StrategyVersion = stable_version
        self._canary: Optional[StrategyVersion] = None
        self._canary_weight: float = 0.0
        self._policy: CanaryPolicy = policy or CanaryPolicy()
        self._state: RolloutState = RolloutState.STABLE_ONLY
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> RolloutState:
        return self._state

    @property
    def canary_weight(self) -> float:
        """Current fraction of capital routed to the canary strategy."""
        return self._canary_weight

    @property
    def stable_weight(self) -> float:
        return 1.0 - self._canary_weight

    @property
    def stable_version(self) -> StrategyVersion:
        return self._stable

    @property
    def canary_version(self) -> Optional[StrategyVersion]:
        return self._canary

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def deploy_canary(
        self,
        canary_version: StrategyVersion,
        policy: Optional[CanaryPolicy] = None,
    ) -> None:
        """Start a canary deployment with *canary_version*.

        Raises
        ------
        RuntimeError
            If a canary is already active.
        """
        if self._state == RolloutState.CANARY_ACTIVE:
            raise RuntimeError(
                "A canary deployment is already active. Rollback or promote first."
            )
        if policy:
            self._policy = policy
        self._canary = canary_version
        self._canary_weight = self._policy.initial_canary_weight
        self._state = RolloutState.CANARY_ACTIVE
        self._record_event("deploy_canary", canary_version.version_id, self._canary_weight)

    def increase_canary_weight(self, step: Optional[float] = None) -> float:
        """Increase canary traffic weight by *step* (defaults to policy step_size).

        Returns the new canary weight.

        Raises
        ------
        RuntimeError
            If no canary is currently active.
        """
        if self._state != RolloutState.CANARY_ACTIVE:
            raise RuntimeError("No active canary deployment.")
        increment = step if step is not None else self._policy.step_size
        self._canary_weight = min(
            self._canary_weight + increment, self._policy.max_canary_weight
        )
        if self._canary_weight >= self._policy.max_canary_weight:
            self._state = RolloutState.PROMOTING
        self._record_event("increase_weight", None, self._canary_weight)
        return self._canary_weight

    def promote_canary(self) -> StrategyVersion:
        """Promote the canary to stable (replaces the current stable version).

        Returns the promoted (now-stable) version.

        Raises
        ------
        RuntimeError
            If no canary is active.
        """
        if self._canary is None:
            raise RuntimeError("No canary to promote.")
        promoted = self._canary
        self._stable = promoted
        self._canary = None
        self._canary_weight = 0.0
        self._state = RolloutState.STABLE_ONLY
        self._record_event("promote", promoted.version_id, 0.0)
        return promoted

    def rollback(self) -> None:
        """Discard the canary and restore 100 % traffic to the stable version.

        Safe to call even when no canary is deployed (no-op in that case).
        """
        if self._canary is not None:
            self._record_event("rollback", self._canary.version_id, 0.0)
        self._canary = None
        self._canary_weight = 0.0
        self._state = RolloutState.STABLE_ONLY

    # ------------------------------------------------------------------
    # Signal routing
    # ------------------------------------------------------------------

    def route_signals(
        self,
        bar: Dict,
        total_capital: float,
        max_position_pct: float = 0.10,
    ) -> Tuple[List[Order], List[Order]]:
        """Route a bar to both strategies and return their orders.

        Auto-promotes canary if its weight has reached
        ``policy.auto_promote_threshold``.

        Parameters
        ----------
        bar:
            Current market bar dict.
        total_capital:
            Total portfolio capital available for order sizing.
        max_position_pct:
            Max fraction of capital per position.

        Returns
        -------
        Tuple[List[Order], List[Order]]
            ``(stable_orders, canary_orders)`` – callers should execute
            *stable_orders* using ``stable_weight * total_capital`` and
            *canary_orders* using ``canary_weight * total_capital``.
        """
        # Auto-promote if weight is at or above threshold
        if (
            self._state in (RolloutState.CANARY_ACTIVE, RolloutState.PROMOTING)
            and self._canary_weight >= self._policy.auto_promote_threshold
        ):
            self.promote_canary()

        stable_capital = self.stable_weight * total_capital
        stable_signals: List[Signal] = self._stable.strategy.on_bar(bar)
        stable_orders = self._stable.strategy.signals_to_orders(
            stable_signals, bar, stable_capital, max_position_pct
        )

        canary_orders: List[Order] = []
        if self._canary is not None:
            canary_capital = self._canary_weight * total_capital
            canary_signals: List[Signal] = self._canary.strategy.on_bar(bar)
            canary_orders = self._canary.strategy.signals_to_orders(
                canary_signals, bar, canary_capital, max_position_pct
            )

        return stable_orders, canary_orders

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return a dict describing the current deployment status."""
        return {
            "state": self._state.value,
            "stable_version_id": self._stable.version_id,
            "canary_version_id": self._canary.version_id if self._canary else None,
            "stable_weight": round(self.stable_weight, 4),
            "canary_weight": round(self._canary_weight, 4),
            "history_len": len(self._history),
        }

    def history(self) -> List[Dict]:
        """Return a copy of the deployment event history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _record_event(self, event: str, version_id: Optional[str], weight: float) -> None:
        self._history.append(
            {
                "event": event,
                "version_id": version_id,
                "canary_weight": weight,
                "ts": time.time(),
            }
        )
