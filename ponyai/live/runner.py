"""Live / canary trading runner.

Canary (灰度) deployment
------------------------
Three deployment modes are supported, mirroring a typical staged rollout:

1. **SHADOW** — the strategy runs and its orders are *logged* but never sent
   to the market.  Use this to validate a strategy in a real-time environment
   without any risk.

2. **CANARY** — the strategy's orders are scaled down by ``canary_fraction``
   before execution.  This lets you expose a small portion of capital to a new
   strategy while the majority of the portfolio remains in the existing system.

3. **FULL** — orders are executed at full size; the strategy is fully deployed.

Transitions between modes are explicit, allowing a systematic progression from
shadow ➜ canary ➜ full with observable metrics at each stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from ponyai.data.feed import Bar
from ponyai.strategy.base import Order, OrderSide, Portfolio, Strategy

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Stage of the canary deployment pipeline."""

    SHADOW = "shadow"
    CANARY = "canary"
    FULL = "full"


@dataclass
class ExecutedOrder:
    """Record of a live order (or shadow log entry)."""

    bar: Bar
    order: Order
    mode: DeploymentMode
    executed_quantity: float
    is_shadow: bool


class LiveRunner:
    """Drives a strategy in live (or simulated-live) mode with staged deployment.

    The runner is intentionally broker-agnostic: actual order submission is
    delegated to a user-supplied *executor* callable.  In tests this can be a
    simple lambda; in production it would call a brokerage API.

    Parameters
    ----------
    strategy:
        A concrete :class:`~ponyai.strategy.base.Strategy` instance.
    mode:
        Initial deployment mode (default: ``SHADOW``).
    canary_fraction:
        Fraction of the intended order size to execute in ``CANARY`` mode.
        Must be in (0, 1].  Ignored for other modes.
    initial_capital:
        Starting cash for the internal :class:`~ponyai.strategy.base.Portfolio`.
    commission_rate:
        Per-trade commission fraction.
    executor:
        Callable ``(order: Order, bar: Bar) -> None`` that submits an order to
        the broker.  Defaults to a no-op (useful for paper trading).
    """

    def __init__(
        self,
        strategy: Strategy,
        mode: DeploymentMode = DeploymentMode.SHADOW,
        canary_fraction: float = 0.1,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
        executor: Optional[Callable[[Order, Bar], None]] = None,
    ) -> None:
        if not 0 < canary_fraction <= 1.0:
            raise ValueError(f"canary_fraction must be in (0, 1], got {canary_fraction}")

        self._strategy = strategy
        self.mode = mode
        self.canary_fraction = canary_fraction
        self._portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
        )
        self._executor: Callable[[Order, Bar], None] = executor or (lambda o, b: None)
        self._execution_log: List[ExecutedOrder] = []

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def promote(self) -> None:
        """Advance to the next deployment stage.

        ``SHADOW`` → ``CANARY`` → ``FULL``.  Calling :meth:`promote` on a
        ``FULL`` runner is a no-op.
        """
        transitions = {
            DeploymentMode.SHADOW: DeploymentMode.CANARY,
            DeploymentMode.CANARY: DeploymentMode.FULL,
            DeploymentMode.FULL: DeploymentMode.FULL,
        }
        previous = self.mode
        self.mode = transitions[self.mode]
        if self.mode != previous:
            logger.info("LiveRunner promoted: %s → %s", previous.value, self.mode.value)

    def demote(self) -> None:
        """Revert to the previous deployment stage.

        ``FULL`` → ``CANARY`` → ``SHADOW``.  Calling :meth:`demote` on a
        ``SHADOW`` runner is a no-op.
        """
        transitions = {
            DeploymentMode.FULL: DeploymentMode.CANARY,
            DeploymentMode.CANARY: DeploymentMode.SHADOW,
            DeploymentMode.SHADOW: DeploymentMode.SHADOW,
        }
        previous = self.mode
        self.mode = transitions[self.mode]
        if self.mode != previous:
            logger.info("LiveRunner demoted: %s → %s", previous.value, self.mode.value)

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> List[ExecutedOrder]:
        """Process one incoming :class:`~ponyai.data.feed.Bar`.

        Returns the list of :class:`ExecutedOrder` records created for this
        bar (shadow logs are included with ``is_shadow=True``).
        """
        orders: List[Order] = self._strategy.on_bar(bar, self._portfolio)
        results: List[ExecutedOrder] = []

        for order in orders:
            executed = self._process_order(order, bar)
            results.append(executed)
            self._execution_log.append(executed)

        # Update portfolio equity
        self._portfolio.record_equity({bar.symbol: bar.close})

        return results

    def _process_order(self, order: Order, bar: Bar) -> ExecutedOrder:
        if self.mode == DeploymentMode.SHADOW:
            logger.debug(
                "[SHADOW] %s %s × %.2f @ %.4f",
                order.side.value, order.symbol, order.quantity, bar.close,
            )
            return ExecutedOrder(
                bar=bar,
                order=order,
                mode=self.mode,
                executed_quantity=0.0,
                is_shadow=True,
            )

        executed_qty = order.quantity
        if self.mode == DeploymentMode.CANARY:
            executed_qty = order.quantity * self.canary_fraction

        scaled_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=executed_qty,
        )

        try:
            self._portfolio.fill_order(scaled_order, bar.close)
            self._executor(scaled_order, bar)
            logger.info(
                "[%s] Executed %s %s × %.2f @ %.4f",
                self.mode.value.upper(),
                scaled_order.side.value,
                scaled_order.symbol,
                scaled_order.quantity,
                bar.close,
            )
        except ValueError as exc:
            logger.warning("Order skipped: %s", exc)
            executed_qty = 0.0

        return ExecutedOrder(
            bar=bar,
            order=order,
            mode=self.mode,
            executed_quantity=executed_qty,
            is_shadow=False,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def execution_log(self) -> List[ExecutedOrder]:
        return list(self._execution_log)

    def __repr__(self) -> str:
        return (
            f"LiveRunner(mode={self.mode.value!r}, "
            f"canary_fraction={self.canary_fraction}, "
            f"strategy={type(self._strategy).__name__})"
        )
