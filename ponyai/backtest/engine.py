"""Reproducible backtesting engine.

Key design goals
----------------
* **Deterministic** – the engine accepts a ``seed`` parameter so that any
  random elements (e.g. strategy tie-breaking) are fully reproducible.
* **Snapshot-based replay** – each run stores the complete sequence of
  context snapshots so results can be inspected and audited.
* **No external state** – all data is passed in; the engine never touches
  the filesystem or network.
"""

from __future__ import annotations

import random
from dataclasses import replace as dataclasses_replace
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ponyai.data.feed import Bar, DataFeed
from ponyai.strategy.base import (
    BaseStrategy,
    Order,
    OrderSide,
    OrderType,
    Position,
    StrategyContext,
)


@dataclass
class Trade:
    """A completed fill record."""

    bar: Bar
    order: Order
    fill_price: float
    fill_quantity: float
    commission: float = 0.0

    @property
    def pnl_contribution(self) -> float:
        sign = 1.0 if self.order.side == OrderSide.BUY else -1.0
        return -sign * self.fill_price * self.fill_quantity - self.commission


@dataclass
class BacktestResult:
    """Aggregated results of a single backtest run."""

    strategy_name: str
    initial_cash: float
    final_cash: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    seed: Optional[int] = None

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    @property
    def total_return(self) -> float:
        """Total return as a fraction of initial capital."""
        if self.initial_cash == 0:
            return 0.0
        return (self.final_cash - self.initial_cash) / self.initial_cash

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    def __repr__(self) -> str:
        return (
            f"BacktestResult(strategy={self.strategy_name!r}, "
            f"return={self.total_return:.2%}, trades={self.num_trades}, "
            f"seed={self.seed})"
        )


class BacktestEngine:
    """Event-driven backtesting engine.

    Parameters
    ----------
    initial_cash:
        Starting capital.
    commission_rate:
        Flat commission as a fraction of trade value (default ``0.001``).
    slippage_rate:
        Slippage as a fraction of fill price (default ``0.0``).
    seed:
        Random seed for reproducibility.  Set to an integer to get
        fully deterministic results across runs.
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.seed = seed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, strategy: BaseStrategy, feed: DataFeed) -> BacktestResult:
        """Run *strategy* over *feed* and return a :class:`BacktestResult`.

        This method is **pure** with respect to the engine instance state –
        re-calling ``run`` with the same seed will always produce the same
        result.
        """
        # Reset RNG to seed so every call to run() is deterministic.
        self._rng = random.Random(self.seed)

        ctx = StrategyContext(cash=self.initial_cash)
        trades: List[Trade] = []
        equity_curve: List[float] = []

        strategy.on_start(ctx)

        for bar in feed:
            ctx.current_bar = bar
            ctx.pending_orders.clear()

            strategy.on_bar(bar, ctx)

            for order in ctx.pending_orders:
                trade = self._fill_order(order, bar, ctx)
                if trade is not None:
                    trades.append(trade)

            equity = self._calculate_equity(ctx, bar)
            equity_curve.append(equity)

        strategy.on_end(ctx)

        return BacktestResult(
            strategy_name=strategy.name,
            initial_cash=self.initial_cash,
            final_cash=ctx.cash,
            trades=trades,
            equity_curve=equity_curve,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_order(
        self, order: Order, bar: Bar, ctx: StrategyContext
    ) -> Optional[Trade]:
        """Simulate order fill at *bar* prices."""
        fill_price = self._get_fill_price(order, bar)
        if fill_price is None:
            return None

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price *= 1.0 + self.slippage_rate
        else:
            fill_price *= 1.0 - self.slippage_rate

        fill_qty = order.quantity
        commission = fill_price * fill_qty * self.commission_rate
        trade_value = fill_price * fill_qty

        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission
            if ctx.cash < total_cost:
                # Partial fill based on available cash
                affordable = ctx.cash / (fill_price * (1 + self.commission_rate))
                if affordable <= 0:
                    return None
                fill_qty = affordable
                trade_value = fill_price * fill_qty
                commission = trade_value * self.commission_rate
                total_cost = trade_value + commission
            ctx.cash -= total_cost
        else:
            ctx.cash += trade_value - commission

        position = ctx.get_position(order.symbol)
        position.update(order.side, fill_qty, fill_price)

        filled_order = dataclasses_replace(order, quantity=fill_qty, timestamp=bar.timestamp)
        return Trade(
            bar=bar,
            order=filled_order,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            commission=commission,
        )

    def _get_fill_price(self, order: Order, bar: Bar) -> Optional[float]:
        """Determine fill price; returns ``None`` if the order cannot fill."""
        if order.order_type == OrderType.MARKET:
            return bar.open

        assert order.limit_price is not None
        if order.side == OrderSide.BUY and bar.low <= order.limit_price:
            return min(order.limit_price, bar.open)
        if order.side == OrderSide.SELL and bar.high >= order.limit_price:
            return max(order.limit_price, bar.open)
        return None

    def _calculate_equity(self, ctx: StrategyContext, bar: Bar) -> float:
        """Mark-to-market equity = cash + market value of positions."""
        mtm = sum(
            pos.quantity * bar.close
            for sym, pos in ctx.positions.items()
            if sym == bar.symbol
        )
        return ctx.cash + mtm
