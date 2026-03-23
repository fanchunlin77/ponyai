"""Backtesting engine – event-driven simulation (可回测).

The engine replays a :class:`~ponyai.data.feed.DataFeed` bar by bar,
passes each bar to a strategy, and fills the resulting signals via a
simulated broker.  After the run the full trade log and performance
metrics are stored in a :class:`BacktestResult`.

Design goals
------------
* **Deterministic** – given the same feed, strategy and parameters the
  result is always identical (can be diffed / reproduced).
* **Isolated** – each engine instance is stateful and should only be
  run once; create a fresh instance per back-test.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from ponyai.data.feed import BarData, DataFeed
from ponyai.strategy.base import BaseStrategy, Signal, SignalDirection


# ---------------------------------------------------------------------------
# Portfolio / position tracking
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Current open position for a single symbol."""

    symbol: str
    size: float = 0.0
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.size == 0.0


@dataclass
class Trade:
    """A filled trade record."""

    symbol: str
    dt: datetime.datetime
    direction: SignalDirection
    size: float
    price: float
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# Simulated broker
# ---------------------------------------------------------------------------


class SimulatedBroker:
    """Fill signals against bar close prices (next-bar-open approximation).

    Parameters
    ----------
    initial_cash:
        Starting cash balance.
    commission_rate:
        Fraction of trade value charged as commission (default 0.0).
    """

    def __init__(
        self, initial_cash: float = 1_000_000.0, commission_rate: float = 0.0
    ) -> None:
        self.cash: float = initial_cash
        self.commission_rate: float = commission_rate
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def fill(self, signal: Signal, bar: BarData) -> Trade | None:
        """Attempt to fill *signal* at *bar*.close and return the trade.

        Returns ``None`` when the signal cannot be filled (e.g. not
        enough cash).
        """
        fill_price = bar.close if signal.price is None else signal.price
        pos = self._get_position(signal.symbol)

        if signal.direction == SignalDirection.BUY:
            cost = fill_price * signal.size
            commission = cost * self.commission_rate
            if self.cash < cost + commission:
                return None  # insufficient funds
            self.cash -= cost + commission
            # update position (FIFO average)
            new_size = pos.size + signal.size
            pos.avg_price = (
                (pos.avg_price * pos.size + fill_price * signal.size) / new_size
                if new_size != 0
                else 0.0
            )
            pos.size = new_size
            trade = Trade(
                symbol=signal.symbol,
                dt=bar.dt,
                direction=signal.direction,
                size=signal.size,
                price=fill_price,
                pnl=0.0,
            )

        elif signal.direction in (SignalDirection.SELL, SignalDirection.FLAT):
            close_size = signal.size if signal.direction == SignalDirection.SELL else pos.size
            if close_size <= 0 or pos.size < close_size:
                return None  # nothing to sell
            proceeds = fill_price * close_size
            commission = proceeds * self.commission_rate
            pnl = (fill_price - pos.avg_price) * close_size - commission
            self.cash += proceeds - commission
            pos.size -= close_size
            if pos.size == 0.0:
                pos.avg_price = 0.0
            trade = Trade(
                symbol=signal.symbol,
                dt=bar.dt,
                direction=signal.direction,
                size=close_size,
                price=fill_price,
                pnl=pnl,
            )
        else:
            return None

        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def equity(self, latest_prices: dict[str, float]) -> float:
        """Return total equity = cash + mark-to-market positions."""
        mtm = sum(
            p.size * latest_prices.get(p.symbol, p.avg_price)
            for p in self.positions.values()
        )
        return self.cash + mtm


# ---------------------------------------------------------------------------
# Back-test result
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Immutable record of a completed back-test run.

    Attributes
    ----------
    strategy_desc:
        Strategy description dict (from :meth:`BaseStrategy.describe`).
    feed_name:
        Name of the :class:`DataFeed` that was replayed.
    trades:
        Ordered list of all executed trades.
    equity_curve:
        List of ``(datetime, equity)`` snapshots, one per bar.
    final_cash:
        Cash balance at end of simulation.
    final_equity:
        Total equity (cash + open positions) at end of simulation.
    total_pnl:
        Sum of realised P&L across all trades.
    """

    strategy_desc: dict[str, Any]
    feed_name: str
    trades: list[Trade]
    equity_curve: list[tuple[datetime.datetime, float]]
    final_cash: float
    final_equity: float

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_trades(self) -> int:
        return len(self.trades)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven back-test engine.

    Parameters
    ----------
    strategy:
        The :class:`~ponyai.strategy.base.BaseStrategy` instance to test.
    feed:
        The :class:`~ponyai.data.feed.DataFeed` to replay.
    initial_cash:
        Starting cash (forwarded to :class:`SimulatedBroker`).
    commission_rate:
        Per-trade commission fraction (forwarded to
        :class:`SimulatedBroker`).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        feed: DataFeed,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.0,
    ) -> None:
        self._strategy = strategy
        self._feed = feed
        self._broker = SimulatedBroker(
            initial_cash=initial_cash, commission_rate=commission_rate
        )
        self._ran = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the back-test and return the result.

        Raises
        ------
        RuntimeError
            If called more than once on the same engine instance.
        """
        if self._ran:
            raise RuntimeError(
                "BacktestEngine.run() may only be called once per instance. "
                "Create a new engine for each run."
            )
        self._ran = True

        equity_curve: list[tuple[datetime.datetime, float]] = []
        latest_prices: dict[str, float] = {}

        self._strategy.on_start()

        for bar in self._feed:
            latest_prices[bar.symbol] = bar.close
            signals = self._strategy.on_bar(bar)
            for signal in signals:
                self._broker.fill(signal, bar)
            equity_curve.append((bar.dt, self._broker.equity(latest_prices)))

        self._strategy.on_stop()

        final_equity = self._broker.equity(latest_prices)

        return BacktestResult(
            strategy_desc=self._strategy.describe(),
            feed_name=self._feed.name,
            trades=list(self._broker.trades),
            equity_curve=equity_curve,
            final_cash=self._broker.cash,
            final_equity=final_equity,
        )
