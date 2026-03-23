"""Event-driven backtesting engine.

Reproducibility guarantees
--------------------------
The engine itself is stateless between runs: constructing a new
:class:`BacktestEngine` with the *same* :class:`~ponyai.data.feed.DataFeed`
(same ``snapshot_id``), the *same* :class:`~ponyai.strategy.base.Strategy`
(same ``seed``), and the same parameters always produces identical results.

The ``snapshot_id`` of the feed and the strategy ``seed`` are both captured in
the :class:`BacktestResult` so that every result is fully self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ponyai.data.feed import Bar, DataFeed
from ponyai.strategy.base import Order, OrderSide, Portfolio, Strategy
from ponyai.utils.metrics import compute_metrics


@dataclass
class BacktestResult:
    """The output of a completed backtest run.

    Attributes
    ----------
    snapshot_id:
        The :attr:`~ponyai.data.feed.DataFeed.snapshot_id` of the data used.
    strategy_seed:
        The :attr:`~ponyai.strategy.base.Strategy.seed` used (``None`` if the
        strategy was not seeded).
    initial_capital:
        Starting portfolio value.
    final_equity:
        Portfolio value at the end of the backtest.
    equity_curve:
        Per-bar equity values.
    metrics:
        Dictionary of standard performance metrics (see
        :func:`~ponyai.utils.metrics.compute_metrics`).
    trade_log:
        List of executed trade records: ``{"bar_index", "symbol", "side",
        "quantity", "price", "commission"}``.
    """

    snapshot_id: str
    strategy_seed: Optional[int]
    initial_capital: float
    final_equity: float
    equity_curve: List[float]
    metrics: Dict[str, float]
    trade_log: List[dict] = field(default_factory=list)


class BacktestEngine:
    """Drives a :class:`~ponyai.strategy.base.Strategy` over a
    :class:`~ponyai.data.feed.DataFeed` in chronological order.

    Parameters
    ----------
    feed:
        Historical bar data.
    strategy:
        A concrete :class:`~ponyai.strategy.base.Strategy` instance.
    initial_capital:
        Starting cash for the portfolio.
    commission_rate:
        Per-trade commission as a fraction of trade value.
    risk_free_rate:
        Annualised risk-free rate used when computing the Sharpe ratio.
    periods_per_year:
        Number of bars per calendar year (252 = daily, 52 = weekly, etc.).
    """

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        self._feed = feed
        self._strategy = strategy
        self._initial_capital = initial_capital
        self._commission_rate = commission_rate
        self._risk_free_rate = risk_free_rate
        self._periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the full backtest and return a :class:`BacktestResult`."""
        portfolio = Portfolio(
            initial_capital=self._initial_capital,
            commission_rate=self._commission_rate,
        )
        trade_log: List[dict] = []
        latest_prices: Dict[str, float] = {}

        for bar_index, bar in enumerate(self._feed.iter_bars()):
            latest_prices[bar.symbol] = bar.close

            # Ask the strategy for orders
            orders: List[Order] = self._strategy.on_bar(bar, portfolio)

            # Fill all market orders at the bar's closing price
            for order in orders:
                fill_price = bar.close
                try:
                    commission = portfolio.fill_order(order, fill_price)
                    trade_log.append(
                        {
                            "bar_index": bar_index,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "quantity": order.quantity,
                            "price": fill_price,
                            "commission": commission,
                        }
                    )
                except ValueError:
                    # Insufficient funds or short-sell attempt — skip the order
                    pass

            # Record equity at close prices
            portfolio.record_equity(latest_prices)

        equity_curve = portfolio.equity_curve
        metrics = compute_metrics(
            equity_curve,
            risk_free_rate=self._risk_free_rate,
            periods_per_year=self._periods_per_year,
        )

        return BacktestResult(
            snapshot_id=self._feed.snapshot_id,
            strategy_seed=self._strategy.seed,
            initial_capital=self._initial_capital,
            final_equity=equity_curve[-1] if equity_curve else self._initial_capital,
            equity_curve=equity_curve,
            metrics=metrics,
            trade_log=trade_log,
        )
