"""BacktestEngine: orchestrates data, strategy, and broker."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ponyai.broker.paper import PaperBroker
from ponyai.broker.portfolio import Portfolio
from ponyai.data.feed import DataFeed
from ponyai.risk.metrics import compute_metrics
from ponyai.strategy.base import Strategy


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes:
        portfolio: Final portfolio state.
        equity_curve: Daily equity values indexed by date.
        metrics: Risk/performance metrics dictionary.
    """

    portfolio: Portfolio
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Run a strategy over historical data and collect results.

    Args:
        feed: A :class:`DataFeed` (already loaded or not).
        strategy: A concrete :class:`Strategy` instance.
        broker: A :class:`PaperBroker` (created automatically if *None*).
        initial_cash: Starting cash for the portfolio.
    """

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        broker: PaperBroker | None = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        portfolio = Portfolio(cash=initial_cash)
        self.broker = broker or PaperBroker(portfolio=portfolio)

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        if self.feed.df.empty:
            self.feed.load()

        df = self.feed.df.copy()
        self.strategy.on_init(df)

        equity: list[float] = []
        dates: list[object] = []

        for _, bar in df.iterrows():
            signals = self.strategy.on_bar(bar)
            for sig in signals:
                self.broker.execute(
                    signal=sig,
                    price=bar["close"],
                    timestamp=bar["date"],
                )
            # Record equity using close prices of held symbols
            prices = {sym: bar["close"] for sym in self.broker.portfolio.positions}
            equity.append(self.broker.portfolio.market_value(prices))
            dates.append(bar["date"])

        self.strategy.on_finish()

        equity_curve = pd.Series(equity, index=pd.DatetimeIndex(dates), name="equity")
        metrics = compute_metrics(equity_curve)

        return BacktestResult(
            portfolio=self.broker.portfolio,
            equity_curve=equity_curve,
            metrics=metrics,
        )
