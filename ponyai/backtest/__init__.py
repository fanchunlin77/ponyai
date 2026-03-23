"""Event-driven backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ponyai.broker import PaperBroker, Portfolio
from ponyai.data import Bar, DataFeed
from ponyai.risk import Metrics, compute_metrics
from ponyai.strategy import Strategy


@dataclass
class BacktestResult:
    """Output of a single backtest run."""

    strategy_name: str
    strategy_version: str
    feed_version: str
    seed: int
    equity_curve: pd.Series
    metrics: Metrics
    fills: list

    def __str__(self) -> str:
        lines = [
            f"Strategy : {self.strategy_name} v{self.strategy_version}",
            f"Feed     : {self.feed_version}",
            f"Seed     : {self.seed}",
            str(self.metrics),
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Reproducible, event-driven backtesting engine.

    The engine iterates over bars in chronological order, calls
    ``strategy.on_bar`` for each bar, and executes returned orders via the
    paper broker.  The same ``(feed_version, strategy_version, seed)`` triplet
    always produces the same result, guaranteeing reproducibility.

    Parameters
    ----------
    broker:
        Paper broker used for order execution; created with defaults if omitted.
    initial_cash:
        Starting portfolio cash.
    """

    def __init__(
        self,
        broker: Optional[PaperBroker] = None,
        initial_cash: float = 1_000_000.0,
    ) -> None:
        self._broker = broker or PaperBroker()
        self._initial_cash = initial_cash

    def run(
        self,
        strategy: Strategy,
        feed: DataFeed,
        symbol: Optional[str] = None,
    ) -> BacktestResult:
        """Run a backtest and return a :class:`BacktestResult`.

        Parameters
        ----------
        strategy:
            The strategy to test; its state is reset before the run starts.
        feed:
            Data feed providing OHLCV bars.
        symbol:
            If given, only bars for this symbol are processed; otherwise the
            first symbol in the feed is used.
        """
        if symbol is None:
            if not feed.symbols:
                raise ValueError("DataFeed contains no bars")
            symbol = feed.symbols[0]

        strategy.reset()
        strategy.on_start()

        portfolio = Portfolio(initial_cash=self._initial_cash)
        equity_records: List[Dict] = []

        bars = feed.bars(symbol=symbol)

        for bar in bars:
            prices = {bar.symbol: bar.close}
            orders = strategy.on_bar(bar, portfolio)
            for order in orders:
                self._broker.execute(order, bar, portfolio)
            equity_records.append(
                {"timestamp": bar.timestamp, "equity": portfolio.equity(prices)}
            )

        strategy.on_end()

        equity_curve = pd.Series(
            [r["equity"] for r in equity_records],
            index=pd.DatetimeIndex([r["timestamp"] for r in equity_records]),
            name="equity",
        )

        metrics = compute_metrics(equity_curve, fills=portfolio.fills)

        return BacktestResult(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            feed_version=feed.version,
            seed=strategy.seed,
            equity_curve=equity_curve,
            metrics=metrics,
            fills=portfolio.fills,
        )
