"""Canary (gray) deployment — gradually shift capital between strategy versions.

The :class:`CanaryRouter` splits incoming bars between a *stable* strategy and
a *candidate* strategy according to a configurable allocation weight.  This
enables safe, incremental live roll-outs:

* ``candidate_weight=0.0``  → 100 % stable (no canary exposure)
* ``candidate_weight=0.5``  → 50 / 50 split
* ``candidate_weight=1.0``  → 100 % candidate (full promotion)

Capital is allocated proportionally, so the combined portfolio is always
fully invested in one of the two strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

from ponyai.broker import PaperBroker, Portfolio
from ponyai.data import Bar, DataFeed
from ponyai.risk import Metrics, compute_metrics
from ponyai.strategy import Order, Strategy


@dataclass
class CanaryResult:
    """Outcome of a canary deployment run."""

    stable_weight: float
    candidate_weight: float
    stable_equity: pd.Series
    candidate_equity: pd.Series
    combined_equity: pd.Series
    stable_metrics: Metrics
    candidate_metrics: Metrics
    combined_metrics: Metrics

    def __str__(self) -> str:
        lines = [
            f"Stable   weight={self.stable_weight:.0%}",
            str(self.stable_metrics),
            f"\nCandidate weight={self.candidate_weight:.0%}",
            str(self.candidate_metrics),
            "\nCombined (weighted)",
            str(self.combined_metrics),
        ]
        return "\n".join(lines)


class CanaryRouter:
    """Routes bars to both a stable and a candidate strategy simultaneously.

    Parameters
    ----------
    stable:
        The currently deployed (stable) strategy.
    candidate:
        The new (candidate) strategy under evaluation.
    candidate_weight:
        Fraction of capital allocated to the candidate (0.0–1.0).
    broker:
        Shared paper broker; two independent portfolios are maintained.
    initial_cash:
        Total notional capital; split between stable and candidate.
    """

    def __init__(
        self,
        stable: Strategy,
        candidate: Strategy,
        candidate_weight: float = 0.1,
        broker: Optional[PaperBroker] = None,
        initial_cash: float = 1_000_000.0,
    ) -> None:
        if not 0.0 <= candidate_weight <= 1.0:
            raise ValueError("candidate_weight must be in [0, 1]")
        self._stable = stable
        self._candidate = candidate
        self._candidate_weight = candidate_weight
        self._stable_weight = 1.0 - candidate_weight
        self._broker = broker or PaperBroker()
        self._initial_cash = initial_cash

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def candidate_weight(self) -> float:
        return self._candidate_weight

    @candidate_weight.setter
    def candidate_weight(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("candidate_weight must be in [0, 1]")
        self._candidate_weight = value
        self._stable_weight = 1.0 - value

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        feed: DataFeed,
        symbol: Optional[str] = None,
    ) -> CanaryResult:
        """Simultaneously backtest both strategies with their capital splits.

        Parameters
        ----------
        feed:
            Shared data feed.
        symbol:
            Symbol to trade; defaults to the first symbol in the feed.

        Returns
        -------
        CanaryResult
            Per-strategy and combined equity curves and metrics.
        """
        if symbol is None:
            if not feed.symbols:
                raise ValueError("DataFeed contains no bars")
            symbol = feed.symbols[0]

        self._stable.reset()
        self._candidate.reset()
        self._stable.on_start()
        self._candidate.on_start()

        stable_cash = self._initial_cash * self._stable_weight
        candidate_cash = self._initial_cash * self._candidate_weight

        stable_portfolio = Portfolio(initial_cash=stable_cash)
        candidate_portfolio = Portfolio(initial_cash=candidate_cash)

        stable_equity_vals: List[float] = []
        candidate_equity_vals: List[float] = []
        timestamps: List[pd.Timestamp] = []

        bars = feed.bars(symbol=symbol)

        for bar in bars:
            prices = {bar.symbol: bar.close}

            s_orders = self._stable.on_bar(bar, stable_portfolio)
            for order in s_orders:
                self._broker.execute(order, bar, stable_portfolio)

            c_orders = self._candidate.on_bar(bar, candidate_portfolio)
            for order in c_orders:
                self._broker.execute(order, bar, candidate_portfolio)

            stable_equity_vals.append(stable_portfolio.equity(prices))
            candidate_equity_vals.append(candidate_portfolio.equity(prices))
            timestamps.append(bar.timestamp)

        self._stable.on_end()
        self._candidate.on_end()

        idx = pd.DatetimeIndex(timestamps)
        stable_equity = pd.Series(stable_equity_vals, index=idx, name="stable_equity")
        candidate_equity = pd.Series(
            candidate_equity_vals, index=idx, name="candidate_equity"
        )
        combined_equity = stable_equity + candidate_equity
        combined_equity.name = "combined_equity"

        return CanaryResult(
            stable_weight=self._stable_weight,
            candidate_weight=self._candidate_weight,
            stable_equity=stable_equity,
            candidate_equity=candidate_equity,
            combined_equity=combined_equity,
            stable_metrics=compute_metrics(
                stable_equity, fills=stable_portfolio.fills
            ),
            candidate_metrics=compute_metrics(
                candidate_equity, fills=candidate_portfolio.fills
            ),
            combined_metrics=compute_metrics(
                combined_equity,
                fills=stable_portfolio.fills + candidate_portfolio.fills,
            ),
        )
