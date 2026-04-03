"""PaperBroker: simulated order execution for backtesting."""

from __future__ import annotations

from datetime import datetime

from ponyai.broker.fill import Fill
from ponyai.broker.portfolio import Portfolio
from ponyai.strategy.base import Signal


class PaperBroker:
    """Simulate order fills against OHLCV bars.

    The broker fills every signal at the bar's *close* price (a simple
    but common assumption for backtesting).

    Args:
        portfolio: The portfolio to update on each fill.
        commission_rate: Proportional commission (e.g. 0.001 = 0.1 %).
    """

    def __init__(
        self,
        portfolio: Portfolio | None = None,
        commission_rate: float = 0.001,
    ) -> None:
        self.portfolio = portfolio or Portfolio()
        self.commission_rate = commission_rate

    def execute(self, signal: Signal, price: float, timestamp: datetime) -> Fill:
        """Fill a signal at the given price.

        Args:
            signal: The trading signal to execute.
            price: Execution price (typically bar close).
            timestamp: Execution timestamp.

        Returns:
            The resulting :class:`Fill`.
        """
        commission = self.commission_rate * signal.quantity * price
        fill = Fill(
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
        )
        self.portfolio.apply_fill(fill)
        return fill
