"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Signal:
    """A trading signal emitted by a strategy.

    Attributes:
        symbol: Ticker or instrument identifier.
        side: ``"buy"`` or ``"sell"``.
        quantity: Number of units to trade.
        metadata: Optional extra information (e.g. limit price).
    """

    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class that every trading strategy must implement."""

    @abstractmethod
    def on_bar(self, bar: pd.Series) -> list[Signal]:
        """Process a single OHLCV bar and return zero or more signals.

        Args:
            bar: A pandas Series with at least ``open``, ``high``, ``low``,
                ``close``, ``volume`` fields plus a ``date`` or datetime index.

        Returns:
            A list of :class:`Signal` objects (may be empty).
        """

    def on_init(self, history: pd.DataFrame) -> None:
        """Optional hook called once before the first bar.

        Override this to pre-compute indicators or warm up look-back windows.
        """

    def on_finish(self) -> None:
        """Optional hook called after the last bar has been processed."""
