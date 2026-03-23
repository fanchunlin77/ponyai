"""Data feed – historical bar data and live tick simulation."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.volume < 0:
            raise ValueError(f"volume must be non-negative, got {self.volume}")


@dataclass
class DataFeed:
    """In-memory data feed backed by a list of :class:`Bar` objects.

    Parameters
    ----------
    bars:
        Pre-loaded bars (ordered chronologically).
    """

    bars: List[Bar] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, text: str, symbol: str, date_format: str = "%Y-%m-%d") -> "DataFeed":
        """Create a :class:`DataFeed` from a CSV string.

        Expected columns (header required): ``date,open,high,low,close,volume``
        """
        reader = csv.DictReader(io.StringIO(text.strip()))
        bars: List[Bar] = []
        for row in reader:
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=datetime.strptime(row["date"], date_format),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
        return cls(bars=bars)

    @classmethod
    def from_bars(cls, bars: Iterable[Bar]) -> "DataFeed":
        """Create a :class:`DataFeed` from an iterable of :class:`Bar` objects."""
        return cls(bars=list(bars))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Bar]:
        return iter(self.bars)

    def __len__(self) -> int:
        return len(self.bars)

    def slice(self, start: Optional[datetime] = None, end: Optional[datetime] = None) -> "DataFeed":
        """Return a new :class:`DataFeed` filtered to ``[start, end]``."""
        result = [
            b
            for b in self.bars
            if (start is None or b.timestamp >= start) and (end is None or b.timestamp <= end)
        ]
        return DataFeed(bars=result)
