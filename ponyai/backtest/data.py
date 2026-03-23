"""Lightweight historical data containers for backtesting."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class BarData:
    """A single OHLCV bar."""

    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d: Dict = {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        if self.extra:
            d.update(self.extra)
        return d


class DataFeed:
    """In-memory data feed backed by a list of :class:`BarData` objects.

    Supports simple filtering and slicing so tests and notebooks can quickly
    build a feed from CSV text or a plain list of dicts.

    Example
    -------
    >>> feed = DataFeed.from_csv(csv_text)
    >>> bars = feed.bars("AAPL", start="2023-01-01", end="2023-12-31")
    """

    def __init__(self, bars: Optional[List[BarData]] = None) -> None:
        self._bars: List[BarData] = bars or []

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(cls, rows: Iterable[Dict]) -> "DataFeed":
        """Build a feed from an iterable of dicts."""
        bar_list = []
        for row in rows:
            bar_list.append(
                BarData(
                    timestamp=str(row.get("timestamp", "")),
                    symbol=str(row.get("symbol", "")),
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=float(row.get("volume", 0)),
                    extra={
                        k: v
                        for k, v in row.items()
                        if k not in {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
                    },
                )
            )
        return cls(bar_list)

    @classmethod
    def from_csv(cls, text: str) -> "DataFeed":
        """Build a feed from CSV text (first row must be a header)."""
        reader = csv.DictReader(io.StringIO(text.strip()))
        return cls.from_dicts(reader)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def bars(
        self,
        symbol: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict]:
        """Return bars as dicts, optionally filtered by symbol and date range."""
        result = []
        for b in self._bars:
            if symbol and b.symbol != symbol:
                continue
            if start and b.timestamp < start:
                continue
            if end and b.timestamp > end:
                continue
            result.append(b.to_dict())
        return result

    def symbols(self) -> List[str]:
        return sorted({b.symbol for b in self._bars})

    def __iter__(self) -> Iterator[BarData]:
        return iter(self._bars)

    def __len__(self) -> int:
        return len(self._bars)
