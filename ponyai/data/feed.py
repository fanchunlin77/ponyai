"""Data feed – reproducible market data layer.

BarData represents a single OHLCV candlestick bar.
DataFeed wraps a list of bars and is iterable in chronological order.
A DataFeed can be serialised to / from a plain list-of-dicts so that the
exact same data set can be replayed in multiple back-test or live runs
(可复现).
"""

from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass, field, asdict
from typing import Iterator, Sequence


@dataclass(frozen=True)
class BarData:
    """A single OHLCV bar for one instrument."""

    symbol: str
    dt: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dt"] = self.dt.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BarData":
        d = dict(d)
        d["dt"] = datetime.datetime.fromisoformat(d["dt"])
        return cls(**d)


@dataclass
class DataFeed:
    """An ordered, reproducible sequence of :class:`BarData` bars.

    Parameters
    ----------
    bars:
        The raw bar list, sorted ascending by ``dt``.
    name:
        Optional label that identifies the data set (e.g. "SPY-1D-2020").
    """

    bars: list[BarData] = field(default_factory=list)
    name: str = ""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(cls, records: Sequence[dict], name: str = "") -> "DataFeed":
        """Build a feed from a list of plain dicts (JSON-friendly).

        This is the primary deserialization path that guarantees
        reproducibility – the same ``records`` will always produce an
        identical feed.
        """
        bars = [BarData.from_dict(r) for r in records]
        bars.sort(key=lambda b: b.dt)
        return cls(bars=bars, name=name)

    # ------------------------------------------------------------------
    # Iteration / access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.bars)

    def __iter__(self) -> Iterator[BarData]:
        return iter(self.bars)

    def __getitem__(self, index: int) -> BarData:
        return self.bars[index]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dicts(self) -> list[dict]:
        """Export to a list of plain dicts (JSON-serialisable)."""
        return [b.to_dict() for b in self.bars]

    def copy(self) -> "DataFeed":
        """Return a deep copy so callers can modify without side-effects."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------

    def since(self, dt: datetime.datetime) -> "DataFeed":
        """Return a new feed containing only bars with dt >= *dt*."""
        return DataFeed(
            bars=[b for b in self.bars if b.dt >= dt],
            name=self.name,
        )

    def until(self, dt: datetime.datetime) -> "DataFeed":
        """Return a new feed containing only bars with dt <= *dt*."""
        return DataFeed(
            bars=[b for b in self.bars if b.dt <= dt],
            name=self.name,
        )
