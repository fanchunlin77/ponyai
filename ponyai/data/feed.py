"""Data feed: loading and iterating historical OHLCV bar data.

Reproducibility is ensured by pinning data to a ``snapshot_id``.  Every
``DataFeed`` carries an immutable snapshot identifier so that two runs using
the same snapshot always iterate over exactly the same bars in the same order.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator, List, Optional

import pandas as pd


@dataclass(frozen=True)
class Bar:
    """A single OHLCV candlestick bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed:
    """An ordered, versioned sequence of :class:`Bar` objects.

    Parameters
    ----------
    bars:
        Ordered list of :class:`Bar` objects.
    snapshot_id:
        An opaque string that uniquely identifies this particular data
        snapshot.  When ``None`` the snapshot id is computed deterministically
        from the bar data so that the same data always yields the same id.
    """

    def __init__(
        self,
        bars: List[Bar],
        snapshot_id: Optional[str] = None,
    ) -> None:
        self._bars: List[Bar] = list(bars)
        self.snapshot_id: str = snapshot_id or self._compute_snapshot_id()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        snapshot_id: Optional[str] = None,
    ) -> "DataFeed":
        """Build a :class:`DataFeed` from a :class:`pandas.DataFrame`.

        The dataframe must contain columns ``timestamp``, ``open``, ``high``,
        ``low``, ``close``, ``volume``.  The ``timestamp`` column is
        interpreted as UTC if it is already a :class:`datetime` object,
        otherwise it is parsed with :func:`pandas.to_datetime`.
        """
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        df = df.sort_values("timestamp").reset_index(drop=True)
        bars: List[Bar] = []
        for row in df.itertuples(index=False):
            ts = row.timestamp
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                )
            )
        return cls(bars, snapshot_id=snapshot_id)

    @classmethod
    def from_csv(
        cls,
        path: str,
        symbol: str,
        snapshot_id: Optional[str] = None,
    ) -> "DataFeed":
        """Build a :class:`DataFeed` from a CSV file.

        The CSV must have a header row with the columns listed in
        :meth:`from_dataframe`.
        """
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return cls.from_dataframe(df, symbol=symbol, snapshot_id=snapshot_id)

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict],
        symbol: str,
        snapshot_id: Optional[str] = None,
    ) -> "DataFeed":
        """Build a :class:`DataFeed` from an iterable of dicts."""
        return cls.from_dataframe(
            pd.DataFrame(list(records)),
            symbol=symbol,
            snapshot_id=snapshot_id,
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def iter_bars(self) -> Iterator[Bar]:
        """Yield every :class:`Bar` in chronological order."""
        return iter(self._bars)

    # ------------------------------------------------------------------
    # Serialisation helpers (used for snapshot id)
    # ------------------------------------------------------------------

    def to_csv_string(self) -> str:
        """Return a canonical CSV representation of all bars."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        for b in self._bars:
            writer.writerow(
                [b.symbol, b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume]
            )
        return buf.getvalue()

    def _compute_snapshot_id(self) -> str:
        """Derive a deterministic snapshot id from the bar data."""
        digest = hashlib.sha256(self.to_csv_string().encode()).hexdigest()
        return digest[:16]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        symbol = repr(self._bars[0].symbol) if self._bars else "N/A"
        return (
            f"DataFeed(snapshot_id={self.snapshot_id!r}, "
            f"bars={len(self._bars)}, "
            f"symbol={symbol})"
        )
