"""Data feed module — versioned, reproducible market data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Bar:
    """A single OHLCV price bar."""

    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class DataFeed:
    """Versioned, reproducible data feed.

    Every feed carries a *version* string (SHA-256 of its content) so that
    backtests can be reproduced exactly by pinning the version.
    """

    def __init__(self, bars: List[Bar], name: str = "feed") -> None:
        self._name = name
        self._bars: List[Bar] = sorted(bars, key=lambda b: (b.symbol, b.timestamp))
        self._version: str = self._compute_version()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        """Deterministic content hash; pin this to guarantee reproducibility."""
        return self._version

    @property
    def symbols(self) -> List[str]:
        return sorted({b.symbol for b in self._bars})

    def bars(
        self,
        symbol: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> List[Bar]:
        """Return bars, optionally filtered by symbol and time range."""
        result = self._bars
        if symbol is not None:
            result = [b for b in result if b.symbol == symbol]
        if start is not None:
            result = [b for b in result if b.timestamp >= start]
        if end is not None:
            result = [b for b in result if b.timestamp <= end]
        return result

    def to_dataframe(
        self,
        symbol: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Return bars as a pandas DataFrame indexed by timestamp."""
        bars = self.bars(symbol=symbol, start=start, end=end)
        if not bars:
            return pd.DataFrame(
                columns=["symbol", "open", "high", "low", "close", "volume"]
            )
        records = [b.to_dict() for b in bars]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        name: str = "feed",
    ) -> "DataFeed":
        """Build a DataFeed from a DataFrame with OHLCV columns."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        bars: List[Bar] = []
        for ts, row in df.iterrows():
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=pd.Timestamp(ts),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
        return cls(bars, name=name)

    @classmethod
    def synthetic(
        cls,
        symbol: str = "SYN",
        n_bars: int = 252,
        start: str = "2023-01-01",
        freq: str = "B",
        seed: int = 42,
        name: str = "synthetic",
    ) -> "DataFeed":
        """Generate a *deterministic* synthetic price series for testing."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start=start, periods=n_bars, freq=freq)
        log_returns = rng.normal(0.0005, 0.015, size=n_bars)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        noise = rng.uniform(0.001, 0.005, size=n_bars)
        high = close * (1 + noise)
        low = close * (1 - noise)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = rng.integers(100_000, 1_000_000, size=n_bars).astype(float)
        bars = [
            Bar(
                symbol=symbol,
                timestamp=pd.Timestamp(dates[i]),
                open=float(open_[i]),
                high=float(high[i]),
                low=float(low[i]),
                close=float(close[i]),
                volume=float(volume[i]),
            )
            for i in range(n_bars)
        ]
        return cls(bars, name=name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_version(self) -> str:
        payload = json.dumps(
            [b.to_dict() for b in self._bars], sort_keys=True
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]
