"""Tests for the data feed module."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from ponyai.data import Bar, DataFeed


class TestBar:
    def test_to_dict_round_trip(self):
        bar = Bar(
            symbol="AAPL",
            timestamp=pd.Timestamp("2023-01-03"),
            open=130.0,
            high=132.0,
            low=129.0,
            close=131.0,
            volume=50000.0,
        )
        d = bar.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["open"] == 130.0
        assert d["close"] == 131.0


class TestDataFeed:
    def _make_feed(self, n: int = 10) -> DataFeed:
        return DataFeed.synthetic(n_bars=n, seed=0)

    def test_synthetic_is_deterministic(self):
        feed1 = DataFeed.synthetic(n_bars=20, seed=42)
        feed2 = DataFeed.synthetic(n_bars=20, seed=42)
        assert feed1.version == feed2.version

    def test_different_seed_different_version(self):
        feed1 = DataFeed.synthetic(seed=1)
        feed2 = DataFeed.synthetic(seed=2)
        assert feed1.version != feed2.version

    def test_bars_count(self):
        feed = self._make_feed(50)
        assert len(feed.bars()) == 50

    def test_bars_symbol_filter(self):
        feed = self._make_feed(20)
        symbol = feed.symbols[0]
        bars = feed.bars(symbol=symbol)
        assert all(b.symbol == symbol for b in bars)

    def test_bars_time_filter(self):
        feed = DataFeed.synthetic(n_bars=50, start="2023-01-01", seed=0)
        start_ts = pd.Timestamp("2023-03-01")
        bars = feed.bars(start=start_ts)
        assert all(b.timestamp >= start_ts for b in bars)

    def test_to_dataframe_columns(self):
        feed = self._make_feed(10)
        df = feed.to_dataframe()
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_from_dataframe_round_trip(self):
        feed = DataFeed.synthetic(n_bars=30, seed=7)
        df = feed.to_dataframe()
        symbol = feed.symbols[0]
        feed2 = DataFeed.from_dataframe(df, symbol=symbol)
        assert len(feed2.bars()) == 30

    def test_from_dataframe_missing_column(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing columns"):
            DataFeed.from_dataframe(df, symbol="X")

    def test_version_is_hex(self):
        feed = self._make_feed(5)
        int(feed.version, 16)  # should not raise
