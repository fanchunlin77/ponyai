"""Tests for the data feed module."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ponyai.data.feed import Bar, DataFeed


def _make_df(n: int = 5, symbol: str = "AAPL") -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 1_000_000.0 + i * 100,
            }
        )
    return pd.DataFrame(rows)


class TestBar:
    def test_bar_is_frozen(self) -> None:
        bar = Bar(
            symbol="X",
            timestamp=datetime(2024, 1, 1),
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.5,
            volume=500.0,
        )
        with pytest.raises(Exception):
            bar.close = 99.0  # type: ignore[misc]


class TestDataFeed:
    def test_from_dataframe_length(self) -> None:
        feed = DataFeed.from_dataframe(_make_df(10), symbol="AAPL")
        assert len(feed) == 10

    def test_iter_bars_order(self) -> None:
        df = _make_df(5)
        # Shuffle the dataframe — from_dataframe must re-sort
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        feed = DataFeed.from_dataframe(shuffled, symbol="AAPL")
        bars = list(feed.iter_bars())
        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps), "Bars must be in chronological order"

    def test_snapshot_id_determinism(self) -> None:
        df = _make_df(5)
        feed1 = DataFeed.from_dataframe(df, symbol="AAPL")
        feed2 = DataFeed.from_dataframe(df, symbol="AAPL")
        assert feed1.snapshot_id == feed2.snapshot_id

    def test_snapshot_id_differs_for_different_data(self) -> None:
        df1 = _make_df(5)
        df2 = _make_df(6)
        feed1 = DataFeed.from_dataframe(df1, symbol="AAPL")
        feed2 = DataFeed.from_dataframe(df2, symbol="AAPL")
        assert feed1.snapshot_id != feed2.snapshot_id

    def test_explicit_snapshot_id_is_preserved(self) -> None:
        feed = DataFeed.from_dataframe(_make_df(3), symbol="AAPL", snapshot_id="my-snap-001")
        assert feed.snapshot_id == "my-snap-001"

    def test_from_records(self) -> None:
        records = [
            {
                "timestamp": datetime(2024, 1, i + 1),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100.0,
            }
            for i in range(4)
        ]
        feed = DataFeed.from_records(records, symbol="BTC")
        assert len(feed) == 4

    def test_from_dataframe_missing_column_raises(self) -> None:
        df = _make_df(3).drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing columns"):
            DataFeed.from_dataframe(df, symbol="AAPL")

    def test_bar_values(self) -> None:
        feed = DataFeed.from_dataframe(_make_df(1), symbol="AAPL")
        bar = list(feed)[0]
        assert bar.symbol == "AAPL"
        assert bar.open == pytest.approx(100.0)
        assert bar.close == pytest.approx(102.0)
