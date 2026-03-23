"""Tests for ponyai.data.feed."""

from __future__ import annotations

import datetime

import pytest

from ponyai.data.feed import BarData, DataFeed


# ---------------------------------------------------------------------------
# BarData
# ---------------------------------------------------------------------------


class TestBarData:
    def test_round_trip(self):
        dt = datetime.datetime(2024, 3, 1, 9, 30)
        bar = BarData("SPY", dt, 500.0, 502.0, 499.0, 501.0, 1_000_000.0)
        assert BarData.from_dict(bar.to_dict()) == bar

    def test_immutable(self):
        dt = datetime.datetime(2024, 3, 1)
        bar = BarData("SPY", dt, 500.0, 502.0, 499.0, 501.0, 1e6)
        with pytest.raises((AttributeError, TypeError)):
            bar.close = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DataFeed
# ---------------------------------------------------------------------------


class TestDataFeed:
    def _make_feed(self, n: int = 5) -> DataFeed:
        base = datetime.datetime(2024, 1, 2)
        records = [
            {
                "symbol": "A",
                "dt": (base + datetime.timedelta(days=i)).isoformat(),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
            }
            for i in range(n)
        ]
        return DataFeed.from_dicts(records, name="test")

    def test_len(self):
        feed = self._make_feed(5)
        assert len(feed) == 5

    def test_order(self):
        """Bars must be sorted ascending by dt regardless of insertion order."""
        base = datetime.datetime(2024, 1, 2)
        records = [
            {
                "symbol": "A",
                "dt": (base + datetime.timedelta(days=i)).isoformat(),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
            }
            for i in [4, 2, 0, 3, 1]  # shuffled
        ]
        feed = DataFeed.from_dicts(records)
        dts = [b.dt for b in feed]
        assert dts == sorted(dts)

    def test_round_trip(self):
        """to_dicts → from_dicts must be lossless (reproducibility)."""
        feed = self._make_feed(3)
        restored = DataFeed.from_dicts(feed.to_dicts(), name=feed.name)
        assert len(restored) == len(feed)
        assert all(a == b for a, b in zip(feed, restored))

    def test_since(self):
        feed = self._make_feed(5)
        cutoff = feed[2].dt
        sliced = feed.since(cutoff)
        assert all(b.dt >= cutoff for b in sliced)
        assert len(sliced) == 3

    def test_until(self):
        feed = self._make_feed(5)
        cutoff = feed[2].dt
        sliced = feed.until(cutoff)
        assert all(b.dt <= cutoff for b in sliced)
        assert len(sliced) == 3

    def test_copy_is_independent(self):
        feed = self._make_feed(3)
        copy = feed.copy()
        copy.bars.clear()
        assert len(feed) == 3
