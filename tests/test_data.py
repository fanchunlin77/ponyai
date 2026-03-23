"""Tests for the data feed layer."""

from datetime import datetime

import pytest

from ponyai.data.feed import Bar, DataFeed


CSV_DATA = """\
date,open,high,low,close,volume
2024-01-01,100.0,105.0,98.0,103.0,10000
2024-01-02,103.0,108.0,101.0,107.0,12000
2024-01-03,107.0,110.0,104.0,106.0,9000
"""


class TestBar:
    def test_valid_bar(self):
        bar = Bar("AAPL", datetime(2024, 1, 1), 100, 105, 98, 103, 10000)
        assert bar.symbol == "AAPL"
        assert bar.close == 103

    def test_high_less_than_low_raises(self):
        with pytest.raises(ValueError, match="high"):
            Bar("AAPL", datetime(2024, 1, 1), 100, 90, 98, 103, 10000)

    def test_negative_volume_raises(self):
        with pytest.raises(ValueError, match="volume"):
            Bar("AAPL", datetime(2024, 1, 1), 100, 105, 98, 103, -1)


class TestDataFeed:
    def test_from_csv(self):
        feed = DataFeed.from_csv(CSV_DATA, "AAPL")
        assert len(feed) == 3
        bars = list(feed)
        assert bars[0].close == 103.0
        assert bars[2].volume == 9000

    def test_from_bars(self):
        bars = [Bar("X", datetime(2024, 1, i), 10, 11, 9, 10, 100) for i in range(1, 4)]
        feed = DataFeed.from_bars(bars)
        assert len(feed) == 3

    def test_slice(self):
        feed = DataFeed.from_csv(CSV_DATA, "AAPL")
        sliced = feed.slice(start=datetime(2024, 1, 2), end=datetime(2024, 1, 2))
        assert len(sliced) == 1
        assert list(sliced)[0].timestamp == datetime(2024, 1, 2)

    def test_slice_no_bounds(self):
        feed = DataFeed.from_csv(CSV_DATA, "AAPL")
        assert len(feed.slice()) == 3
