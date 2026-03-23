"""Shared test fixtures and helpers."""

from __future__ import annotations

import datetime

import pytest

from ponyai.data.feed import BarData, DataFeed
from ponyai.strategy.base import BaseStrategy, Signal, SignalDirection


# ---------------------------------------------------------------------------
# A minimal deterministic strategy used across tests
# ---------------------------------------------------------------------------


class AlwaysBuyStrategy(BaseStrategy):
    """Buys 1 unit on every bar (used in engine tests)."""

    def on_bar(self, bar: BarData) -> list[Signal]:
        return [Signal(symbol=bar.symbol, direction=SignalDirection.BUY, size=1.0)]


class AlwaysFlatStrategy(BaseStrategy):
    """Emits no signals – passive observer."""

    def on_bar(self, bar: BarData) -> list[Signal]:
        return []


# ---------------------------------------------------------------------------
# Fixture: a small 5-bar feed
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_feed() -> DataFeed:
    base = datetime.datetime(2024, 1, 2, 9, 30)
    bars = [
        BarData(
            symbol="TEST",
            dt=base + datetime.timedelta(days=i),
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=101.0 + i,
            volume=1000.0,
        )
        for i in range(5)
    ]
    return DataFeed(bars=bars, name="test-feed")
