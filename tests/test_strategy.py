"""Tests for ponyai.strategy.base."""

from __future__ import annotations

import datetime

import pytest

from ponyai.data.feed import BarData
from ponyai.strategy.base import BaseStrategy, Signal, SignalDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MA2Strategy(BaseStrategy):
    """Simple 2-bar moving-average crossover for testing."""

    def __init__(self, params=None):
        super().__init__(params)
        self._prices: list[float] = []
        self._position: bool = False

    def on_bar(self, bar: BarData) -> list[Signal]:
        self._prices.append(bar.close)
        if len(self._prices) < 2:
            return []
        fast = self._prices[-1]
        slow = sum(self._prices[-2:]) / 2
        if fast > slow and not self._position:
            self._position = True
            return [Signal(symbol=bar.symbol, direction=SignalDirection.BUY)]
        if fast < slow and self._position:
            self._position = False
            return [Signal(symbol=bar.symbol, direction=SignalDirection.FLAT)]
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseStrategy:
    def _bar(self, close: float) -> BarData:
        return BarData("X", datetime.datetime(2024, 1, 1), 0, 0, 0, close, 0)

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy()  # type: ignore[abstract]

    def test_describe(self):
        strat = _MA2Strategy(params={"window": 2})
        desc = strat.describe()
        assert desc["strategy"] == "_MA2Strategy"
        assert desc["params"] == {"window": 2}

    def test_name(self):
        assert _MA2Strategy().name == "_MA2Strategy"

    def test_no_signal_on_first_bar(self):
        strat = _MA2Strategy()
        assert strat.on_bar(self._bar(100.0)) == []

    def test_buy_signal_on_rising(self):
        strat = _MA2Strategy()
        strat.on_bar(self._bar(100.0))
        signals = strat.on_bar(self._bar(101.0))
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY

    def test_flat_signal_on_falling(self):
        strat = _MA2Strategy()
        strat.on_bar(self._bar(100.0))
        strat.on_bar(self._bar(101.0))  # buy
        signals = strat.on_bar(self._bar(100.0))  # fall → flat
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.FLAT


class TestSignal:
    def test_defaults(self):
        sig = Signal("AAPL", SignalDirection.BUY)
        assert sig.size == 1.0
        assert sig.price is None
        assert sig.meta == {}

    def test_meta_is_independent_per_instance(self):
        s1 = Signal("A", SignalDirection.BUY)
        s2 = Signal("A", SignalDirection.BUY)
        s1.meta["x"] = 1
        assert "x" not in s2.meta
