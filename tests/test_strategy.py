"""Tests for ponyai.strategy.base."""

import pandas as pd

from ponyai.strategy.base import Signal, Strategy


class BuyAndHold(Strategy):
    """Trivial strategy: buy once on the first bar."""

    def __init__(self, symbol: str = "TEST") -> None:
        self.symbol = symbol
        self._bought = False

    def on_bar(self, bar: pd.Series) -> list[Signal]:
        if not self._bought:
            self._bought = True
            return [Signal(symbol=self.symbol, side="buy", quantity=10)]
        return []


class TestStrategy:
    def test_buy_and_hold_emits_one_signal(self) -> None:
        strategy = BuyAndHold()
        bar = pd.Series({"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000})
        signals = strategy.on_bar(bar)
        assert len(signals) == 1
        assert signals[0].side == "buy"

    def test_buy_and_hold_no_repeat(self) -> None:
        strategy = BuyAndHold()
        bar = pd.Series({"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000})
        strategy.on_bar(bar)
        assert strategy.on_bar(bar) == []

    def test_signal_metadata(self) -> None:
        sig = Signal(symbol="AAPL", side="buy", quantity=5, metadata={"limit": 150.0})
        assert sig.metadata["limit"] == 150.0
