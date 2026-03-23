"""Shared test helpers and fixtures."""

from __future__ import annotations

from typing import Dict, List

from ponyai.core.strategy import Signal, SignalType, Strategy


# ---------------------------------------------------------------------------
# Minimal concrete strategy for tests
# ---------------------------------------------------------------------------

class BuyAndHoldStrategy(Strategy):
    """Always generates a BUY signal on the first bar, HOLD thereafter."""

    def generate_signals(self, bar: Dict) -> List[Signal]:
        if self._bars_seen == 1:
            return [Signal(symbol=bar["symbol"], signal_type=SignalType.BUY)]
        return [Signal(symbol=bar["symbol"], signal_type=SignalType.HOLD)]


class MomentumStrategy(Strategy):
    """Buys when price rises above a moving-average window, sells otherwise.

    Parameters
    ----------
    window:
        Look-back period for the simple moving average.
    """

    def __init__(self, name: str = "momentum", window: int = 3) -> None:
        super().__init__(name=name, params={"window": window})
        self._prices: List[float] = []

    def generate_signals(self, bar: Dict) -> List[Signal]:
        price = bar["close"]
        self._prices.append(price)
        window = self.params["window"]
        if len(self._prices) < window:
            return [Signal(symbol=bar["symbol"], signal_type=SignalType.HOLD)]
        sma = sum(self._prices[-window:]) / window
        if price > sma:
            return [Signal(symbol=bar["symbol"], signal_type=SignalType.BUY, strength=0.5)]
        return [Signal(symbol=bar["symbol"], signal_type=SignalType.SELL)]


def make_bars(n: int = 20, symbol: str = "TEST", start_price: float = 100.0) -> List[Dict]:
    """Generate a deterministic sequence of fake OHLCV bars."""
    bars = []
    price = start_price
    for i in range(n):
        close = price + (i % 5) - 2  # simple deterministic pattern
        bars.append(
            {
                "timestamp": f"2024-01-{i + 1:02d}",
                "symbol": symbol,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 10_000.0,
            }
        )
    return bars
