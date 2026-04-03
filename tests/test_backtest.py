"""Tests for ponyai.backtest.engine (integration)."""

from pathlib import Path

import pandas as pd
import pytest

from ponyai.backtest.engine import BacktestEngine
from ponyai.data.feed import DataFeed
from ponyai.strategy.base import Signal, Strategy


class SimpleMA(Strategy):
    """Buy when close > 5-bar SMA, sell when below."""

    def __init__(self, symbol: str = "TEST", window: int = 3) -> None:
        self.symbol = symbol
        self.window = window
        self._closes: list[float] = []
        self._in_position = False

    def on_bar(self, bar: pd.Series) -> list[Signal]:
        self._closes.append(bar["close"])
        if len(self._closes) < self.window:
            return []
        sma = sum(self._closes[-self.window :]) / self.window
        signals: list[Signal] = []
        if bar["close"] > sma and not self._in_position:
            signals.append(Signal(symbol=self.symbol, side="buy", quantity=10))
            self._in_position = True
        elif bar["close"] < sma and self._in_position:
            signals.append(Signal(symbol=self.symbol, side="sell", quantity=10))
            self._in_position = False
        return signals


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "prices.csv"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "open": range(100, 120),
            "high": range(105, 125),
            "low": range(95, 115),
            "close": [100 + i * (1 if i % 3 else -2) for i in range(20)],
            "volume": [1000] * 20,
        }
    )
    df.to_csv(path, index=False)
    return path


class TestBacktestEngine:
    def test_runs_without_error(self, csv_path: Path) -> None:
        feed = DataFeed(path=csv_path)
        strategy = SimpleMA(symbol="TEST", window=3)
        engine = BacktestEngine(feed=feed, strategy=strategy, initial_cash=100_000)
        result = engine.run()
        assert len(result.equity_curve) == 20
        assert "sharpe_ratio" in result.metrics

    def test_initial_cash_preserved_no_trades(self, csv_path: Path) -> None:
        class DoNothing(Strategy):
            def on_bar(self, bar: pd.Series) -> list[Signal]:
                return []

        feed = DataFeed(path=csv_path)
        engine = BacktestEngine(feed=feed, strategy=DoNothing(), initial_cash=50_000)
        result = engine.run()
        assert result.portfolio.cash == 50_000
        assert result.metrics["total_return"] == 0.0
