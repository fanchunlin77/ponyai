"""Tests for the backtesting engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from ponyai.backtest.engine import BacktestEngine, BacktestResult
from ponyai.data.feed import Bar, DataFeed
from ponyai.strategy.base import Order, OrderSide, Portfolio, Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feed(closes: List[float], symbol: str = "AAPL") -> DataFeed:
    records = [
        {
            "timestamp": datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            "open": c,
            "high": c + 1,
            "low": c - 1,
            "close": c,
            "volume": 1_000.0,
        }
        for i, c in enumerate(closes)
    ]
    return DataFeed.from_records(records, symbol=symbol, snapshot_id=f"test-{symbol}")


class BuyOnceStrategy(Strategy):
    """Buys 10 shares on the very first bar, then holds."""

    def __init__(self, quantity: float = 10.0, seed: int = None) -> None:
        super().__init__(seed=seed)
        self.quantity = quantity
        self._bought = False

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
        if not self._bought:
            self._bought = True
            return [Order(symbol=bar.symbol, side=OrderSide.BUY, quantity=self.quantity)]
        return []


class HoldStrategy(Strategy):
    def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    def test_result_has_correct_snapshot_id(self) -> None:
        feed = _make_feed([100, 101, 102])
        engine = BacktestEngine(feed, HoldStrategy())
        result = engine.run()
        assert result.snapshot_id == "test-AAPL"

    def test_hold_strategy_preserves_capital(self) -> None:
        feed = _make_feed([100, 110, 120])
        engine = BacktestEngine(feed, HoldStrategy(), initial_capital=10_000.0)
        result = engine.run()
        # No trades → equity equals cash = initial capital
        assert result.final_equity == pytest.approx(10_000.0)

    def test_buy_once_increases_equity_on_rising_market(self) -> None:
        feed = _make_feed([100, 110, 120, 130], symbol="AAPL")
        engine = BacktestEngine(
            feed,
            BuyOnceStrategy(quantity=10.0),
            initial_capital=10_000.0,
            commission_rate=0.0,
        )
        result = engine.run()
        # Bought 10 @ 100 → cost 1000, cash = 9000
        # At bar 4 close = 130: equity = 9000 + 10×130 = 10300
        assert result.final_equity == pytest.approx(10_300.0)

    def test_trade_log_has_one_entry_for_buy_once(self) -> None:
        feed = _make_feed([100, 101, 102])
        engine = BacktestEngine(feed, BuyOnceStrategy(quantity=5.0), commission_rate=0.0)
        result = engine.run()
        assert len(result.trade_log) == 1
        assert result.trade_log[0]["side"] == "BUY"
        assert result.trade_log[0]["quantity"] == pytest.approx(5.0)

    def test_equity_curve_length_equals_bar_count(self) -> None:
        n = 20
        feed = _make_feed(list(range(100, 100 + n)))
        engine = BacktestEngine(feed, HoldStrategy(), initial_capital=50_000.0)
        result = engine.run()
        assert len(result.equity_curve) == n

    def test_metrics_computed(self) -> None:
        feed = _make_feed([100, 101, 102, 103, 104])
        engine = BacktestEngine(feed, HoldStrategy())
        result = engine.run()
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics

    def test_reproducibility_same_seed_same_result(self) -> None:
        """Running the same engine twice must produce identical results."""
        closes = [100, 105, 103, 108, 112, 107, 115]
        feed = _make_feed(closes)

        def _run():
            return BacktestEngine(
                _make_feed(closes),
                BuyOnceStrategy(seed=0),
                initial_capital=10_000.0,
                commission_rate=0.001,
            ).run()

        r1 = _run()
        r2 = _run()
        assert r1.final_equity == pytest.approx(r2.final_equity)
        assert r1.equity_curve == pytest.approx(r2.equity_curve)
        assert r1.snapshot_id == r2.snapshot_id

    def test_insufficient_cash_order_skipped(self) -> None:
        feed = _make_feed([100, 110])

        class TryToBuyTooMuch(Strategy):
            def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
                # Try to buy more than capital allows
                return [Order(symbol=bar.symbol, side=OrderSide.BUY, quantity=1_000_000.0)]

        engine = BacktestEngine(
            feed,
            TryToBuyTooMuch(),
            initial_capital=1_000.0,
            commission_rate=0.0,
        )
        result = engine.run()
        # No trades should have been executed
        assert result.trade_log == []

    def test_strategy_seed_captured_in_result(self) -> None:
        feed = _make_feed([100, 101])
        engine = BacktestEngine(feed, BuyOnceStrategy(seed=99))
        result = engine.run()
        assert result.strategy_seed == 99
