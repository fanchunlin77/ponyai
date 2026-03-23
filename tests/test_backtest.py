"""Tests for ponyai.backtest.engine."""

from __future__ import annotations

import datetime

import pytest

from ponyai.backtest.engine import BacktestEngine, BacktestResult, SimulatedBroker, Trade
from ponyai.data.feed import BarData, DataFeed
from ponyai.strategy.base import BaseStrategy, Signal, SignalDirection
from tests.conftest import AlwaysBuyStrategy, AlwaysFlatStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BuyThenSellStrategy(BaseStrategy):
    """Buys on bar 0, sells (flat) on bar 2."""

    def __init__(self):
        super().__init__()
        self._count = 0

    def on_bar(self, bar: BarData) -> list[Signal]:
        self._count += 1
        if self._count == 1:
            return [Signal(symbol=bar.symbol, direction=SignalDirection.BUY, size=10.0)]
        if self._count == 3:
            return [Signal(symbol=bar.symbol, direction=SignalDirection.FLAT)]
        return []


# ---------------------------------------------------------------------------
# SimulatedBroker
# ---------------------------------------------------------------------------


class TestSimulatedBroker:
    def _bar(self, close=100.0) -> BarData:
        return BarData("T", datetime.datetime(2024, 1, 1), close, close, close, close, 0)

    def test_buy_reduces_cash(self):
        broker = SimulatedBroker(initial_cash=10_000)
        bar = self._bar(100.0)
        broker.fill(Signal("T", SignalDirection.BUY, size=5.0), bar)
        assert broker.cash == pytest.approx(10_000 - 500.0)

    def test_sell_restores_cash(self):
        broker = SimulatedBroker(initial_cash=10_000)
        bar = self._bar(100.0)
        broker.fill(Signal("T", SignalDirection.BUY, size=5.0), bar)
        broker.fill(Signal("T", SignalDirection.SELL, size=5.0), bar)
        assert broker.cash == pytest.approx(10_000.0)

    def test_insufficient_cash_returns_none(self):
        broker = SimulatedBroker(initial_cash=100)
        bar = self._bar(1_000.0)
        trade = broker.fill(Signal("T", SignalDirection.BUY, size=1.0), bar)
        assert trade is None

    def test_equity_with_open_position(self):
        broker = SimulatedBroker(initial_cash=10_000)
        bar = self._bar(100.0)
        broker.fill(Signal("T", SignalDirection.BUY, size=5.0), bar)
        assert broker.equity({"T": 110.0}) == pytest.approx(10_000 - 500 + 5 * 110)

    def test_commission(self):
        broker = SimulatedBroker(initial_cash=10_000, commission_rate=0.01)
        bar = self._bar(100.0)
        broker.fill(Signal("T", SignalDirection.BUY, size=10.0), bar)
        expected_cash = 10_000 - 100 * 10 * 1.01
        assert broker.cash == pytest.approx(expected_cash)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    def test_basic_run(self, simple_feed):
        strat = AlwaysBuyStrategy()
        engine = BacktestEngine(strat, simple_feed, initial_cash=100_000)
        result = engine.run()
        assert isinstance(result, BacktestResult)
        assert result.total_trades == len(simple_feed)

    def test_equity_curve_length(self, simple_feed):
        engine = BacktestEngine(AlwaysBuyStrategy(), simple_feed, initial_cash=100_000)
        result = engine.run()
        assert len(result.equity_curve) == len(simple_feed)

    def test_flat_strategy_no_trades(self, simple_feed):
        engine = BacktestEngine(AlwaysFlatStrategy(), simple_feed)
        result = engine.run()
        assert result.total_trades == 0

    def test_run_only_once(self, simple_feed):
        engine = BacktestEngine(AlwaysFlatStrategy(), simple_feed)
        engine.run()
        with pytest.raises(RuntimeError, match="once"):
            engine.run()

    def test_buy_then_sell_pnl(self, simple_feed):
        strat = BuyThenSellStrategy()
        engine = BacktestEngine(strat, simple_feed, initial_cash=100_000)
        result = engine.run()
        # Bar 0 close=101, Bar 2 close=103 → pnl = (103-101)*10 = 20
        assert result.total_pnl == pytest.approx(20.0)

    def test_result_strategy_desc(self, simple_feed):
        strat = AlwaysBuyStrategy(params={"k": "v"})
        engine = BacktestEngine(strat, simple_feed)
        result = engine.run()
        assert result.strategy_desc["strategy"] == "AlwaysBuyStrategy"
        assert result.strategy_desc["params"] == {"k": "v"}

    def test_feed_name_in_result(self, simple_feed):
        engine = BacktestEngine(AlwaysFlatStrategy(), simple_feed)
        result = engine.run()
        assert result.feed_name == "test-feed"

    def test_reproducibility(self, simple_feed):
        """Running the same config twice must produce identical results."""
        def _run():
            feed = simple_feed.copy()
            strat = BuyThenSellStrategy()
            return BacktestEngine(strat, feed, initial_cash=50_000).run()

        r1, r2 = _run(), _run()
        assert r1.total_pnl == r2.total_pnl
        assert r1.final_equity == r2.final_equity
        assert len(r1.trades) == len(r2.trades)
