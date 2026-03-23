"""Tests for the backtesting engine – reproducibility and correctness."""

from datetime import datetime, timedelta

import pytest

from ponyai.data.feed import Bar, DataFeed
from ponyai.backtest.engine import BacktestEngine, BacktestResult
from ponyai.strategy.base import BaseStrategy, Order, OrderSide, OrderType, StrategyContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feed(n: int = 10, symbol: str = "TEST") -> DataFeed:
    bars = []
    base = datetime(2024, 1, 1)
    for i in range(n):
        price = 100.0 + i
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=base + timedelta(days=i),
                open=price,
                high=price + 2,
                low=price - 1,
                close=price + 1,
                volume=1000.0,
            )
        )
    return DataFeed.from_bars(bars)


class BuyAndHoldStrategy(BaseStrategy):
    """Buy one unit on the first bar, never sell."""

    name = "BuyAndHold"
    _bought = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bought = False

    def on_bar(self, bar: Bar, context: StrategyContext) -> None:
        if not self._bought:
            context.submit_order(
                Order(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=1.0,
                )
            )
            self._bought = True


class DoNothingStrategy(BaseStrategy):
    """Never submits any orders."""

    name = "DoNothing"

    def on_bar(self, bar: Bar, context: StrategyContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    def test_no_trades_do_nothing(self):
        engine = BacktestEngine(initial_cash=100_000.0)
        result = engine.run(DoNothingStrategy(), _make_feed())
        assert result.num_trades == 0
        assert result.final_cash == pytest.approx(100_000.0)

    def test_buy_and_hold_reduces_cash(self):
        engine = BacktestEngine(initial_cash=100_000.0, commission_rate=0.0)
        result = engine.run(BuyAndHoldStrategy(), _make_feed())
        # One buy at open price of first bar (100.0), quantity 1
        assert result.num_trades == 1
        assert result.final_cash == pytest.approx(100_000.0 - 100.0)

    def test_equity_curve_length(self):
        feed = _make_feed(n=5)
        engine = BacktestEngine()
        result = engine.run(DoNothingStrategy(), feed)
        assert len(result.equity_curve) == 5

    def test_result_repr_contains_strategy_name(self):
        engine = BacktestEngine()
        result = engine.run(DoNothingStrategy(), _make_feed())
        assert "DoNothing" in repr(result)


class TestReproducibility:
    """Verify that identical seeds produce identical results."""

    def test_same_seed_same_result(self):
        feed = _make_feed()
        engine1 = BacktestEngine(initial_cash=50_000.0, seed=42)
        engine2 = BacktestEngine(initial_cash=50_000.0, seed=42)
        r1 = engine1.run(BuyAndHoldStrategy(), feed)
        r2 = engine2.run(BuyAndHoldStrategy(), feed)
        assert r1.final_cash == r2.final_cash
        assert r1.equity_curve == r2.equity_curve
        assert r1.num_trades == r2.num_trades

    def test_repeated_run_same_engine_same_result(self):
        """Calling run() twice on the same engine instance is deterministic."""
        feed = _make_feed()
        engine = BacktestEngine(initial_cash=50_000.0, seed=99)
        r1 = engine.run(BuyAndHoldStrategy(), feed)
        r2 = engine.run(BuyAndHoldStrategy(), feed)
        assert r1.final_cash == r2.final_cash
        assert r1.equity_curve == r2.equity_curve

    def test_different_seeds_recorded(self):
        feed = _make_feed()
        r1 = BacktestEngine(seed=1).run(DoNothingStrategy(), feed)
        r2 = BacktestEngine(seed=2).run(DoNothingStrategy(), feed)
        assert r1.seed == 1
        assert r2.seed == 2

    def test_none_seed_recorded(self):
        feed = _make_feed()
        result = BacktestEngine(seed=None).run(DoNothingStrategy(), feed)
        assert result.seed is None


class TestLimitOrders:
    class LimitBuyStrategy(BaseStrategy):
        name = "LimitBuy"
        _bought = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._bought = False

        def on_bar(self, bar: Bar, context: StrategyContext) -> None:
            if not self._bought:
                context.submit_order(
                    Order(
                        symbol=bar.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=1.0,
                        limit_price=200.0,  # very high limit – always fills
                    )
                )
                self._bought = True

    def test_limit_order_fills_when_low_below_limit(self):
        feed = _make_feed(n=5)
        engine = BacktestEngine(initial_cash=100_000.0, commission_rate=0.0)
        result = engine.run(self.LimitBuyStrategy(), feed)
        assert result.num_trades == 1
