"""Tests for the backtesting engine — reproducibility is the key property."""

from __future__ import annotations

import pytest

from ponyai.backtest import BacktestEngine
from ponyai.data import DataFeed
from ponyai.strategy.examples import BuyAndHold, MovingAverageCrossover


class TestReproducibility:
    """Same feed + same strategy + same seed → identical results every time."""

    def test_same_inputs_same_result(self):
        feed = DataFeed.synthetic(n_bars=100, seed=42)
        engine = BacktestEngine()

        r1 = engine.run(BuyAndHold(seed=0), feed)
        r2 = engine.run(BuyAndHold(seed=0), feed)

        assert r1.feed_version == r2.feed_version
        assert r1.seed == r2.seed
        assert list(r1.equity_curve) == list(r2.equity_curve)

    def test_different_seed_same_equity_for_deterministic_strategy(self):
        """BuyAndHold is deterministic regardless of seed."""
        feed = DataFeed.synthetic(n_bars=60, seed=1)
        engine = BacktestEngine()
        r1 = engine.run(BuyAndHold(seed=10), feed)
        r2 = engine.run(BuyAndHold(seed=99), feed)
        # BuyAndHold doesn't use the RNG, so curves must be identical
        assert list(r1.equity_curve) == list(r2.equity_curve)


class TestBacktestEngine:
    def test_equity_curve_length(self):
        feed = DataFeed.synthetic(n_bars=50)
        engine = BacktestEngine()
        result = engine.run(BuyAndHold(quantity=10), feed)
        assert len(result.equity_curve) == 50

    def test_equity_increases_with_rising_prices(self):
        """After buying on the first bar of a monotonically rising series the
        equity should be above the initial cash by the end."""
        import numpy as np
        import pandas as pd
        from ponyai.data import Bar

        dates = pd.bdate_range("2023-01-01", periods=20)
        prices = np.linspace(100, 200, 20)
        bars = [
            Bar(
                symbol="UP",
                timestamp=pd.Timestamp(d),
                open=p,
                high=p,
                low=p,
                close=p,
                volume=1e6,
            )
            for d, p in zip(dates, prices)
        ]
        feed = DataFeed(bars, name="rising")
        engine = BacktestEngine(initial_cash=1_000_000)
        result = engine.run(BuyAndHold(quantity=100), feed)
        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]

    def test_metrics_are_populated(self):
        feed = DataFeed.synthetic(n_bars=100)
        engine = BacktestEngine()
        result = engine.run(MovingAverageCrossover(fast_period=5, slow_period=15), feed)
        m = result.metrics
        assert isinstance(m.total_return, float)
        assert isinstance(m.sharpe_ratio, float)
        assert m.max_drawdown <= 0.0

    def test_result_str_contains_strategy_name(self):
        feed = DataFeed.synthetic(n_bars=50)
        engine = BacktestEngine()
        result = engine.run(BuyAndHold(), feed)
        assert "BuyAndHold" in str(result)

    def test_no_bars_raises(self):
        from ponyai.data import DataFeed

        engine = BacktestEngine()
        empty_feed = DataFeed([], name="empty")
        with pytest.raises(ValueError, match="no bars"):
            engine.run(BuyAndHold(), empty_feed)
