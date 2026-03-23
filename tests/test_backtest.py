"""Tests for the backtest engine and runner."""

from __future__ import annotations

import pytest

from ponyai.backtest import BacktestRunner, DataFeed
from ponyai.core.engine import BacktestEngine
from tests.conftest import BuyAndHoldStrategy, MomentumStrategy, make_bars


# ---------------------------------------------------------------------------
# BacktestEngine basic run
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_run_returns_result(self):
        engine = BacktestEngine(initial_capital=100_000, seed=0)
        strategy = BuyAndHoldStrategy(name="bah")
        bars = make_bars(10)
        result = engine.run(strategy, bars)
        assert result is not None
        assert result.run_id.startswith("run_")
        assert len(result.equity_curve) == 10
        assert len(result.timestamps) == 10

    def test_config_snapshot_contains_required_keys(self):
        engine = BacktestEngine(seed=99)
        strategy = BuyAndHoldStrategy(name="bah", params={"a": 1})
        bars = make_bars(5)
        result = engine.run(strategy, bars)
        snap = result.config_snapshot
        assert snap["seed"] == 99
        assert snap["strategy_class"] == "BuyAndHoldStrategy"
        assert snap["strategy_name"] == "bah"
        assert snap["strategy_params"] == {"a": 1}
        assert snap["bar_count"] == 5

    def test_metrics_are_computed(self):
        engine = BacktestEngine(initial_capital=100_000, seed=0)
        strategy = MomentumStrategy(window=3)
        bars = make_bars(20)
        result = engine.run(strategy, bars)
        # total_return can be negative or positive; just verify it's a float
        assert isinstance(result.total_return, float)
        assert result.max_drawdown >= 0.0
        assert 0.0 <= result.win_rate <= 1.0

    def test_run_produces_trades(self):
        engine = BacktestEngine(initial_capital=100_000, seed=0)
        strategy = MomentumStrategy(window=3)
        bars = make_bars(20)
        result = engine.run(strategy, bars)
        assert result.total_trades == len(result.trades)


# ---------------------------------------------------------------------------
# BacktestRunner convenience wrapper
# ---------------------------------------------------------------------------

class TestBacktestRunner:
    def test_runner_delegates_to_engine(self):
        runner = BacktestRunner(initial_capital=50_000, seed=1)
        strategy = BuyAndHoldStrategy(name="bah")
        bars = make_bars(10)
        result = runner.run(strategy, bars)
        assert result.config_snapshot["initial_capital"] == 50_000
        assert result.config_snapshot["seed"] == 1

    def test_extra_config_included(self):
        runner = BacktestRunner(seed=42)
        strategy = BuyAndHoldStrategy(name="bah")
        bars = make_bars(5)
        result = runner.run(strategy, bars, extra_config={"data_version": "v2"})
        assert result.config_snapshot["data_version"] == "v2"


# ---------------------------------------------------------------------------
# DataFeed
# ---------------------------------------------------------------------------

class TestDataFeed:
    CSV = """\
timestamp,symbol,open,high,low,close,volume
2024-01-01,AAPL,150,155,148,153,1000000
2024-01-02,AAPL,153,158,152,157,1200000
2024-01-03,AAPL,157,160,155,159,900000
2024-01-01,TSLA,200,205,198,203,500000
"""

    def test_from_csv_parses_correctly(self):
        feed = DataFeed.from_csv(self.CSV)
        assert len(feed) == 4
        assert set(feed.symbols()) == {"AAPL", "TSLA"}

    def test_filter_by_symbol(self):
        feed = DataFeed.from_csv(self.CSV)
        aapl = feed.bars("AAPL")
        assert all(b["symbol"] == "AAPL" for b in aapl)
        assert len(aapl) == 3

    def test_filter_by_date_range(self):
        feed = DataFeed.from_csv(self.CSV)
        bars = feed.bars(start="2024-01-02", end="2024-01-02")
        assert len(bars) == 1  # only the AAPL bar on 2024-01-02 (TSLA is 2024-01-01)
