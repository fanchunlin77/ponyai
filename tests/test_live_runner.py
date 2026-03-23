"""Tests for the live runner (canary deployment)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from ponyai.data.feed import Bar
from ponyai.live.runner import DeploymentMode, ExecutedOrder, LiveRunner
from ponyai.strategy.base import Order, OrderSide, Portfolio, Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(close: float = 100.0, symbol: str = "AAPL") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1_000.0,
    )


class AlwaysBuyStrategy(Strategy):
    def __init__(self, quantity: float = 10.0) -> None:
        super().__init__()
        self.quantity = quantity

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
        return [Order(symbol=bar.symbol, side=OrderSide.BUY, quantity=self.quantity)]


class HoldStrategy(Strategy):
    def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeploymentModeTransitions:
    def test_promote_shadow_to_canary(self) -> None:
        runner = LiveRunner(HoldStrategy(), mode=DeploymentMode.SHADOW)
        runner.promote()
        assert runner.mode == DeploymentMode.CANARY

    def test_promote_canary_to_full(self) -> None:
        runner = LiveRunner(HoldStrategy(), mode=DeploymentMode.CANARY)
        runner.promote()
        assert runner.mode == DeploymentMode.FULL

    def test_promote_full_stays_full(self) -> None:
        runner = LiveRunner(HoldStrategy(), mode=DeploymentMode.FULL)
        runner.promote()
        assert runner.mode == DeploymentMode.FULL

    def test_demote_full_to_canary(self) -> None:
        runner = LiveRunner(HoldStrategy(), mode=DeploymentMode.FULL)
        runner.demote()
        assert runner.mode == DeploymentMode.CANARY

    def test_demote_canary_to_shadow(self) -> None:
        runner = LiveRunner(HoldStrategy(), mode=DeploymentMode.CANARY)
        runner.demote()
        assert runner.mode == DeploymentMode.SHADOW

    def test_demote_shadow_stays_shadow(self) -> None:
        runner = LiveRunner(HoldStrategy(), mode=DeploymentMode.SHADOW)
        runner.demote()
        assert runner.mode == DeploymentMode.SHADOW


class TestShadowMode:
    def test_shadow_does_not_execute_orders(self) -> None:
        executed_orders: List[Order] = []
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=10.0),
            mode=DeploymentMode.SHADOW,
            initial_capital=100_000.0,
            executor=lambda o, b: executed_orders.append(o),
        )
        runner.on_bar(_bar(close=100.0))
        # executor must NOT have been called in shadow mode
        assert executed_orders == []

    def test_shadow_marks_orders_as_shadow(self) -> None:
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=10.0),
            mode=DeploymentMode.SHADOW,
        )
        results = runner.on_bar(_bar())
        assert len(results) == 1
        assert results[0].is_shadow is True
        assert results[0].executed_quantity == pytest.approx(0.0)

    def test_shadow_does_not_change_portfolio_cash(self) -> None:
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=10.0),
            mode=DeploymentMode.SHADOW,
            initial_capital=50_000.0,
        )
        initial_cash = runner.portfolio.cash
        runner.on_bar(_bar(close=100.0))
        # Shadow mode must not move any cash
        assert runner.portfolio.cash == pytest.approx(initial_cash)


class TestCanaryMode:
    def test_canary_scales_down_quantity(self) -> None:
        executed: List[Order] = []
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=100.0),
            mode=DeploymentMode.CANARY,
            canary_fraction=0.1,
            initial_capital=1_000_000.0,
            commission_rate=0.0,
            executor=lambda o, b: executed.append(o),
        )
        runner.on_bar(_bar(close=100.0))
        assert len(executed) == 1
        assert executed[0].quantity == pytest.approx(10.0)  # 100 × 0.1

    def test_canary_is_not_shadow(self) -> None:
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=10.0),
            mode=DeploymentMode.CANARY,
            canary_fraction=0.5,
            initial_capital=1_000_000.0,
        )
        results = runner.on_bar(_bar())
        assert results[0].is_shadow is False

    def test_canary_fraction_validation(self) -> None:
        with pytest.raises(ValueError):
            LiveRunner(HoldStrategy(), canary_fraction=0.0)
        with pytest.raises(ValueError):
            LiveRunner(HoldStrategy(), canary_fraction=1.5)


class TestFullMode:
    def test_full_executes_at_full_quantity(self) -> None:
        executed: List[Order] = []
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=50.0),
            mode=DeploymentMode.FULL,
            initial_capital=1_000_000.0,
            commission_rate=0.0,
            executor=lambda o, b: executed.append(o),
        )
        runner.on_bar(_bar(close=100.0))
        assert len(executed) == 1
        assert executed[0].quantity == pytest.approx(50.0)


class TestExecutionLog:
    def test_execution_log_accumulates(self) -> None:
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=5.0),
            mode=DeploymentMode.SHADOW,
        )
        runner.on_bar(_bar())
        runner.on_bar(_bar())
        runner.on_bar(_bar())
        assert len(runner.execution_log) == 3

    def test_promote_and_execute(self) -> None:
        """Promote from shadow to canary mid-stream and verify execution changes."""
        executed: List[Order] = []
        runner = LiveRunner(
            AlwaysBuyStrategy(quantity=100.0),
            mode=DeploymentMode.SHADOW,
            canary_fraction=0.2,
            initial_capital=1_000_000.0,
            commission_rate=0.0,
            executor=lambda o, b: executed.append(o),
        )
        # Bar 1 – shadow: executor not called
        runner.on_bar(_bar(close=100.0))
        assert len(executed) == 0

        # Promote to canary
        runner.promote()
        # Bar 2 – canary: executor called with scaled quantity
        runner.on_bar(_bar(close=100.0))
        assert len(executed) == 1
        assert executed[0].quantity == pytest.approx(20.0)  # 100 × 0.2
