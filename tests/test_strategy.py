"""Tests for the strategy base and portfolio."""

from __future__ import annotations

from datetime import datetime
from typing import List

import pytest

from ponyai.data.feed import Bar
from ponyai.strategy.base import Order, OrderSide, Portfolio, Position, Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(close: float = 100.0, symbol: str = "AAPL") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1_000.0,
    )


class AlwaysBuyStrategy(Strategy):
    """Buys a fixed number of shares on every bar."""

    def __init__(self, quantity: float = 10.0, seed: int = None) -> None:
        super().__init__(seed=seed)
        self.quantity = quantity

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
        return [Order(symbol=bar.symbol, side=OrderSide.BUY, quantity=self.quantity)]


class HoldStrategy(Strategy):
    """Never trades."""

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
        return []


# ---------------------------------------------------------------------------
# Order tests
# ---------------------------------------------------------------------------


class TestOrder:
    def test_valid_order(self) -> None:
        o = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5.0)
        assert o.quantity == 5.0

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError):
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=0.0)

    def test_negative_quantity_raises(self) -> None:
        with pytest.raises(ValueError):
            Order(symbol="AAPL", side=OrderSide.SELL, quantity=-1.0)


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------


class TestPortfolio:
    def test_initial_state(self) -> None:
        p = Portfolio(initial_capital=500_000.0)
        assert p.cash == pytest.approx(500_000.0)
        assert p.positions == {}

    def test_buy_reduces_cash(self) -> None:
        p = Portfolio(initial_capital=100_000.0, commission_rate=0.0)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10.0)
        p.fill_order(order, price=100.0)
        assert p.cash == pytest.approx(99_000.0)
        assert p.get_position("AAPL").quantity == pytest.approx(10.0)

    def test_sell_increases_cash(self) -> None:
        p = Portfolio(initial_capital=100_000.0, commission_rate=0.0)
        buy = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10.0)
        p.fill_order(buy, price=100.0)
        sell = Order(symbol="AAPL", side=OrderSide.SELL, quantity=5.0)
        p.fill_order(sell, price=110.0)
        assert p.get_position("AAPL").quantity == pytest.approx(5.0)
        assert p.cash == pytest.approx(100_000.0 - 1_000.0 + 550.0)

    def test_commission_deducted(self) -> None:
        p = Portfolio(initial_capital=100_000.0, commission_rate=0.01)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10.0)
        commission = p.fill_order(order, price=100.0)
        assert commission == pytest.approx(10.0)
        # 10 shares × 100 + 10 commission = 1010 deducted
        assert p.cash == pytest.approx(98_990.0)

    def test_insufficient_cash_raises(self) -> None:
        p = Portfolio(initial_capital=100.0, commission_rate=0.0)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100.0)
        with pytest.raises(ValueError, match="Insufficient cash"):
            p.fill_order(order, price=100.0)

    def test_record_equity(self) -> None:
        p = Portfolio(initial_capital=10_000.0, commission_rate=0.0)
        buy = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10.0)
        p.fill_order(buy, price=100.0)
        equity = p.record_equity({"AAPL": 110.0})
        # cash = 10000 - 1000 = 9000, holdings = 10×110 = 1100 → total = 10100
        assert equity == pytest.approx(10_100.0)

    def test_equity_curve_grows(self) -> None:
        p = Portfolio(initial_capital=10_000.0, commission_rate=0.0)
        p.record_equity({"AAPL": 100.0})
        p.record_equity({"AAPL": 200.0})
        assert len(p.equity_curve) == 2


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestStrategy:
    def test_seeded_strategy_is_deterministic(self) -> None:
        """Two strategies with the same seed must make the same random choices."""

        class RandomStrategy(Strategy):
            def on_bar(self, bar: Bar, portfolio: Portfolio) -> List[Order]:
                # Use the per-instance RNG, not the global random module
                qty = self.rng.uniform(1, 10)
                return [Order(symbol=bar.symbol, side=OrderSide.BUY, quantity=qty)]

        s1 = RandomStrategy(seed=42)
        s2 = RandomStrategy(seed=42)
        bar = _bar()
        portfolio = Portfolio()
        orders1 = s1.on_bar(bar, portfolio)
        orders2 = s2.on_bar(bar, portfolio)
        assert orders1[0].quantity == pytest.approx(orders2[0].quantity)

    def test_abstract_on_bar_must_be_implemented(self) -> None:
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]
