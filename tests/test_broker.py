"""Tests for the paper broker."""

from __future__ import annotations

import pandas as pd
import pytest

from ponyai.broker import PaperBroker, Portfolio
from ponyai.data import Bar
from ponyai.strategy import Order, OrderSide, OrderType


def _make_bar(close: float = 100.0, symbol: str = "SYN") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=pd.Timestamp("2023-01-03"),
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=1_000_000.0,
    )


class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(initial_cash=500_000)
        assert p.cash == 500_000
        assert p.positions == {}

    def test_equity_no_positions(self):
        p = Portfolio(1_000_000)
        assert p.equity({}) == 1_000_000

    def test_apply_buy_fill_updates_cash(self):
        from ponyai.broker import Fill

        p = Portfolio(100_000)
        fill = Fill(
            symbol="A",
            side=OrderSide.BUY,
            quantity=10,
            price=100.0,
            timestamp=pd.Timestamp("2023-01-03"),
            commission=1.0,
        )
        p.apply_fill(fill)
        assert p.cash == pytest.approx(100_000 - 10 * 100.0 - 1.0)
        assert p.position("A") == 10

    def test_apply_sell_fill_updates_cash(self):
        from ponyai.broker import Fill

        p = Portfolio(0)
        buy = Fill("A", OrderSide.BUY, 10, 100.0, pd.Timestamp("2023-01-03"), 0)
        p.cash = 1_500.0
        p.apply_fill(buy)
        sell = Fill("A", OrderSide.SELL, 10, 110.0, pd.Timestamp("2023-01-04"), 1.0)
        p.apply_fill(sell)
        assert p.position("A") == 0
        assert p.cash > 0


class TestPaperBroker:
    def test_buy_order_creates_fill(self):
        broker = PaperBroker(commission_rate=0.0, slippage_rate=0.0)
        portfolio = Portfolio(10_000)
        bar = _make_bar(100.0)
        order = Order(symbol="SYN", side=OrderSide.BUY, quantity=10)
        fill = broker.execute(order, bar, portfolio)
        assert fill is not None
        assert fill.quantity == 10
        assert fill.price == pytest.approx(100.0)

    def test_sell_more_than_held_clamps(self):
        broker = PaperBroker(commission_rate=0.0, slippage_rate=0.0)
        portfolio = Portfolio(10_000)
        bar = _make_bar(100.0)
        # Buy 5
        broker.execute(Order("SYN", OrderSide.BUY, 5), bar, portfolio)
        # Try to sell 20 — should only sell 5
        fill = broker.execute(Order("SYN", OrderSide.SELL, 20), bar, portfolio)
        assert fill is not None
        assert fill.quantity == 5

    def test_buy_without_cash_returns_none(self):
        broker = PaperBroker(commission_rate=0.0, slippage_rate=0.0)
        portfolio = Portfolio(0)
        bar = _make_bar(100.0)
        fill = broker.execute(Order("SYN", OrderSide.BUY, 100), bar, portfolio)
        assert fill is None

    def test_limit_buy_not_triggered_above_price(self):
        broker = PaperBroker(commission_rate=0.0, slippage_rate=0.0)
        portfolio = Portfolio(50_000)
        bar = _make_bar(100.0)
        order = Order("SYN", OrderSide.BUY, 10, OrderType.LIMIT, limit_price=90.0)
        fill = broker.execute(order, bar, portfolio)
        assert fill is None

    def test_wrong_symbol_returns_none(self):
        broker = PaperBroker()
        portfolio = Portfolio(50_000)
        bar = _make_bar(100.0, symbol="SYN")
        order = Order("OTHER", OrderSide.BUY, 10)
        fill = broker.execute(order, bar, portfolio)
        assert fill is None
