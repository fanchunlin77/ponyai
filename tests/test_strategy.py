"""Tests for example strategies."""

from __future__ import annotations

import pandas as pd
import pytest

from ponyai.data import Bar, DataFeed
from ponyai.broker import Portfolio
from ponyai.strategy.examples import BuyAndHold, MovingAverageCrossover
from ponyai.strategy import OrderSide


def _bar(close: float, symbol: str = "SYN", ts: str = "2023-01-03") -> Bar:
    return Bar(symbol=symbol, timestamp=pd.Timestamp(ts), open=close,
               high=close, low=close, close=close, volume=1e6)


class TestBuyAndHold:
    def test_buys_on_first_bar(self):
        strategy = BuyAndHold(quantity=50)
        p = Portfolio(100_000)
        orders = strategy.on_bar(_bar(100), p)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == 50

    def test_no_order_after_first_bar(self):
        strategy = BuyAndHold(quantity=50)
        p = Portfolio(100_000)
        strategy.on_bar(_bar(100), p)
        orders = strategy.on_bar(_bar(101), p)
        assert orders == []

    def test_reset_clears_bought_flag(self):
        strategy = BuyAndHold(quantity=50)
        p = Portfolio(100_000)
        strategy.on_bar(_bar(100), p)
        strategy.reset()
        orders = strategy.on_bar(_bar(100), p)
        assert len(orders) == 1


class TestMovingAverageCrossover:
    def test_no_signal_before_slow_period(self):
        strategy = MovingAverageCrossover(fast_period=5, slow_period=10)
        p = Portfolio(100_000)
        for i in range(9):
            orders = strategy.on_bar(_bar(100 + i, ts=f"2023-01-{i+3:02d}"), p)
            assert orders == []

    def test_buy_signal_on_uptrend(self):
        """Rising prices → fast MA > slow MA → BUY."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5, quantity=10)
        p = Portfolio(100_000)
        prices = list(range(90, 110))  # monotonically rising
        dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
        orders_all = []
        for price, ts in zip(prices, dates):
            orders = strategy.on_bar(_bar(float(price), ts=str(ts.date())), p)
            orders_all.extend(orders)
        buy_orders = [o for o in orders_all if o.side == OrderSide.BUY]
        assert len(buy_orders) >= 1
