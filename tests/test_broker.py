"""Tests for ponyai.broker (Fill, Portfolio, PaperBroker)."""

from datetime import datetime

from ponyai.broker.fill import Fill
from ponyai.broker.paper import PaperBroker
from ponyai.broker.portfolio import Portfolio
from ponyai.strategy.base import Signal


class TestFill:
    def test_buy_cost(self) -> None:
        fill = Fill("X", "buy", 10, 50.0, datetime(2024, 1, 1))
        assert fill.cost == 500.0

    def test_sell_cost(self) -> None:
        fill = Fill("X", "sell", 10, 50.0, datetime(2024, 1, 1))
        assert fill.cost == -500.0

    def test_commission_included_in_cost(self) -> None:
        fill = Fill("X", "buy", 10, 50.0, datetime(2024, 1, 1), commission=5.0)
        assert fill.cost == 505.0

    def test_buy_pnl(self) -> None:
        fill = Fill("X", "buy", 10, 50.0, datetime(2024, 1, 1))
        assert fill.pnl == -500.0

    def test_sell_pnl(self) -> None:
        fill = Fill("X", "sell", 10, 50.0, datetime(2024, 1, 1))
        assert fill.pnl == 500.0


class TestPortfolio:
    def test_apply_buy(self) -> None:
        port = Portfolio(cash=10_000)
        fill = Fill("X", "buy", 10, 100.0, datetime(2024, 1, 1))
        port.apply_fill(fill)
        assert port.positions["X"] == 10
        assert port.cash == 9_000

    def test_apply_sell(self) -> None:
        port = Portfolio(cash=10_000)
        buy = Fill("X", "buy", 10, 100.0, datetime(2024, 1, 1))
        sell = Fill("X", "sell", 10, 110.0, datetime(2024, 1, 2))
        port.apply_fill(buy)
        port.apply_fill(sell)
        assert port.positions["X"] == 0
        assert port.cash == 10_100.0

    def test_market_value(self) -> None:
        port = Portfolio(cash=5_000)
        fill = Fill("X", "buy", 10, 100.0, datetime(2024, 1, 1))
        port.apply_fill(fill)
        mv = port.market_value({"X": 120.0})
        assert mv == 5_000 - 1_000 + 10 * 120.0  # 4000 + 1200 = 5200

    def test_total_commission(self) -> None:
        port = Portfolio()
        port.apply_fill(Fill("X", "buy", 10, 100.0, datetime(2024, 1, 1), commission=1.0))
        port.apply_fill(Fill("X", "sell", 10, 100.0, datetime(2024, 1, 2), commission=1.5))
        assert port.total_commission == 2.5


class TestPaperBroker:
    def test_execute_creates_fill(self) -> None:
        broker = PaperBroker(commission_rate=0.001)
        sig = Signal(symbol="Y", side="buy", quantity=5)
        fill = broker.execute(sig, price=200.0, timestamp=datetime(2024, 1, 1))
        assert fill.symbol == "Y"
        assert fill.commission == 0.001 * 5 * 200.0

    def test_portfolio_updated(self) -> None:
        broker = PaperBroker(commission_rate=0.0)
        sig = Signal(symbol="Y", side="buy", quantity=5)
        broker.execute(sig, price=200.0, timestamp=datetime(2024, 1, 1))
        assert broker.portfolio.positions["Y"] == 5
        assert broker.portfolio.cash == 100_000 - 1_000
