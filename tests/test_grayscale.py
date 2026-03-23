"""Tests for the grayscale router and live engine."""

from datetime import datetime

import pytest

from ponyai.data.feed import Bar
from ponyai.live.engine import GrayscaleRouter, LiveEngine, StrategySlot
from ponyai.strategy.base import BaseStrategy, Order, OrderSide, OrderType, StrategyContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(symbol: str = "AAPL") -> Bar:
    return Bar(symbol, datetime(2024, 1, 1), 100, 105, 98, 103, 10000)


class CountingStrategy(BaseStrategy):
    """Records every bar it receives."""

    name = "Counting"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar_count = 0

    def on_bar(self, bar: Bar, context: StrategyContext) -> None:
        self.bar_count += 1


class OrderingStrategy(BaseStrategy):
    """Always submits a market buy order."""

    name = "Ordering"

    def on_bar(self, bar: Bar, context: StrategyContext) -> None:
        context.submit_order(
            Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
            )
        )


# ---------------------------------------------------------------------------
# GrayscaleRouter tests
# ---------------------------------------------------------------------------


class TestGrayscaleRouter:
    def test_single_slot_always_routes_there(self):
        strat = CountingStrategy()
        router = GrayscaleRouter(slots=[StrategySlot("prod", strat, weight=1.0)])
        for symbol in ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]:
            assert router.route(symbol).name == "prod"

    def test_routing_is_deterministic(self):
        s1, s2 = CountingStrategy(), CountingStrategy()
        router = GrayscaleRouter(
            slots=[
                StrategySlot("prod", s1, weight=0.9),
                StrategySlot("canary", s2, weight=0.1),
            ],
            seed=42,
        )
        results = [router.route("AAPL").name for _ in range(10)]
        # Same key always maps to same slot
        assert len(set(results)) == 1

    def test_same_seed_same_routing(self):
        s1a, s1b = CountingStrategy(), CountingStrategy()
        s2a, s2b = CountingStrategy(), CountingStrategy()
        router_a = GrayscaleRouter(
            slots=[
                StrategySlot("prod", s1a, weight=0.8),
                StrategySlot("canary", s2a, weight=0.2),
            ],
            seed=7,
        )
        router_b = GrayscaleRouter(
            slots=[
                StrategySlot("prod", s1b, weight=0.8),
                StrategySlot("canary", s2b, weight=0.2),
            ],
            seed=7,
        )
        symbols = [f"SYM{i}" for i in range(20)]
        routes_a = [router_a.route(s).name for s in symbols]
        routes_b = [router_b.route(s).name for s in symbols]
        assert routes_a == routes_b

    def test_different_seeds_may_differ(self):
        s1, s2 = CountingStrategy(), CountingStrategy()
        router1 = GrayscaleRouter(
            slots=[
                StrategySlot("prod", s1, weight=0.5),
                StrategySlot("canary", s2, weight=0.5),
            ],
            seed=1,
        )
        router2 = GrayscaleRouter(
            slots=[
                StrategySlot("prod", s1, weight=0.5),
                StrategySlot("canary", s2, weight=0.5),
            ],
            seed=999,
        )
        symbols = [f"SYM{i}" for i in range(50)]
        routes1 = [router1.route(s).name for s in symbols]
        routes2 = [router2.route(s).name for s in symbols]
        # With 50 symbols and 50/50 split, different seeds should yield
        # at least some difference.
        assert routes1 != routes2

    def test_weight_update_changes_routing(self):
        s1, s2 = CountingStrategy(), CountingStrategy()
        router = GrayscaleRouter(
            slots=[
                StrategySlot("prod", s1, weight=1.0),
                StrategySlot("canary", s2, weight=0.0),
            ],
            seed=0,
        )
        # Before update: all traffic goes to prod
        for sym in [f"X{i}" for i in range(20)]:
            assert router.route(sym).name == "prod"

        # After update: all traffic goes to canary
        router.update_weight("prod", 0.0)
        router.update_weight("canary", 1.0)
        for sym in [f"X{i}" for i in range(20)]:
            assert router.route(sym).name == "canary"

    def test_update_weight_unknown_slot_raises(self):
        router = GrayscaleRouter(slots=[StrategySlot("prod", CountingStrategy(), weight=1.0)])
        with pytest.raises(KeyError):
            router.update_weight("nonexistent", 0.5)

    def test_empty_slots_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            GrayscaleRouter(slots=[])

    def test_zero_total_weight_raises(self):
        with pytest.raises(ValueError, match="Total weight"):
            GrayscaleRouter(slots=[StrategySlot("a", CountingStrategy(), weight=0.0)])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            StrategySlot("a", CountingStrategy(), weight=-1.0)


# ---------------------------------------------------------------------------
# LiveEngine tests
# ---------------------------------------------------------------------------


class TestLiveEngine:
    def _make_engine(self) -> LiveEngine:
        prod = CountingStrategy()
        canary = CountingStrategy()
        router = GrayscaleRouter(
            slots=[
                StrategySlot("prod", prod, weight=1.0),
                StrategySlot("canary", canary, weight=0.0),
            ],
            seed=0,
        )
        return LiveEngine(router=router, initial_cash=100_000.0)

    def test_on_bar_returns_slot_name_and_orders(self):
        engine = self._make_engine()
        engine.start()
        slot_name, orders = engine.on_bar(_bar("AAPL"))
        assert slot_name == "prod"
        assert orders == []

    def test_ordering_strategy_emits_order(self):
        ordering = OrderingStrategy()
        router = GrayscaleRouter(
            slots=[StrategySlot("prod", ordering, weight=1.0)], seed=0
        )
        engine = LiveEngine(router=router, initial_cash=100_000.0)
        engine.start()
        _, orders = engine.on_bar(_bar("AAPL"))
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_context_isolated_per_slot(self):
        prod = OrderingStrategy()
        canary = CountingStrategy()
        router = GrayscaleRouter(
            slots=[
                StrategySlot("prod", prod, weight=0.5),
                StrategySlot("canary", canary, weight=0.5),
            ],
            seed=0,
        )
        engine = LiveEngine(router=router, initial_cash=50_000.0)
        ctx_prod = engine.get_context("prod")
        ctx_canary = engine.get_context("canary")
        assert ctx_prod is not ctx_canary
        assert ctx_prod.cash == 50_000.0
        assert ctx_canary.cash == 50_000.0

    def test_start_stop_lifecycle(self):
        engine = self._make_engine()
        engine.start()
        engine.stop()  # should not raise

    def test_grayscale_ramp_up(self):
        """Simulate a canary ramp from 0 -> 10% -> 100%."""
        prod = CountingStrategy()
        canary = CountingStrategy()
        router = GrayscaleRouter(
            slots=[
                StrategySlot("prod", prod, weight=1.0),
                StrategySlot("canary", canary, weight=0.0),
            ],
            seed=42,
        )
        engine = LiveEngine(router=router)
        engine.start()

        symbols = [f"SYM{i}" for i in range(100)]

        # Phase 1: 0% canary
        phase1_canary = sum(
            1 for s in symbols if router.route(s).name == "canary"
        )
        assert phase1_canary == 0

        # Phase 2: 10% canary
        router.update_weight("canary", 0.1)
        router.update_weight("prod", 0.9)
        phase2_canary = sum(
            1 for s in symbols if router.route(s).name == "canary"
        )
        assert 0 < phase2_canary < len(symbols)

        # Phase 3: 100% canary
        router.update_weight("prod", 0.0)
        router.update_weight("canary", 1.0)
        phase3_canary = sum(
            1 for s in symbols if router.route(s).name == "canary"
        )
        assert phase3_canary == len(symbols)

        engine.stop()
