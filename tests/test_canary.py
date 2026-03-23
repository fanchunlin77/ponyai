"""Tests for ponyai.deploy.canary (grey-release / canary deployment)."""

from __future__ import annotations

import datetime

import pytest

from ponyai.data.feed import BarData
from ponyai.deploy.canary import CanaryRouter, StrategyRegistry
from ponyai.strategy.base import BaseStrategy, Signal, SignalDirection
from tests.conftest import AlwaysBuyStrategy, AlwaysFlatStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(symbol: str = "T", day: int = 1) -> BarData:
    dt = datetime.datetime(2024, 1, day)
    return BarData(symbol, dt, 100, 101, 99, 100, 1000)


# ---------------------------------------------------------------------------
# StrategyRegistry
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_register_and_retrieve(self):
        reg = StrategyRegistry()
        s = AlwaysBuyStrategy()
        reg.register("champ", s, weight=1.0)
        assert "champ" in reg
        assert reg.get_strategy("champ") is s
        assert reg.get_weight("champ") == 1.0

    def test_update_weight(self):
        reg = StrategyRegistry()
        reg.register("champ", AlwaysBuyStrategy(), weight=1.0)
        reg.update_weight("champ", 0.5)
        assert reg.get_weight("champ") == 0.5

    def test_update_weight_unknown(self):
        reg = StrategyRegistry()
        with pytest.raises(KeyError):
            reg.update_weight("nonexistent", 0.5)

    def test_negative_weight_raises(self):
        reg = StrategyRegistry()
        with pytest.raises(ValueError):
            reg.register("x", AlwaysBuyStrategy(), weight=-0.1)

    def test_describe(self):
        reg = StrategyRegistry()
        reg.register("champ", AlwaysBuyStrategy(params={"p": 1}), weight=0.9)
        desc = reg.describe()
        assert len(desc) == 1
        assert desc[0]["name"] == "champ"
        assert desc[0]["weight"] == 0.9
        assert desc[0]["params"] == {"p": 1}

    def test_names(self):
        reg = StrategyRegistry()
        reg.register("a", AlwaysBuyStrategy())
        reg.register("b", AlwaysFlatStrategy())
        assert set(reg.names()) == {"a", "b"}


# ---------------------------------------------------------------------------
# CanaryRouter
# ---------------------------------------------------------------------------


class TestCanaryRouter:
    def _registry(self, champ_w=0.8, chal_w=0.2) -> StrategyRegistry:
        reg = StrategyRegistry()
        reg.register("champion", AlwaysBuyStrategy(), weight=champ_w)
        reg.register("challenger", AlwaysFlatStrategy(), weight=chal_w)
        return reg

    def test_route_returns_name_and_signals(self):
        reg = self._registry()
        router = CanaryRouter(reg, seed=42)
        name, signals = router.route(_bar())
        assert name in ("champion", "challenger")
        assert isinstance(signals, list)

    def test_route_all_weight_on_one(self):
        """When one strategy has weight=0 it is never chosen."""
        reg = StrategyRegistry()
        reg.register("only", AlwaysBuyStrategy(), weight=1.0)
        reg.register("never", AlwaysFlatStrategy(), weight=0.0)
        router = CanaryRouter(reg, seed=7)
        for day in range(1, 21):
            name, _ = router.route(_bar(day=day))
            assert name == "only"

    def test_route_is_deterministic(self):
        """Same seed + weights → same routing decisions."""
        reg1 = self._registry()
        reg2 = self._registry()
        r1 = CanaryRouter(reg1, seed=99)
        r2 = CanaryRouter(reg2, seed=99)
        bars = [_bar(day=d) for d in range(1, 11)]
        routes1 = [r1.route(b)[0] for b in bars]
        routes2 = [r2.route(b)[0] for b in bars]
        assert routes1 == routes2

    def test_all_strategies_receive_bar(self):
        """All registered strategies must see every bar, not just the active one."""

        class CountingStrategy(BaseStrategy):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def on_bar(self, bar: BarData) -> list[Signal]:
                self.calls += 1
                return []

        s1, s2 = CountingStrategy(), CountingStrategy()
        reg = StrategyRegistry()
        reg.register("a", s1, weight=1.0)
        reg.register("b", s2, weight=0.0)
        router = CanaryRouter(reg, seed=0)
        for day in range(1, 6):
            router.route(_bar(day=day))
        assert s1.calls == 5
        assert s2.calls == 5

    def test_empty_registry_raises(self):
        reg = StrategyRegistry()
        router = CanaryRouter(reg)
        with pytest.raises(RuntimeError, match="empty"):
            router.route(_bar())

    def test_all_zero_weights_raises(self):
        reg = StrategyRegistry()
        reg.register("z", AlwaysBuyStrategy(), weight=0.0)
        router = CanaryRouter(reg)
        with pytest.raises(RuntimeError, match="weights"):
            router.route(_bar())

    def test_gradual_rollout(self):
        """Challenger traffic share grows as weight increases."""
        bars = [_bar(symbol=f"S{i}", day=1) for i in range(200)]
        results: dict[float, float] = {}
        for challenger_w in (0.0, 0.2, 0.5, 0.8, 1.0):
            reg = self._registry(champ_w=1 - challenger_w, chal_w=challenger_w)
            router = CanaryRouter(reg, seed=42)
            challenger_hits = sum(
                1 for b in bars if router.route(b)[0] == "challenger"
            )
            results[challenger_w] = challenger_hits / len(bars)
        # Monotone increase
        weights = sorted(results)
        for w1, w2 in zip(weights, weights[1:]):
            assert results[w1] <= results[w2]
