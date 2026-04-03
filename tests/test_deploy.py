"""Tests for ponyai.deploy.canary."""

import pandas as pd
import pytest

from ponyai.deploy.canary import CanaryRouter
from ponyai.strategy.base import Signal, Strategy


class AlwaysBuy(Strategy):
    def on_bar(self, bar: pd.Series) -> list[Signal]:
        return [Signal(symbol="A", side="buy", quantity=1)]


class AlwaysSell(Strategy):
    def on_bar(self, bar: pd.Series) -> list[Signal]:
        return [Signal(symbol="A", side="sell", quantity=1)]


class TestCanaryRouter:
    def test_all_baseline(self) -> None:
        router = CanaryRouter(
            baseline=AlwaysBuy(),
            canary=AlwaysSell(),
            canary_weight=0.0,
            seed=0,
        )
        bar = pd.Series({"close": 100})
        signals = router.route(bar)
        assert all(s.side == "buy" for s in signals)

    def test_all_canary(self) -> None:
        router = CanaryRouter(
            baseline=AlwaysBuy(),
            canary=AlwaysSell(),
            canary_weight=1.0,
            seed=0,
        )
        bar = pd.Series({"close": 100})
        signals = router.route(bar)
        assert all(s.side == "sell" for s in signals)

    def test_mixed_routing(self) -> None:
        router = CanaryRouter(
            baseline=AlwaysBuy(),
            canary=AlwaysSell(),
            canary_weight=0.5,
            seed=42,
        )
        bar = pd.Series({"close": 100})
        sides = [router.route(bar)[0].side for _ in range(100)]
        assert "buy" in sides and "sell" in sides

    def test_set_weight_validation(self) -> None:
        router = CanaryRouter(baseline=AlwaysBuy(), canary=AlwaysSell())
        with pytest.raises(ValueError):
            router.set_canary_weight(1.5)
        with pytest.raises(ValueError):
            router.set_canary_weight(-0.1)
