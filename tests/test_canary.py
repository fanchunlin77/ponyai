"""Tests for canary (gray) deployment router."""

from __future__ import annotations

import pytest

from ponyai.data import DataFeed
from ponyai.deploy import CanaryRouter
from ponyai.strategy.examples import BuyAndHold, MovingAverageCrossover


class TestCanaryRouter:
    def _default_feed(self) -> DataFeed:
        return DataFeed.synthetic(n_bars=100, seed=42)

    def test_full_stable_weight(self):
        """candidate_weight=0 → all capital in stable."""
        feed = self._default_feed()
        router = CanaryRouter(
            stable=BuyAndHold(quantity=10),
            candidate=MovingAverageCrossover(),
            candidate_weight=0.0,
            initial_cash=1_000_000,
        )
        result = router.run(feed)
        assert result.candidate_weight == 0.0
        assert result.stable_weight == 1.0
        # Candidate portfolio has 0 cash → its equity is zero throughout
        assert all(v == 0.0 for v in result.candidate_equity)

    def test_full_candidate_weight(self):
        """candidate_weight=1 → all capital in candidate."""
        feed = self._default_feed()
        router = CanaryRouter(
            stable=BuyAndHold(quantity=10),
            candidate=MovingAverageCrossover(),
            candidate_weight=1.0,
            initial_cash=1_000_000,
        )
        result = router.run(feed)
        assert all(v == 0.0 for v in result.stable_equity)

    def test_split_equity_sums_to_combined(self):
        feed = self._default_feed()
        router = CanaryRouter(
            stable=BuyAndHold(quantity=10),
            candidate=MovingAverageCrossover(),
            candidate_weight=0.3,
            initial_cash=1_000_000,
        )
        result = router.run(feed)
        combined = result.stable_equity + result.candidate_equity
        for a, b in zip(combined, result.combined_equity):
            assert a == pytest.approx(b)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="candidate_weight"):
            CanaryRouter(
                stable=BuyAndHold(),
                candidate=MovingAverageCrossover(),
                candidate_weight=1.5,
            )

    def test_weight_setter_validates(self):
        router = CanaryRouter(
            stable=BuyAndHold(),
            candidate=MovingAverageCrossover(),
            candidate_weight=0.1,
        )
        with pytest.raises(ValueError):
            router.candidate_weight = -0.1

    def test_weight_update_takes_effect(self):
        router = CanaryRouter(
            stable=BuyAndHold(),
            candidate=MovingAverageCrossover(),
            candidate_weight=0.1,
        )
        router.candidate_weight = 0.5
        assert router.candidate_weight == 0.5

    def test_result_contains_all_metrics(self):
        feed = self._default_feed()
        router = CanaryRouter(
            stable=BuyAndHold(quantity=10),
            candidate=MovingAverageCrossover(),
            candidate_weight=0.2,
        )
        result = router.run(feed)
        assert hasattr(result.stable_metrics, "sharpe_ratio")
        assert hasattr(result.candidate_metrics, "total_return")
        assert hasattr(result.combined_metrics, "max_drawdown")

    def test_equity_length_matches_bars(self):
        n = 80
        feed = DataFeed.synthetic(n_bars=n, seed=5)
        router = CanaryRouter(
            stable=BuyAndHold(quantity=5),
            candidate=BuyAndHold(quantity=3),
            candidate_weight=0.4,
        )
        result = router.run(feed)
        assert len(result.stable_equity) == n
        assert len(result.candidate_equity) == n
        assert len(result.combined_equity) == n
