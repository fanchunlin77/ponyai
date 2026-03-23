"""Tests for the canary (灰度) deployment manager."""

from __future__ import annotations

import pytest

from ponyai.canary import CanaryManager, CanaryPolicy, RolloutState, StrategyVersion
from tests.conftest import BuyAndHoldStrategy, MomentumStrategy, make_bars


def _make_version(name: str = "strategy") -> StrategyVersion:
    strategy = BuyAndHoldStrategy(name=name)
    return StrategyVersion.create(strategy)


class TestCanaryManager:
    """Basic lifecycle tests."""

    def test_initial_state_is_stable_only(self):
        manager = CanaryManager(_make_version("stable"))
        assert manager.state == RolloutState.STABLE_ONLY
        assert manager.canary_weight == 0.0
        assert manager.stable_weight == 1.0

    def test_deploy_canary_changes_state(self):
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"))
        assert manager.state == RolloutState.CANARY_ACTIVE
        assert manager.canary_weight == 0.10  # default initial weight
        assert manager.stable_weight == pytest.approx(0.90)

    def test_deploy_canary_with_custom_policy(self):
        policy = CanaryPolicy(initial_canary_weight=0.20)
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"), policy=policy)
        assert manager.canary_weight == pytest.approx(0.20)

    def test_deploy_canary_twice_raises(self):
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary1"))
        with pytest.raises(RuntimeError, match="already active"):
            manager.deploy_canary(_make_version("canary2"))

    def test_increase_weight(self):
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"))
        new_weight = manager.increase_canary_weight()
        assert new_weight == pytest.approx(0.20)

    def test_increase_weight_capped_at_max(self):
        policy = CanaryPolicy(initial_canary_weight=0.90, max_canary_weight=1.0, step_size=0.20)
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"), policy=policy)
        manager.increase_canary_weight()
        assert manager.canary_weight == pytest.approx(1.0)

    def test_promote_canary(self):
        stable_version = _make_version("stable")
        canary_version = _make_version("canary")
        manager = CanaryManager(stable_version)
        manager.deploy_canary(canary_version)
        promoted = manager.promote_canary()
        assert promoted.version_id == canary_version.version_id
        assert manager.state == RolloutState.STABLE_ONLY
        assert manager.canary_version is None
        assert manager.stable_version.version_id == canary_version.version_id

    def test_rollback_restores_stable(self):
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"))
        manager.rollback()
        assert manager.state == RolloutState.STABLE_ONLY
        assert manager.canary_version is None
        assert manager.canary_weight == 0.0

    def test_rollback_no_canary_is_noop(self):
        manager = CanaryManager(_make_version("stable"))
        manager.rollback()  # should not raise
        assert manager.state == RolloutState.STABLE_ONLY

    def test_promote_without_canary_raises(self):
        manager = CanaryManager(_make_version("stable"))
        with pytest.raises(RuntimeError):
            manager.promote_canary()

    def test_increase_weight_without_canary_raises(self):
        manager = CanaryManager(_make_version("stable"))
        with pytest.raises(RuntimeError):
            manager.increase_canary_weight()


class TestCanaryRouting:
    """Signal routing tests."""

    def test_route_returns_two_order_lists(self):
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"))
        bar = make_bars(1)[0]
        stable_orders, canary_orders = manager.route_signals(bar, total_capital=100_000)
        # Both lists are returned; stable should have orders (BuyAndHold buys on bar 1)
        assert isinstance(stable_orders, list)
        assert isinstance(canary_orders, list)

    def test_route_with_no_canary_returns_empty_canary_orders(self):
        manager = CanaryManager(_make_version("stable"))
        bar = make_bars(1)[0]
        stable_orders, canary_orders = manager.route_signals(bar, total_capital=100_000)
        assert canary_orders == []

    def test_auto_promote_at_threshold(self):
        policy = CanaryPolicy(
            initial_canary_weight=1.0,
            max_canary_weight=1.0,
            step_size=0.10,
            auto_promote_threshold=1.0,
        )
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"), policy=policy)
        bar = make_bars(1)[0]
        # First call with weight == threshold should auto-promote
        manager.route_signals(bar, total_capital=100_000)
        assert manager.state == RolloutState.STABLE_ONLY
        assert manager.canary_version is None


class TestCanaryStatus:
    """Status and history tests."""

    def test_status_contains_expected_keys(self):
        manager = CanaryManager(_make_version("stable"))
        status = manager.status()
        assert "state" in status
        assert "stable_version_id" in status
        assert "canary_version_id" in status
        assert "stable_weight" in status
        assert "canary_weight" in status

    def test_history_records_events(self):
        manager = CanaryManager(_make_version("stable"))
        manager.deploy_canary(_make_version("canary"))
        manager.increase_canary_weight()
        manager.rollback()
        history = manager.history()
        events = [h["event"] for h in history]
        assert "deploy_canary" in events
        assert "increase_weight" in events
        assert "rollback" in events


class TestCanaryPolicy:
    """Validation tests for CanaryPolicy."""

    def test_invalid_initial_weight_raises(self):
        with pytest.raises(ValueError):
            CanaryPolicy(initial_canary_weight=0.0)

    def test_invalid_step_size_raises(self):
        with pytest.raises(ValueError):
            CanaryPolicy(step_size=1.5)

    def test_invalid_max_weight_raises(self):
        with pytest.raises(ValueError):
            CanaryPolicy(max_canary_weight=0.0)
