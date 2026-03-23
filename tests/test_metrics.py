"""Tests for performance metrics."""

from __future__ import annotations

import math

import pytest

from ponyai.utils.metrics import compute_metrics, _compute_max_drawdown


class TestComputeMaxDrawdown:
    def test_no_drawdown(self) -> None:
        assert _compute_max_drawdown([100, 110, 120]) == pytest.approx(0.0)

    def test_full_loss(self) -> None:
        dd = _compute_max_drawdown([100, 50, 0.001])
        assert dd < -0.99

    def test_partial_drawdown(self) -> None:
        # Peak 120, trough 90 → drawdown = (90-120)/120 = -0.25
        dd = _compute_max_drawdown([100, 120, 90])
        assert dd == pytest.approx(-0.25)


class TestComputeMetrics:
    def test_single_bar_returns_zeros(self) -> None:
        m = compute_metrics([100.0])
        assert m["total_return"] == 0.0
        assert m["sharpe_ratio"] == 0.0

    def test_flat_equity_curve(self) -> None:
        m = compute_metrics([100.0, 100.0, 100.0, 100.0])
        assert m["total_return"] == pytest.approx(0.0)
        assert m["annualised_volatility"] == pytest.approx(0.0)
        assert m["max_drawdown"] == pytest.approx(0.0)

    def test_rising_equity_positive_return(self) -> None:
        # 100 → 200 over 252 periods → total return = 100 %
        curve = [100.0 + i * (100.0 / 252) for i in range(253)]
        m = compute_metrics(curve, periods_per_year=252)
        assert m["total_return"] == pytest.approx(1.0, abs=0.01)
        assert m["annualised_return"] > 0

    def test_max_drawdown_negative(self) -> None:
        curve = [100, 120, 80, 90, 110]
        m = compute_metrics(curve)
        assert m["max_drawdown"] < 0

    def test_sharpe_positive_for_trending_up(self) -> None:
        curve = [100 + i * 0.5 for i in range(100)]
        m = compute_metrics(curve, periods_per_year=252)
        assert m["sharpe_ratio"] > 0

    def test_all_keys_present(self) -> None:
        m = compute_metrics([100, 101, 102])
        expected_keys = {
            "total_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
        }
        assert expected_keys.issubset(m.keys())

    def test_calmar_ratio_zero_when_no_drawdown(self) -> None:
        curve = [100 + i for i in range(5)]
        m = compute_metrics(curve)
        assert m["calmar_ratio"] == pytest.approx(0.0)
