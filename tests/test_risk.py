"""Tests for risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ponyai.risk import compute_metrics


def _make_equity(values, start="2023-01-01"):
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates)


class TestComputeMetrics:
    def test_flat_curve_zero_return(self):
        eq = _make_equity([100.0] * 50)
        m = compute_metrics(eq)
        assert m.total_return == pytest.approx(0.0, abs=1e-10)

    def test_rising_curve_positive_return(self):
        values = np.linspace(100, 200, 100)
        m = compute_metrics(_make_equity(values))
        assert m.total_return == pytest.approx(1.0)  # 100% return
        assert m.annualized_return > 0

    def test_max_drawdown_non_positive(self):
        values = [100, 110, 90, 95, 105]
        m = compute_metrics(_make_equity(values))
        assert m.max_drawdown <= 0.0

    def test_no_drawdown_on_monotone_rising(self):
        values = np.linspace(100, 200, 50)
        m = compute_metrics(_make_equity(values))
        assert m.max_drawdown == pytest.approx(0.0, abs=1e-10)

    def test_sharpe_positive_for_rising(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0.001, 0.01, 200)
        eq = 100.0 * np.exp(np.cumsum(rets))
        m = compute_metrics(_make_equity(eq))
        assert m.sharpe_ratio > 0

    def test_short_curve_raises(self):
        eq = _make_equity([100.0])
        with pytest.raises(ValueError, match="at least 2"):
            compute_metrics(eq)

    def test_metrics_str_contains_return(self):
        eq = _make_equity(np.linspace(100, 150, 60))
        m = compute_metrics(eq)
        assert "Total Return" in str(m)
