"""Tests for ponyai.risk.metrics."""

import numpy as np
import pandas as pd
import pytest

from ponyai.risk.metrics import compute_metrics


class TestComputeMetrics:
    def test_flat_equity(self) -> None:
        equity = pd.Series([100.0] * 10)
        m = compute_metrics(equity)
        assert m["total_return"] == 0.0
        assert m["max_drawdown"] == 0.0

    def test_positive_return(self) -> None:
        equity = pd.Series([100.0, 110.0, 121.0])
        m = compute_metrics(equity)
        assert m["total_return"] == pytest.approx(0.21, rel=1e-6)
        assert m["annual_return"] > 0

    def test_max_drawdown(self) -> None:
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        m = compute_metrics(equity)
        # Peak = 120, trough = 90 → drawdown = (90-120)/120 = -0.25
        assert m["max_drawdown"] == pytest.approx(-0.25, rel=1e-6)

    def test_single_value(self) -> None:
        equity = pd.Series([100.0])
        m = compute_metrics(equity)
        assert m["sharpe_ratio"] == 0.0

    def test_sharpe_positive(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        equity = pd.Series(100 * np.cumprod(1 + returns))
        m = compute_metrics(equity)
        assert m["sharpe_ratio"] > 0
