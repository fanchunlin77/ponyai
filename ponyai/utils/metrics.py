"""Performance metrics computed from an equity curve."""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def compute_metrics(
    equity_curve: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Compute standard performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve:
        Sequence of portfolio equity values, one per period.
    risk_free_rate:
        Annualised risk-free rate (default 0).
    periods_per_year:
        Number of periods in a year (252 for daily, 52 for weekly, etc.).

    Returns
    -------
    dict
        ``total_return``, ``annualised_return``, ``annualised_volatility``,
        ``sharpe_ratio``, ``max_drawdown``, ``calmar_ratio``.
    """
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "annualised_return": 0.0,
            "annualised_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    initial = equity_curve[0]
    final = equity_curve[-1]

    # ------------------------------------------------------------------ #
    # Total and annualised return
    # ------------------------------------------------------------------ #
    total_return = (final - initial) / initial
    n_periods = len(equity_curve) - 1
    annualised_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # ------------------------------------------------------------------ #
    # Period returns
    # ------------------------------------------------------------------ #
    period_returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
    ]

    # ------------------------------------------------------------------ #
    # Volatility & Sharpe
    # ------------------------------------------------------------------ #
    n = len(period_returns)
    mean_r = sum(period_returns) / n
    variance = sum((r - mean_r) ** 2 for r in period_returns) / (n - 1) if n > 1 else 0.0
    std_r = math.sqrt(variance)
    annualised_volatility = std_r * math.sqrt(periods_per_year)

    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = [r - period_rf for r in period_returns]
    mean_excess = sum(excess_returns) / len(excess_returns)
    sharpe_ratio = (
        (mean_excess / std_r) * math.sqrt(periods_per_year) if std_r > 0 else 0.0
    )

    # ------------------------------------------------------------------ #
    # Max drawdown
    # ------------------------------------------------------------------ #
    max_drawdown = _compute_max_drawdown(equity_curve)

    # ------------------------------------------------------------------ #
    # Calmar ratio
    # ------------------------------------------------------------------ #
    calmar_ratio = annualised_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }


def _compute_max_drawdown(equity_curve: List[float]) -> float:
    """Return the maximum peak-to-trough drawdown (negative number)."""
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        if drawdown < max_dd:
            max_dd = drawdown
    return max_dd
