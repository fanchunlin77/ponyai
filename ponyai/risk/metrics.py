"""Risk / performance metrics for equity curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def compute_metrics(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute standard risk and performance metrics.

    Args:
        equity: A time-indexed series of portfolio equity values.
        risk_free_rate: Annualised risk-free rate (default 0).

    Returns:
        Dictionary with the following keys:

        * ``total_return`` – cumulative return over the period.
        * ``annual_return`` – annualised compound return.
        * ``annual_volatility`` – annualised standard deviation of returns.
        * ``sharpe_ratio`` – annualised Sharpe ratio.
        * ``max_drawdown`` – largest peak-to-trough decline.
        * ``calmar_ratio`` – annual return / max drawdown.
    """
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    returns = equity.pct_change().dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_days = len(returns)
    annual_return = float((1 + total_return) ** (_TRADING_DAYS / max(n_days, 1)) - 1)
    annual_volatility = float(returns.std() * np.sqrt(_TRADING_DAYS))

    excess_return = annual_return - risk_free_rate
    sharpe_ratio = (
        float(excess_return / annual_volatility) if annual_volatility != 0 else 0.0
    )

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = float(drawdown.min())

    calmar_ratio = (
        float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
    )

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }
