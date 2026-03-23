"""Performance and risk metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    """Summary statistics for a single backtest run."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    total_trades: int

    def __str__(self) -> str:
        lines = [
            "--- Performance Metrics ---",
            f"  Total Return        : {self.total_return:+.2%}",
            f"  Annualized Return   : {self.annualized_return:+.2%}",
            f"  Annualized Vol      : {self.annualized_volatility:.2%}",
            f"  Sharpe Ratio        : {self.sharpe_ratio:.3f}",
            f"  Max Drawdown        : {self.max_drawdown:.2%}",
            f"  Calmar Ratio        : {self.calmar_ratio:.3f}",
            f"  Win Rate            : {self.win_rate:.2%}",
            f"  Total Trades        : {self.total_trades}",
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_curve: pd.Series,
    fills: Optional[list] = None,
    trading_days_per_year: int = 252,
) -> Metrics:
    """Compute standard performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve:
        Series of portfolio equity values indexed by date (one entry per bar).
    fills:
        Optional list of :class:`~ponyai.broker.Fill` objects used to compute
        win rate.
    trading_days_per_year:
        Used for annualisation; default is 252 (equity markets).
    """
    if len(equity_curve) < 2:
        raise ValueError("equity_curve must have at least 2 data points")

    initial = equity_curve.iloc[0]
    if initial == 0:
        # Portfolio has no capital — return a zero-filled metrics object
        return Metrics(
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            total_trades=len(fills) if fills else 0,
        )

    daily_returns = equity_curve.pct_change().dropna()
    n = len(daily_returns)

    # Total / annualised return
    total_return = equity_curve.iloc[-1] / initial - 1.0
    years = n / trading_days_per_year
    annualized_return = (1 + total_return) ** (1 / max(years, 1e-9)) - 1

    # Volatility
    annualized_volatility = float(daily_returns.std() * np.sqrt(trading_days_per_year))

    # Sharpe (risk-free = 0 for simplicity)
    mean_daily = float(daily_returns.mean())
    std_daily = float(daily_returns.std())
    sharpe_ratio = (
        (mean_daily / std_daily) * np.sqrt(trading_days_per_year)
        if std_daily > 0
        else 0.0
    )

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Calmar
    calmar_ratio = (
        annualized_return / abs(max_drawdown) if max_drawdown < 0 else float("inf")
    )

    # Win rate from fills
    win_rate = 0.0
    total_trades = 0
    if fills:
        from ponyai.broker import OrderSide

        sell_pnls = [f.pnl for f in fills if f.side == OrderSide.SELL]
        total_trades = len(fills)
        if sell_pnls:
            win_rate = sum(1 for p in sell_pnls if p > 0) / len(sell_pnls)

    return Metrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        total_trades=total_trades,
    )
