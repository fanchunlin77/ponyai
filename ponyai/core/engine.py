"""Backtest engine with built-in reproducibility support.

Reproducibility is guaranteed by:
1. Storing a ``seed`` in the run config so random draws are deterministic.
2. Capturing a full snapshot of the config at run-time (strategy name, params,
   data range, seed) in :class:`BacktestResult` so every result is self-describing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .strategy import Order, OrderSide, Strategy


@dataclass
class TradeRecord:
    """A single executed trade."""

    timestamp: Any
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Full result of a backtest run, including a reproducibility snapshot."""

    run_id: str
    config_snapshot: Dict
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[Any] = field(default_factory=list)

    # --- summary metrics (computed after run) ---
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0

    def summary(self) -> Dict:
        return {
            "run_id": self.run_id,
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "config_snapshot": self.config_snapshot,
        }


class BacktestEngine:
    """Event-driven backtest engine.

    Parameters
    ----------
    initial_capital:
        Starting cash in the account.
    commission_rate:
        Fraction of trade value charged as commission (e.g. 0.001 = 0.1 %).
    seed:
        Random seed for reproducibility.  Pass the same seed to get identical
        results across runs.
    slippage_pct:
        Simulated market-impact slippage as a fraction of price.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
        seed: int = 42,
        slippage_pct: float = 0.0005,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.seed = seed
        self.slippage_pct = slippage_pct

        # Seed both stdlib and numpy RNGs for full reproducibility
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: Strategy,
        bars: Sequence[Dict],
        max_position_pct: float = 0.10,
        extra_config: Optional[Dict] = None,
    ) -> BacktestResult:
        """Run *strategy* over the provided *bars* sequence.

        Parameters
        ----------
        strategy:
            An instance of a :class:`~ponyai.core.strategy.Strategy` sub-class.
        bars:
            An ordered sequence of bar dicts, each containing at minimum:
            ``timestamp``, ``symbol``, ``open``, ``high``, ``low``, ``close``,
            ``volume``.
        max_position_pct:
            Passed through to :meth:`Strategy.signals_to_orders`.
        extra_config:
            Any additional key/value pairs to embed in the reproducibility
            snapshot (e.g. data source version, feature flags).

        Returns
        -------
        BacktestResult
            Self-describing result including config snapshot and metrics.
        """
        # --- build reproducibility snapshot ---
        config = self._build_config(strategy, bars, max_position_pct, extra_config)
        run_id = self._make_run_id(config)

        result = BacktestResult(run_id=run_id, config_snapshot=config)

        # --- re-seed so each run() call is isolated even if reused ---
        random.seed(self.seed)
        np.random.seed(self.seed)

        cash = self.initial_capital
        strategy_copy = copy.deepcopy(strategy)
        strategy_copy.on_start()

        cost_basis: Dict[str, float] = {}  # symbol -> avg cost for PnL calculation

        for bar in bars:
            signals = strategy_copy.on_bar(bar)
            orders = strategy_copy.signals_to_orders(
                signals, bar, cash, max_position_pct=max_position_pct
            )

            for order in orders:
                cash, trade = self._execute_order(order, bar, cash, cost_basis)
                if trade:
                    result.trades.append(trade)
                    strategy_copy.update_position(order, trade.fill_price)

            # Mark-to-market equity
            pos_value = sum(
                pos.quantity * bar.get("close", pos.avg_cost)
                for sym, pos in strategy_copy.positions.items()
                if bar.get("symbol") == sym or len(strategy_copy.positions) == 1
            )
            result.equity_curve.append(cash + pos_value)
            result.timestamps.append(bar.get("timestamp"))

        strategy_copy.on_end()

        self._compute_metrics(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_order(
        self,
        order: Order,
        bar: Dict,
        cash: float,
        cost_basis: Dict[str, float],
    ):
        """Simulate order execution with slippage and commission."""
        price = bar.get("close", 0.0)
        if price <= 0:
            return cash, None

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        trade_value = fill_price * order.quantity
        commission = trade_value * self.commission_rate

        if order.side == OrderSide.BUY:
            cost = trade_value + commission
            if cost > cash:
                # Scale down quantity to fit available cash
                affordable = cash / (fill_price * (1 + self.commission_rate))
                if affordable <= 0:
                    return cash, None
                order.quantity = affordable
                trade_value = fill_price * order.quantity
                commission = trade_value * self.commission_rate
                cost = trade_value + commission
            cash -= cost
            cost_basis[order.symbol] = fill_price
            pnl = 0.0
        else:
            avg_cost = cost_basis.get(order.symbol, fill_price)
            pnl = (fill_price - avg_cost) * order.quantity - commission
            cash += trade_value - commission

        trade = TradeRecord(
            timestamp=bar.get("timestamp"),
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            pnl=pnl,
        )
        return cash, trade

    def _build_config(
        self,
        strategy: Strategy,
        bars: Sequence[Dict],
        max_position_pct: float,
        extra_config: Optional[Dict],
    ) -> Dict:
        """Build a fully-descriptive config snapshot for reproducibility."""
        first_bar = bars[0] if bars else {}
        last_bar = bars[-1] if bars else {}
        config: Dict[str, Any] = {
            "strategy_class": type(strategy).__name__,
            "strategy_name": strategy.name,
            "strategy_params": copy.deepcopy(strategy.params),
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "slippage_pct": self.slippage_pct,
            "seed": self.seed,
            "max_position_pct": max_position_pct,
            "bar_count": len(bars),
            "start_timestamp": str(first_bar.get("timestamp", "")),
            "end_timestamp": str(last_bar.get("timestamp", "")),
        }
        if extra_config:
            config.update(extra_config)
        return config

    @staticmethod
    def _make_run_id(config: Dict) -> str:
        """Create a deterministic run ID from the config snapshot."""
        raw = json.dumps(config, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        ts = int(time.time())
        return f"run_{ts}_{digest}"

    @staticmethod
    def _compute_metrics(result: BacktestResult) -> None:
        """Compute summary performance metrics in-place."""
        curve = result.equity_curve
        result.total_trades = len(result.trades)

        if not curve:
            return

        result.total_return = (curve[-1] - curve[0]) / curve[0] if curve[0] else 0.0

        # Daily returns
        returns = [
            (curve[i] - curve[i - 1]) / curve[i - 1]
            for i in range(1, len(curve))
            if curve[i - 1] > 0
        ]

        if returns:
            arr = np.array(returns)
            std = arr.std()
            result.sharpe_ratio = float((arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0)

        # Max drawdown
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd

        # Win rate
        winning = [t for t in result.trades if t.pnl > 0]
        result.win_rate = len(winning) / result.total_trades if result.total_trades else 0.0
