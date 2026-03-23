"""High-level backtest runner that ties together the engine and data feed."""

from __future__ import annotations

from typing import Dict, List, Optional

from ponyai.core.engine import BacktestEngine, BacktestResult
from ponyai.core.strategy import Strategy


class BacktestRunner:
    """Convenience wrapper that configures a :class:`BacktestEngine` and runs
    a strategy against a pre-built list of bar dicts.

    This is the main entry point for most users.

    Example
    -------
    >>> runner = BacktestRunner(initial_capital=500_000, seed=0)
    >>> result = runner.run(strategy, bars)
    >>> print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
        seed: int = 42,
        slippage_pct: float = 0.0005,
    ) -> None:
        self._engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            seed=seed,
            slippage_pct=slippage_pct,
        )

    def run(
        self,
        strategy: Strategy,
        bars: List[Dict],
        max_position_pct: float = 0.10,
        extra_config: Optional[Dict] = None,
    ) -> BacktestResult:
        """Run *strategy* over *bars* and return a :class:`BacktestResult`."""
        return self._engine.run(
            strategy=strategy,
            bars=bars,
            max_position_pct=max_position_pct,
            extra_config=extra_config,
        )
