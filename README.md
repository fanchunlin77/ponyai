# PonyAI — 可复现、可回测、可灰度上线的量化平台

PonyAI is a **reproducible**, **backtestable**, and **canary-deployable** quantitative trading platform written in Python.

## Core Principles

| Principle | Implementation |
|-----------|----------------|
| 可复现 (Reproducible) | Versioned data snapshots (`snapshot_id`), per-instance seeded RNGs in every `Strategy`. |
| 可回测 (Backtestable) | Event-driven `BacktestEngine` with transaction costs, equity curve, and performance metrics. |
| 可灰度上线 (Canary deployment) | `LiveRunner` supports three deployment modes: `SHADOW` → `CANARY` → `FULL`. |

---

## Quick Start

```python
from datetime import datetime, timezone
import pandas as pd
from ponyai import DataFeed, BacktestEngine, LiveRunner, DeploymentMode
from ponyai.strategy.base import Order, OrderSide, Portfolio, Strategy

# 1. Load historical data (reproducibility: pin snapshot_id)
df = pd.read_csv("data/AAPL.csv", parse_dates=["timestamp"])
feed = DataFeed.from_dataframe(df, symbol="AAPL", snapshot_id="aapl-2024-v1")

# 2. Define a strategy (reproducibility: fix seed)
class MovingAverageCrossStrategy(Strategy):
    def __init__(self, short=5, long=20, seed=42):
        super().__init__(seed=seed)
        self.short = short
        self.long = long
        self._prices = []

    def on_bar(self, bar, portfolio):
        self._prices.append(bar.close)
        if len(self._prices) < self.long:
            return []
        short_ma = sum(self._prices[-self.short:]) / self.short
        long_ma  = sum(self._prices[-self.long:])  / self.long
        pos = portfolio.get_position(bar.symbol)
        if short_ma > long_ma and pos.quantity == 0:
            qty = portfolio.cash // bar.close
            return [Order(bar.symbol, OrderSide.BUY, qty)] if qty > 0 else []
        if short_ma < long_ma and pos.quantity > 0:
            return [Order(bar.symbol, OrderSide.SELL, pos.quantity)]
        return []

# 3. Backtest
engine = BacktestEngine(feed, MovingAverageCrossStrategy(), initial_capital=1_000_000)
result = engine.run()
print(f"snapshot_id : {result.snapshot_id}")
print(f"final equity: {result.final_equity:,.2f}")
print(f"total return: {result.metrics['total_return']:.2%}")
print(f"Sharpe ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"max drawdown: {result.metrics['max_drawdown']:.2%}")

# 4. Canary deployment
runner = LiveRunner(
    MovingAverageCrossStrategy(),
    mode=DeploymentMode.SHADOW,   # start in shadow
    canary_fraction=0.1,          # canary uses 10 % of intended size
)

for bar in feed:                  # replace with a real market data stream
    runner.on_bar(bar)

# Promote once you are satisfied with shadow behaviour
runner.promote()                  # SHADOW → CANARY
# ...monitor...
runner.promote()                  # CANARY → FULL
```

---

## Project Structure

```
ponyai/
├── ponyai/
│   ├── data/
│   │   └── feed.py          # Bar, DataFeed (versioned, snapshot-pinned)
│   ├── strategy/
│   │   └── base.py          # Order, Portfolio, Strategy (seeded RNG)
│   ├── backtest/
│   │   └── engine.py        # BacktestEngine, BacktestResult
│   ├── live/
│   │   └── runner.py        # LiveRunner, DeploymentMode (canary pipeline)
│   └── utils/
│       └── metrics.py       # compute_metrics (Sharpe, drawdown, …)
└── tests/
    ├── test_data_feed.py
    ├── test_strategy.py
    ├── test_backtest.py
    ├── test_live_runner.py
    └── test_metrics.py
```

---

## Installation

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Key APIs

### `DataFeed`

```python
DataFeed.from_dataframe(df, symbol="AAPL", snapshot_id="v1")
DataFeed.from_csv("data.csv", symbol="AAPL")
DataFeed.from_records([{"timestamp": ..., "open": ..., ...}], symbol="BTC")
```

Every `DataFeed` has a `snapshot_id` — a deterministic hash of its contents (or an explicit string you supply). Pinning the `snapshot_id` guarantees that the same data is used across runs.

### `Strategy`

Subclass `Strategy`, implement `on_bar(bar, portfolio) -> List[Order]`, and pass a `seed` for deterministic behaviour:

```python
class MyStrategy(Strategy):
    def on_bar(self, bar, portfolio):
        qty = self.rng.randint(1, 10)  # use self.rng (not global random)
        return [Order(bar.symbol, OrderSide.BUY, qty)]
```

### `BacktestEngine`

```python
result = BacktestEngine(feed, strategy, initial_capital=1_000_000).run()
result.snapshot_id      # data provenance
result.strategy_seed    # strategy provenance
result.metrics          # Sharpe, drawdown, returns, …
result.trade_log        # every executed trade
result.equity_curve     # per-bar equity
```

### `LiveRunner` — Canary Deployment

```python
runner = LiveRunner(strategy, mode=DeploymentMode.SHADOW, canary_fraction=0.1)
runner.on_bar(bar)   # SHADOW: logs orders, no execution
runner.promote()     # → CANARY: executes at 10 % of size
runner.promote()     # → FULL:   executes at full size
runner.demote()      # roll back if needed
```
