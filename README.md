# PonyAI – 可复现、可回测、可灰度上线的量化平台

PonyAI is a lightweight, pure-Python quantitative trading platform built around three core principles:

| Principle | Chinese | Description |
|-----------|---------|-------------|
| **Reproducible** | 可复现 | Every backtest run is deterministic when a seed is provided. Results are archived and can be replayed exactly. |
| **Backtestable** | 可回测 | An event-driven engine replays historical OHLCV bars against any strategy, tracking positions, cash, and equity curve. |
| **Grayscale deployment** | 可灰度上线 | A router assigns live bar traffic to strategy slots by weight, enabling canary releases with configurable ramp-up. |

---

## Quick Start

```bash
pip install -e ".[dev]"
```

### 1 – Define a strategy

```python
from ponyai import BaseStrategy, Order, OrderSide, OrderType

class SMACrossStrategy(BaseStrategy):
    name = "SMACross"

    def __init__(self):
        super().__init__()
        self._prices = []

    def on_bar(self, bar, context):
        self._prices.append(bar.close)
        if len(self._prices) < 20:
            return
        short_ma = sum(self._prices[-5:]) / 5
        long_ma  = sum(self._prices[-20:]) / 20
        pos = context.get_position(bar.symbol)
        if short_ma > long_ma and pos.is_flat:
            context.submit_order(Order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
            ))
```

### 2 – Load data and backtest

```python
from ponyai import DataFeed, BacktestEngine

feed   = DataFeed.from_csv(open("data/AAPL.csv").read(), symbol="AAPL")
engine = BacktestEngine(initial_cash=1_000_000.0, seed=42)   # seed → reproducible
result = engine.run(SMACrossStrategy(), feed)

print(result)
# BacktestResult(strategy='SMACross', return=12.34%, trades=18, seed=42)
print("Equity curve:", result.equity_curve[-5:])
```

### 3 – Grayscale live deployment

```python
from ponyai.live.engine import GrayscaleRouter, LiveEngine, StrategySlot

router = GrayscaleRouter(slots=[
    StrategySlot("production", prod_strategy, weight=0.9),
    StrategySlot("canary",     new_strategy,  weight=0.1),
], seed=0)

engine = LiveEngine(router=router, initial_cash=1_000_000.0)
engine.start()

for bar in live_bar_stream:
    slot_name, orders = engine.on_bar(bar)
    print(f"Bar routed to {slot_name!r}, {len(orders)} orders")

# Ramp canary to 50 %
router.update_weight("production", 0.5)
router.update_weight("canary",     0.5)

# Full roll-out
router.update_weight("production", 0.0)
router.update_weight("canary",     1.0)

engine.stop()
```

---

## Architecture

```
ponyai/
├── data/
│   └── feed.py          # Bar dataclass, DataFeed (CSV loader, slice)
├── strategy/
│   └── base.py          # BaseStrategy, Order, Position, StrategyContext
├── backtest/
│   └── engine.py        # BacktestEngine (deterministic, seeded)
└── live/
    └── engine.py        # GrayscaleRouter, LiveEngine
tests/
├── test_data.py
├── test_backtest.py     # reproducibility & correctness tests
└── test_grayscale.py    # routing, ramp-up, lifecycle tests
```

---

## Running Tests

```bash
pytest tests/ -v
```
