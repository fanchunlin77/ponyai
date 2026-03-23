# ponyai

**可复现、可回测、可灰度上线的量化平台**

A lightweight, event-driven quantitative trading platform built around three core principles:

| 特性 | Feature | Description |
|------|---------|-------------|
| 可复现 | Reproducible | Serialisable data feeds + strategy params → identical results every run |
| 可回测 | Backtestable | Event-driven engine replays historical bars through any strategy |
| 可灰度上线 | Canary deployment | Gradually shift live traffic from champion to challenger strategy |

---

## Architecture

```
ponyai/
├── data/
│   └── feed.py          # BarData, DataFeed – reproducible data layer
├── strategy/
│   └── base.py          # BaseStrategy, Signal, SignalDirection
├── backtest/
│   └── engine.py        # BacktestEngine, SimulatedBroker, BacktestResult
└── deploy/
    └── canary.py        # StrategyRegistry, CanaryRouter (grey release)
```

---

## Quick-start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Define a strategy

```python
from ponyai.data.feed import BarData
from ponyai.strategy.base import BaseStrategy, Signal, SignalDirection

class SMACrossover(BaseStrategy):
    """Buy when fast SMA > slow SMA, sell otherwise."""

    def __init__(self, fast=5, slow=20):
        super().__init__(params={"fast": fast, "slow": slow})
        self._prices = []

    def on_bar(self, bar: BarData) -> list[Signal]:
        self._prices.append(bar.close)
        fast = self.params["fast"]
        slow = self.params["slow"]
        if len(self._prices) < slow:
            return []
        fast_avg = sum(self._prices[-fast:]) / fast
        slow_avg = sum(self._prices[-slow:]) / slow
        if fast_avg > slow_avg:
            return [Signal(bar.symbol, SignalDirection.BUY)]
        return [Signal(bar.symbol, SignalDirection.FLAT)]
```

### 3. Back-test (可回测)

```python
from ponyai.data.feed import DataFeed
from ponyai.backtest.engine import BacktestEngine

feed = DataFeed.from_dicts(records, name="SPY-1D-2023")  # records = list of dicts
engine = BacktestEngine(SMACrossover(fast=5, slow=20), feed, initial_cash=100_000)
result = engine.run()

print(result.total_pnl)        # realised P&L
print(result.final_equity)     # cash + open positions
print(result.total_trades)     # number of fills
```

### 4. Reproducibility (可复现)

```python
# Serialise the feed to JSON and restore it later – results are identical
import json
snapshot = json.dumps(feed.to_dicts())
restored_feed = DataFeed.from_dicts(json.loads(snapshot), name=feed.name)
```

### 5. Canary / grey-release deployment (可灰度上线)

```python
from ponyai.deploy.canary import StrategyRegistry, CanaryRouter

registry = StrategyRegistry()
registry.register("champion",   SMACrossover(fast=5,  slow=20), weight=0.8)
registry.register("challenger", SMACrossover(fast=3,  slow=10), weight=0.2)

router = CanaryRouter(registry, seed=42)  # seed → deterministic routing

for bar in live_feed:
    active_name, signals = router.route(bar)
    # act on signals …

# Gradually increase challenger traffic as confidence grows:
registry.update_weight("champion",   0.5)
registry.update_weight("challenger", 0.5)
```

---

## Running the tests

```bash
pytest
```

All 42 tests must pass.
