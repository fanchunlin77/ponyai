# ponyai

**可复现、可回测、可灰度上线的量化平台** — A quantitative trading platform that is reproducible, backtestable, and supports canary (灰度) deployment.

---

## Features

| Feature | Description |
|---|---|
| 🔁 **Reproducible (可复现)** | Every backtest run stores a full config snapshot (seed, strategy params, data range, engine settings). Re-running with the same snapshot always produces identical results. |
| 📈 **Backtestable (可回测)** | Event-driven backtest engine with OHLCV data feeds, commission, slippage simulation, and built-in performance metrics (return, Sharpe, drawdown, win-rate). |
| 🚦 **Canary Deployment (可灰度上线)** | `CanaryManager` routes live capital between a stable and a candidate strategy version. Gradually increase the canary's traffic share, then promote or rollback — all with a full audit trail. |

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Write a strategy

```python
from typing import Dict, List
from ponyai.core.strategy import Strategy, Signal, SignalType

class MyStrategy(Strategy):
    def generate_signals(self, bar: Dict) -> List[Signal]:
        # Buy when the close is above a fixed threshold
        if bar["close"] > self.params.get("threshold", 100):
            return [Signal(symbol=bar["symbol"], signal_type=SignalType.BUY)]
        return [Signal(symbol=bar["symbol"], signal_type=SignalType.SELL)]
```

### 3. Run a backtest

```python
from ponyai.backtest import BacktestRunner, DataFeed

csv_text = """timestamp,symbol,open,high,low,close,volume
2024-01-01,TEST,99,102,98,101,10000
2024-01-02,TEST,101,105,100,104,12000
2024-01-03,TEST,104,106,102,103,9000"""

feed = DataFeed.from_csv(csv_text)
bars = feed.bars("TEST")

runner = BacktestRunner(initial_capital=100_000, seed=42)
result = runner.run(MyStrategy("my_strat", params={"threshold": 100}), bars)
print(result.summary())
```

**Output (deterministic across runs with the same seed):**
```json
{
  "run_id": "run_..._abc123",
  "total_return": 0.0012,
  "sharpe_ratio": 1.85,
  "max_drawdown": 0.0005,
  "total_trades": 2,
  "win_rate": 1.0,
  "config_snapshot": { "seed": 42, "strategy_params": {"threshold": 100}, ... }
}
```

### 4. Canary (灰度) deployment

```python
from ponyai.canary import CanaryManager, CanaryPolicy, StrategyVersion

stable_v  = StrategyVersion.create(MyStrategy("v1", params={"threshold": 100}))
canary_v  = StrategyVersion.create(MyStrategy("v2", params={"threshold": 102}))

manager = CanaryManager(stable_v)

# Start with 10 % traffic going to the new version
policy = CanaryPolicy(initial_canary_weight=0.10, step_size=0.10)
manager.deploy_canary(canary_v, policy=policy)

print(manager.status())
# {'state': 'CANARY_ACTIVE', 'stable_weight': 0.9, 'canary_weight': 0.1, ...}

# Gradually increase canary weight after observing good performance
for _ in range(8):
    manager.increase_canary_weight()

# Promote when satisfied
manager.promote_canary()
print(manager.status())  # state: STABLE_ONLY, all traffic on new version

# Or rollback if something goes wrong
# manager.rollback()
```

---

## Architecture

```
ponyai/
├── core/
│   ├── strategy.py     # Abstract Strategy base, Signal, Order, Position
│   └── engine.py       # BacktestEngine — reproducible event-driven simulation
├── backtest/
│   ├── data.py         # BarData + DataFeed (CSV / dict loading, filtering)
│   └── runner.py       # BacktestRunner convenience wrapper
├── canary/
│   └── rollout.py      # CanaryManager, CanaryPolicy, StrategyVersion, RolloutState
└── utils/
    └── config.py       # PlatformConfig — serialisable platform-wide settings
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 37 tests should pass.
