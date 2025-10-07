# QRAgent_Bench - Factor Strategy Development Environment

A reinforcement learning environment for AI agents to develop and improve quantitative factor strategies through iterative observation, analysis, and enhancement.

## Overview

This environment allows RL agents to:
- **Observe** market data and factor performance
- **Improve** factor strategies through programmatic modifications
- **Learn** from performance feedback and reward signals

## Key Features

- **Customizable Reward System**: JSON-configurable reward functions
- **Factor DSL**: JSON-based domain-specific language for factor definition
- **Real-time Backtesting**: In-sample performance evaluation with baseline comparison
- **Data Analysis Tools**: Built-in tools for market data exploration
- **Validation System**: Comprehensive factor program validation

## Quick Start

```python
from envs.factor_env import FactorImproveEnv

# Initialize environment
env = FactorImproveEnv(
    data_path="data/ff25_value_weighted.csv",
    test_train_split=0.8,
    timesteps=10,
    baseline_path="factors/baseline.json",
    plot_path="results/plots"
)

# Reset and start episode
obs, info = env.reset()

# Take actions
action = {"type": "OBSERVE", "tool": "describe_data"}
obs, reward, terminated, truncated, info = env.step(action)

action = {"type": "FACTOR_IMPROVE", "new_program": factor_program}
obs, reward, terminated, truncated, info = env.step(action)

action = {"type": "STOP"}
obs, reward, terminated, truncated, info = env.step(action)
```

## Installation

```bash
pip install -r requirements.txt
```

Download Fama-French 25 Portfolios data and save as `data/ff25_value_weighted.csv`.

## Project Structure

```
├── envs/
│   ├── factor_env.py              # Main RL environment
│   ├── reward_calculator.py       # Reward calculation
│   └── default_reward_config.json # Reward configuration
├── engine/
│   ├── backtester.py             # Backtesting engine
│   ├── data_loader.py            # Data loading
│   ├── data_analysis.py          # Data analysis tools
│   ├── metrics.py                # Performance metrics
│   └── plot_backtest_results.py  # Plotting utilities
├── factors/
│   ├── program.py                # Factor DSL operations
│   ├── validate.py               # Validation logic
│   └── baseline.json             # Baseline factor program
├── training/
│   └── sample_training_loop.py   # Example training script
└── data/
    └── ff25_value_weighted.csv   # Market data
```

## Action Types

- **OBSERVE**: Analyze data using built-in tools
  - `describe_data`: Get data statistics and shape
  - `plot_returns`: Generate return distribution plots
  - `analyze_factor_performance`: Analyze factor performance
- **FACTOR_IMPROVE**: Propose new factor strategies
- **STOP**: End episode and trigger final evaluation

## Reward Configuration

Customize rewards by editing `envs/default_reward_config.json`:

```json
{
  "factor_improve": {
    "base_reward_multiplier": 2.0
  },
  "observe": {
    "success_reward": 0.1,
    "failure_reward": -10.0
  },
  "stop": {
    "base_multiplier": 0.7,
    "cost_per_turnover": 0.06,
    "cost_per_step": 0.01
  },
  "validation_error": {
    "reward": -10.0
  }
}
```

## Factor DSL

Define factors using JSON with a directed acyclic graph (DAG) structure:

```json
{
  "nodes": [
    {"id": "x0", "op": "rolling_return", "n": 126},
    {"id": "x1", "op": "rolling_return", "n": 21},
    {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
    {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
    {"id": "score", "op": "zscore_xs", "src": "x3"}
  ],
  "output": "score"
}
```

### Available Operations

- **Time Series**: `rolling_return`, `ema`, `delay`
- **Statistical**: `zscore_xs`, `demean_xs`, `winsor_quantile`, `clip`
- **Arithmetic**: `add`, `sub`, `mul`
- **Combination**: `combine`

## Usage Examples

### Basic Training Loop

```python
from envs.factor_env import FactorImproveEnv

env = FactorImproveEnv(
    data_path="data/ff25_value_weighted.csv",
    test_train_split=0.8,
    timesteps=10,
    baseline_path="factors/baseline.json",
    plot_path="training_plots"
)

obs, info = env.reset()
total_reward = 0.0

while True:
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"Episode Complete! Total Reward: {total_reward:.3f}")
        break
```

### Factor Validation

```python
from factors.validate import validate_program

program = {
    "nodes": [
        {"id": "x0", "op": "rolling_return", "n": 63},
        {"id": "score", "op": "zscore_xs", "src": "x0"}
    ],
    "output": "score"
}

is_valid, errors = validate_program(program)
if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

## Data Format

The environment expects CSV data with:
- **Rows**: Time periods (daily observations)
- **Columns**: Portfolio returns
- **Index**: Datetime index
- **Values**: Decimal returns (e.g., 0.01 for 1% return)

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Turnover**: Portfolio rebalancing frequency
- **Information Ratio**: Active return vs tracking error
- **Maximum Drawdown**: Largest peak-to-trough decline

## License

MIT License