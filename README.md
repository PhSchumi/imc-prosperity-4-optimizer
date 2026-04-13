# IMC Prosperity 4 Optimizer

Hyperparameter optimizer for [IMC Prosperity 4](https://prosperity.imc.com/) trading algorithms, based on [Optuna](https://optuna.org/).

**Warning: Overfitting hyperparameters on training data can lead to poor performance on test data. Always optimize on multiple days and use few parameters.**

## Features

- **TPE optimization** - Efficient Bayesian optimization via Optuna
- **Grid search** - Exhaustive search over parameter space
- **Multi-objective** - Maximize profit while minimizing risk
- **Hyperparameter importance** - Identifies which parameters actually matter
- **Auto-export** - Save optimized algorithm with best parameters

## Installation

```bash
# From local directory
pip install -e /path/to/imc-prosperity-4-optimizer

# Requires prosperity4bt backtester
pip install prosperity4bt
```

## Quick Start

### 1. Annotate hyperparameters in your algorithm

```python
class Trader:
    # Only optimize the parameters that matter
    FAIR_VALUE_ROLLING_WEIGHT = 0.8  # opt: float(0.6, 0.95, step=0.05)
    FAIR_VALUE_MICRO_WEIGHT = 0.2    # opt: float(0.05, 0.4, step=0.05)
    POSITION_LIMIT = 60              # opt: int(40, 80)
```

Supported types after `# opt:`:
- `int(low, high, step=1)` - Integer range
- `float(low, high, step=0.1)` - Float range (step required for grid search)
- `categorical(["a", "b", "c"])` - Discrete choices

### 2. Run optimization

```bash
# Basic optimization on round 0
prosperity4opt algorithm.py 0 --trials 100

# Multiple rounds to avoid overfitting
prosperity4opt algorithm.py 0 1 2 --trials 200

# Export optimized algorithm
prosperity4opt algorithm.py 0 1 2 --trials 200 --output-algo best_algorithm.py
```

## Options

```bash
prosperity4opt algorithm.py DAYS... [OPTIONS]

Arguments:
  DAYS                    Days to backtest (e.g., 0 for all of round 0, 0-1 for specific day)

Options:
  --trials INTEGER        Max number of trials to run
  --metric TEXT           Optimization metric: min (worst day), sum (total), mean (average)
  --multi-objective TEXT  Multi-objective: std (minimize variance) or drawdown
  --output-algo PATH      Save optimized algorithm to file
  --grid                  Use grid search instead of TPE
  --jobs INTEGER          Parallel trials (-1 for all CPU cores)
  --out PATH              Log file path (default: prosperity4opt.log)
  --match-trades MODE     Trade matching: all, worse, none
```

## Avoiding Overfitting

### DO:
- ✅ Train on **multiple days** (`0 1 2` instead of just `0`)
- ✅ Use **3-7 parameters** max (check importance scores)
- ✅ Use enough trials (200+ for TPE)

### DON'T:
- ❌ Train on a single day
- ❌ Optimize 20+ parameters without enough data
- ❌ Trust results without validating on unseen days

### Understanding Hyperparameter Importance

After optimization, check which parameters matter:

```
HYPERPARAMETER IMPORTANCE
  FAIR_VALUE_ROLLING_WEIGHT: 0.381  ← Important
  FAIR_VALUE_MICRO_WEIGHT: 0.149    ← Important
  POSITION_LIMIT: 0.012             ← Noise, remove
```

Parameters with importance < 0.05 are likely noise - remove them and re-run.

## Metrics Explained

| Metric | Optimizes | Best For |
|--------|-----------|----------|
| `min` | Worst day's profit | Conservative, avoids blowouts |
| `sum` | Total profit | Maximum total gain |
| `mean` | Average profit | Balanced approach |

**Recommendation:** Use `sum` when training on 3+ days. Use `min` when training on fewer days or when you want a more conservative strategy.

## Multi-Objective Optimization

Maximize profit while minimizing risk:

```bash
# Minimize profit variance (more consistent returns)
prosperity4opt algorithm.py 0 1 2 --multi-objective std --trials 300

# Minimize max drawdown
prosperity4opt algorithm.py 0 1 2 --multi-objective drawdown --trials 300
```

Multi-objective optimization returns a **Pareto front** of non-dominated solutions. The optimizer automatically exports the first (best) solution.

## Viewing Results

```bash
# Interactive dashboard
optuna-dashboard prosperity4opt.log

# Then open http://localhost:8080
```

The dashboard shows:
- Trial history and parameter relationships
- Best parameters and their values
- Hyperparameter importance (if scikit-learn is installed)

## Examples

```bash
# Simple optimization
prosperity4opt algo.py 0 --trials 100

# Multiple rounds, export result
prosperity4opt algo.py 0 1 2 --trials 200 --output-algo best.py

# Grid search (exhaustive)
prosperity4opt algo.py 0 --grid

# Multi-objective with custom metric
prosperity4opt algo.py 0 1 2 --multi-objective std --metric sum --trials 500

# Parallel execution
prosperity4opt algo.py 0 1 2 --trials 500 --jobs 8
```

## Development

```bash
# Clone and install in editable mode
git clone https://github.com/PhSchumi/imc-prosperity-4-optimizer
cd imc-prosperity-4-optimizer
pip install -e .
```

## License

MIT License - See [LICENSE](LICENSE) for details.
