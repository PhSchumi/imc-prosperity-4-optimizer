# Important

This is an unfinished project I started during the tutorial round of Prosperity 3 with the intention of open-sourcing it, but I didn't get it to a state where I was comfortable with doing so. It's in a usable state, I used it for some rounds, but it's a bit rough around the edges, partially undocumented, and lacking some key optimizations. Below you can find the readme in the state it was at the end of the competition, with no guarantees about its correctness.

---

# IMC Prosperity 3 Optimizer

[![Build Status](https://github.com/jmerle/imc-prosperity-3-optimizer/workflows/Build/badge.svg)](https://github.com/jmerle/imc-prosperity-3-optimizer/actions/workflows/build.yml)
[![PyPI Version](https://img.shields.io/pypi/v/prosperity3opt)](https://pypi.org/project/prosperity3opt/)

**Warning: overfitting your hyperparameters on training data may lead to unexpected results on the test data. Use at your own risk.**

This repository contains a hyperparameter optimizer for [IMC Prosperity 3](https://prosperity.imc.com/) algorithms, based on [Optuna](https://optuna.org/) and my [Prosperity 3 backtester](https://github.com/jmerle/imc-prosperity-3-backtester).

## Usage

Run `pip install -U prosperity3opt` to install or update the optimizer.

Hyperparameters that need to be optimized must be annotated in your code like this:
```python
RAINFOREST_RESIN_VALUE = 10_000 # opt: int(10_000 - 3, 10_000 + 3)
```

You can use any of [Optuna's `trial.suggest_*` methods](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial) that are not deprecated after the "# opt: " comment, without the "trial.suggest_" part nor the name of the parameter:
- [`# opt: categorical(choices)`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical): suggest a value for a categorical parameter
- [`# opt: float(low, high, *[, step, log])`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float): suggest a value for a floating point parameter
- [`# opt: int(low, high, *[, step, log])`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int): suggest a value for an integer parameter

"# opt:" comments must be placed on variable assignments, which do not necessarily have to be global variable assignments:
```python
RAINFOREST_RESIN_VALUE = 10_000 # opt: int(10_000 - 3, 10_000 + 3)

class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        rainforest_resin_value = 10_000 # opt: int(10_000 - 3, 10_000 + 3)
        return rainforest_resin_value
```

The name of the hyperparameter (i.e. the `name` in Optuna's `trial.suggest_*(name, ...)` methods) is inferred from the name of the variable that is being assigned. Parameters passed to these methods are `eval`'ed on their own, meaning they must not depend on functions or variables defined elsewhere in your algorithm.

After annotating your hyperparameters you can run the optimizer like you run a backtest:
```sh
# Optimize on all days from round 1
$ prosperity3opt example/starter.py 1

# Optimize on round 1 day 0
$ prosperity3opt example/starter.py 1-0

# Optimize on round 1 day -1 and round 1 day 0
$ prosperity3opt example/starter.py 1--1 1-0

# Optimize on all days from rounds 1 and 2
$ prosperity3opt example/starter.py 1 2

# You get the idea
```

You can find all available options by running `prosperity3opt --help`.

During and after performing an optimization you can run `optuna-dashboard prosperity3opt.log` to visualize the results in the [Optuna Dashboard](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html) (which is automatically installed when you install the optimizer).

## Grid Search

By default the optimizer uses Optuna's [tree-structured Parzen Estimator sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html). You can change this to its grid search sampler using the `--grid` flag. This requires any floating point hyperparameters to have their step size explicitly defined using the `step` keyword argument, i.e. `float(low, high, step=<value>)`.

## Development

Follow these steps if you want to make changes to the backtester:
1. Install [uv](https://docs.astral.sh/uv/).
2. Clone (or fork and clone) this repository.
3. Open a terminal in your clone of the repository.
4. Create a venv with `uv venv` and activate it.
5. Run `uv sync`.
6. Any changes you make are now automatically taken into account the next time you run `prosperity3opt` inside the venv.
