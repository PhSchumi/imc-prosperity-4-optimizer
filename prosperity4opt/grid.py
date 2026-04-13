import sys

import numpy as np
import optuna
from optuna import Trial
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from prosperity4opt.objective import ObjectiveRunner


def get_possible_values(d: BaseDistribution) -> list[CategoricalChoiceType]:
    if isinstance(d, IntDistribution):
        return list(range(d.low, d.high + d.step, d.step))
    elif isinstance(d, FloatDistribution):
        if d.step is None:
            raise ValueError(
                "Step size is undefined, please define it using the step=<value> keyword argument in the parameter definition"
            )

        return np.arange(d.low, d.high + d.step, d.step).tolist()
    elif isinstance(d, CategoricalDistribution):
        return list(d.choices)

    raise ValueError(f"Unsupported distribution {d}")


def get_grid_search_space(runner: ObjectiveRunner) -> dict[str, list[CategoricalChoiceType]]:
    # Use public API instead of internal storage access
    study = optuna.create_study(study_name="extracting-distributions")
    trial = study.ask()  # Public API to create a trial

    space = {}
    for name, func in runner.params.items():
        print(f"Extracting distribution for parameter '{name}' with definition '{runner.param_definitions[name]}'")
        eval(func)

        try:
            space[name] = get_possible_values(trial.distributions[name])
            print(f"Extracted {len(space[name])} possible values for parameter '{name}'")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    return space
