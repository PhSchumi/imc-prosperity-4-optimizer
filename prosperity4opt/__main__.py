# import concurrent.futures as f

# f.ThreadPoolExecutor = f.ProcessPoolExecutor

import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Annotated, Generator, Optional

import optuna
from optuna.samplers import BaseSampler, GridSampler, TPESampler
from optuna.storages import BaseStorage, InMemoryStorage, JournalStorage
from optuna.storages.journal import JournalFileBackend

from prosperity4bt.models import TradeMatchingMode
from typer import Argument, Option, Typer

from prosperity4opt.grid import get_grid_search_space
from prosperity4opt.objective import ObjectiveRunner


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp())

    try:
        yield path
    finally:
        shutil.rmtree(path)

## likely needs adjustment
def version_callback(value: bool) -> None:
    if value:
        print(f"prosperity4opt {metadata.version(__package__)}")
        sys.exit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    algorithm: Annotated[Path, 
                         Argument(
                             help="Path to the Python file containing the algorithm to optimize.", 
                             show_default=False, 
                             exists=True, 
                             file_okay=True, 
                             dir_okay=False, 
                             resolve_path=True
                             )],
    days: Annotated[list[str], 
                    Argument(
                        help="The days to backtest on. <round>-<day> for a single day, <round> for all days in a round.", 
                        show_default=False
                        )],
    out: Annotated[Path,
                   Option(help="Path to save optimization results to.",
                          show_default=False,
                          dir_okay=False,
                          resolve_path=True)] = Path("prosperity4opt.log"),
    no_out: Annotated[bool,
                      Option("--no-out",
                             help="Skip saving optimization results.")] = False,
    output_algo: Annotated[Optional[Path],
                          Option(
                              "--output-algo",
                              help="Path to save algorithm file with best parameters (e.g., best_algo.py). If not specified, no file is saved.",
                              show_default=False,
                              dir_okay=False,
                              resolve_path=True)] = None,
    jobs: Annotated[int, 
                    Option(help="Number of backtests to run in parallel (-1 to use number of CPU cores).")] = -1,
    minimize: Annotated[bool,
                        Option("--min",
                               help="Minimize the total profit rather than maximizing it.")] = False,
    metric: Annotated[str,
                      Option(help="Optimization metric: 'min' (worst day), 'sum' (total profit), or 'mean' (average profit). Default: 'min'")] = "min",
    multi_objective: Annotated[Optional[str],
                               Option(
                                   "--multi-objective",
                                   help="Enable multi-objective optimization with a risk metric: 'std' (minimize profit variance) or 'drawdown' (minimize max drawdown). Maximizes profit while minimizing risk.")] = None,
    trials: Annotated[Optional[int], 
                      Option(
                          help="Maximum number of trials to run (defaults to infinity).", 
                          show_default=False)] = None,
    seconds: Annotated[Optional[int], 
                       Option(
                           help="Maximum number of seconds to run for (defaults to infinity).", 
                           show_default=False)] = None,
    grid: Annotated[bool, 
                    Option("--grid", 
                           help="Perform a grid search. Requires all floating point parameters to have a step size.")] = False,
    match_trades: Annotated[TradeMatchingMode, 
                            Option(
                                help="How to match orders against market trades. 'all' matches trades with prices equal to or worse than your quotes, 'worse' matches trades with prices worse than your quotes, 'none' does not match trades against orders at all.")] = TradeMatchingMode.all,
    version: Annotated[bool, 
                       Option("--version", "-v", 
                              help="Show the program's version number and exit.", 
                              is_eager=True, callback=version_callback)] = False,
) -> None:  # fmt: skip
    """
    Optimize an IMC Prosperity 4 algorithm using Optuna and prosperity4bt.
    See https://github.com/jmerle/imc-prosperity-3-optimizer for usage documentation.
    """
    # Validate metric parameter
    valid_metrics = {"min", "sum", "mean"}
    if metric not in valid_metrics:
        print(f"Error: --metric must be one of {valid_metrics}, got '{metric}'")
        sys.exit(1)

    # Validate multi_objective parameter
    valid_multi_objective = {"std", "drawdown", None}
    if multi_objective not in valid_multi_objective:
        print(f"Error: --multi-objective must be one of {valid_multi_objective - {None}}, got '{multi_objective}'")
        sys.exit(1)

    if out is not None and no_out:
        print("Error: --out and --no-out are mutually exclusive")
        sys.exit(1)

    backtester_args = ["--match-trades", match_trades.value]

    with temporary_directory() as temp_dir:
        runner = ObjectiveRunner(temp_dir, algorithm, days, backtester_args, metric=metric, multi_objective=multi_objective)

        if len(runner.params) == 0:
            print("Error: no hyperparameters found")
            sys.exit(1)

        storage: BaseStorage
        if no_out:
            storage = InMemoryStorage()
        else:
            storage = JournalStorage(JournalFileBackend(str(out)))

        sampler: BaseSampler
        if grid:
            search_space = get_grid_search_space(runner)
            sampler = GridSampler(search_space)

            num_combinations = 1
            for v in search_space.values():
                num_combinations *= len(v)

            print(f"Running grid search on {num_combinations:,.0f} possible hyperparameter combinations")
        else:
            sampler = TPESampler(multivariate=True, seed=42)

        # Multi-objective or single-objective study
        if multi_objective is not None:
            # For multi-objective: maximize profit, minimize risk
            directions = ["maximize" if not minimize else "minimize", "minimize"]
            study = optuna.create_study(
                storage=storage,
                sampler=sampler,
                directions=directions,
                study_name=datetime.now().strftime("prosperity4opt_%Y-%m-%d_%H-%M-%S"),
            )
            print(f"Multi-objective optimization: maximizing {metric} profit, minimizing {multi_objective}")
        else:
            study = optuna.create_study(
                storage=storage,
                sampler=sampler,
                direction="minimize" if minimize else "maximize",
                study_name=datetime.now().strftime("prosperity4opt_%Y-%m-%d_%H-%M-%S"),
            )

        try:
            study.optimize(runner.objective, n_jobs=jobs, n_trials=trials, timeout=seconds, show_progress_bar=True)
        except KeyboardInterrupt:
            print("\nStopping optimization...")
        finally:
            print()
            print("=" * 60)
            print("OPTIMIZATION RESULTS")
            print("=" * 60)
            print(f"Total trials completed: {len(study.trials)}")

            if multi_objective is not None:
                # Multi-objective: show Pareto front
                print("\nBest trials (Pareto front):")
                print("-" * 60)
                try:
                    best_trials = study.best_trials
                    for i, trial in enumerate(best_trials[:10]):  # Show top 10
                        profit_val = trial.values[0]
                        risk_val = trial.values[1]
                        print(f"\n  Trial {i+1}:")
                        print(f"    Profit ({metric}): {profit_val:,.0f}")
                        print(f"    Risk ({multi_objective}): {risk_val:,.2f}")
                        print(f"    Parameters: {trial.params}")
                except Exception as e:
                    print(f"Could not retrieve Pareto front: {e}")
            else:
                # Single-objective
                try:
                    print(f"\nBest profit: {study.best_value:,.0f}")
                    print(f"Best parameters: {study.best_params}")
                except Exception:
                    pass

            # Hyperparameter importance
            if len(study.trials) >= 2:  # Importance requires at least 2 trials
                try:
                    print("\n" + "=" * 60)
                    print("HYPERPARAMETER IMPORTANCE")
                    print("=" * 60)

                    if multi_objective is not None:
                        # For multi-objective, get importance for each objective
                        for obj_idx, obj_name in enumerate([f"profit ({metric})", f"risk ({multi_objective})"]):
                            try:
                                importance = optuna.importance.get_param_importances(
                                    study, target=lambda t: t.values[obj_idx]
                                )
                                print(f"\nImportance for {obj_name}:")
                                for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
                                    print(f"  {param}: {imp:.3f}")
                            except Exception as e:
                                print(f"Could not compute importance for {obj_name}: {e}")
                    else:
                        importance = optuna.importance.get_param_importances(study)
                        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
                            print(f"  {param}: {imp:.3f}")
                except Exception as e:
                    print(f"Could not compute hyperparameter importance: {e}")

        # Save optimized algorithm file if requested
        if output_algo is not None:
            try:
                if multi_objective is not None:
                    # For multi-objective, use the best trial from Pareto front
                    best_params = study.best_trials[0].params
                else:
                    best_params = study.best_params
                runner.save_optimized_algorithm(output_algo, best_params)
            except Exception as e:
                print(f"\nWarning: Could not save optimized algorithm: {e}")

        print("\n" + "=" * 60)
        if not no_out:
            print(f"Results saved to: {out}")
            print("Run 'optuna-dashboard {}' to visualize".format(out))
        print("=" * 60)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
