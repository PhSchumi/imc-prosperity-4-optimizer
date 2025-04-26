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
from prosperity3bt.models import TradeMatchingMode
from typer import Argument, Option, Typer

from prosperity3opt.grid import get_grid_search_space
from prosperity3opt.objective import ObjectiveRunner


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp())

    try:
        yield path
    finally:
        shutil.rmtree(path)


def version_callback(value: bool) -> None:
    if value:
        print(f"prosperity3opt {metadata.version(__package__)}")
        sys.exit(0)


app = Typer(context_settings={"help_option_names": ["--help", "-h"]})


@app.command()
def cli(
    algorithm: Annotated[Path, Argument(help="Path to the Python file containing the algorithm to optimize.", show_default=False, exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    days: Annotated[list[str], Argument(help="The days to backtest on. <round>-<day> for a single day, <round> for all days in a round.", show_default=False)],
    out: Annotated[Path, Option(help="Path to save optimization results to.", show_default=False, dir_okay=False, resolve_path=True)] = Path("prosperity3opt.log"),
    no_out: Annotated[bool, Option("--no-out", help="Skip saving optimization results.")] = False,
    jobs: Annotated[int, Option(help="Number of backtests to run in parallel (-1 to use number of CPU cores).")] = -1,
    minimize: Annotated[bool, Option("--min", help="Minimize the total profit rather than maximizing it.")] = False,
    trials: Annotated[Optional[int], Option(help="Maximum number of trials to run (defaults to infinity).", show_default=False)] = None,
    seconds: Annotated[Optional[int], Option(help="Maximum number of seconds to run for (defaults to infinity).", show_default=False)] = None,
    grid: Annotated[bool, Option("--grid", help="Perform a grid search. Requires all floating point parameters to have a step size.")] = False,
    match_trades: Annotated[TradeMatchingMode, Option(help="How to match orders against market trades. 'all' matches trades with prices equal to or worse than your quotes, 'worse' matches trades with prices worse than your quotes, 'none' does not match trades against orders at all.")] = TradeMatchingMode.all,
    version: Annotated[bool, Option("--version", "-v", help="Show the program's version number and exit.", is_eager=True, callback=version_callback)] = False,
) -> None:  # fmt: skip
    """
    Optimize an IMC Prosperity 3 algorithm using Optuna and prosperity3bt.
    See https://github.com/jmerle/imc-prosperity-3-optimizer for usage documentation.
    """
    if out is not None and no_out:
        print("Error: --out and --no-out are mutually exclusive")
        sys.exit(1)

    backtester_args = ["--match-trades", match_trades.value]

    with temporary_directory() as temp_dir:
        runner = ObjectiveRunner(temp_dir, algorithm, days, backtester_args)

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
            sampler = TPESampler()

        study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            direction="minimize" if minimize else "maximize",
            study_name=datetime.now().strftime("prosperity3opt_%Y-%m-%d_%H-%M-%S"),
        )

        try:
            study.optimize(runner.objective, n_jobs=jobs, n_trials=trials, timeout=seconds)
        except KeyboardInterrupt:
            print("Stopping optimization...")
        finally:
            try:
                print(f"Best profit: {study.best_value:,.0f}")
                print(f"Best parameters: {study.best_params}")
            except Exception:
                pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
