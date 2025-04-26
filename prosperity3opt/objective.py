import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from multiprocessing import Lock
from pathlib import Path
from typing import Callable, Generator

from optuna import Trial, TrialPruned
from optuna.distributions import CategoricalChoiceType


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp())

    try:
        yield path
    finally:
        shutil.rmtree(path)


class ObjectiveRunner:
    def __init__(self, temp_dir: Path, algorithm_file: Path, days: list[str], backtester_args: list[str]) -> None:
        self._temp_dir = temp_dir
        self._days = days
        self._backtester_args = backtester_args

        self.params = dict[str, Callable[[Trial], CategoricalChoiceType]]()
        self.param_definitions = dict[str, str]()

        self._algorithm_file = self._process_algorithm_file(algorithm_file)

        self._params_seen = set[str]()
        self._params_seen_lock = Lock()

    def _process_algorithm_file(self, algorithm_file: Path) -> Path:
        shutil.copyfile(algorithm_file.parent / "datamodel.py", self._temp_dir / "datamodel.py")
        output_file = self._temp_dir / "algorithm.py"

        with algorithm_file.open("r", encoding="utf-8") as fin, output_file.open("w+", encoding="utf-8") as fout:
            fout.write(
                """
import json as prosperity3opt_json
import os as prosperity3opt_os
prosperity3opt_params = prosperity3opt_json.loads(prosperity3opt_os.environ["PROSPERITY3OPT_PARAMS"])
            """.strip()
            )
            fout.write("\n\n")

            pattern = re.compile(r"^\s*([^ ]+)\s*=\s*(.*?)\s*#\s*opt:\s*((categorical|float|int).*)\s*$")

            for i, line in enumerate(fin):
                fout.write(pattern.sub(self._process_opt_match, line))

        return output_file

    def _process_opt_match(self, match: re.Match) -> str:
        var_name = match.group(1)
        param_name = self._get_param_name(var_name)

        param_definition = match.group(3)
        param_definition_with_name = param_definition.replace("(", f'("{param_name}", ', 1)

        self.param_definitions[param_name] = param_definition
        self.params[param_name] = f"trial.suggest_{param_definition_with_name}"

        print(f"Hyperparameter: {var_name=}, {param_name=}, {param_definition=}")

        line = match.group(0)
        value_start = match.start(2)
        value_end = match.end(2)
        new_value = f'prosperity3opt_params["{param_name}"]'

        return line[:value_start] + new_value + line[value_end:]

    def _get_param_name(self, var_name: str) -> str:
        if var_name not in self.params:
            return var_name

        suffix = 2
        while f"{var_name}{suffix}" in self.params:
            suffix += 1

        return f"{var_name}{suffix}"

    def objective(self, trial: Trial) -> int:
        params = json.dumps(
            {name: eval(value, {"trial": trial}, {}) for name, value in self.params.items()}, sort_keys=True
        )

        with self._params_seen_lock:
            if params in self._params_seen:
                raise TrialPruned("Same parameters as previous trial.")

            self._params_seen.add(params)

        env = os.environ.copy()
        env["PROSPERITY3OPT_PARAMS"] = params

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "prosperity3bt",
                str(self._algorithm_file),
                *self._days,
                *self._backtester_args,
                "--no-out",
                "--no-progress",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        stdout = proc.stdout.decode("utf-8")

        if proc.returncode != 0:
            raise RuntimeError(f"prosperity3bt exited with status code {proc.returncode}. Output:\n{stdout}")

        day_profits = []
        for line in stdout.splitlines():
            if line.startswith("Total profit: "):
                day_profits.append(int(line.split(": ")[1].replace(",", "")))

            if line == "Profit summary:":
                break

        return min(day_profits)
