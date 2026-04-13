import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator, Optional, Set, Dict, List, Union, Tuple, Any

from optuna import Trial, TrialPruned, Study
from optuna.distributions import CategoricalChoiceType

# Windows-compatible file locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp())

    try:
        yield path
    finally:
        shutil.rmtree(path)


class ObjectiveRunner:
    def __init__(
        self,
        temp_dir: Path,
        algorithm_file: Path,
        days: List[str],
        backtester_args: List[str],
        seen_file: Optional[Path] = None,
        metric: str = "min",
        multi_objective: Optional[str] = None
    ) -> None:
        self._temp_dir = temp_dir
        self._days = days
        self._backtester_args = backtester_args
        self._original_algorithm_file = algorithm_file  # Store original for output

        self.params = dict[str, Callable[[Trial], CategoricalChoiceType]]()
        self.param_definitions = dict[str, str]()

        self._algorithm_file = self._process_algorithm_file(algorithm_file)

        # File-based storage for cross-process duplicate detection
        self._seen_file = seen_file if seen_file is not None else temp_dir / ".params_seen.json"
        self._seen_lock = temp_dir / ".params_seen.lock"
        self._metric = metric  # "min", "sum", or "mean"
        self._multi_objective = multi_objective  # "std", "drawdown", or None

    def _process_algorithm_file(self, algorithm_file: Path) -> Path:
        # Try to find and copy datamodel.py if it exists
        datamodel_sources = [
            algorithm_file.parent / "datamodel.py",
            algorithm_file.parent.parent / "datamodel.py",
            algorithm_file.parent / "datamodels.py",  # alternate name
        ]
        datamodel_copied = False
        for datamodel_path in datamodel_sources:
            if datamodel_path.exists():
                shutil.copyfile(datamodel_path, self._temp_dir / "datamodel.py")
                datamodel_copied = True
                break

        # If not found locally, try to get from prosperity4bt package
        if not datamodel_copied:
            try:
                import prosperity4bt.datamodel as dm
                import os
                datamodel_src = Path(dm.__file__)
                if datamodel_src.exists():
                    shutil.copyfile(datamodel_src, self._temp_dir / "datamodel.py")
                    datamodel_copied = True
            except ImportError:
                pass

        if not datamodel_copied:
            print("Warning: datamodel.py not found, algorithm may fail if it depends on it")

        output_file = self._temp_dir / "algorithm.py"

        with algorithm_file.open("r", encoding="utf-8") as fin, output_file.open("w+", encoding="utf-8") as fout:
            fout.write("""import json as prosperity4opt_json
import os as prosperity4opt_os
prosperity4opt_params = prosperity4opt_json.loads(prosperity4opt_os.environ["PROSPERITY4OPT_PARAMS"])

""")

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
        new_value = f'prosperity4opt_params["{param_name}"]'

        return line[:value_start] + new_value + line[value_end:]

    def _get_param_name(self, var_name: str) -> str:
        if var_name not in self.params:
            return var_name

        suffix = 2
        while f"{var_name}{suffix}" in self.params:
            suffix += 1

        return f"{var_name}{suffix}"

    def _check_and_mark_params_seen(self, params: str) -> bool:
        """Check if params have been seen before using file-based locking.
        Returns True if params were already seen, False otherwise."""
        # Create lock file and directories if needed
        self._seen_lock.parent.mkdir(parents=True, exist_ok=True)

        lock_file = open(self._seen_lock, 'w')

        # Acquire exclusive lock
        if HAS_FCNTL:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif HAS_MSVCRT:
            # Windows locking - lock a region of the file
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        # If no locking available, proceed without lock (best effort)

        try:
            # Load existing seen params
            seen = set()
            if self._seen_file.exists():
                with open(self._seen_file, 'r') as f:
                    seen = set(json.load(f))

            # Check if already seen
            if params in seen:
                return True

            # Mark as seen
            seen.add(params)

            # Write back
            with open(self._seen_file, 'w') as f:
                json.dump(list(seen), f)

            return False
        finally:
            # Release lock
            if HAS_FCNTL:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif HAS_MSVCRT:
                lock_file.close()  # Closing releases the lock on Windows
            else:
                lock_file.close()

    def objective(self, trial: Trial) -> Union[int, Tuple[int, float]]:
        params = json.dumps(
            {name: eval(value, {"trial": trial}, {}) for name, value in self.params.items()}, sort_keys=True
        )

        # Check for duplicates using file-based locking
        if self._check_and_mark_params_seen(params):
            raise TrialPruned("Same parameters as previous trial.")

        env = os.environ.copy()
        env["PROSPERITY4OPT_PARAMS"] = params

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "prosperity4bt",
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
            # Check for common errors
            if "No module named 'prosperity4bt'" in stdout:
                raise RuntimeError(
                    "prosperity4bt is not installed. Please install it with: pip install prosperity4bt"
                )
            raise RuntimeError(f"prosperity4bt exited with status code {proc.returncode}. Output:\n{stdout}")

        day_profits = []
        for line in stdout.splitlines():
            if line.startswith("Total profit: "):
                day_profits.append(int(line.split(": ")[1].replace(",", "")))

            if line == "Profit summary:":
                break

        if not day_profits:
            raise RuntimeError(
                f"No profit data found in backtester output. The output may have changed format or no days were matched. "
                f"Days: {self._days}"
            )

        # Calculate primary metric (profit)
        if self._metric == "sum":
            profit_value = sum(day_profits)
        elif self._metric == "mean":
            profit_value = int(sum(day_profits) / len(day_profits))
        else:  # "min" (default)
            profit_value = min(day_profits)

        # Multi-objective: return (profit, risk) tuple
        if self._multi_objective is not None:
            if self._multi_objective == "std":
                # Standard deviation (lower is better for consistency)
                if len(day_profits) > 1:
                    mean_val = sum(day_profits) / len(day_profits)
                    variance = sum((x - mean_val) ** 2 for x in day_profits) / len(day_profits)
                    risk_value = math.sqrt(variance)
                else:
                    risk_value = 0.0
            elif self._multi_objective == "drawdown":
                # Max drawdown from peak (lower is better)
                cumulative = 0
                peak = float('-inf')
                max_drawdown = 0
                for profit in day_profits:
                    cumulative += profit
                    peak = max(peak, cumulative)
                    drawdown = peak - cumulative
                    max_drawdown = max(max_drawdown, drawdown)
                risk_value = float(max_drawdown)
            else:
                risk_value = 0.0

            return (profit_value, risk_value)

        return profit_value

    def save_optimized_algorithm(self, output_path: Path, best_params: Dict[str, Any]) -> None:
        """Save the algorithm file with optimized parameters baked in."""
        # Read the original algorithm file (not the modified temp one)
        with self._original_algorithm_file.open("r", encoding="utf-8") as f:
            original_lines = f.readlines()

        # Process each line, replacing optimized parameters
        output_lines = []
        for line in original_lines:
            # Check if this line has an # opt: annotation
            match = re.match(r"^(\s*)([^ ]+)\s*=\s*(.*?)\s*#\s*opt:.*$", line)
            if match:
                indent, var_name, _ = match.groups()
                # Use var_name directly as the param name (they match)
                if var_name in best_params:
                    value = best_params[var_name]
                    # Format the value appropriately
                    if isinstance(value, float):
                        # Check if it's essentially an integer
                        if value == int(value):
                            formatted_value = str(int(value))
                        else:
                            formatted_value = f"{value:.3f}".rstrip('0').rstrip('.')
                    else:
                        formatted_value = str(value)
                    # Write the line with the optimized value
                    output_lines.append(f'{indent}{var_name} = {formatted_value}\n')
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

        # Write to output file
        with output_path.open("w", encoding="utf-8") as f:
            f.writelines(output_lines)

        print(f"\nOptimized algorithm saved to: {output_path}")
