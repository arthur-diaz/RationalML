"""Small terminal helpers and scoped Optuna logging; no ML decisions live here."""

from contextlib import contextmanager
import logging
import sys
from typing import Iterator

import optuna


@contextmanager
def optuna_verbosity(verbose: int) -> Iterator[None]:
    """Temporarily change levels only, retaining handlers and user configuration."""
    # Initialize through the public API before taking a snapshot. Save explicit
    # levels (including NOTSET), rather than only the effective inherited level.
    optuna.logging.get_verbosity()
    root = logging.getLogger("optuna")
    loggers = [root] + [
        logger for name, logger in list(logging.Logger.manager.loggerDict.items())
        if name.startswith("optuna.") and isinstance(logger, logging.Logger)
    ]
    saved = [(logger, logger.level) for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.INFO if verbose >= 2 else logging.WARNING)
        yield
    finally:
        for logger, level in saved:
            logger.setLevel(level)


def print_run_info(task: str, metric: str, models: int, cv: int, trials: int, n_jobs: int) -> None:
    label = task if task == "regression" else f"{task} classification"
    cpu = "all (backend defaults)" if n_jobs == -1 else str(n_jobs)
    print(f"RationalML - {label}\n")
    print(f"Models        {models}\nCV folds      {cv}\nTrials/model  {trials}\nCPU           {cpu}\nMetric        {metric}\n")


class TrialProgress:
    """Count terminal trial states, including failures that abort optimization.

    Terminals redraw in place. Redirected output gets one line per model and
    a final line, while the internal counter still updates after every trial.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.finished = 0
        self.failed = 0
        self.pruned = 0
        self._seen: set[tuple[optuna.Study, int]] = set()
        self._terminal = sys.stdout.isatty()
        print("Optimizing models")
        self._render()

    def update(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        key = (study, trial.number)
        if not trial.state.is_finished() or key in self._seen:
            return
        self._seen.add(key)
        self.finished += 1
        self.failed += int(trial.state == optuna.trial.TrialState.FAIL)
        self.pruned += int(trial.state == optuna.trial.TrialState.PRUNED)
        if self._terminal:
            self._render()

    def _render(self, suffix: str = "") -> None:
        width = 24
        filled = min(width, width * self.finished // self.total)
        line = f"[{'#' * filled}{'-' * (width - filled)}] {self.finished}/{self.total}"
        if self.failed or self.pruned:
            line += f" (failed={self.failed}, pruned={self.pruned})"
        print(("\r" if self._terminal else "") + line + suffix,
              end="" if self._terminal else "\n", flush=True)

    def model_finished(self) -> None:
        if not self._terminal and self.finished < self.total:
            self._render()

    def close(self, *, interrupted: bool = False) -> None:
        suffix = " - interrupted" if interrupted else (
            " - budget incomplete (timeout/early stop)" if self.finished < self.total else ""
        )
        self._render(suffix)
        if self._terminal:
            print()
