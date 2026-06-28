"""
Utility classes and functions for mcab package.
"""
import warnings

from joblib import Parallel
from tqdm.auto import tqdm

__all__ = ['ProgressParallel']

def _worker_init_warnings():
    """Suppress known noisy RuntimeWarnings in parallel workers.
    The main process will emit them once via its own filter.
    """
    warnings.filterwarnings(
        'ignore',
        message='Precision loss occurred in moment calculation',
        category=RuntimeWarning
    )
    warnings.filterwarnings(
        'ignore',
        message='divide by zero',
        category=RuntimeWarning
    )

class ProgressParallel(Parallel):
    """
    Wrapper for joblib.Parallel with tqdm progress bar output.
    """
    def __init__(self, use_tqdm=True, total=None, desc=None, unit=None, leave=True, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc  = desc or  ''
        self._unit  = unit or 'it'
        self._leave = leave
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        tqdm_params = dict(
            disable=not self._use_tqdm,
            total=self._total,
            desc=self._desc,
            unit=self._unit,
            leave=self._leave,
            dynamic_ncols=True,
        )
        with tqdm(**tqdm_params) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
