import contextlib
import functools
import multiprocessing as mp
import warnings
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import rich.progress
from absl import logging

T = TypeVar("T")


def suppress_message(fn: Callable[..., T]) -> Callable[..., T | None]:
    @functools.wraps(fn)
    def _fn(args: tuple) -> T | None:
        warnings.simplefilter("ignore")
        logging.set_verbosity(logging.WARNING)

        try:
            return fn(*args)

        except Exception as e:
            logging.error(f"Error in {fn.__name__}: {e}")
            return None

    return _fn


def map_fn(
    fn: Callable,
    tasks: list[Any],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
) -> Iterable[Any]:
    if num_workers == 0:
        return map(fn, tasks)

    fn = suppress_message(fn)
    pool = stack.enter_context(mp.Pool(processes=num_workers))
    results = pool.imap_unordered(fn, tasks)
    results = rich.progress.track(results, total=len(tasks))

    return results
