import contextlib
import functools
import itertools
import multiprocessing as mp
import warnings
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import rich.progress
from absl import logging

T = TypeVar("T")


def wrap_fn(fn: Callable[..., T]) -> Callable[..., T | None]:
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
    fn: Callable[..., T],
    tasks: list[Any],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
) -> Iterable[T | None]:
    if num_workers == 0:
        return itertools.starmap(fn, tasks)

    pool = stack.enter_context(mp.Pool(processes=num_workers))
    wrapped_fn: Callable[..., T | None] = wrap_fn(fn)

    results = pool.imap_unordered(wrapped_fn, tasks)
    results = rich.progress.track(results, total=len(tasks))

    return results
