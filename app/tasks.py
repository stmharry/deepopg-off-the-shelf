import contextlib
import itertools
import multiprocessing as mp
import warnings
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import rich.progress
from absl import logging

T = TypeVar("T")


def _map_fn(
    args: tuple[
        Callable[..., T],
        tuple[Any, ...],
    ]
) -> T | None:
    warnings.simplefilter("ignore")
    logging.set_verbosity(logging.WARNING)

    fn, task = args
    try:
        return fn(*task)

    except Exception as e:
        logging.error(f"Error in {fn.__name__}: {e}")
        return None


def map_fn(
    fn: Callable[..., T],
    tasks: list[tuple],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
) -> Iterable[T | None]:
    if num_workers == 0:
        return itertools.starmap(fn, tasks)

    pool = stack.enter_context(mp.Pool(processes=num_workers))
    results = pool.imap_unordered(_map_fn, [(fn, task) for task in tasks])
    results = rich.progress.track(results, total=len(tasks))

    return results
