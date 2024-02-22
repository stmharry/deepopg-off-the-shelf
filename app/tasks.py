import contextlib
import multiprocessing as mp
from collections.abc import Callable, Iterable
from typing import Any

import rich.progress


def map_fn(
    fn: Callable,
    tasks: list[Any],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
) -> Iterable[Any]:
    if num_workers == 0:
        return map(fn, tasks)

    pool = stack.enter_context(mp.Pool(processes=num_workers))
    results = pool.imap_unordered(fn, tasks)
    results = rich.progress.track(results, total=len(tasks))

    return results
