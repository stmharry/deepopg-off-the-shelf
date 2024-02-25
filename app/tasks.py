import contextlib
import multiprocessing as mp
import multiprocessing.context as mp_context
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeAlias, TypeVar

import rich.progress
from absl import logging

T = TypeVar("T")

TaskFn: TypeAlias = Callable[..., T]
Task: TypeAlias = (
    tuple[TaskFn[T], tuple[Any, ...], dict[str, Any]]
    | tuple[TaskFn[T], tuple[Any, ...]]
    | tuple[TaskFn[T], dict[str, Any]]
    | tuple[TaskFn[T]]
)
Result: TypeAlias = T | None


def _do_task(task: Task[T]) -> Result[T]:
    fn: TaskFn[T]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}
    match task:
        case (fn, tuple() as args, dict() as kwargs):
            ...
        case (fn, tuple() as args):
            ...
        case (fn, dict() as kwargs):
            ...
        case (fn,):
            ...

    return fn(*args, **kwargs)


def do_task(task: Task[T]) -> Result[T]:
    warnings.simplefilter("ignore")
    logging.set_verbosity(logging.WARNING)

    try:
        return _do_task(task)

    except Exception as e:
        fn: TaskFn[T] = task[0]
        logging.error(f"Error in {fn.__name__}: {e}")
        return None


def map_task(
    tasks: list[Task[T]],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
    method: Literal["fork", "spawn", "forkserver"] | None = "fork",
) -> Iterable[Result[T]]:
    if num_workers == 0:
        return map(_do_task, tasks)

    context: mp_context.BaseContext = mp.get_context(method)
    pool = stack.enter_context(context.Pool(processes=num_workers))
    results = pool.imap_unordered(do_task, tasks)
    results = rich.progress.track(results, total=len(tasks))

    return results


# backwards compatibility


def map_fn(
    fn: TaskFn[T],
    tasks: list[tuple[Any, ...]],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
    method: Literal["fork", "spawn", "forkserver"] | None = "fork",
) -> Iterable[Result[T]]:
    return map_task(
        [(fn, task) for task in tasks],
        stack=stack,
        num_workers=num_workers,
        method=method,
    )
