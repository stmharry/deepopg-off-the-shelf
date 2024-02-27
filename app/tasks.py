import contextlib
import dataclasses
import multiprocessing as mp
import multiprocessing.context as mp_context
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import rich.progress
from absl import logging

T = TypeVar("T")

TaskFn: TypeAlias = Callable[..., T]
Result: TypeAlias = T | None


@dataclasses.dataclass
class Task(Generic[T]):
    fn: TaskFn[T]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __call__(self) -> Result[T]:
        return self.fn(*self.args, **self.kwargs)


def run_task(task: Task[T]) -> Result[T]:
    return task.__call__()


def run_task_with_message_suppressed(
    task: Task[T], verbosity: int = logging.WARNING
) -> Result[T]:
    original_verbosity = logging.get_verbosity()

    logging.set_verbosity(verbosity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            result: Result[T] = task.__call__()

        except Exception as e:
            logging.error(f"Error in {task.fn.__name__}: {e}")
            return None

    logging.set_verbosity(original_verbosity)
    return result


def map_task(
    tasks: list[Task[T]],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
    method: Literal["fork", "spawn", "forkserver"] | None = "fork",
) -> Iterable[Result[T]]:
    if num_workers == 0:
        return map(run_task, tasks)

    context: mp_context.BaseContext = mp.get_context(method)
    pool = stack.enter_context(context.Pool(processes=num_workers))
    results = pool.imap_unordered(run_task_with_message_suppressed, tasks)
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
        [Task(fn, args=task) for task in tasks],
        stack=stack,
        num_workers=num_workers,
        method=method,
    )
