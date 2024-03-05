import contextlib
import dataclasses
import multiprocessing as mp
import multiprocessing.context as mp_context
import multiprocessing.pool as mp_pool
import warnings
from collections.abc import Callable, Iterable, Sized
from typing import Any, Generic, Literal, TypeAlias, TypeVar, overload

import pipe
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

    def __call__(self, *args: Any, **kwargs: Any) -> "Task":
        return Task(self.fn, args=self.args + args, kwargs=self.kwargs | kwargs)

    def run(self) -> Result[T]:
        return self.fn(*self.args, **self.kwargs)


def run_task(task: Task[T]) -> Result[T]:
    return task.run()


def run_task_with_message_suppressed(
    task: Task[T], verbosity: int = logging.WARNING
) -> Result[T]:
    original_verbosity = logging.get_verbosity()

    logging.set_verbosity(verbosity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            result: Result[T] = task.run()

        except Exception as e:
            logging.error(f"Error in {task.fn.__name__}: {e}")
            return None

    logging.set_verbosity(original_verbosity)
    return result


@overload
def map_task(
    tasks: Iterable[Task[T]],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
    filter_none: Literal[True] = ...,
    method: Literal["fork", "spawn", "forkserver"] | None = "fork",
) -> Iterable[T]: ...


@overload
def map_task(
    tasks: Iterable[Task[T]],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
    filter_none: Literal[False] = ...,
    method: Literal["fork", "spawn", "forkserver"] | None = "fork",
) -> Iterable[Result[T]]: ...


def map_task(
    tasks: Iterable[Task[T]],
    stack: contextlib.ExitStack,
    num_workers: int = 0,
    filter_none: bool = True,
    method: Literal["fork", "spawn", "forkserver"] | None = "fork",
) -> Iterable[Result[T]] | Iterable[T]:
    if num_workers == 0:
        results = map(run_task, tasks)

    else:
        context: mp_context.BaseContext = mp.get_context(method)
        pool = stack.enter_context(context.Pool(processes=num_workers))
        results = pool.imap_unordered(run_task_with_message_suppressed, tasks)

        results = rich.progress.track(
            results, total=len(tasks) if isinstance(tasks, Sized) else None
        )

    if filter_none:
        results = filter(None, results)

    return results


@dataclasses.dataclass(kw_only=True)
class ParallelPipe(Generic[T]):
    fn: TaskFn[T]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    _mp_pool: mp_pool.Pool | None = None
    _unpack_input: bool = False
    _allow_unordered: bool = False

    def run(self) -> Result[T]:
        return self.fn(*self.args, **self.kwargs)

    def run_with_message_suppressed(
        self, verbosity: int = logging.WARNING
    ) -> Result[T]:
        original_verbosity = logging.get_verbosity()

        logging.set_verbosity(verbosity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                result: Result[T] = self.run()

            except Exception as e:
                logging.error(f"Error in {self.fn.__name__}: {e}")
                return None

        logging.set_verbosity(original_verbosity)
        return result

    def __ror__(self, iterable: Iterable[Any]) -> Iterable[Result[T]]:
        args_list: Iterable[tuple[Any, ...]]
        if self._unpack_input:
            args_list = (tuple(item) for item in iterable)
        else:
            args_list = ((item,) for item in iterable)

        pipes: Iterable[ParallelPipe[T]] = (
            ParallelPipe(
                fn=self.fn,
                args=args + self.args,
                kwargs=self.kwargs,
            )
            for args in args_list
        )

        if self._mp_pool is None:
            logging.warning("No multiprocessing pool provided. Running in serial mode.")
            return map(ParallelPipe.run, pipes)

        else:
            if self._allow_unordered:
                return self._mp_pool.imap_unordered(
                    ParallelPipe.run_with_message_suppressed, pipes
                )
            else:
                return self._mp_pool.imap(
                    ParallelPipe.run_with_message_suppressed, pipes
                )

    def __call__(self, *args: Any, **kwargs: Any) -> "ParallelPipe":
        return ParallelPipe(
            fn=self.fn,
            args=args + self.args,
            kwargs=kwargs | self.kwargs,
            _mp_pool=self._mp_pool,
            _unpack_input=self._unpack_input,
            _allow_unordered=self._allow_unordered,
        )


@dataclasses.dataclass
class Pool(object):
    num_workers: int = 0
    method: Literal["fork", "spawn", "forkserver"] | None = "fork"

    _mp_pool: mp_pool.Pool | None = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        if self.num_workers > 0:
            context: mp_context.BaseContext = mp.get_context(self.method)
            self._mp_pool = context.Pool(processes=self.num_workers)

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.num_workers > 0:
            assert self._mp_pool is not None

            self._mp_pool.close()
            self._mp_pool.join()
            self._mp_pool = None

    def pipe(
        self, fn: TaskFn[T], unpack_input: bool = True, allow_unordered: bool = True
    ) -> pipe.Pipe | ParallelPipe[T]:
        return ParallelPipe(
            fn=fn,
            _mp_pool=self._mp_pool,
            _unpack_input=unpack_input,
            _allow_unordered=allow_unordered,
        )
