import dataclasses
import multiprocessing as mp
import multiprocessing.context as mp_context
import multiprocessing.pool as mp_pool
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import pipe
import rich.progress
from absl import logging

T = TypeVar("T")

TaskFn: TypeAlias = Callable[..., T]
Result: TypeAlias = T | None


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
        if self._unpack_input:
            iterable |= pipe.map(tuple)
        else:
            iterable |= pipe.map(lambda x: (x,))

        iterable |= pipe.map(
            lambda item: ParallelPipe(
                fn=self.fn, args=item + self.args, kwargs=self.kwargs
            )
        )

        if self._mp_pool is None:
            logging.warning("No multiprocessing pool provided. Running in serial mode.")
            iterable |= pipe.map(ParallelPipe.run)

        else:
            if self._allow_unordered:
                iterable = self._mp_pool.imap_unordered(
                    ParallelPipe.run_with_message_suppressed, iterable
                )
            else:
                iterable = self._mp_pool.imap(
                    ParallelPipe.run_with_message_suppressed, iterable
                )

        return iterable

    def __call__(self, *args: Any, **kwargs: Any) -> "ParallelPipe":
        return dataclasses.replace(
            self, args=self.args + args, kwargs=self.kwargs | kwargs
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

    def parallel_pipe(
        self,
        fn: TaskFn[T],
        unpack_input: bool = False,
        allow_unordered: bool = False,
    ) -> ParallelPipe[T]:
        return ParallelPipe[T](
            fn=fn,
            _mp_pool=self._mp_pool,
            _unpack_input=unpack_input,
            _allow_unordered=allow_unordered,
        )


@pipe.Pipe
def track_progress(
    iterable: Iterable[T],
    description: str = "Working...",
) -> Iterable[T]:
    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(elapsed_when_finished=True),
    )
    with progress:
        yield from progress.track(iterable, description=description)


@pipe.Pipe
def filter_none(iterable: Iterable[Result[T]]) -> Iterable[T]:
    return (item for item in iterable if item is not None)


@pipe.Pipe
def tee_logging(iterable: Iterable[T], description: str = "tee_logging") -> Iterable[T]:
    for item in iterable:
        logging.debug(f"{description}: {item!r}")
        yield item
