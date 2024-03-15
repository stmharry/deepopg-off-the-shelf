import abc
import dataclasses
from pathlib import Path
from typing import ClassVar, Generic, Literal, TypeVar, get_args, overload

import pipe
from absl import logging

from detectron2.data import DatasetCatalog

DATA_T = TypeVar("DATA_T")
DRIVER_T = TypeVar("DRIVER_T", bound="BaseDatasetDriver")


@dataclasses.dataclass  # type: ignore
class BaseDatasetDriver(Generic[DATA_T], metaclass=abc.ABCMeta):
    PREFIX: ClassVar[str]
    SPLITS: ClassVar[list[str]] = []

    root_dir: Path
    _type_T: type[DATA_T] = dataclasses.field(init=False)

    def __post_init__(self):
        self._type_T = get_args(self.__orig_bases__[0])[0]  # type: ignore

    @classmethod
    @abc.abstractmethod
    def register(cls: type[DRIVER_T], root_dir: Path) -> DRIVER_T: ...

    @classmethod
    def get_dataset_name(cls, split: str | None) -> str:
        if split is None:
            return cls.PREFIX

        return f"{cls.PREFIX}_{split}"

    @classmethod
    def get_split_name(cls, dataset_name: str) -> str | None:
        if dataset_name == cls.PREFIX:
            return None

        match dataset_name.rsplit("_", maxsplit=1):
            case [cls.PREFIX, split]:
                return split

            case _:
                raise ValueError(f"Invalid dataset name: {dataset_name}")

    @classmethod
    def available_dataset_names(cls) -> list[str]:
        return list((None, *cls.SPLITS) | pipe.map(cls.get_dataset_name))

    @property
    @abc.abstractmethod
    def split_dir(self) -> Path: ...

    def get_file_names(self, dataset_name: str) -> list[str]:
        split_path: Path = Path(self.split_dir, f"{dataset_name}.txt")

        if not split_path.exists():
            raise ValueError(f"Split file does not exist: {split_path}")

        with open(split_path, "r") as f:
            return list(f | pipe.map(str.rstrip))


@dataclasses.dataclass
class BaseDatasetFactory(Generic[DRIVER_T]):
    @classmethod
    @abc.abstractmethod
    def get_subclasses(cls) -> list[type[DRIVER_T]]: ...

    @overload
    @classmethod
    def register_by_name(
        cls, dataset_name: str, root_dir: Path, allow_missing: Literal[False] = ...
    ) -> DRIVER_T: ...

    @overload
    @classmethod
    def register_by_name(
        cls, dataset_name: str, root_dir: Path, allow_missing: Literal[True]
    ) -> DRIVER_T | None: ...

    @classmethod
    def register_by_name(
        cls, dataset_name: str, root_dir: Path, allow_missing: bool = False
    ) -> DRIVER_T | None:
        data_driver: DRIVER_T | None = None

        subclass: type[DRIVER_T]
        for subclass in cls.get_subclasses():
            if dataset_name not in subclass.available_dataset_names():
                continue

            if dataset_name in DatasetCatalog.list():
                logging.info(f"Dataset {dataset_name!s} already registered!")
                continue

            data_driver = subclass.register(root_dir=root_dir)

        if (not allow_missing) and (data_driver is None):
            raise ValueError(f"Dataset {dataset_name} not found.")

        return data_driver

    @classmethod
    def available_dataset_names(cls) -> list[str]:
        return list(
            cls.get_subclasses()
            | pipe.map(lambda subclass: subclass.available_dataset_names())
            | pipe.chain
        )
