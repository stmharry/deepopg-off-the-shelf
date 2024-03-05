import abc
import dataclasses
import functools
import json
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import ijson
import numpy as np
import pipe
from absl import logging
from matplotlib import colormaps, colors
from pydantic import TypeAdapter

from app.coco.schemas import CocoCategory, CocoData, CocoImage
from app.types import PathLike
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json

DATA_T = TypeVar("DATA_T", bound="CocoData")
DRIVER_T = TypeVar("DRIVER_T", bound="CocoDatasetDriver")


@dataclasses.dataclass  # type: ignore
class CocoDatasetDriver(Generic[DATA_T], metaclass=abc.ABCMeta):
    PREFIX: ClassVar[str]
    SPLITS: ClassVar[list[str]] = []

    root_dir: Path

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

    #

    @classmethod
    def get_coco_categories(cls, coco_path: PathLike) -> list[CocoCategory]:
        with open(coco_path) as f:
            return list(
                ijson.items(f, "categories.item")
                | pipe.map(TypeAdapter(CocoCategory).validate_python)
                | pipe.sort(key=lambda category: category.id)
            )

    @classmethod
    def get_coco_images(cls, coco_path: PathLike) -> list[CocoImage]:
        with open(coco_path) as f:
            return list(
                ijson.items(f, "images.item")
                | pipe.map(TypeAdapter(CocoImage).validate_python)
            )

    @classmethod
    def get_colors(cls, num_classes: int) -> list[np.ndarray]:
        cmap: colors.Colormap = colormaps.get_cmap("gist_rainbow")

        return list(
            (
                # background color
                (np.asarray((0, 0, 0), dtype=np.uint8),),
                # foreground colors
                (
                    range(num_classes - 1)
                    | pipe.map(
                        lambda i: np.asarray(cmap(i)[:3]).__mul__(255).astype(np.uint8)
                    )
                ),
            )
            | pipe.chain
        )

    #

    @property
    def image_dir(self) -> Path:
        return Path(self.root_dir, "images")

    @property
    @abc.abstractmethod
    def split_dir(self) -> Path: ...

    @property
    def coco_path(self) -> Path | None:
        return None

    @property
    def coco_paths(self) -> list[Path]:
        if self.coco_path is None:
            logging.warning(
                f"InstanceDetection {self.__class__.__name__} does not have a"
                " coco_path!"
            )
            return []

        return [self.coco_path]

    @functools.cached_property
    def coco_categories(self) -> list[CocoCategory]:
        categories: list[CocoCategory] | None = None
        for coco_path in self.coco_paths:
            if not coco_path.exists():
                raise ValueError(f"Coco dataset does not exist: {coco_path!s}!")

            _categories = self.get_coco_categories(coco_path)
            if (categories is not None) and categories != _categories:
                raise ValueError(
                    f"Categories from {coco_path} do not match previous categories!"
                )

            categories = _categories

        if categories is None:
            raise ValueError(f"No categories found for {self.PREFIX}!")

        return categories

    @functools.cached_property
    def coco_dataset(self) -> list[DATA_T]:
        return list(
            (
                load_coco_json(
                    coco_path, image_root=self.image_dir, dataset_name=self.PREFIX
                )
                for coco_path in self.coco_paths
            )
            | pipe.chain
            | pipe.map(TypeAdapter(DATA_T).validate_python)
        )

    def get_file_names(self, dataset_name: str) -> list[str]:
        split_path: Path = Path(self.split_dir, f"{dataset_name}.txt")

        if not split_path.exists():
            raise ValueError(f"Split file does not exist: {split_path}")

        with open(split_path, "r") as f:
            return list(f | pipe.map(str.rstrip))

    def get_coco_dataset(self, dataset_name: str) -> list[DATA_T]:
        if dataset_name == self.PREFIX:
            return self.coco_dataset

        names: set[str] = set(self.get_file_names(dataset_name=dataset_name))

        return list(
            self.coco_dataset
            | pipe.filter(
                lambda data: str(
                    data.file_name.with_suffix("").relative_to(self.image_dir)
                )
                in names
            )
        )

    def get_coco_dataset_as_jsons(self, dataset_name: str) -> list[dict[str, Any]]:
        return list(
            self.get_coco_dataset(dataset_name=dataset_name)
            | pipe.map(lambda data: json.loads(data.model_dump_json()))
        )


@dataclasses.dataclass
class CocoDatasetFactory(Generic[DRIVER_T]):
    @classmethod
    @abc.abstractmethod
    def get_subclasses(cls) -> list[type[DRIVER_T]]: ...

    @classmethod
    def register_by_name(cls, dataset_name: str, root_dir: Path) -> DRIVER_T:
        data_driver: DRIVER_T | None = None

        subclass: type[DRIVER_T]
        for subclass in cls.get_subclasses():
            if dataset_name not in subclass.available_dataset_names():
                continue

            if dataset_name in DatasetCatalog.list():
                logging.info(f"Dataset {dataset_name!s} already registered!")
                continue

            data_driver = subclass.register(root_dir=root_dir)

        if data_driver is None:
            raise ValueError(f"Dataset {dataset_name} not found.")

        return data_driver

    @classmethod
    def available_dataset_names(cls) -> list[str]:
        return list(
            cls.get_subclasses()
            | pipe.map(lambda subclass: subclass.available_dataset_names())
            | pipe.chain
        )
