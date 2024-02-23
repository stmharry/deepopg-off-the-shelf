import abc
import dataclasses
import itertools
from pathlib import Path
from typing import Any, ClassVar, TypeVar, cast

import ijson
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from absl import logging
from pydantic import parse_obj_as

from app.schemas import CocoCategory, CocoImage
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json

T = TypeVar("T", bound="CocoDataset")


@dataclasses.dataclass  # type: ignore
class CocoDataset(metaclass=abc.ABCMeta):
    PREFIX: ClassVar[str]
    SPLITS: ClassVar[list[str]] = []

    root_dir: Path

    _coco_dataset: list[dict[str, Any]] | None = dataclasses.field(default=None)

    @classmethod
    def available_dataset_names(cls: type[T]) -> list[str]:
        dataset_names: list[str] = []

        subclass: type[T]
        for subclass in cls.get_subclasses():  # type: ignore
            dataset_names.extend(subclass.get_dataset_names())

        return dataset_names

    @classmethod
    def register_by_name(cls: type[T], dataset_name: str, root_dir: Path) -> T | None:
        data_driver: T | None = None

        subclass: type[T]
        for subclass in cls.get_subclasses():  # type: ignore
            if dataset_name not in subclass.get_dataset_names():
                continue

            if dataset_name in DatasetCatalog.list():
                logging.info(f"Dataset {dataset_name!s} already registered!")
                continue

            data_driver = subclass.register(root_dir=root_dir)

        return data_driver

    @classmethod
    @abc.abstractmethod
    def register(cls: type[T], root_dir: Path) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def get_subclasses(cls) -> list[type]:
        ...

    @classmethod
    def get_dataset_names(cls) -> list[str]:
        return [cls.PREFIX, *[f"{cls.PREFIX}_{split}" for split in cls.SPLITS]]

    @classmethod
    def get_coco_categories(cls, coco_path: str | Path) -> list[CocoCategory]:
        with open(coco_path) as f:
            categories: list[CocoCategory] = parse_obj_as(
                list[CocoCategory], ijson.items(f, "categories.item")
            )
            sorted_categories: list[CocoCategory] = sorted(
                categories, key=lambda category: cast(int, category.id)
            )

        return sorted_categories

    @classmethod
    def get_coco_images(cls, coco_path: str | Path) -> list[CocoImage]:
        with open(coco_path) as f:
            images: list[CocoImage] = parse_obj_as(
                list[CocoImage], ijson.items(f, "images.item")
            )

        return images

    @classmethod
    def get_dataset(cls, self: "CocoDataset", coco_path: Path) -> list[dict[str, Any]]:
        return load_coco_json(
            coco_path, image_root=self.image_dir, dataset_name=self.PREFIX
        )

    @classmethod
    def get_colors(cls, num_classes: int) -> np.ndarray:
        return (
            np.r_[
                [(0, 0, 0)],
                cm.gist_rainbow(np.arange(num_classes - 1))[:, :3],  # type: ignore
            ]
            .__mul__(255)
            .astype(np.uint8)
        )

    @property
    def image_dir(self) -> Path:
        return Path(self.root_dir, "images")

    @property
    @abc.abstractmethod
    def mask_dir(self) -> Path:
        return Path(self.root_dir, "masks")

    @property
    @abc.abstractmethod
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits")

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

    @property
    def dataset(self) -> list[dict[str, Any]]:
        for coco_path in self.coco_paths:
            if not coco_path.exists():
                raise ValueError(f"Coco dataset does not exist: {coco_path!s}!")

        if self._coco_dataset is None:
            self._coco_dataset = list(
                itertools.chain.from_iterable(
                    self.__class__.get_dataset(self, coco_path)
                    for coco_path in self.coco_paths
                )
            )

        return self._coco_dataset

    def get_split_file_names(self, split: str) -> list[str]:
        split_path: Path = Path(self.split_dir, f"{self.PREFIX}_{split}.txt")
        file_names: list[str] = (
            pd.read_csv(split_path, header=None).squeeze().tolist()  # type: ignore
        )

        return file_names

    def get_split(self, split: str) -> list[dict[str, Any]]:
        file_names: list[str] = self.get_split_file_names(split=split)
        file_paths: set[Path] = set(
            Path(self.image_dir, f"{file_name}.jpg") for file_name in file_names
        )
        coco_dataset: list[dict[str, Any]] = [
            data for data in self.dataset if Path(data["file_name"]) in file_paths
        ]

        return coco_dataset
