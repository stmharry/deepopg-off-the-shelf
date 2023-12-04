import abc
import dataclasses
import functools
import itertools
from pathlib import Path
from typing import Any, ClassVar, TypeVar, cast

import ijson
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt
import pandas as pd
from absl import logging
from pydantic import parse_obj_as

from app.instance_detection.schemas import InstanceDetectionData
from app.schemas import CocoCategory, CocoImage
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

T = TypeVar("T", bound="CocoDataset")


@dataclasses.dataclass  # type: ignore
class CocoDataset(metaclass=abc.ABCMeta):
    SPLITS: ClassVar[list[str]] = []

    root_dir: Path

    _coco_dataset: list[dict[str, Any]] | None = dataclasses.field(default=None)

    @classmethod
    def register(cls: type[T], root_dir: Path) -> T:
        logging.info(f"Registering {cls.__name__!s} dataset...")

        self = cls(root_dir=root_dir)

        categories: list[CocoCategory] | None = None
        for coco_path in self.coco_paths:
            _categories = self.get_coco_categories(coco_path)

            if categories is None:
                categories = _categories

            elif categories != _categories:
                raise ValueError(
                    f"Categories from {coco_path!s} do not match previous categories!"
                )

        if categories is None:
            raise ValueError(f"No categories found in {self.coco_paths!s}!")

        thing_classes: list[str] = [category.name for category in categories]
        thing_colors: npt.NDArray[np.uint8] = (
            np.r_[
                [(0, 0, 0)],
                cm.gist_rainbow(np.arange(len(thing_classes) - 1))[:, :3],  # type: ignore
            ]
            .__mul__(255)
            .astype(np.uint8)
        )

        for split in self.SPLITS:
            name: str = f"pano_{split}"

            DatasetCatalog.register(
                name, functools.partial(self.get_split, split=split, as_schema=False)
            )
            MetadataCatalog.get(name).set(
                thing_classes=thing_classes,
                thing_colors=thing_colors,
                json_file=self.coco_path,
                evaluator_type="coco",
            )

        return self

    @property
    def dataset(self) -> list[dict[str, Any]]:
        for coco_path in self.coco_paths:
            if not coco_path.exists():
                raise ValueError(
                    f"Coco dataset does not exist: {coco_path!s}, please run "
                    f"`InstanceDetection.prepare_coco` first!"
                )

        if self._coco_dataset is None:
            self._coco_dataset = list(
                itertools.chain.from_iterable(
                    load_coco_json(
                        coco_path, image_root=self.image_dir, dataset_name="pano"
                    )
                    for coco_path in self.coco_paths
                )
            )

        return self._coco_dataset

    @property
    @abc.abstractmethod
    def image_dir(self) -> Path:
        ...

    @property
    @abc.abstractmethod
    def mask_dir(self) -> Path:
        ...

    @property
    @abc.abstractmethod
    def split_dir(self) -> Path:
        ...

    @property
    def coco_path(self) -> Path | None:
        return None

    @property
    def coco_paths(self) -> list[Path]:
        if self.coco_path is None:
            raise ValueError(
                f"InstanceDetection {self.__class__.__name__} does not have a coco_path!"
            )

        return [self.coco_path]

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

    def get_split_file_names(self, split: str | None = None) -> list[str]:
        if split is None:
            split = "all"

        split_path: Path = Path(self.split_dir, f"{split}.txt")
        file_names: list[str] = (
            pd.read_csv(split_path, header=None).squeeze().tolist()  # type: ignore
        )

        return file_names

    def get_split(
        self, split: str | None = None, as_schema: bool = True
    ) -> list[dict[str, Any]] | list[InstanceDetectionData]:
        file_names: list[str] = self.get_split_file_names(split=split)
        file_paths: set[Path] = set(
            Path(self.image_dir, f"{file_name}.jpg") for file_name in file_names
        )
        coco_dataset: list[dict[str, Any]] = [
            data for data in self.dataset if Path(data["file_name"]) in file_paths
        ]

        if as_schema:
            return parse_obj_as(list[InstanceDetectionData], coco_dataset)
        else:
            return coco_dataset
