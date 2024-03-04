import abc
import dataclasses
import functools
from pathlib import Path
from typing import ClassVar, TypeVar

import numpy as np
from absl import logging
from pydantic.fields import _Unset

from app.coco.datasets import CocoDatasetDriver, CocoDatasetFactory
from app.semantic_segmentation.schemas import SemanticSegmentationData
from detectron2.data import DatasetCatalog, MetadataCatalog

T = TypeVar("T", bound="SemanticSegmentation")


@dataclasses.dataclass
class SemanticSegmentation(CocoDatasetDriver[SemanticSegmentationData]):
    @classmethod
    def register(cls: type[T], root_dir: Path) -> T:
        logging.info(f"Registering {cls.__name__} dataset...")

        self = cls(root_dir=root_dir)

        stuff_classes: list[str] = ["BACKGROUND"] + [
            category.name for category in self.coco_categories
        ]
        stuff_colors: list[np.ndarray] = cls.get_colors(len(stuff_classes))

        for split in cls.SPLITS:
            name: str = cls.get_dataset_name(split)

            DatasetCatalog.register(
                name, functools.partial(self.get_coco_dataset_as_jsons, split=split)
            )
            MetadataCatalog.get(name).set(
                stuff_classes=stuff_classes,
                stuff_colors=stuff_colors,
                ignore_label=0,
                json_file=self.coco_path,
                evaluator_type="sem_seg",
            )

        return self

    @property
    @abc.abstractmethod
    def mask_dir(self) -> Path: ...

    @functools.cached_property
    def coco_dataset(self) -> list[SemanticSegmentationData]:
        dataset: list[SemanticSegmentationData] = []
        for data in super().coco_dataset:
            sem_seg_file_name: Path = Path(
                self.mask_dir,
                Path(data.file_name).relative_to(self.image_dir).with_suffix(".png"),
            )
            if not sem_seg_file_name.exists():
                data = data.model_copy(update={"sem_seg_file_name": _Unset})

            dataset.append(data)

        return dataset


@dataclasses.dataclass
class SemanticSegmentationV4(SemanticSegmentation):
    PREFIX: ClassVar[str] = "pano_semseg_v4"
    SPLITS: ClassVar[list[str]] = ["train", "eval", "test", "debug"]

    @property
    def mask_dir(self) -> Path:
        return Path(self.root_dir, "masks", "segmentation-v4")

    @property
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits", "segmentation-v4")

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "semantic-segmentation-v4.json")


@dataclasses.dataclass
class SemanticSegmentationV4NTUH(SemanticSegmentationV4):
    PREFIX: ClassVar[str] = "pano_semseg_v4_ntuh"
    SPLITS: ClassVar[list[str]] = ["test"]

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "semantic-segmentation-v4-ntuh.json")


@dataclasses.dataclass
class SemanticSegmentationFactory(CocoDatasetFactory[SemanticSegmentation]):
    @classmethod
    def get_subclasses(cls) -> list[type[SemanticSegmentation]]:
        return [
            SemanticSegmentationV4,
            SemanticSegmentationV4NTUH,
        ]
