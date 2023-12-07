import dataclasses
import functools
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
from absl import logging

from app.datasets import CocoDataset
from app.schemas import CocoCategory
from app.semantic_segmentation.schemas import SemanticSegmentationData
from detectron2.data import DatasetCatalog, MetadataCatalog

T = TypeVar("T", bound="SemanticSegmentation")


@dataclasses.dataclass
class SemanticSegmentation(CocoDataset):
    PREFIX: ClassVar[str] = "pano_semseg"

    @classmethod
    def register_by_name(cls: type[T], dataset_name: str, root_dir: Path) -> T | None:
        data_driver: SemanticSegmentation | None = None
        if dataset_name in [
            f"{cls.PREFIX}_train",
        ]:
            data_driver = SemanticSegmentationV4.register(root_dir=root_dir)

        return data_driver  # type: ignore

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

        stuff_classes: list[str] = [category.name for category in categories]
        stuff_colors: npt.NDArray[np.uint8] = cls.get_colors(len(stuff_classes))

        for split in cls.SPLITS:
            name: str = f"{cls.PREFIX}_{split}"

            DatasetCatalog.register(
                name, functools.partial(self.get_split, split=split)
            )
            MetadataCatalog.get(name).set(
                stuff_classes=stuff_classes,
                stuff_colors=stuff_colors,
                json_file=self.coco_path,
                evaluator_type="coco",
            )

        return self

    @classmethod
    def get_dataset(cls, self: "CocoDataset", coco_path: Path) -> list[dict[str, Any]]:
        dataset = super().get_dataset(self=self, coco_path=coco_path)

        # ensure the data is consistent with the schema
        data_schemas: list[SemanticSegmentationData] = [
            SemanticSegmentationData(
                file_name=data["file_name"],
                height=data["height"],
                width=data["width"],
                image_id=data["image_id"],
                sem_seg_file_name=Path(
                    self.mask_dir,
                    (
                        Path(data["file_name"])
                        .relative_to(self.image_dir)
                        .with_suffix(".png")
                    ),
                ),
            )
            for data in dataset
        ]

        return [data_schema.dict() for data_schema in data_schemas]


@dataclasses.dataclass
class SemanticSegmentationV4(SemanticSegmentation):
    PREFIX: ClassVar[str] = "pano_semseg_v4"
    SPLITS: ClassVar[list[str]] = ["train", "eval"]

    @property
    def mask_dir(self) -> Path:
        return Path(self.root_dir, "masks", "segmentation-v4")

    @property
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits", "segmentation-v4")

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "semantic-segmentation-v4.json")
