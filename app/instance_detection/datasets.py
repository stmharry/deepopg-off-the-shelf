import dataclasses
import functools
from pathlib import Path
from typing import ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
from absl import logging

from app.coco.datasets import CocoDataset
from app.coco.schemas import CocoCategory
from detectron2.data import DatasetCatalog, MetadataCatalog

T = TypeVar("T", bound="InstanceDetection")


@dataclasses.dataclass
class InstanceDetection(CocoDataset):
    CATEGORY_MAPPING_RE: ClassVar[dict[str, str] | None] = None

    @classmethod
    def get_subclasses(cls) -> list[type["InstanceDetection"]]:
        return [
            InstanceDetectionV1,
            InstanceDetectionV1NTUH,
            InstanceDetectionOdontoAI,
        ]

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
        thing_colors: npt.NDArray[np.uint8] = cls.get_colors(len(thing_classes))

        for split in cls.SPLITS:
            name: str = f"{cls.PREFIX}_{split}"

            DatasetCatalog.register(
                name, functools.partial(self.get_split, split=split)
            )
            MetadataCatalog.get(name).set(
                thing_classes=thing_classes,
                thing_colors=thing_colors,
                json_file=self.coco_path,
                evaluator_type="coco",
            )

        return self


class InstanceDetectionV1(InstanceDetection):
    PREFIX: ClassVar[str] = "pano"
    SPLITS: ClassVar[list[str]] = ["train", "eval", "test", "debug"]
    CATEGORY_MAPPING_RE: ClassVar[dict[str, str] | None] = {
        r"TOOTH_(\d+)": r"TOOTH_\1",
        r"DENTAL_IMPLANT_(\d+)": "IMPLANT",
        r"ROOT_REMNANTS_(\d+)": "ROOT_REMNANTS",
        r"METAL_CROWN_(\d+)": "CROWN_BRIDGE",
        r"NON_METAL_CROWN_(\d+)": "CROWN_BRIDGE",
        r"METAL_FILLING_(\d+)": "FILLING",
        r"NON_METAL_FILLING_(\d+)": "FILLING",
        r"ROOT_CANAL_FILLING_(\d+)": "ENDO",
        r"CARIES_(\d+)": "CARIES",
        r"PERIAPICAL_RADIOLUCENT_(\d+)": "PERIAPICAL_RADIOLUCENT",
    }

    @property
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits", "instance-detection-v1")

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "instance-detection-v1-promaton.json")


@dataclasses.dataclass
class InstanceDetectionV1NTUH(InstanceDetectionV1):
    PREFIX: ClassVar[str] = "pano_ntuh"
    SPLITS: ClassVar[list[str]] = ["test", "debug"]

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "instance-detection-v1-ntuh.json")


@dataclasses.dataclass
class InstanceDetectionOdontoAI(InstanceDetection):
    PREFIX: ClassVar[str] = "pano_odontoai"
    SPLITS: ClassVar[list[str]] = ["train", "val", "test"]

    @property
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits", "instance-detection-odontoai")

    @property
    def coco_paths(self) -> list[Path]:
        return [
            Path(self.root_dir, "coco", "instance-detection-odontoai-train.json"),
            Path(self.root_dir, "coco", "instance-detection-odontoai-val.json"),
        ]
