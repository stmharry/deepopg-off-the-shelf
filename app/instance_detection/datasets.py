import dataclasses
import functools
import re
from pathlib import Path
from typing import ClassVar, TypeVar

import numpy as np
from absl import logging

from app.coco.datasets import CocoDatasetDriver, CocoDatasetFactory
from app.coco.schemas import CocoCategory
from app.instance_detection.schemas import InstanceDetectionData
from detectron2.data import DatasetCatalog, MetadataCatalog

T = TypeVar("T", bound="InstanceDetection")


@dataclasses.dataclass
class InstanceDetection(CocoDatasetDriver[InstanceDetectionData]):
    CATEGORY_NAME_TO_MAPPINGS: ClassVar[dict[str, dict[str, str]] | None] = None

    @classmethod
    def register(cls: type[T], root_dir: Path) -> T:
        logging.info(f"Registering {cls.__name__} dataset driver...")

        self = cls(root_dir=root_dir)

        thing_classes: list[str] = [category.name for category in self.coco_categories]
        thing_colors: list[np.ndarray] = cls.get_colors(len(thing_classes))

        for split in [None, *cls.SPLITS]:
            dataset_name: str = cls.get_dataset_name(split)

            logging.info(f"Registering '{dataset_name}' dataset into catalogs")

            DatasetCatalog.register(
                dataset_name,
                functools.partial(
                    self.get_coco_dataset_as_jsons, dataset_name=dataset_name
                ),
            )
            MetadataCatalog.get(dataset_name).set(
                thing_classes=thing_classes,
                thing_colors=thing_colors,
                json_file=self.coco_path,
                evaluator_type="coco",
            )

        return self

    @classmethod
    def map_categories(
        cls, categories: list[CocoCategory]
    ) -> dict[int, dict[str, str]]:
        if cls.CATEGORY_NAME_TO_MAPPINGS is None:
            return {}

        pattern_to_mappings: dict[re.Pattern, dict[str, str]] = {
            re.compile(pattern): mappings
            for pattern, mappings in cls.CATEGORY_NAME_TO_MAPPINGS.items()
        }

        category_id_to_mapped: dict[int, dict[str, str]] = {}
        for category in categories:
            if category.id is None:
                raise ValueError(f"Category {category.name} has no id!")

            pattern: re.Pattern
            for pattern in pattern_to_mappings.keys():
                match_obj = pattern.match(category.name)

                if match_obj is not None:
                    break

            else:
                # no match found for this category
                continue

            mappings: dict[str, str] = pattern_to_mappings[pattern]
            category_id_to_mapped[category.id] = {
                key: pattern.sub(to_pattern, category.name)
                for key, to_pattern in mappings.items()
            }

        return category_id_to_mapped


@dataclasses.dataclass
class InstanceDetectionRaw(InstanceDetection):
    PREFIX: ClassVar[str] = "pano_raw"
    SPLITS: ClassVar[list[str]] = []

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "promaton.json")


@dataclasses.dataclass
class InstanceDetectionV1(InstanceDetection):
    PREFIX: ClassVar[str] = "pano"
    SPLITS: ClassVar[list[str]] = [
        "train",
        "eval",
        "eval_v2",
        "test",
        "test_v2",
        "test_v2_1",
        "debug",
    ]
    CATEGORY_NAME_TO_MAPPINGS: ClassVar[dict[str, dict[str, str]] | None] = {
        r"TOOTH_(?P<fdi>\d+)": {
            "category": r"TOOTH_\g<fdi>",
            "fdi": r"\g<fdi>",
        },
        r"DENTAL_IMPLANT_(?P<fdi>\d+)": {
            "category": "IMPLANT",
            "fdi": r"\g<fdi>",
        },
        r"ROOT_REMNANTS_(?P<fdi>\d+)": {
            "category": "ROOT_REMNANTS",
            "fdi": r"\g<fdi>",
        },
        r"METAL_CROWN_(?P<fdi>\d+)": {
            "category": "CROWN_BRIDGE",
            "fdi": r"\g<fdi>",
        },
        r"NON_METAL_CROWN_(?P<fdi>\d+)": {
            "category": "CROWN_BRIDGE",
            "fdi": r"\g<fdi>",
        },
        r"METAL_FILLING_(?P<fdi>\d+)": {
            "category": "FILLING",
            "fdi": r"\g<fdi>",
        },
        r"NON_METAL_FILLING_(?P<fdi>\d+)": {
            "category": "FILLING",
            "fdi": r"\g<fdi>",
        },
        r"ROOT_CANAL_FILLING_(?P<fdi>\d+)": {
            "category": "ENDO",
            "fdi": r"\g<fdi>",
        },
        r"CARIES_(?P<fdi>\d+)": {
            "category": "CARIES",
            "fdi": r"\g<fdi>",
        },
        r"PERIAPICAL_RADIOLUCENT_(?P<fdi>\d+)": {
            "category": "PERIAPICAL_RADIOLUCENT",
            "fdi": r"\g<fdi>",
        },
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
    SPLITS: ClassVar[list[str]] = ["test", "test_v2", "test_v2_1", "debug"]

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "instance-detection-v1-ntuh.json")


@dataclasses.dataclass
class InstanceDetectionV2(InstanceDetection):
    PREFIX: ClassVar[str] = "pano_insdet_v2"
    SPLITS: ClassVar[list[str]] = ["train", "eval", "eval_v2", "test_v2_1"]
    CATEGORY_NAME_TO_MAPPINGS: ClassVar[dict[str, dict[str, str]] | None] = {
        r"TOOTH_(?P<fdi>\d+)": {
            "category": r"TOOTH",
            "fdi": r"\g<fdi>",
        },
        r"DENTAL_IMPLANT_(?P<fdi>\d+)": {
            "category": "IMPLANT",
            "fdi": r"\g<fdi>",
        },
        r"ROOT_REMNANTS_(?P<fdi>\d+)": {
            "category": "ROOT_REMNANTS",
            "fdi": r"\g<fdi>",
        },
        r"METAL_CROWN_(?P<fdi>\d+)": {
            "category": "CROWN_BRIDGE",
            "fdi": r"\g<fdi>",
        },
        r"NON_METAL_CROWN_(?P<fdi>\d+)": {
            "category": "CROWN_BRIDGE",
            "fdi": r"\g<fdi>",
        },
        r"METAL_FILLING_(?P<fdi>\d+)": {
            "category": "FILLING",
            "fdi": r"\g<fdi>",
        },
        r"NON_METAL_FILLING_(?P<fdi>\d+)": {
            "category": "FILLING",
            "fdi": r"\g<fdi>",
        },
        r"ROOT_CANAL_FILLING_(?P<fdi>\d+)": {
            "category": "ENDO",
            "fdi": r"\g<fdi>",
        },
        r"CARIES_(?P<fdi>\d+)": {
            "category": "CARIES",
            "fdi": r"\g<fdi>",
        },
        r"PERIAPICAL_RADIOLUCENT_(?P<fdi>\d+)": {
            "category": "PERIAPICAL_RADIOLUCENT",
            "fdi": r"\g<fdi>",
        },
    }

    @property
    def split_dir(self) -> Path:
        return Path(self.root_dir, "splits", "instance-detection-v2")

    @property
    def coco_path(self) -> Path:
        return Path(self.root_dir, "coco", "instance-detection-v2-promaton.json")


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


@dataclasses.dataclass
class InstanceDetectionFactory(CocoDatasetFactory[InstanceDetection]):
    @classmethod
    def get_subclasses(cls) -> list[type[InstanceDetection]]:
        return [
            InstanceDetectionRaw,
            InstanceDetectionV1,
            InstanceDetectionV1NTUH,
            InstanceDetectionV2,
            InstanceDetectionOdontoAI,
        ]
