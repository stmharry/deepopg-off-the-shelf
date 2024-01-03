import dataclasses
import functools
import re
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import ClassVar, TypeVar

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pycocotools.coco
import pycocotools.mask
import tqdm
from absl import logging

from app.datasets import CocoDataset
from app.schemas import Coco, CocoAnnotation, CocoCategory, CocoImage
from detectron2.data import DatasetCatalog, MetadataCatalog

T = TypeVar("T", bound="InstanceDetection")


@dataclasses.dataclass
class InstanceDetection(CocoDataset):
    CATEGORY_MAPPING_RE: ClassVar[dict[str, str] | None] = None
    IMAGE_GLOB: ClassVar[str] = "PROMATON/*.jpg"

    category_mapping: dict[str, str] | None = None

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

    @property
    def mask_dir(self) -> Path:
        return Path(self.root_dir, "masks", "raw", "PROMATON", "objects")

    def _parse_mask_id(self, mask_id: str) -> tuple[str, str, str]:
        names: list[str] = mask_id.split("_")

        image_name: str = names[0]
        supercategory: str = "_".join(names[1:-1])
        fdi: str = names[-1]

        return (image_name, supercategory, fdi)

    def _prepare_categories(self) -> list[CocoCategory]:
        mask_paths: list[Path] = list(self.mask_dir.glob("*.png"))

        category_names: set[str] = set()
        for mask_path in mask_paths:
            (_, supercategory, fdi) = self._parse_mask_id(mask_path.stem)
            category_name: str = f"{supercategory}_{fdi}"
            category_names.add(category_name)

        if self.CATEGORY_MAPPING_RE is not None:
            mapped_category_names: set[str] = set()
            category_mapping: dict[str, str] = {}
            for category_name in category_names:
                for from_pattern, to_pattern in self.CATEGORY_MAPPING_RE.items():
                    match_obj: re.Match | None = re.match(from_pattern, category_name)
                    if match_obj is None:
                        continue

                    mapped_category_name: str = re.sub(
                        from_pattern, to_pattern, category_name
                    )
                    break
                else:
                    # no mapping found, skipping this `category_name`
                    continue

                mapped_category_names.add(mapped_category_name)
                category_mapping[category_name] = mapped_category_name

            category_names = mapped_category_names
            self.category_mapping = category_mapping

        categories: list[CocoCategory] = [
            CocoCategory(name=category_name) for category_name in category_names
        ]

        return sorted(categories, key=lambda category: category.name)

    @staticmethod
    def _prepare_image(image_path: Path, image_dir: Path) -> dict:
        meta: dict = iio.immeta(image_dir / image_path)
        (width, height) = meta["shape"]

        return dict(
            id=image_path.stem,
            file_name=str(image_path),
            width=width,
            height=height,
        )

    def prepare_images(self, n_jobs: int = -1) -> list[CocoImage]:
        image_paths: list[Path] = list(
            image_path.relative_to(self.image_dir)
            for image_path in self.image_dir.glob(self.IMAGE_GLOB)
        )

        images: list[CocoImage] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_id: dict[Future, Path] = {}

            image_path: Path
            for image_path in tqdm.tqdm(image_paths, desc="Image Job Dispatch"):
                future: Future = executor.submit(
                    self._prepare_image, image_path=image_path, image_dir=self.image_dir
                )
                future_to_id[future] = image_path

            for future in tqdm.tqdm(
                as_completed(future_to_id), desc="Image", total=len(future_to_id)
            ):
                try:
                    result = future.result()
                except Exception as e:
                    image_path = future_to_id[future]

                    logging.warning(f"Error raised processing {image_path}: {e}!")
                    continue

                image: CocoImage = CocoImage.parse_obj(result)
                images.append(image)

        return sorted(images, key=lambda image: image.id)

    @staticmethod
    def _prepare_annotation(
        mask_path: Path, mask_name: str, category_name: str, image_name: str
    ) -> dict | None:
        mask: npt.NDArray[np.uint8] = iio.imread(mask_path)
        rows: npt.NDArray[np.bool_] = np.any(mask, axis=1)  # type: ignore
        cols: npt.NDArray[np.bool_] = np.any(mask, axis=0)  # type: ignore
        y: npt.NDArray[np.int64] = np.where(rows)[0]
        x: npt.NDArray[np.int64] = np.where(cols)[0]

        if (len(y) == 0) or (len(x) == 0):
            return None

        bbox: list[int] = [x[0], y[0], x[-1] - x[0] + 1, y[-1] - y[0] + 1]
        rle: pycocotools.mask._EncodedRLE = pycocotools.mask.encode(
            np.asarray(mask, order="F")  # type:ignore
        )
        area: int = pycocotools.mask.area(rle)

        return dict(
            id=mask_name,
            image_id=image_name,
            category_id=category_name,
            bbox=bbox,
            segmentation=rle,
            area=area,
        )

    def prepare_annotations(
        self,
        category_names: set[str],
        image_names: set[str | int],
        n_jobs: int = -1,
    ) -> list[CocoAnnotation]:
        mask_paths: list[Path] = list(self.mask_dir.glob("*.png"))

        annotations: list[CocoAnnotation] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_id: dict[Future, Path] = {}
            for mask_path in tqdm.tqdm(mask_paths, desc="Annotation Job Dispatch"):
                mask_name: str = mask_path.stem
                (image_name, supercategory, fdi) = self._parse_mask_id(mask_name)

                if image_name not in image_names:
                    continue

                category_name: str | None = f"{supercategory}_{fdi}"

                if self.CATEGORY_MAPPING_RE is not None:
                    assert self.category_mapping is not None
                    assert category_name is not None

                    category_name = self.category_mapping.get(category_name)

                if category_name not in category_names:
                    continue

                future: Future = executor.submit(
                    self._prepare_annotation,
                    mask_path=mask_path,
                    mask_name=mask_name,
                    category_name=category_name,
                    image_name=image_name,
                )
                future_to_id[future] = mask_path

            for future in tqdm.tqdm(
                as_completed(future_to_id), desc="Annotation", total=len(future_to_id)
            ):
                try:
                    result = future.result()
                except Exception as e:
                    mask_path = future_to_id[future]
                    logging.warning(f"Error raised processing {mask_path}: {e}!")
                    continue

                if result is None:
                    continue

                annotation: CocoAnnotation = CocoAnnotation.parse_obj(result)
                annotations.append(annotation)

        return sorted(annotations, key=lambda annotation: annotation.id)

    def prepare_coco(self, force: bool = False, n_jobs: int = -1) -> None:
        if (not force) and self.coco_path.exists():
            logging.info(f"Coco dataset exists: {self.coco_path!s}.")
            return

        categories: list[CocoCategory] = self._prepare_categories()
        images: list[CocoImage] = self.prepare_images(n_jobs=n_jobs)
        annotations: list[CocoAnnotation] = self.prepare_annotations(
            category_names=set(category.name for category in categories),
            image_names=set(image.id for image in images),
            n_jobs=n_jobs,
        )

        category_name_to_index: dict[str | int, int] = {}
        for num, category in enumerate(categories):
            category_name_to_index[category.name] = num + 1
            category.id = num + 1

        image_name_to_index: dict[str | int, int] = {}
        for num, image in enumerate(images):
            image_name_to_index[image.id] = num + 1
            image.id = num + 1

        for num, annotation in enumerate(annotations):
            annotation.id = num + 1
            annotation.image_id = image_name_to_index[annotation.image_id]
            annotation.category_id = category_name_to_index[annotation.category_id]

        coco: Coco = Coco(categories=categories, images=images, annotations=annotations)

        with open(self.coco_path, "w") as f:
            f.write(coco.json())


class InstanceDetectionV1(InstanceDetection):
    PREFIX: ClassVar[str] = "pano"
    SPLITS: ClassVar[list[str]] = ["all", "train", "eval", "debug"]
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
