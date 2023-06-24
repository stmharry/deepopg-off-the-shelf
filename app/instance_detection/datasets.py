import abc
import dataclasses
import functools
import re
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import ijson
import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt
import pandas as pd
import pycocotools.coco
import pycocotools.mask
import tqdm
from absl import logging
from pydantic import parse_obj_as

from app.instance_detection.schemas import (
    Coco,
    CocoAnnotation,
    CocoCategory,
    CocoImage,
    InstanceDetectionData,
)
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

T = TypeVar("T", bound="InstanceDetection")


@dataclasses.dataclass  # type: ignore
class InstanceDetection(metaclass=abc.ABCMeta):
    CATEGORY_MAPPING_RE: ClassVar[Optional[Dict[str, str]]] = None
    IMAGE_GLOB: ClassVar[str] = "PROMATON/*.jpg"

    root_dir: Union[Path, str]
    category_mapping: Optional[Dict[str, str]] = None

    _coco_dataset: Optional[List[Dict[str, Any]]] = dataclasses.field(default=None)

    @classmethod
    def register(cls: Type[T], root_dir: Union[Path, str]) -> T:
        self = cls(root_dir=root_dir)

        with open(self.coco_path) as f:
            categories: Iterable[Dict] = ijson.items(f, "categories.item")
            thing_classes: List[str] = [category["name"] for category in categories]

        thing_colors: npt.NDArray[np.uint8] = (
            np.r_[
                [(0, 0, 0)],
                cm.gist_rainbow(np.arange(len(thing_classes) - 1))[:, :3],  # type: ignore
            ]
            .__mul__(255)
            .astype(np.uint8)
        )

        for split in ["all", "train", "eval", "debug"]:
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
    def dataset(self) -> List[Dict[str, Any]]:
        if not self.coco_path.exists():
            raise ValueError(
                f"Coco dataset does not exist: {self.coco_path!s}, please run "
                f"`InstanceDetection.prepare_coco` first!"
            )

        if self._coco_dataset is None:
            self._coco_dataset = load_coco_json(
                self.coco_path, image_root=self.image_dir, dataset_name="pano"
            )

        return self._coco_dataset

    @property
    def image_dir(self) -> Path:
        return Path(self.root_dir, "images")

    @property
    def mask_dir(self) -> Path:
        return Path(self.root_dir, "masks", "raw", "PROMATON", "objects")

    @property
    @abc.abstractmethod
    def split_dir(self) -> Path:
        pass

    @property
    @abc.abstractmethod
    def coco_path(self) -> Path:
        pass

    def _parse_mask_id(self, mask_id: str) -> Tuple[str, str, str]:
        names: List[str] = mask_id.split("_")

        image_name: str = names[0]
        supercategory: str = "_".join(names[1:-1])
        fdi: str = names[-1]

        return (image_name, supercategory, fdi)

    def _prepare_categories(self) -> List[CocoCategory]:
        mask_paths: List[Path] = list(self.mask_dir.glob("*.png"))

        category_names: Set[str] = set()
        for mask_path in mask_paths:
            (_, supercategory, fdi) = self._parse_mask_id(mask_path.stem)
            category_name: str = f"{supercategory}_{fdi}"
            category_names.add(category_name)

        if self.CATEGORY_MAPPING_RE is not None:
            mapped_category_names: Set[str] = set()
            category_mapping: Dict[str, str] = {}
            for category_name in category_names:
                for from_pattern, to_pattern in self.CATEGORY_MAPPING_RE.items():
                    match_obj: Optional[re.Match] = re.match(
                        from_pattern, category_name
                    )
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

        categories: List[CocoCategory] = [
            CocoCategory(name=category_name) for category_name in category_names
        ]

        return sorted(categories, key=lambda category: category.name)

    @staticmethod
    def _prepare_image(image_path: Path, image_dir: Path) -> Dict:
        meta: Dict = iio.immeta(image_dir / image_path)
        (width, height) = meta["shape"]

        return dict(
            id=image_path.stem,
            file_name=str(image_path),
            width=width,
            height=height,
        )

    def prepare_images(self, n_jobs: int = -1) -> List[CocoImage]:
        image_paths: List[Path] = list(
            image_path.relative_to(self.image_dir)
            for image_path in self.image_dir.glob(self.IMAGE_GLOB)
        )

        images: List[CocoImage] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_id: Dict[Future, Path] = {}

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
    ) -> Optional[Dict]:
        mask: npt.NDArray[np.uint8] = iio.imread(mask_path)
        rows: npt.NDArray[np.bool_] = np.any(mask, axis=1)  # type: ignore
        cols: npt.NDArray[np.bool_] = np.any(mask, axis=0)  # type: ignore
        y: npt.NDArray[np.int64] = np.where(rows)[0]
        x: npt.NDArray[np.int64] = np.where(cols)[0]

        if (len(y) == 0) or (len(x) == 0):
            return None

        bbox: List[int] = [x[0], y[0], x[-1] - x[0] + 1, y[-1] - y[0] + 1]
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
        category_names: Set[str],
        image_names: Set[Union[str, int]],
        n_jobs: int = -1,
    ) -> List[CocoAnnotation]:
        mask_paths: List[Path] = list(self.mask_dir.glob("*.png"))

        annotations: List[CocoAnnotation] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_id: Dict[Future, Path] = {}
            for mask_path in tqdm.tqdm(mask_paths, desc="Annotation Job Dispatch"):
                mask_name: str = mask_path.stem
                (image_name, supercategory, fdi) = self._parse_mask_id(mask_name)

                if image_name not in image_names:
                    continue

                category_name: Optional[str] = f"{supercategory}_{fdi}"

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

        categories: List[CocoCategory] = self._prepare_categories()
        images: List[CocoImage] = self.prepare_images(n_jobs=n_jobs)
        annotations: List[CocoAnnotation] = self.prepare_annotations(
            category_names=set(category.name for category in categories),
            image_names=set(image.id for image in images),
            n_jobs=n_jobs,
        )

        category_name_to_index: Dict[Union[str, int], int] = {}
        for num, category in enumerate(categories):
            category_name_to_index[category.name] = num + 1
            category.id = num + 1

        image_name_to_index: Dict[Union[str, int], int] = {}
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

    def get_split_file_names(self, split: Optional[str] = None) -> List[str]:
        if split is None:
            split = "all"

        split_path: Path = self.split_dir / f"{split}.txt"
        file_names: List[str] = (
            pd.read_csv(split_path, header=None).squeeze().tolist()  # type: ignore
        )

        return file_names

    def get_split(
        self, split: Optional[str] = None, as_schema: bool = True
    ) -> Union[List[Dict[str, Any]], List[InstanceDetectionData]]:
        file_names: List[str] = self.get_split_file_names(split=split)
        file_paths: Set[Path] = set(
            Path(self.image_dir, f"{file_name}.jpg") for file_name in file_names
        )
        coco_dataset: List[Dict[str, Any]] = [
            data for data in self.dataset if Path(data["file_name"]) in file_paths
        ]

        if as_schema:
            return parse_obj_as(List[InstanceDetectionData], coco_dataset)
        else:
            return coco_dataset


@dataclasses.dataclass
class InstanceDetectionV1(InstanceDetection):
    CATEGORY_MAPPING_RE: ClassVar[Optional[Dict[str, str]]] = {
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
        return Path(self.root_dir, "coco", "instance-detection-v1.json")
