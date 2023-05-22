import abc
import logging
import re
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt
import pandas as pd
import pycocotools.coco
import pycocotools.mask
import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
from pydantic import BaseModel, parse_obj_as

logger = logging.getLogger(__name__)


# coco


class CocoImage(BaseModel):
    id: str
    file_name: str
    width: int
    height: int


class CocoCategory(BaseModel):
    id: Optional[int] = None
    name: str


class CocoRLE(BaseModel):
    size: List[int]
    counts: str


class CocoAnnotation(BaseModel):
    id: str
    image_id: str
    category_id: Union[int, str]
    bbox: List[int]
    segmentation: CocoRLE
    area: int
    iscrowd: int = 0


class Coco(BaseModel):
    categories: List[CocoCategory]
    images: List[CocoImage]
    annotations: List[CocoAnnotation]


# instance detection


class InstanceDetectionAnnotation(BaseModel):
    bbox: List[int]
    bbox_mode: BoxMode
    category_id: int
    segmentation: CocoRLE
    iscrowd: int


class InstanceDetectionData(BaseModel):
    file_name: Path
    height: int
    width: int
    image_id: str
    annotations: List[InstanceDetectionAnnotation]


@dataclass
class InstanceDetection(metaclass=abc.ABCMeta):
    CATEGORY_MAPPING_RE: ClassVar[Optional[Dict[str, str]]] = None

    root_dir: Union[Path, str]
    category_mapping: Optional[Dict[str, str]] = None

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

        image_id: str = names[0]
        supercategory: str = "_".join(names[1:-1])
        fdi: str = names[-1]

        return (image_id, supercategory, fdi)

    def _prepare_categories(self) -> List[CocoCategory]:
        mask_paths: List[Path] = list(self.mask_dir.glob("*.png"))

        # name_to_category: Dict[str, CocoCategory] = {}
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

        sorted_category_names: List[str] = list(sorted(category_names))

        categories: List[CocoCategory] = []
        for num_category, category_name in enumerate(sorted_category_names):
            categories.append(CocoCategory(id=num_category, name=category_name))

        return categories

    @staticmethod
    def _prepare_image(image_path: Path) -> Dict:
        meta: Dict = iio.immeta(image_path)
        (width, height) = meta["shape"]

        return dict(
            id=image_path.stem,
            file_name=str(image_path),
            width=width,
            height=height,
        )

    def prepare_images(self, n_jobs: int = -1) -> List[CocoImage]:
        image_paths: List[Path] = list(self.image_dir.glob("*.jpg"))

        images: List[CocoImage] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_id: Dict[Future, Path] = {}

            image_path: Path
            for image_path in tqdm.tqdm(image_paths, desc="Image Job Dispatch"):
                future: Future = executor.submit(
                    self._prepare_image, image_path=image_path
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

        return images

    @staticmethod
    def _prepare_annotation(
        mask_path: Path, mask_id: str, category_id: int, image_id: str
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
            id=mask_id,
            image_id=image_id,
            category_id=category_id,
            bbox=bbox,
            segmentation=rle,
            area=area,
        )

    def prepare_annotations(
        self,
        name_to_category: Dict[str, CocoCategory],
        image_ids: Set[str],
        n_jobs: int = -1,
    ) -> List[CocoAnnotation]:
        mask_paths: List[Path] = list(self.mask_dir.glob("*.png"))

        annotations: List[CocoAnnotation] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_id: Dict[Future, Path] = {}
            make_path: Path
            for mask_path in tqdm.tqdm(mask_paths, desc="Annotation Job Dispatch"):
                mask_id: str = mask_path.stem
                (image_id, supercategory, fdi) = self._parse_mask_id(mask_id)

                if image_id not in image_ids:
                    continue

                category_name: Optional[str] = f"{supercategory}_{fdi}"

                if self.CATEGORY_MAPPING_RE is not None:
                    assert self.category_mapping is not None
                    assert category_name is not None

                    category_name = self.category_mapping.get(category_name)

                if category_name not in name_to_category:
                    continue

                category_id: Optional[int] = name_to_category[category_name].id
                assert category_id is not None

                future: Future = executor.submit(
                    self._prepare_annotation,
                    mask_path=mask_path,
                    mask_id=mask_id,
                    category_id=category_id,
                    image_id=image_id,
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

        return annotations

    def prepare_coco(self, force: bool = False, n_jobs: int = -1) -> None:
        if (not force) and self.coco_path.exists():
            logger.info(f"Coco dataset exists: {self.coco_path!s}.")
            return

        categories: List[CocoCategory] = self._prepare_categories()
        images: List[CocoImage] = self.prepare_images(n_jobs=n_jobs)

        name_to_category: Dict[str, CocoCategory] = {
            category.name: category for category in categories
        }
        image_ids: Set[str] = set(image.id for image in images)
        annotations: List[CocoAnnotation] = self.prepare_annotations(
            name_to_category=name_to_category, image_ids=image_ids, n_jobs=n_jobs
        )

        coco: Coco = Coco(categories=categories, images=images, annotations=annotations)

        with open(self.coco_path, "w") as f:
            f.write(coco.json())

    def register(self) -> None:
        coco = Coco.parse_file(self.coco_path)

        thing_classes: List[str] = [category.name for category in coco.categories]
        thing_colors: npt.NDArray[np.uint8] = (
            np.r_[
                [(0, 0, 0)],
                cm.gist_rainbow(np.arange(len(thing_classes) - 1))[:, :3],  # type: ignore
            ]
            .__mul__(255)
            .astype(np.uint8)
        )

        dataset: List[InstanceDetectionData] = parse_obj_as(
            List[InstanceDetectionData],
            load_coco_json(self.coco_path, image_root=self.image_dir),
        )

        for split in ["train", "eval"]:
            name: str = f"pano_{split}"

            file_names: List[str] = (
                pd.read_csv(self.split_dir / f"{split}.txt", header=None)  # type: ignore
                .squeeze()
                .tolist()
            )
            file_paths: Set[Path] = set(
                Path(self.image_dir, f"{filename}.jpg") for filename in file_names
            )
            _dataset: List[Dict[str, Any]] = [
                data.dict() for data in dataset if data.file_name in file_paths
            ]

            DatasetCatalog.register(name, lambda: _dataset)
            MetadataCatalog.get(name).set(
                thing_classes=thing_classes,
                thing_colors=thing_colors,
                json_file=self.coco_path,
                evaluator_type="coco",
            )


@dataclass
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
