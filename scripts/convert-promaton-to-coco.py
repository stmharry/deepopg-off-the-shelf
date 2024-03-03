import contextlib
import dataclasses
from collections.abc import Iterable
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pydicom
import rich.progress
from absl import app, flags, logging

from app.lib.psg import psg_to_npa
from app.masks import Mask
from app.schemas import Coco, CocoAnnotation, CocoCategory, CocoImage
from app.tasks import Task, map_task

flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_string("output_coco", "./data/coco/promaton.json", "Output COCO file.")
flags.DEFINE_integer("num_workers", 0, "Number of processes to use.")
FLAGS = flags.FLAGS


COCO_CATEGORIES: list[CocoCategory] = [
    *[
        CocoCategory(
            supercategory=supercategory, name=f"{supercategory}_{quadrant}{index}"
        )
        for supercategory in [
            "TOOTH",
            "TOOTH_NERVE",
            "DENTAL_IMPLANT",
            "DENTAL_IMPLANT_ABUTMENT",
            "ROOT_REMNANTS",
            "METAL_CROWN",
            "NON_METAL_CROWN",
            "METAL_FILLING",
            "NON_METAL_FILLING",
            "ROOT_CANAL_FILLING",
            "CARIES",
            "PERIAPICAL_RADIOLUCENT",
        ]
        for quadrant in range(1, 5)
        for index in range(1, 9)
    ],
    *[
        CocoCategory(supercategory=supercategory, name=f"{supercategory}_{direction}")
        for supercategory, directions in [
            ("MAXILLARY_SINUS", ["left", "right"]),
            ("INFERIOR_ALVEOLAR_NERVE", ["left", "right"]),
            ("METAL_EXTERNAL_RETENTION_WIRE", ["lower", "upper"]),
        ]
        for direction in directions
    ],
    *[
        CocoCategory(supercategory=supercategory, name=supercategory)
        for supercategory in [
            "LOWER_JAW",
            "UPPER_JAW",
            "METAL_EXTERNAL",
            "METAL_EXTERNAL_BRACES",
            "UNKNOWN_RADIOLUCENT",
            "UNKNOWN_RADIOPAQUE",
        ]
    ],
]


@dataclasses.dataclass
class DicomFile(object):
    path: Path
    root_path: Path

    image_name: str = dataclasses.field(init=False)

    @classmethod
    def list(cls, root_dir: Path, verbose: bool = False) -> Iterable["DicomFile"]:
        study_dirs: Iterable[Path] = Path(root_dir, "PROMATON").iterdir()
        if verbose:
            study_dirs = rich.progress.track(
                list(study_dirs), description="Iterating studies"
            )

        for study_dir in study_dirs:
            if not study_dir.is_dir():
                continue

            yield cls(
                path=Path(study_dir, "scan", f"{study_dir.stem}.dcm"),
                root_path=root_dir,
            )

    def __post_init__(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Dicom file not found: {self.path}")

        match self.path.relative_to(self.root_path).parts:
            case ("PROMATON", image_name, "scan", _):
                self.image_name = image_name

            case _:
                raise ValueError(f"Unexpected path: {self.path}")

    def to_coco_image(self, image_dir: Path, extension: str = "jpg") -> CocoImage:
        image_path: Path = Path(image_dir, "PROMATON", f"{self.image_name}.{extension}")
        if not image_path.exists():
            logging.debug(f"Writing image to {image_path}")

            ds: pydicom.Dataset
            with pydicom.dcmread(self.path) as ds:
                ds.HighBit = 0
                iio.imwrite(image_path, ds.pixel_array)

        image_meta: dict = iio.immeta(image_path)
        match image_meta:
            case {"shape": (width, height), **kwargs}:
                pass

            case _:
                raise ValueError(f"Unexpected image_meta: {image_meta}")

        return CocoImage(
            id=self.image_name,
            file_name=str(image_path.relative_to(image_dir)),
            width=width,
            height=height,
        )


@dataclasses.dataclass
class PSGFile(object):
    path: Path
    root_path: Path

    image_name: str = dataclasses.field(init=False)
    supercategory_name: str = dataclasses.field(init=False)
    category_name: str = dataclasses.field(init=False)

    @classmethod
    def list(cls, root_dir: Path, verbose: bool = False) -> Iterable["PSGFile"]:
        study_dirs: Iterable[Path] = Path(root_dir, "PROMATON").iterdir()
        if verbose:
            study_dirs = rich.progress.track(
                list(study_dirs), description="Iterating studies"
            )

        for study_dir in study_dirs:
            if not study_dir.is_dir():
                continue

            for psg_path in Path(study_dir, "output").glob("**/*.psg"):
                yield cls(path=psg_path, root_path=root_dir)

    def __post_init__(self):
        if not self.path.exists():
            raise FileNotFoundError(f"PSG file not found: {self.path}")

        match self.path.relative_to(self.root_path).parts:
            case ("PROMATON", image_name, "output", supercategory_name, file_name):
                self.image_name = image_name
                self.supercategory_name = supercategory_name
                self.category_name = Path(file_name).stem

            case _:
                raise ValueError(f"Unexpected path: {self.path}")

    def to_coco_category(self) -> CocoCategory:
        return CocoCategory(
            supercategory=self.supercategory_name, name=self.category_name
        )

    def to_coco_annotation(self) -> CocoAnnotation:
        # this logic is suboptimal as you can convert psg to rle directly without converting to npa first
        bitmask: np.ndarray
        bitmask, _, _ = psg_to_npa(self.path)

        mask: Mask = Mask.from_obj(bitmask)

        return CocoAnnotation(
            id=str(self.path.relative_to(self.root_path)),
            image_id=self.image_name,
            category_id=self.category_name,
            bbox=mask.bbox_xywh,
            segmentation=mask.rle,
            area=mask.area,
        )


def main(_):
    raw_dir: Path = Path(FLAGS.data_dir, "raw")

    category_by_name: dict[str, CocoCategory] = {
        category.name: category.copy(update={"id": index})
        for index, category in enumerate(COCO_CATEGORIES, start=1)
    }

    images: list[CocoImage] = []
    with contextlib.ExitStack() as stack:
        tasks: Iterable[Task] = (
            Task(
                fn=DicomFile.to_coco_image,
                args=(dicom,),
                kwargs={"image_dir": Path(FLAGS.data_dir, "images")},
            )
            for dicom in DicomFile.list(root_dir=raw_dir)
        )

        for image in map_task(tasks, stack=stack, num_workers=FLAGS.num_workers):
            if image is None:
                continue

            if not isinstance(image.id, str):
                continue

            images.append(image)

    image_by_name: dict[str, CocoImage] = {
        image.id: image.copy(update={"id": index})  # type: ignore
        for index, image in enumerate(images, start=1)
    }

    annotations: list[CocoAnnotation] = []
    with contextlib.ExitStack() as stack:
        tasks: Iterable[Task] = (
            Task(
                fn=PSGFile.to_coco_annotation,
                args=(psg,),
            )
            for psg in PSGFile.list(root_dir=raw_dir)
        )

        for annotation in map_task(tasks, stack=stack, num_workers=FLAGS.num_workers):
            if annotation is None:
                continue

            category: CocoCategory | None = category_by_name.get(annotation.category_id)  # type: ignore
            if category is None:
                logging.warning(
                    f"Category not found for name: {annotation.category_id}"
                )
                continue

            image: CocoImage | None = image_by_name.get(annotation.image_id)  # type: ignore
            if image is None:
                logging.warning(f"Image not found: {annotation.image_id}")
                continue

            annotations.append(
                annotation.copy(
                    update={"image_id": image.id, "category_id": category.id}
                )
            )

    annotations = [
        annoation.copy(update={"id": index})
        for index, annoation in enumerate(annotations, start=1)
    ]

    coco: Coco = Coco(
        categories=list(category_by_name.values()),
        images=list(image_by_name.values()),
        annotations=annotations,
    )
    coco_path: Path = Path(FLAGS.output_coco)

    logging.info(f"Writing COCO to {coco_path}")
    with open(coco_path, "w") as f:
        f.write(coco.json())


if __name__ == "__main__":
    app.run(main)
