import dataclasses
from collections.abc import Iterable
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pipe
import pydicom
from absl import app, flags, logging

from app.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from app.lib.psg import psg_to_npa
from app.masks import Mask
from app.tasks import Pool, filter_none, track_progress

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
    def iterdir(cls, root_dir: Path) -> Iterable["DicomFile"]:
        return (
            list(Path(root_dir, "PROMATON").iterdir())
            | track_progress(description="Dicom Files")
            | pipe.filter(Path.is_dir)
            | pipe.map(
                lambda study_dir: cls(
                    path=Path(study_dir, "scan", f"{study_dir.stem}.dcm"),
                    root_path=root_dir,
                )
            )
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
            case {"shape": (int() as width, int() as height)}:
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
    def iterdir(cls, root_dir: Path) -> Iterable["PSGFile"]:
        return (
            list(Path(root_dir, "PROMATON").iterdir())
            | track_progress(description="PSG Files")
            | pipe.filter(Path.is_dir)
            | pipe.map(
                lambda study_dir: (
                    cls(path=psg_path, root_path=root_dir)
                    for psg_path in Path(study_dir, "output").glob("**/*.psg")
                )
            )
            | pipe.chain
        )

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

    images: list[CocoImage] = []
    annotations: list[CocoAnnotation] = []
    with Pool(num_workers=FLAGS.num_workers) as pool:
        images = list(
            DicomFile.iterdir(root_dir=raw_dir)
            | pool.parallel_pipe(DicomFile.to_coco_image, allow_unordered=True)(
                image_dir=Path(FLAGS.data_dir, "images")
            )
            | filter_none
        )

        annotations = list(
            PSGFile.iterdir(root_dir=raw_dir)
            | pool.parallel_pipe(PSGFile.to_coco_annotation, allow_unordered=True)
            | filter_none
        )

    coco: Coco = Coco.create(
        categories=COCO_CATEGORIES, images=images, annotations=annotations
    )
    coco_path: Path = Path(FLAGS.output_coco)

    logging.info(f"Writing COCO to {coco_path}")
    with open(coco_path, "w") as f:
        f.write(coco.model_dump_json())


if __name__ == "__main__":
    app.run(main)
