from pathlib import Path
from typing import Any

from absl import app, flags

from app.instance_detection.types import InstanceDetectionV1Category as Category
from app.schemas import Coco, CocoAnnotation, CocoCategory, CocoImage

flags.DEFINE_string("coco", None, "Input COCO file.")
flags.DEFINE_string("output_coco", None, "Output COCO file.")
flags.DEFINE_string("prefix", "NTUH/", "Prefix for image file paths.")
FLAGS = flags.FLAGS

CATEGORY_MAPPING: dict[str, str] = {
    "Caries": Category.CARIES,
    "CrownBridge": Category.CROWN_BRIDGE,
    "Endo": Category.ENDO,
    "Restoration": Category.FILLING,
    "Implant": Category.IMPLANT,
    "ApicalLesion": Category.PERIAPICAL_RADIOLUCENT,
    "_": Category.ROOT_REMNANTS,  # not labeled as category but as metadata
    "UNS8": "TOOTH_11",
    "UNS7": "TOOTH_12",
    "UNS6": "TOOTH_13",
    "UNS5": "TOOTH_14",
    "UNS4": "TOOTH_15",
    "UNS3": "TOOTH_16",
    "UNS2": "TOOTH_17",
    "UNS1": "TOOTH_18",
    "UNS9": "TOOTH_21",
    "UNS10": "TOOTH_22",
    "UNS11": "TOOTH_23",
    "UNS12": "TOOTH_24",
    "UNS13": "TOOTH_25",
    "UNS14": "TOOTH_26",
    "UNS15": "TOOTH_27",
    "UNS16": "TOOTH_28",
    "UNS24": "TOOTH_31",
    "UNS23": "TOOTH_32",
    "UNS22": "TOOTH_33",
    "UNS21": "TOOTH_34",
    "UNS20": "TOOTH_35",
    "UNS19": "TOOTH_36",
    "UNS18": "TOOTH_37",
    "UNS17": "TOOTH_38",
    "UNS25": "TOOTH_41",
    "UNS26": "TOOTH_42",
    "UNS27": "TOOTH_43",
    "UNS28": "TOOTH_44",
    "UNS29": "TOOTH_45",
    "UNS30": "TOOTH_46",
    "UNS31": "TOOTH_47",
    "UNS32": "TOOTH_48",
}


def main(_):
    coco: Coco = Coco.parse_file(FLAGS.coco)

    categories: list[CocoCategory] = [
        CocoCategory(id=i, name=name)
        for i, name in enumerate(CATEGORY_MAPPING.values(), start=1)
    ]
    category_by_name: dict[str, CocoCategory] = {
        category.name: category for category in categories
    }
    category_id_mapping: dict[Any, Any] = {
        category.id: category_by_name[CATEGORY_MAPPING[category.name]].id
        for category in coco.categories
    }

    images: list[CocoImage] = []
    for image in coco.images:
        file_name: str = Path(image.file_name).stem

        _image: CocoImage = image.copy(
            update={
                "id": image.id,
                "file_name": f"{FLAGS.prefix}{file_name}.jpg",
            }
        )
        images.append(_image)

    annotations: list[CocoAnnotation] = []
    for annotation in coco.annotations:
        _annotation: CocoAnnotation

        _annotation = annotation.copy(
            update={
                "id": len(annotations) + 1,
                "category_id": category_id_mapping[annotation.category_id],
                "metadata": None,
            }
        )
        annotations.append(_annotation)

        # root remnants are not labeled as category but as metadata, so we need to add them manually
        metadata: dict[str, Any] = annotation.metadata or {}
        if metadata.get("RR") == "Y":
            _annotation = annotation.copy(
                update={
                    "id": len(annotations) + 1,
                    "category_id": category_by_name[Category.ROOT_REMNANTS].id,
                    "metadata": None,
                }
            )
            annotations.append(_annotation)

    coco: Coco = Coco(categories=categories, images=images, annotations=annotations)
    json_str: str = coco.json(indent=2)
    with open(FLAGS.output_coco, "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    app.run(main)
