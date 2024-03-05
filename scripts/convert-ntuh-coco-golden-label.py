from pathlib import Path

from absl import app, flags

from app.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from app.coco.schemas import ID
from app.instance_detection import InstanceDetectionV1Category as Category

flags.DEFINE_string("coco", "./data/raw/NTUH/ntuh-opg-12.json", "Input COCO file path.")
flags.DEFINE_string(
    "output_coco",
    "./data/coco/instance-detection-v1-ntuh.json",
    "Output COCO file path.",
)
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
    with open(FLAGS.coco, "r") as f:
        coco: Coco = Coco.model_validate_json(f.read())

    raw_category_id_to_name: dict[ID, str] = {}
    for raw_category in coco.categories:
        category_name: str | None = CATEGORY_MAPPING.get(raw_category.name)

        if category_name is None:
            raise ValueError(f"Unknown category: {raw_category.name}")

        if raw_category.id is None:
            raise ValueError(f"Category id is None: {raw_category}")

        raw_category_id_to_name[raw_category.id] = category_name

    categories: list[CocoCategory] = [
        CocoCategory(name=name) for name in raw_category_id_to_name.values()
    ]

    images: list[CocoImage] = [
        image.model_copy(update={"file_name": f"NTUH/{Path(image.file_name).stem}.jpg"})
        for image in coco.images
    ]

    annotations: list[CocoAnnotation] = []
    for annotation in coco.annotations:
        annotations.append(
            annotation.model_copy(
                update={
                    "category_id": raw_category_id_to_name[annotation.category_id],
                    "metadata": None,
                }
            )
        )

        match annotation.metadata:
            # root remnants are not labeled as category but as metadata, so we need to add them manually
            case {"RR": "Y"}:
                annotations.append(
                    annotation.model_copy(
                        update={
                            "category_id": Category.ROOT_REMNANTS,
                            "metadata": None,
                        }
                    )
                )

    coco: Coco = Coco.create(
        categories=categories,
        images=images,
        annotations=annotations,
        sort_category=True,
    )
    with open(FLAGS.output_coco, "w") as f:
        f.write(coco.model_dump_json())


if __name__ == "__main__":
    app.run(main)
