from typing import Any

from absl import app, flags

from app.instance_detection.schemas import Coco, CocoAnnotation, CocoCategory, CocoImage

flags.DEFINE_string("input", None, "Input COCO file.")
flags.DEFINE_string("output", None, "Output COCO file.")
FLAGS = flags.FLAGS


def main(_):
    coco = Coco.parse_file(FLAGS.input)

    category_name_mapping = {
        "Caries": "CARIES",
        "CrownBridge": "CROWN_BRIDGE",
        "Endo": "ENDO",
        "Restoration": "FILLING",
        "Implant": "IMPLANT",
        "ApicalLesion": "PERIAPICAL_RADIOLUCENT",
        "_": "ROOT_REMNANTS",  # not labeled as category but as metadata
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

    categories: list[CocoCategory] = [
        CocoCategory(id=i, name=name)
        for i, name in enumerate(category_name_mapping.values(), start=1)
    ]
    category_by_name = {category.name: category for category in categories}

    category_id_mapping: dict[Any, Any] = {
        category.id: category_by_name[category_name_mapping[category.name]].id
        for category in coco.categories
    }

    images: list[CocoImage] = []
    image_id_mapping: dict[Any, Any] = {}
    for image in coco.images:
        _image: CocoImage = image.copy(
            update={
                "id": len(images) + 1,
                "file_name": f"NTUH/{image.file_name}",
            }
        )
        images.append(_image)

        image_id_mapping[image.id] = _image.id

    annotations: list[CocoAnnotation] = []
    for annotation in coco.annotations:
        _annotation: CocoAnnotation

        _annotation = annotation.copy(
            update={
                "id": len(annotations) + 1,
                "image_id": image_id_mapping[annotation.image_id],
                "category_id": category_id_mapping[annotation.category_id],
                "metadata": None,
            }
        )
        annotations.append(_annotation)

        # root remnants are not labeled as category but as metadata, so we need to add them manually
        metadata: dict = annotation.metadata or {}
        if metadata.get("RR") == "Y":
            _annotation = annotation.copy(
                update={
                    "id": len(annotations) + 1,
                    "image_id": image_id_mapping[annotation.image_id],
                    "category_id": category_by_name["ROOT_REMNANTS"].id,
                    "metadata": None,
                }
            )
            annotations.append(_annotation)

    coco = Coco(categories=categories, images=images, annotations=annotations)
    json_str: str = coco.json(indent=2)
    with open(FLAGS.output, "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    app.run(main)
