import json
from pathlib import Path

from absl import app, flags, logging

from app.coco import ID, Coco, CocoAnnotation, CocoCategory, CocoImage
from app.finding_summary import FindingLabel, FindingLabelList
from app.instance_detection import InstanceDetection, InstanceDetectionFactory

flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_enum(
    "dataset_prefix",
    "pano",
    InstanceDetectionFactory.available_dataset_prefixes(),
    "Dataset prefix.",
)
flags.DEFINE_string("coco", "./data/coco/promaton.json", "Input coco json file.")
flags.DEFINE_string("output_golden_label", None, "Output golden label csv file path.")
FLAGS = flags.FLAGS


def convert_to_coco(
    coco: Coco,
    category_id_to_mapped: dict[int, dict[str, str]],
    coco_path: Path,
) -> None:
    category_names: set[str] = set(
        mapped["category"] for mapped in category_id_to_mapped.values()
    )
    categories: list[CocoCategory] = [
        CocoCategory(name=name) for name in category_names
    ]

    annotations: list[CocoAnnotation] = []
    for annotation in coco.annotations:
        if annotation.category_id not in category_id_to_mapped:
            continue

        mapped: dict[str, str] = category_id_to_mapped[annotation.category_id]
        category_name: str = mapped["category"]
        annotations.append(annotation.model_copy(update={"category_id": category_name}))

    coco = Coco.create(
        categories=categories,
        images=coco.images,
        annotations=annotations,
        sort_category=True,
    )

    logging.info(f"Saving to {coco_path}")
    with open(coco_path, "w") as f:
        f.write(coco.model_dump_json())


def convert_to_finding_labels(
    coco: Coco,
    category_id_to_mapped: dict[int, dict[str, str]],
) -> None:
    id_to_image: dict[ID, CocoImage] = {image.id: image for image in coco.images}

    finding_labels: list[FindingLabel] = []
    for annotation in coco.annotations:
        if annotation.image_id not in id_to_image:
            continue

        if annotation.category_id not in category_id_to_mapped:
            continue

        image: CocoImage = id_to_image[annotation.image_id]
        mapped: dict[str, str] = category_id_to_mapped[annotation.category_id]

        finding_labels.append(
            FindingLabel(
                file_name=Path(image.file_name).stem,
                fdi=int(mapped["fdi"]),
                finding=mapped["category"],
            )
        )

    FindingLabelList.model_validate(finding_labels).to_csv(FLAGS.output_golden_label)


def main(_):
    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_prefix,
        root_dir=FLAGS.data_dir,
        into_detectron2_catalog=False,
    )
    if data_driver.CATEGORY_NAME_TO_MAPPINGS is None:
        logging.error(
            f"InstanceDetection {data_driver.__class__.__name__} does not have"
            " CATEGORY_NAME_TO_MAPPINGS!"
        )
        return

    if data_driver.coco_path is None:
        logging.error(
            f"InstanceDetection {data_driver.__class__.__name__} does not have a"
            " coco_path!"
        )
        return

    with open(FLAGS.coco, "r") as f:
        json_obj = json.load(f)

    coco: Coco = Coco.model_validate(json_obj)
    category_id_to_mapped: dict[int, dict[str, str]] = data_driver.map_categories(
        categories=coco.categories
    )

    convert_to_coco(
        coco,
        category_id_to_mapped=category_id_to_mapped,
        coco_path=Path(data_driver.coco_path),
    )

    if FLAGS.output_golden_label is not None:
        convert_to_finding_labels(
            coco,
            category_id_to_mapped=category_id_to_mapped,
        )


if __name__ == "__main__":
    app.run(main)
