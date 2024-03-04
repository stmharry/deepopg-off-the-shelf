from pathlib import Path
from typing import Any

import numpy as np
import yaml
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection import InstanceDetection, InstanceDetectionData
from app.masks import Mask
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_enum("dataset_prefix", "pano", ["pano", "pano_ntuh"], "Dataset prefix.")
flags.DEFINE_string("yolo_dir", "yolo", "Yolo directory (relative to `data_dir`.")

FLAGS = flags.FLAGS


def main(_):
    directory_name: str
    match FLAGS.dataset_prefix:
        case "pano":
            directory_name = "PROMATON"

        case "pano_ntuh":
            directory_name = "NTUH"

        case _:
            raise ValueError(f"Unknown dataset name: {FLAGS.dataset_prefix}")

    data_driver: InstanceDetection | None = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_prefix, root_dir=FLAGS.data_dir
    )
    if data_driver is None:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_prefix}")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_prefix)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_prefix)

    # Write names.txt

    yolo_dir: Path = Path(FLAGS.data_dir, FLAGS.yolo_dir)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    for split in data_driver.SPLITS:
        names_path: Path = Path(yolo_dir, f"{data_driver.PREFIX}_{split}.txt")

        logging.info(f"Writing to {names_path!s}...")
        names_path.write_text(
            "\n".join(
                [
                    str(Path(FLAGS.data_dir, "images", f"{file_name}.jpg"))
                    for file_name in data_driver.get_split_file_names(split)
                ]
            )
        )

    # Write labels

    label_dir: Path = Path(FLAGS.data_dir, "labels", directory_name)
    label_dir.mkdir(parents=True, exist_ok=True)

    for data in dataset:
        label_path: Path = Path(label_dir, data.file_name.stem + ".txt")

        logging.info(f"Converting '{data.file_name.stem}' into {label_path!s}...")

        with open(label_path, "w") as f:
            for annotation in data.annotations:
                mask: Mask = Mask.from_obj(
                    annotation.segmentation, width=data.width, height=data.height
                )
                polygon: list[int] | None = mask.polygon
                if polygon is None:
                    continue

                polygon_array: np.ndarray = np.array(polygon).reshape(-1, 2) / np.array(
                    [data.width, data.height]
                )
                line: str = " ".join(map("{:.4f}".format, polygon_array.flatten()))

                f.write(f"{annotation.category_id} {line}\n")

    if FLAGS.dataset_prefix != "pano":
        logging.info(f"Skip writing metadata.yaml for dataset: {FLAGS.dataset_prefix}")
        return

    # Write metadata.yaml

    yolo_metadata: dict[str, Any] = {
        "path": FLAGS.data_dir,
        "train": str(Path(yolo_dir.relative_to(FLAGS.data_dir), "pano_train.txt")),
        "val": str(Path(yolo_dir.relative_to(FLAGS.data_dir), "pano_eval.txt")),
        "names": {
            index: category for index, category in enumerate(metadata.thing_classes)
        },
    }
    yaml_path: Path = Path(yolo_dir, "metadata.yaml")

    logging.info(f"Writing metadata to {yaml_path!s}...")
    with open(yaml_path, "w") as f:
        yaml.dump(yolo_metadata, f)


if __name__ == "__main__":
    app.run(main)
