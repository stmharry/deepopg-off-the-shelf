from pathlib import Path
from typing import Any

import numpy as np
import ultralytics.data.converter
import yaml
from absl import app, flags, logging
from pydantic import parse_obj_as

from app import utils
from app.instance_detection.datasets import (
    InstanceDetection,
    InstanceDetectionV1,
    InstanceDetectionV1NTUH,
)
from app.instance_detection.schemas import InstanceDetectionData
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("yolo_dir", "yolo", "Yolo directory (relative to `data_dir`.")

FLAGS = flags.FLAGS


def main(_):
    directory_name: str
    data_driver: InstanceDetection
    if FLAGS.dataset_name == "pano_all":
        data_driver = InstanceDetectionV1.register(root_dir=FLAGS.data_dir)
        directory_name = "PROMATON"
        splits = ["train", "eval"]

    elif FLAGS.dataset_name == "pano_ntuh":
        data_driver = InstanceDetectionV1NTUH.register(root_dir=FLAGS.data_dir)
        directory_name = "NTUH"
        splits = ["ntuh"]

    else:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_name}")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    # Write names.txt

    yolo_dir: Path = Path(FLAGS.data_dir, FLAGS.yolo_dir, directory_name)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        names_path: Path = Path(yolo_dir, f"{split}.txt")
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
        logging.info(f"Converting {data.file_name.stem}...")

        label_path: Path = Path(label_dir, data.file_name.stem + ".txt")

        with open(label_path, "w") as f:
            for annotation in data.annotations:
                polygons, _ = utils.convert_to_polygon(annotation.segmentation)

                if len(polygons) == 0:
                    continue

                if len(polygons) == 1:
                    polygon = np.reshape(polygons[0], (-1, 2))

                else:
                    polygons = ultralytics.data.converter.merge_multi_segment(polygons)
                    polygon = np.concatenate(polygons, axis=0)

                polygon = (polygon / np.array([data.width, data.height])).flatten()
                line: str = " ".join(map("{:.4f}".format, polygon))

                f.write(f"{annotation.category_id} {line}\n")

    if FLAGS.dataset_name != "pano_all":
        logging.info(f"Skip writing metadata.yaml for dataset: {FLAGS.dataset_name}")
        return

    # Write metadata.yaml

    yolo_metadata: dict[str, Any] = {
        "path": FLAGS.data_dir,
        "train": str(Path(yolo_dir.relative_to(FLAGS.data_dir), "train.txt")),
        "val": str(Path(yolo_dir.relative_to(FLAGS.data_dir), "eval.txt")),
        "names": {
            index: category for index, category in enumerate(metadata.thing_classes)
        },
    }
    yaml_path: Path = Path(yolo_dir, "metadata.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yolo_metadata, f)


if __name__ == "__main__":
    app.run(main)
