from pathlib import Path
from typing import Any

import numpy as np
import ultralytics.data.converter
import yaml
from absl import app, flags, logging
from pydantic import parse_obj_as

from app import utils
from app.instance_detection.datasets import InstanceDetectionV1
from app.instance_detection.schemas import InstanceDetectionData
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("yolo_dir", "yolo", "Yolo directory (relative to `data_dir`.")

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.dataset_name == "pano_all":
        data_driver = InstanceDetectionV1.register(root_dir=FLAGS.data_dir)
    else:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_name}")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    #

    yolo_dir: Path = Path(FLAGS.data_dir, FLAGS.yolo_dir, "PROMATON")

    for split in ["train", "eval"]:
        names_path: Path = Path(yolo_dir, f"{split}.txt")
        names_path.write_text(
            "\n".join(
                [
                    str(Path(FLAGS.data_dir, "images", f"{file_name}.jpg"))
                    for file_name in data_driver.get_split_file_names(split)
                ]
            )
        )

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

    #

    label_dir: Path = Path(FLAGS.data_dir, "labels", "PROMATON")
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


if __name__ == "__main__":
    app.run(main)
