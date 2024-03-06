from pathlib import Path
from typing import Any

import numpy as np
import yaml
from absl import app, flags, logging

from app.instance_detection import (
    InstanceDetection,
    InstanceDetectionData,
    InstanceDetectionFactory,
)
from app.masks import Mask
from app.tasks import Pool, track_progress
from detectron2.data import Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_enum("dataset_prefix", "pano", ["pano", "pano_ntuh"], "Dataset prefix.")
flags.DEFINE_string("yolo_dir", "yolo", "Yolo directory (relative to `data_dir`.")
flags.DEFINE_boolean("force", False, "Overwrite existing files.")
flags.DEFINE_integer("num_workers", 0, "Number of processes to use.")
FLAGS = flags.FLAGS


def process_data(
    data: InstanceDetectionData,
    *,
    label_dir: Path,
) -> None:
    label_path: Path = Path(label_dir, f"{data.file_name.stem}.txt")
    if (not FLAGS.force) and label_path.exists():
        logging.info(f"Skip writing to {label_path!s}...")
        return

    lines: list[str] = []
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

        lines.append(f"{annotation.category_id} {line}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def main(_):
    directory_name: str
    match FLAGS.dataset_prefix:
        case "pano":
            directory_name = "PROMATON"

        case "pano_ntuh":
            directory_name = "NTUH"

        case _:
            raise ValueError(f"Unknown dataset name: {FLAGS.dataset_prefix}")

    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_prefix, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = data_driver.get_coco_dataset(
        dataset_name=FLAGS.dataset_prefix
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_prefix)

    # Write names.txt

    yolo_dir: Path = Path(FLAGS.data_dir, FLAGS.yolo_dir)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    for split in data_driver.SPLITS:
        dataset_name: str = data_driver.get_dataset_name(split=split)
        names_path: Path = Path(yolo_dir, f"{data_driver.PREFIX}_{split}.txt")

        logging.info(f"Writing to {names_path!s}...")
        names_path.write_text(
            "\n".join(
                str(Path(FLAGS.data_dir, "images", f"{file_name}.jpg"))
                for file_name in data_driver.get_file_names(dataset_name=dataset_name)
            )
        )

    # Write labels

    label_dir: Path = Path(FLAGS.data_dir, "labels", directory_name)
    label_dir.mkdir(parents=True, exist_ok=True)

    with Pool(num_workers=FLAGS.num_workers) as pool:
        list(
            dataset
            | track_progress
            | pool.parallel_pipe(process_data, allow_unordered=True)(
                label_dir=label_dir
            )
        )

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
