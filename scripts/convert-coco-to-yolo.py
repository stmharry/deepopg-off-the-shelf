from pathlib import Path
from typing import Any

import numpy as np
import yaml
from absl import app, flags, logging

from app.instance_detection import (
    InstanceDetection,
    InstanceDetectionData,
    InstanceDetectionFactory,
    InstanceDetectionV1,
    InstanceDetectionV1NTUH,
    InstanceDetectionV2,
)
from app.masks import Mask
from app.tasks import Pool, track_progress
from detectron2.data import Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_enum(
    "dataset_prefix",
    "pano",
    InstanceDetectionFactory.available_dataset_prefixes(),
    "Dataset prefix.",
)
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
    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_prefix, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = data_driver.get_coco_dataset(
        dataset_name=FLAGS.dataset_prefix
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_prefix)

    directory_name: str
    match data_driver:
        case InstanceDetectionV1() | InstanceDetectionV2():
            directory_name = "PROMATON"

        case InstanceDetectionV1NTUH():
            directory_name = "NTUH"

        case _:
            raise ValueError(f"Unknown dataset name: {FLAGS.dataset_prefix}")

    # See: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    # YOLO expects a directory with the following structure:
    #
    #   root_dir/
    #   ├── metadata.txt
    #   ├── images/
    #   │   ├── image1.jpg
    #   │   ├── image2.jpg
    #   │   └── ...
    #   └── labels/
    #       ├── label1.txt
    #       ├── label2.txt
    #       └── ...
    #
    # We will write the following files:
    #   - metadata.yaml
    #   - names.txt: representing the list of image paths
    #   - images: we use a symbolic link to the original images
    #   - labels: these are generated

    yolo_dir: Path = Path(FLAGS.data_dir, "yolo", FLAGS.dataset_prefix)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    # metadata.yaml

    yolo_metadata: dict[str, Any] = {
        "path": FLAGS.data_dir,
        "train": str(
            Path(
                yolo_dir,
                data_driver.get_dataset_name("train") + ".txt",
            ).relative_to(FLAGS.data_dir)
        ),
        "val": str(
            Path(
                yolo_dir,
                data_driver.get_dataset_name("eval") + ".txt",
            ).relative_to(FLAGS.data_dir)
        ),
        "names": {
            index: category for index, category in enumerate(metadata.thing_classes)
        },
    }
    yaml_path: Path = Path(yolo_dir, "metadata.yaml")

    logging.info(f"Writing metadata to {yaml_path!s}...")
    with open(yaml_path, "w") as f:
        yaml.dump(yolo_metadata, f)

    # images

    image_dir: Path = Path(yolo_dir, "images")
    image_dir.symlink_to(Path(FLAGS.data_dir, "images"))

    # names.txt

    for split in data_driver.SPLITS:
        dataset_name: str = data_driver.get_dataset_name(split=split)
        names_path: Path = Path(yolo_dir, f"{data_driver.PREFIX}_{split}.txt")

        logging.info(f"Writing to {names_path!s}...")
        names_path.write_text(
            "\n".join(
                str(Path(image_dir, f"{file_name}.jpg"))
                for file_name in data_driver.get_file_names(dataset_name=dataset_name)
            )
        )

    # labels

    label_dir: Path = Path(yolo_dir, "labels", directory_name)
    label_dir.mkdir(parents=True, exist_ok=True)

    with Pool(num_workers=FLAGS.num_workers) as pool:
        list(
            dataset
            | track_progress
            | pool.parallel_pipe(process_data, allow_unordered=True)(
                label_dir=label_dir
            )
        )


if __name__ == "__main__":
    app.run(main)
