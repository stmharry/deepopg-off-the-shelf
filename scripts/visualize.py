import contextlib
import re
from pathlib import Path

import numpy as np
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionList,
)
from app.tasks import map_fn
from app.utils import read_image
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import VisImage, Visualizer

flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_enum(
    "dataset_name", "pano", InstanceDetection.available_dataset_names(), "Dataset name."
)
flags.DEFINE_string("visualize_dir", "visualize", "Visualizer directory.")
flags.DEFINE_string(
    "prediction", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_bool(
    "use_gt_as_prediction",
    False,
    "Set to true to perform command on ground truth. Useful when we do not have ground"
    " truth finding summary but only ground truth segmentation.",
)
flags.DEFINE_bool(
    "visualize_subset",
    False,
    "Set to true to visualize tooth-only, m3-only, and findings-only objects.",
)
flags.DEFINE_float("min_score", 0.0, "Minimum score to visualize.")
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
FLAGS = flags.FLAGS


def visualize_data(
    data: InstanceDetectionData,
    prediction: InstanceDetectionPrediction,
    metadata: Metadata,
    category_re_groups: dict[str, str],
    visualize_dir: Path,
) -> None:
    for group_name, re_pattern in category_re_groups.items():
        image_path: Path
        if group_name == "all":
            image_path = Path(
                visualize_dir, f"{data.file_name.stem}{data.file_name.suffix}"
            )
        else:
            image_path = Path(
                visualize_dir,
                f"{data.file_name.stem}_{group_name}{data.file_name.suffix}",
            )

        if image_path.exists():
            logging.info(f"Skipping {data.image_id} as it already exists.")
            continue

        category_ids: list[int] = [
            category_id
            for (category_id, category) in enumerate(metadata.thing_classes)
            if re.match(re_pattern, category)
        ]

        instances: Instances = prediction.to_detectron2_instances(
            height=data.height,
            width=data.width,
            category_ids=category_ids,
            min_score=FLAGS.min_score,
        )

        image_rgb: np.ndarray = read_image(data.file_name)
        visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0)
        image_vis: VisImage = visualizer.draw_instance_predictions(instances)

        logging.info(f"Saving to {image_path}.")
        image_vis.save(image_path)


def main(_):
    data_driver: InstanceDetection | None = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    if data_driver is None:
        raise ValueError(f"Dataset {FLAGS.dataset_name} not found.")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    category_re_groups: dict[str, str]
    if FLAGS.visualize_subset:
        category_re_groups = {
            "all": ".*",
            "tooth": r"TOOTH_\d+",
            "m3": r"TOOTH_(18|28|38|48)",
            "findings": r"(?!TOOTH_\d+)",
        }
    else:
        category_re_groups = {"all": ".*"}

    predictions: list[InstanceDetectionPrediction]
    if FLAGS.use_gt_as_prediction:
        predictions = [
            InstanceDetectionPrediction.from_instance_detection_data(data)
            for data in dataset
        ]
    else:
        predictions = InstanceDetectionPredictionList.from_detectron2_detection_pth(
            Path(FLAGS.result_dir, FLAGS.prediction)
        )

    id_to_prediction: dict[str | int, InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    visualize_dir: Path = Path(FLAGS.result_dir, FLAGS.visualize_dir)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        tasks: list[tuple] = [
            (
                data,
                id_to_prediction[data.image_id],
                metadata,
                category_re_groups,
                visualize_dir,
            )
            for data in dataset
            if data.image_id in id_to_prediction
        ]

        for _ in map_fn(
            visualize_data, tasks=tasks, stack=stack, num_workers=FLAGS.num_workers
        ):
            ...


if __name__ == "__main__":
    app.run(main)
