import re
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
)
from app.utils import (
    instance_detection_data_to_prediction,
    load_predictions,
    prediction_to_detectron2_instances,
)
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import VisImage, Visualizer

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("visualize_dir", "visualize", "Visualizer directory.")
flags.DEFINE_string(
    "prediction_name", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_bool(
    "use_gt_as_prediction",
    False,
    "Set to true to perform command on ground truth. Useful when we do not have ground truth "
    "finding summary but only ground truth segmentation.",
)
flags.DEFINE_bool(
    "visualize_subset",
    False,
    "Set to true to visualize tooth-only, m3-only, and findings-only objects.",
)
FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.INFO)

    _: InstanceDetection = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
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
            instance_detection_data_to_prediction(instance_detection_data=data)
            for data in dataset
        ]
    else:
        predictions = load_predictions(
            prediction_path=Path(FLAGS.result_dir, FLAGS.prediction_name)
        )

    id_to_prediction: dict[str | int, InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    visualize_dir: Path = Path(FLAGS.result_dir, FLAGS.visualize_dir)
    Path(visualize_dir).mkdir(parents=True, exist_ok=True)

    for data in dataset:
        if data.image_id not in id_to_prediction:
            logging.warning(f"Image id {data.image_id} not found in predictions.")
            continue

        logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

        prediction: InstanceDetectionPrediction = id_to_prediction[data.image_id]

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

            instances: Instances = prediction_to_detectron2_instances(
                prediction,
                height=data.height,
                width=data.width,
                category_ids=category_ids,
            )

            image: np.ndarray = iio.imread(data.file_name)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)

            image_rgb: np.ndarray
            if image.shape[2] == 1:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image_rgb = image
            else:
                raise NotImplementedError

            visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0)
            image_vis: VisImage = visualizer.draw_instance_predictions(instances)

            logging.info(f"Saving to {image_path}.")
            image_vis.save(image_path)


if __name__ == "__main__":
    app.run(main)
