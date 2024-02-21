from pathlib import Path

import numpy as np
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
    InstanceDetectionPredictionList,
)
from app.masks import Mask
from detectron2.data import DatasetCatalog

flags.DEFINE_string("data_dir", None, "Data directory")
flags.DEFINE_string("result_dir", None, "Result directory")
flags.DEFINE_string("dataset_name", None, "Dataset name")
flags.DEFINE_string(
    "prediction", "instances_predictions.pth", "Input prediction file name."
)
FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.INFO)

    data_driver: InstanceDetection | None = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    if data_driver is None:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_name}")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )

    # Assemble predictions

    predictions: list[InstanceDetectionPrediction] = []
    for data in dataset:
        image_name: str = data.file_name.stem

        label_file_path: Path = Path(FLAGS.result_dir, "labels", f"{image_name}.txt")
        with open(label_file_path, "r") as f:
            lines = f.readlines()

        instances: list[InstanceDetectionPredictionInstance] = []
        for line in lines:
            items: list[str] = line.split()

            try:
                category_id: int = int(items[0])
                score: float = float(items[-1])
                polygon: np.ndarray = np.fromiter(
                    items[1:-1],  # type: ignore
                    dtype=np.float32,
                ).reshape(-1, 2) * np.array([data.width, data.height])

                if polygon.size == 0:
                    raise ValueError("Empty segmentation")

                mask: Mask = Mask.from_obj(
                    [polygon.flatten().tolist()],
                    height=data.height,
                    width=data.width,
                )

                bbox: list[int] = mask.bbox_xywh
                if bbox[0] >= data.width or bbox[1] >= data.height:
                    raise ValueError("bbox out of bound")

                if bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError("bbox has zero or negative size")

            except ValueError as e:
                logging.warning(
                    f"Error when parsing line from {label_file_path!s}: {line.strip()} due to '{e}', skipping..."
                )
                continue

            instance: InstanceDetectionPredictionInstance = (
                InstanceDetectionPredictionInstance(
                    image_id=data.image_id,
                    bbox=bbox,
                    category_id=category_id,
                    segmentation=mask.polygons,
                    score=score,
                )
            )
            instances.append(instance)

        prediction: InstanceDetectionPrediction = InstanceDetectionPrediction(
            image_id=data.image_id,
            instances=instances,
        )
        predictions.append(prediction)

    predictions_path: Path = Path(FLAGS.result_dir, FLAGS.prediction)
    logging.info(f"Saving predictions to {predictions_path}")

    InstanceDetectionPredictionList.to_detectron2_detection_pth(
        predictions, path=predictions_path
    )


if __name__ == "__main__":
    app.run(main)
