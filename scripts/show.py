from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pycocotools.mask
import torch
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection import schemas
from app.instance_detection.datasets import InstanceDetectionV1
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.structures import BoxMode, Instances
from detectron2.utils.visualizer import VisImage, Visualizer

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_boolean("show_visualizer", False, "Visualize the results to images.")
FLAGS = flags.FLAGS


def as_detectron2_instances(
    prediction: schemas.InstanceDetectionPrediction,
    image_size: Tuple[int, int],
    min_score: float,
) -> Instances:
    scores: List[float] = []
    pred_boxes: List[List[int]] = []
    pred_classes: List[int] = []
    pred_masks: List[npt.NDArray[np.uint8]] = []

    for instance in prediction.instances:
        score: float = instance.score

        if score < min_score:
            continue

        scores.append(score)
        pred_boxes.append(
            BoxMode.convert(instance.bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        )
        pred_classes.append(instance.category_id)
        pred_masks.append(pycocotools.mask.decode(instance.segmentation.dict()))

    return Instances(
        image_size=image_size,
        scores=np.array(scores),
        pred_boxes=np.array(pred_boxes),
        pred_classes=np.array(pred_classes),
        pred_masks=np.array(pred_masks),
    )


def show_visualizer(
    dataset: List[schemas.InstanceDetectionData],
    metadata: Metadata,
    prediction_name: str = "instances_predictions.pth",
) -> None:
    prediction_path: Path = Path(FLAGS.result_dir, prediction_name)

    predictions_obj = torch.load(prediction_path)
    predictions = parse_obj_as(
        List[schemas.InstanceDetectionPrediction], predictions_obj
    )
    id_to_prediction: Dict[Union[str, int], schemas.InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    visualize_dir: Path = Path(FLAGS.result_dir, "visualize")
    Path(visualize_dir).mkdir(exist_ok=True)

    for data in dataset:
        visualize_path: Path = Path(visualize_dir, data.file_name.name)

        if visualize_path.exists():
            logging.info(f"Skipping {data.image_id} as it already exists.")
            continue

        if data.image_id not in id_to_prediction:
            logging.warning(f"Image id {data.image_id} not found in predictions.")
            continue

        logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

        prediction: schemas.InstanceDetectionPrediction = id_to_prediction[
            data.image_id
        ]
        instances: Instances = as_detectron2_instances(
            prediction, image_size=(data.height, data.width), min_score=0.05
        )

        image_bw: np.ndarray = iio.imread(data.file_name)
        image_rgb: np.ndarray = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)

        visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0)
        image_vis: VisImage = visualizer.draw_instance_predictions(instances)
        image_vis.save(visualize_path)


def main(_):
    InstanceDetectionV1.register(root_dir=FLAGS.data_dir)

    dataset: List[schemas.InstanceDetectionData] = parse_obj_as(
        List[schemas.InstanceDetectionData], DatasetCatalog.get("pano_eval")
    )
    metadata: Metadata = MetadataCatalog.get("pano_eval")

    if FLAGS.show_visualizer:
        show_visualizer(dataset=dataset, metadata=metadata)


if __name__ == "__main__":
    app.run(main)
