from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import imageio.v3 as iio
import numpy as np
import pycocotools.mask
import torch
from absl import app, flags, logging
from detectron2.data import Metadata, MetadataCatalog
from detectron2.structures import BoxMode, Instances
from detectron2.utils.visualizer import VisImage, Visualizer
from pydantic import parse_obj_as

from app.data import InstanceDetectionV1
from app.schemas import InstanceDetectionPrediction

flags.DEFINE_string(
    "pred_path", "./output/inference/instances_predictions.pth", "Prediction file path."
)
flags.DEFINE_string("data_dir", "/mnt/hdd/PANO/data", "Data directory.")
flags.DEFINE_string(
    "output_dir", "/mnt/hdd/PANO/data/debug/instances", "Output directory."
)
FLAGS = flags.FLAGS


def as_detectron2_instances(
    prediction: InstanceDetectionPrediction,
    image_size: Tuple[int, int],
    min_score: float,
) -> Instances:
    scores = np.stack([instance.score for instance in prediction.instances])
    pred_boxes = np.stack([instance.bbox for instance in prediction.instances])
    pred_boxes = BoxMode.convert(pred_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    pred_classes = np.stack([instance.category_id for instance in prediction.instances])
    pred_masks = np.stack(
        [
            pycocotools.mask.decode(instance.segmentation.dict())
            for instance in prediction.instances
        ]
    )
    selected = scores > min_score

    return Instances(
        image_size=image_size,
        scores=scores[selected],
        pred_boxes=pred_boxes[selected],
        pred_classes=pred_classes[selected],
        pred_masks=pred_masks[selected],
    )


def main(_):
    dataset = InstanceDetectionV1(root_dir=FLAGS.data_dir)
    dataset.load()

    eval_dataset = dataset.get_split("eval")
    metadata: Metadata = MetadataCatalog.get("pano_eval")

    predictions_obj = torch.load(FLAGS.pred_path)
    predictions = parse_obj_as(List[InstanceDetectionPrediction], predictions_obj)
    id_to_prediction: Dict[Union[str, int], InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    Path(FLAGS.output_dir).mkdir(exist_ok=True)

    for data in eval_dataset:
        logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

        prediction: InstanceDetectionPrediction = id_to_prediction[data.image_id]
        instances: Instances = as_detectron2_instances(
            prediction, image_size=(data.height, data.width), min_score=0.1
        )

        image_bw: np.ndarray = iio.imread(data.file_name)
        image_rgb: np.ndarray = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)

        visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0)
        image_vis: VisImage = visualizer.draw_instance_predictions(instances)
        image_vis.save(Path(FLAGS.output_dir, data.file_name.name))


if __name__ == "__main__":
    app.run(main)
