from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pycocotools.mask
import pyomo.environ as pyo
import torch
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetectionV1
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
)
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
FLAGS = flags.FLAGS


def calculate_iou_bbox(bbox1: List[int], bbox2: List[int]) -> float:
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2

    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2

    xA = max(x1_1, x1_2)
    yA = max(y1_1, y1_2)
    xB = min(x2_1, x2_2)
    yB = min(y2_1, y2_2)

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    area_1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    return intersection_area / float(area_1 + area_2 - intersection_area)


def calculate_iou_mask(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    return np.sum(intersection) / np.sum(union)


def do_assignment(
    iou: np.ndarray, score: np.ndarray, epsilon: float = 1e-3
) -> np.ndarray:
    (num_instances, num_categories) = score.shape
    assert iou.shape == (num_instances, num_instances)

    p: Dict[Tuple[int, int], float] = {
        (n, c): score[n, c] for n in range(num_instances) for c in range(num_categories)
    }
    q: Dict[Tuple[int, int], float] = {
        (n1, n2): iou[n1, n2]
        for n1 in range(num_instances)
        for n2 in range(num_instances)
    }

    model: pyo.ConcreteModel = pyo.ConcreteModel("QuadraticAssignment")
    model.N = pyo.RangeSet(0, num_instances - 1)
    model.C = pyo.RangeSet(0, num_categories - 1)
    model.P = pyo.Param(model.N, model.C, initialize=p)
    model.Q = pyo.Param(model.N, model.N, initialize=q)

    model.x = pyo.Var(model.N, model.C, domain=pyo.Binary)

    reward = sum(model.P[n, c] * model.x[n, c] for n in model.N for c in model.C)

    penalty = sum(
        model.Q[n1, n2]
        * sum(model.x[n1, c1] for c1 in model.C)
        * sum(model.x[n2, c2] for c2 in model.C)
        if n1 != n2
        else 0
        for n1 in model.N
        for n2 in model.N
    )
    model.obj = pyo.Objective(expr=reward - penalty, sense=pyo.maximize)

    model.single_assignment = pyo.ConstraintList()
    for n in model.N:
        model.single_assignment.add(sum(model.x[n, c] for c in model.C) <= 1)
    for c in model.C:
        model.single_assignment.add(sum(model.x[n, c] for n in model.N) <= 1)

    solver = pyo.SolverFactory("ipopt")
    solver.solve(model)

    assignment = np.zeros((num_instances, num_categories), dtype=np.bool_)
    for n in model.N:
        for c in model.C:
            value: float = model.x[n, c].value

            if abs(value - 1.0) < epsilon:
                assignment[n, c] = True
            elif abs(value - 0.0) < epsilon:
                assignment[n, c] = False
            else:
                raise ValueError(f"Invalid value: {value}")

    return assignment


def postprocess(
    input_prediction_name: str = "instances_predictions.pth",
    output_prediction_name: str = "instances_predictions.postprocessed.pth",
) -> None:
    data_driver = InstanceDetectionV1.register(root_dir=FLAGS.data_dir)
    dataset: List[InstanceDetectionData] = parse_obj_as(
        List[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)
    num_categories: int = len(metadata.thing_classes)
    is_teeth_class: List[bool] = [
        category.startswith("TOOTH") for category in metadata.thing_classes
    ]

    input_prediction_path: Path = Path(FLAGS.result_dir, input_prediction_name)
    predictions_obj = torch.load(input_prediction_path)
    predictions = parse_obj_as(List[InstanceDetectionPrediction], predictions_obj)
    id_to_prediction: Dict[Union[str, int], InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    for data in dataset:
        if data.image_id not in id_to_prediction:
            logging.warning(f"Image id {data.image_id} not found in predictions.")
            continue

        logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

        prediction: InstanceDetectionPrediction = id_to_prediction[data.image_id]
        instances: List[InstanceDetectionPredictionInstance] = prediction.instances
        num_instances: int = len(instances)

        masks: List[np.ndarray] = []
        for instance in instances:
            mask = pycocotools.mask.decode(instance.segmentation.dict())
            masks.append(mask)

        iou: np.ndarray = np.eye(num_instances)
        for i in range(num_instances):
            for j in range(num_instances):
                if i >= j:
                    continue

                iou_bbox = calculate_iou_bbox(instances[i].bbox, instances[j].bbox)
                if iou_bbox == 0:
                    continue

                iou_mask = calculate_iou_mask(masks[i], masks[j])
                logging.debug(f"IoU of instances {i} and {j} is {iou_mask}.")

                iou[i, j] = iou[j, i] = iou_mask

        score: np.ndarray = np.zeros((num_instances, num_categories))
        for i in range(num_instances):
            score[i, instances[i].category_id] = instances[i].score

        assignment: np.ndarray = do_assignment(iou, score[:, is_teeth_class])
        instance_assignment: np.ndarray = np.any(assignment, axis=1)

        assigned_instances: List[InstanceDetectionPredictionInstance] = []
        for i in range(num_instances):
            # the instance is either not a tooth or not assigned to any tooth
            if is_teeth_class[instances[i].category_id] and not instance_assignment[i]:
                continue

            assigned_instances.append(instances[i])

        prediction.instances = assigned_instances

    output_prediction_path: Path = Path(FLAGS.result_dir, output_prediction_name)
    torch.save(
        [prediction.dict() for prediction in predictions], output_prediction_path
    )


def main(_):
    logging.set_verbosity(logging.INFO)
    postprocess()


if __name__ == "__main__":
    app.run(main)
