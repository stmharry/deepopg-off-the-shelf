from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
flags.DEFINE_string(
    "input_prediction_name", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_string(
    "output_prediction_name",
    "instances_predictions.postprocessed.pth",
    "Output prediction file name.",
)
FLAGS = flags.FLAGS


def calculate_iom_bbox(bbox1: List[int], bbox2: List[int]) -> float:
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

    area_1: int = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_2: int = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    intersection_area: int = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return intersection_area / min(area_1, area_2)


def calculate_iom_mask(mask1: np.ndarray, mask2: np.ndarray) -> float:
    area_1: int = np.sum(mask1)
    area_2: int = np.sum(mask2)

    intersection: np.ndarray = np.logical_and(mask1, mask2)

    return np.sum(intersection) / min(area_1, area_2)


def do_teeth_assignment(
    iom: np.ndarray, score: np.ndarray, epsilon: float = 1e-3
) -> np.ndarray:
    (num_instances, num_categories) = score.shape
    assert iom.shape == (num_instances, num_instances)

    p: Dict[Tuple[int, int], float] = {
        (n, c): score[n, c] for n in range(num_instances) for c in range(num_categories)
    }
    q: Dict[Tuple[int, int], float] = {
        (n1, n2): iom[n1, n2]
        for n1 in range(num_instances)
        for n2 in range(num_instances)
    }

    model: pyo.ConcreteModel = pyo.ConcreteModel("TeethAssignment")
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
    has_invalid_value: bool = False
    for n in model.N:
        for c in model.C:
            value: float = model.x[n, c].value

            if abs(value - 1.0) < epsilon:
                assignment[n, c] = True
            elif abs(value - 0.0) < epsilon:
                assignment[n, c] = False
            else:
                has_invalid_value = True
                assignment[n, c] = False

    if has_invalid_value:
        logging.warning("Invalid value detected in assignment.")

    return assignment


def do_assignment(
    reward: np.ndarray,
    quadratic_penalty: np.ndarray,
    penalty_group_ids: Optional[List[Optional[int]]] = None,
    unique_group_ids: Optional[List[Optional[int]]] = None,
    assignment_penalty: float = 0.01,
    epsilon: float = 1e-3,
) -> np.ndarray:
    assignment: np.ndarray

    num: int = len(reward)
    if num == 0:
        assignment = np.zeros(0, dtype=np.bool_)
        return assignment

    if penalty_group_ids is None:
        penalty_group_ids = [None] * num

    if unique_group_ids is None:
        unique_group_ids = [None] * num

    assert quadratic_penalty.shape == (num, num)
    assert penalty_group_ids is not None
    assert unique_group_ids is not None

    p: Dict[int, float] = {n: reward[n] for n in range(num)}
    q: Dict[Tuple[int, int], float] = {
        (n1, n2): quadratic_penalty[n1, n2] for n1 in range(num) for n2 in range(num)
    }

    model: pyo.ConcreteModel = pyo.ConcreteModel("QuadraticAssignment")
    model.N = pyo.RangeSet(0, num - 1)
    model.P = pyo.Param(model.N, initialize=p)
    model.Q = pyo.Param(model.N, model.N, initialize=q)

    model.x = pyo.Var(model.N, domain=pyo.Binary)

    objectives: List[pyo.Expression] = []
    for n1 in model.N:
        objectives.append(model.P[n1] * model.x[n1])

        for n2 in model.N:
            # only deal with upper triangular matrix
            if n1 > n2:
                continue

            penalty_coefficient: float = 0.0
            if n1 == n2:
                penalty_coefficient = assignment_penalty
            elif (
                penalty_group_ids[n1] is not None
                and penalty_group_ids[n2] is not None
                and penalty_group_ids[n1] == penalty_group_ids[n2]
            ):
                penalty_coefficient = model.Q[n1, n2]

            if penalty_coefficient == 0.0:
                continue

            objectives.append(-penalty_coefficient * model.x[n1] * model.x[n2])

    constraints: List[pyo.Expression] = []
    if unique_group_ids is not None:
        for unique_constraint_id in list(set(unique_group_ids)):
            if unique_constraint_id is None:
                continue

            constraints.append(
                sum(
                    model.x[n]
                    for n in model.N
                    if unique_group_ids[n] == unique_constraint_id
                )
                <= 1
            )

    model.obj = pyo.Objective(expr=sum(objectives), sense=pyo.maximize)
    if len(constraints) > 0:
        model.constraints = pyo.ConstraintList()
        for constraint in constraints:
            model.constraints.add(constraint)

    solver = pyo.SolverFactory("ipopt")
    solver.solve(model)

    assignment = np.zeros(num, dtype=np.bool_)
    has_invalid_value: bool = False
    for n in model.N:
        value: float = model.x[n].value

        if abs(value - 1.0) < epsilon:
            assignment[n] = True
        elif abs(value - 0.0) < epsilon:
            assignment[n] = False
        else:
            has_invalid_value = True
            assignment[n] = False

    if has_invalid_value:
        logging.warning("Invalid value detected in assignment.")

    return assignment


def postprocess() -> None:
    data_driver = InstanceDetectionV1.register(root_dir=FLAGS.data_dir)
    dataset: List[InstanceDetectionData] = parse_obj_as(
        List[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    input_prediction_path: Path = Path(FLAGS.result_dir, FLAGS.input_prediction_name)
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

        iom: np.ndarray = np.eye(num_instances)
        for i in range(num_instances):
            for j in range(num_instances):
                if i >= j:
                    continue

                iom_bbox = calculate_iom_bbox(instances[i].bbox, instances[j].bbox)
                if iom_bbox == 0:
                    continue

                iom_mask = calculate_iom_mask(masks[i], masks[j])
                logging.debug(f"IoM of instances {i} and {j} is {iom_mask}.")

                iom[i, j] = iom[j, i] = iom_mask

        scores: List[float] = []
        penalty_group_ids: List[Optional[int]] = []
        unique_group_ids: List[Optional[int]] = []
        for instance in instances:
            category_name: str = metadata.thing_classes[instance.category_id]

            scores.append(instance.score)

            penalty_group_id: Optional[int] = None
            if category_name.startswith("TOOTH"):
                penalty_group_id = 1
            elif category_name == "ROOT_REMNANTS":
                penalty_group_id = 2
            else:
                penalty_group_id = 3
            penalty_group_ids.append(penalty_group_id)

            unique_constraint_id: Optional[int] = None
            if category_name.startswith("TOOTH"):
                unique_constraint_id = instance.category_id
            unique_group_ids.append(unique_constraint_id)

        assignment: np.ndarray = do_assignment(
            reward=np.array(scores),
            quadratic_penalty=iom,
            penalty_group_ids=penalty_group_ids,
            unique_group_ids=unique_group_ids,
        )
        logging.info(f"Removing {np.sum(~assignment)} instances.")

        prediction.instances = np.array(instances)[assignment].tolist()

    output_prediction_path: Path = Path(FLAGS.result_dir, FLAGS.output_prediction_name)
    torch.save(
        [prediction.dict() for prediction in predictions], output_prediction_path
    )


def main(_):
    logging.set_verbosity(logging.INFO)
    postprocess()


if __name__ == "__main__":
    app.run(main)
