from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pycocotools.mask
import torch
from pydantic import parse_obj_as

from app.instance_detection.schemas import (
    CocoAnnotation,
    CocoCategory,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
)
from detectron2.structures import BoxMode, Instances


def calculate_iom_bbox(bbox1: list[int], bbox2: list[int]) -> float:
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


def prediction_to_detectron2_instances(
    prediction: InstanceDetectionPrediction,
    image_size: tuple[int, int],
    category_ids: list[int] | None = None,
) -> Instances:
    scores: list[float] = []
    pred_boxes: list[list[int]] = []
    pred_classes: list[int] = []
    pred_masks: list[npt.NDArray[np.uint8]] = []

    for instance in prediction.instances:
        score: float = instance.score

        if (category_ids is not None) and (instance.category_id not in category_ids):
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


def prediction_to_coco_annotations(
    prediction: InstanceDetectionPrediction,
    coco_categories: list[CocoCategory],
    start_id: int = 0,
) -> list[CocoAnnotation]:
    instances: list[InstanceDetectionPredictionInstance] = prediction.instances

    coco_annotations: list[CocoAnnotation] = []
    id: int = start_id
    for instance in instances:
        category_id: int | None = coco_categories[instance.category_id].id
        assert category_id is not None

        rle_obj: dict[str, Any] = instance.segmentation.dict()
        segmentation: npt.NDArray[np.uint8] = pycocotools.mask.decode(rle_obj)
        contours: list[npt.NDArray[np.int32]]
        contours, _ = cv2.findContours(
            segmentation[..., None], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        coco_annotations.append(
            CocoAnnotation(
                id=id,
                image_id=instance.image_id,
                category_id=category_id,
                bbox=instance.bbox,
                segmentation=[contour.flatten().tolist() for contour in contours],
                area=pycocotools.mask.area(rle_obj),
                metadata={"score": instance.score},
            )
        )
        id += 1

    return coco_annotations


def load_predictions(prediction_path: Path) -> list[InstanceDetectionPrediction]:
    predictions_obj = torch.load(prediction_path)
    predictions = parse_obj_as(list[InstanceDetectionPrediction], predictions_obj)

    return predictions


def save_predictions(
    predictions: list[InstanceDetectionPrediction], prediction_path: Path
) -> None:
    predictions_obj: list[dict[str, Any]] = [
        prediction.dict() for prediction in predictions
    ]
    torch.save(predictions_obj, prediction_path)
