from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import parse_file_as

from app.schemas import ID
from app.semantic_segmentation.schemas import (
    SemanticSegmentationPrediction,
    SemanticSegmentationPredictionInstance,
)
from app.utils import Mask
from detectron2.structures import BoxMode, Instances


def parse_predictions_json(
    path: Path, file_name_to_image_id: dict[Any, ID]
) -> list[SemanticSegmentationPrediction]:
    instances: list[SemanticSegmentationPredictionInstance] = parse_file_as(
        list[SemanticSegmentationPredictionInstance], path
    )

    file_name: Path | None = None
    image_id: ID | None = None
    _instances: list[SemanticSegmentationPredictionInstance] = []

    predictions: list[SemanticSegmentationPrediction] = []
    for instance in instances:
        if instance.file_name != file_name:
            image_id = file_name_to_image_id.get(instance.file_name, None)

            if file_name is not None:
                predictions.append(
                    SemanticSegmentationPrediction(
                        image_id=image_id, file_name=file_name, instances=_instances
                    )
                )

            file_name = instance.file_name
            _instances = []

        _instances.append(instance.copy(update={"image_id": image_id}))

    if file_name is not None:
        predictions.append(
            SemanticSegmentationPrediction(
                image_id=image_id, file_name=file_name, instances=_instances
            )
        )

    return predictions


def prediction_to_detectron2_instances(
    prediction: SemanticSegmentationPrediction,
    height: int,
    width: int,
    category_ids: list[int] | None = None,
    background_id: int = 0,
) -> Instances:
    scores: list[float] = []
    pred_boxes: list[list[int]] = []
    pred_classes: list[int] = []
    pred_masks: list[npt.NDArray[np.uint8]] = []

    for instance in prediction.instances:
        category_id: int = int(instance.category_id)

        if (category_ids is not None) and (category_id not in category_ids):
            continue

        if category_id == background_id:
            continue

        score: float = 1.0
        mask: Mask = Mask.from_obj(
            instance.segmentation,
            height=height,
            width=width,
        )

        scores.append(score)
        pred_boxes.append(
            BoxMode.convert(mask.bbox_xywh, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        )
        pred_classes.append(category_id)
        pred_masks.append(mask.bitmask)

    return Instances(
        image_size=(height, width),
        scores=np.array(scores),
        pred_boxes=np.array(pred_boxes),
        pred_classes=np.array(pred_classes),
        pred_masks=np.array(pred_masks),
    )
