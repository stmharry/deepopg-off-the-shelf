from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, TypeAdapter

from app.coco import ID, CocoAnnotation, CocoCategory, CocoData, CocoRLE
from app.masks import Mask
from detectron2.structures import BoxMode, Instances


class InstanceDetectionAnnotation(BaseModel):
    bbox: list[int]
    bbox_mode: BoxMode
    category_id: int
    segmentation: CocoRLE | list[list[int]]
    iscrowd: int


class InstanceDetectionData(CocoData):
    annotations: list[InstanceDetectionAnnotation]


class InstanceDetectionPredictionInstance(BaseModel):
    image_id: ID
    bbox: list[int]
    category_id: int
    segmentation: CocoRLE | list[list[int]]
    score: float


class InstanceDetectionPrediction(BaseModel):
    image_id: ID
    instances: list[InstanceDetectionPredictionInstance]

    @classmethod
    def from_instance_detection_data(
        cls,
        instance_detection_data: InstanceDetectionData,
    ) -> "InstanceDetectionPrediction":
        return InstanceDetectionPrediction(
            image_id=instance_detection_data.image_id,
            instances=[
                InstanceDetectionPredictionInstance(
                    image_id=instance_detection_data.image_id,
                    bbox=annotation.bbox,
                    category_id=annotation.category_id,
                    segmentation=annotation.segmentation,
                    score=1.0,
                )
                for annotation in instance_detection_data.annotations
            ],
        )

    def to_coco_annotations(
        self,
        coco_categories: list[CocoCategory],
        start_id: int = 0,
    ) -> list[CocoAnnotation]:
        coco_annotations: list[CocoAnnotation] = []
        id: int = start_id
        for instance in self.instances:
            category_id: int | None = coco_categories[instance.category_id].id
            assert category_id is not None

            mask: Mask = Mask.from_obj(instance.segmentation)

            coco_annotations.append(
                CocoAnnotation(
                    id=id,
                    image_id=instance.image_id,
                    category_id=category_id,
                    bbox=instance.bbox,
                    segmentation=mask.polygons,
                    area=mask.area,
                    metadata={"score": instance.score},
                )
            )
            id += 1

        return coco_annotations

    def to_detectron2_instances(
        self,
        height: int,
        width: int,
        category_ids: list[int] | None = None,
        min_score: float = 0.0,
    ) -> Instances:
        scores: list[float] = []
        pred_boxes: list[list[int]] = []
        pred_classes: list[int] = []
        pred_masks: list[npt.NDArray[np.uint8]] = []

        for instance in self.instances:
            score: float = instance.score

            if score < min_score:
                continue

            if (category_ids is not None) and (
                instance.category_id not in category_ids
            ):
                continue

            scores.append(score)
            pred_boxes.append(
                BoxMode.convert(instance.bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            )
            pred_classes.append(instance.category_id)

            mask: Mask = Mask.from_obj(
                instance.segmentation,
                height=height,
                width=width,
            )

            pred_masks.append(mask.bitmask.astype(np.uint8))

        return Instances(
            image_size=(height, width),
            scores=np.array(scores),
            pred_boxes=np.array(pred_boxes),
            pred_classes=np.array(pred_classes),
            pred_masks=np.array(pred_masks),
        )


class InstanceDetectionPredictionList(object):
    @classmethod
    def from_detectron2_detection_pth(
        cls, path: Path, image_ids: set[ID] | None = None
    ) -> list[InstanceDetectionPrediction]:
        predictions_obj = torch.load(path)
        if image_ids is not None:
            predictions_obj = [
                prediction
                for prediction in predictions_obj
                if prediction["image_id"] in image_ids
            ]

        return TypeAdapter(list[InstanceDetectionPrediction]).validate_python(
            predictions_obj
        )

    @classmethod
    def to_detectron2_detection_pth(
        cls,
        instance_detection_predictions: list[InstanceDetectionPrediction],
        path: Path,
    ) -> None:
        predictions_obj: list[dict[str, Any]] = [
            instance_detection_prediction.model_dump()
            for instance_detection_prediction in instance_detection_predictions
        ]
        torch.save(predictions_obj, path)
