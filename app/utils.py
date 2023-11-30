import dataclasses
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pycocotools.mask
import torch
import ultralytics.data.converter
from pydantic import parse_obj_as

from app.instance_detection.schemas import (
    CocoAnnotation,
    CocoCategory,
    CocoRLE,
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
)
from detectron2.structures import BoxMode, Instances, polygons_to_bitmask


def uns_to_fdi(uns: int) -> int:
    quadrant: int = (uns - 1) // 8 + 1
    index: int = (uns - 1) % 8 + 1 if quadrant % 2 == 0 else 9 - ((uns - 1) % 8 + 1)
    fdi: int = quadrant * 10 + index
    return fdi


def calculate_iom_bbox(
    bbox1: list[int], bbox2: list[int], epsilon1: float = 1e-3, epsilon2: float = 1e-6
) -> float:
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

    return (intersection_area + epsilon2) / (min(area_1, area_2) + epsilon1)


def calculate_iom_mask(
    mask1: np.ndarray, mask2: np.ndarray, epsilon1: float = 1e-3, epsilon2: float = 1e-6
) -> float:
    area_1: int = np.sum(mask1)
    area_2: int = np.sum(mask2)

    intersection: np.ndarray = np.logical_and(mask1, mask2)

    return (np.sum(intersection) + epsilon2) / (min(area_1, area_2) + epsilon1)


@dataclasses.dataclass
class Mask(object):
    _rle: CocoRLE | None = None
    _polygons: list[list[int]] | None = None
    _bitmask: npt.NDArray[np.uint8] | None = None

    _height: int | None = None
    _width: int | None = None

    def __post_init__(self) -> None:
        if all(
            [
                self._rle is None,
                self._polygons is None,
                self._bitmask is None,
            ]
        ):
            raise ValueError("Either rle, polygons or bitmask must be provided")

    @classmethod
    def from_obj(
        cls,
        obj: Any,
        height: int | None = None,
        width: int | None = None,
    ) -> "Mask":
        if isinstance(obj, CocoRLE):
            return cls(_rle=obj, _height=height, _width=width)

        elif isinstance(obj, dict):
            return cls(_rle=CocoRLE.parse_obj(obj), _height=height, _width=width)

        elif isinstance(obj, list):
            return cls(_polygons=obj, _height=height, _width=width)

        elif isinstance(obj, np.ndarray):
            return cls(_bitmask=obj, _height=height, _width=width)

        else:
            raise NotImplementedError

    # We have the following four conversions:
    #   1. rle -> bitmask
    #   2. bitmask -> rle
    #   3. polygons -> bitmask
    #   4. bitmask -> polygons

    @classmethod
    def convert_rle_to_bitmask(cls, rle: CocoRLE) -> npt.NDArray[np.uint8]:
        rle_obj: dict = rle.dict()
        bitmask: npt.NDArray[np.uint8] = pycocotools.mask.decode(rle_obj)
        return bitmask

    @classmethod
    def convert_bitmask_to_rle(cls, bitmask: npt.NDArray[np.uint8]) -> CocoRLE:
        rle_obj: dict = pycocotools.mask.encode(bitmask.ravel(order="F"))
        rle: CocoRLE = CocoRLE.parse_obj(rle_obj)
        return rle

    @classmethod
    def convert_polygons_to_bitmask(
        cls, polygons: list[list[int]], height: int, width: int
    ) -> npt.NDArray[np.uint8]:
        bitmask: npt.NDArray[np.uint8] = polygons_to_bitmask(polygons, height, width)
        return bitmask

    @classmethod
    def convert_bitmask_to_polygons(
        cls, bitmask: npt.NDArray[np.uint8]
    ) -> list[list[int]]:
        contours: list[npt.NDArray[np.int32]]
        contours, _ = cv2.findContours(
            bitmask[..., None], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons: list[list[int]] = [contour.flatten().tolist() for contour in contours]
        return polygons

    @property
    def rle(self) -> CocoRLE:
        if self._rle is not None:
            return self._rle

        # note this calls self.bitmask()
        bitmask: npt.NDArray[np.uint8] = self.bitmask
        return self.convert_bitmask_to_rle(bitmask)

    @property
    def polygon(self) -> list[int] | None:
        polygons: list[list[int]] = self.polygons

        if len(polygons) == 0:
            return None

        if len(polygons) == 1:
            return polygons[0]

        polygons = ultralytics.data.converter.merge_multi_segment(polygons)
        polygon = np.concatenate(polygons, axis=0)

        return list(polygon.flatten().tolist())

    @property
    def polygons(self) -> list[list[int]]:
        if self._polygons is not None:
            return self._polygons

        bitmask: npt.NDArray[np.uint8]
        if self._bitmask is not None:
            bitmask = self._bitmask

        else:
            assert self._rle is not None

            bitmask = self.convert_rle_to_bitmask(self._rle)

        return self.convert_bitmask_to_polygons(bitmask)

    def calculate_bitmask(
        self, allow_unspecified_shape: bool = False
    ) -> npt.NDArray[np.uint8]:
        if self._bitmask is not None:
            return self._bitmask

        if self._rle is not None:
            return self.convert_rle_to_bitmask(self._rle)

        assert self._polygons is not None

        height: int
        width: int
        if (self._height is None) or (self._width is None):
            if allow_unspecified_shape:
                height = np.array(self._polygons).reshape(-1, 2)[:, 1].max()
                width = np.array(self._polygons).reshape(-1, 2)[:, 0].max()

            else:
                raise ValueError("Height and width must be provided")
        else:
            height = self._height
            width = self._width

        return self.convert_polygons_to_bitmask(
            self._polygons, height=height, width=width
        )

    @property
    def bitmask(self) -> npt.NDArray[np.uint8]:
        return self.calculate_bitmask(allow_unspecified_shape=False)

    @property
    def area(self) -> int:
        # we don't actually need the extra padding suppose height and width are not provided
        bitmask: npt.NDArray[np.uint8] = self.calculate_bitmask(
            allow_unspecified_shape=True
        )
        return np.sum(bitmask)

    @property
    def bbox_xywh(self) -> list[int]:
        polygons: list[list[int]] = self.polygons

        x_min: int = min(
            [np.array(polygon).reshape(-1, 2)[:, 0].min() for polygon in polygons]
        )
        y_min: int = min(
            [np.array(polygon).reshape(-1, 2)[:, 1].min() for polygon in polygons]
        )
        x_max: int = max(
            [np.array(polygon).reshape(-1, 2)[:, 0].max() for polygon in polygons]
        )
        y_max: int = max(
            [np.array(polygon).reshape(-1, 2)[:, 1].max() for polygon in polygons]
        )

        return [
            x_min,
            y_min,
            x_max - x_min,
            y_max - y_min,
        ]


def prediction_to_detectron2_instances(
    prediction: InstanceDetectionPrediction,
    height: int,
    width: int,
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

        mask: Mask = Mask.from_obj(
            instance.segmentation,
            height=height,
            width=width,
        )

        pred_masks.append(mask.bitmask)

    return Instances(
        image_size=(height, width),
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


def instance_detection_data_to_prediction(
    instance_detection_data: InstanceDetectionData,
) -> InstanceDetectionPrediction:
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
