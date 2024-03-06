from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pipe
from pydantic import BaseModel, RootModel, TypeAdapter

from app.coco.schemas import ID, CocoData, CocoRLE
from app.masks import Mask
from detectron2.structures import BoxMode, Instances


class SemanticSegmentationData(CocoData):
    sem_seg_file_name: Path | None = None


class SemanticSegmentationPredictionInstance(BaseModel):
    image_id: ID | None = None
    file_name: Path
    category_id: ID
    segmentation: CocoRLE


class SemanticSegmentationPrediction(BaseModel):
    image_id: ID | None = None
    file_name: Path
    instances: list[SemanticSegmentationPredictionInstance]

    def to_detectron2_instances(
        self,
        height: int,
        width: int,
        category_ids: list[int] | None = None,
        background_id: int = 0,
    ) -> Instances:
        scores: list[float] = []
        pred_boxes: list[list[int]] = []
        pred_classes: list[int] = []
        pred_masks: list[npt.NDArray[np.uint8]] = []

        for instance in self.instances:
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
            pred_masks.append(mask.bitmask.astype(np.uint8))

        return Instances(
            image_size=(height, width),
            scores=np.array(scores),
            pred_boxes=np.array(pred_boxes),
            pred_classes=np.array(pred_classes),
            pred_masks=np.array(pred_masks),
        )

    def to_semseg_mask(
        self,
        height: int,
        width: int,
        background_id: int = 0,
    ) -> np.ndarray:
        bitmasks: list[np.ndarray] = []
        category_ids: list[int] = []
        for instance in self.instances:
            category_id: int = int(instance.category_id)

            bitmask: np.ndarray
            if category_id == background_id:
                bitmask = np.zeros((height, width), dtype=np.uint8)

            else:
                mask: Mask = Mask.from_obj(
                    instance.segmentation,
                    height=height,
                    width=width,
                )
                bitmask = mask.bitmask

            bitmasks.append(bitmask)
            category_ids.append(category_id)

        semseg_mask: np.ndarray = np.asarray(category_ids, dtype=np.int32)[
            np.argmax(bitmasks, axis=0)
        ]

        return semseg_mask


class SemanticSegmentationPredictionList(
    RootModel[list[SemanticSegmentationPrediction]]
):
    @classmethod
    def from_detectron2_semseg_output_json(
        cls, path: Path, file_name_to_image_id: dict[Any, ID]
    ) -> "SemanticSegmentationPredictionList":
        with open(path, "r") as f:
            instances: list[SemanticSegmentationPredictionInstance] = TypeAdapter(
                list[SemanticSegmentationPredictionInstance]
            ).validate_json(f.read())

        predictions: list[SemanticSegmentationPrediction] = list(
            instances
            | pipe.groupby(lambda instance: instance.file_name)
            | pipe.filter(lambda kv: kv[0] in file_name_to_image_id)
            | pipe.map(
                lambda kv: SemanticSegmentationPrediction(
                    image_id=file_name_to_image_id[kv[0]],
                    file_name=kv[0],
                    instances=kv[1],
                )
            )
        )

        return cls.model_validate(predictions)
