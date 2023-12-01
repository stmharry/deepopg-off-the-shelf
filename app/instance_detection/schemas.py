from pathlib import Path

from pydantic import BaseModel

from app.instance_detection.types import InstanceDetectionV1Category
from app.schemas import ID, CocoRLE
from detectron2.structures import BoxMode


class InstanceDetectionAnnotation(BaseModel):
    bbox: list[int]
    bbox_mode: BoxMode
    category_id: int
    segmentation: CocoRLE | list[list[int]]
    iscrowd: int


class InstanceDetectionData(BaseModel):
    file_name: Path
    height: int
    width: int
    image_id: ID
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


class Finding(BaseModel):
    file_name: str
    fdi: int
    finding: InstanceDetectionV1Category
