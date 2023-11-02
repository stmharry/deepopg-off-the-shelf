from pathlib import Path
from typing import Any

from pydantic import BaseModel

from detectron2.structures import BoxMode


class CocoRLE(BaseModel):
    size: list[int]
    counts: str


class CocoCategory(BaseModel):
    id: int | None = None
    name: str


class CocoImage(BaseModel):
    id: int | str
    file_name: str
    width: int
    height: int


class CocoAnnotation(BaseModel):
    id: int | str
    image_id: int | str
    category_id: int | str
    bbox: list[int]
    segmentation: CocoRLE | list[list[int]]
    area: int
    iscrowd: int = 0
    metadata: dict[str, Any] | None = None


class Coco(BaseModel):
    categories: list[CocoCategory]
    images: list[CocoImage]
    annotations: list[CocoAnnotation]


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
    image_id: int | int
    annotations: list[InstanceDetectionAnnotation]


class InstanceDetectionPredictionInstance(BaseModel):
    image_id: int | str
    bbox: list[int]
    category_id: int
    segmentation: CocoRLE | list[list[int]]
    score: float


class InstanceDetectionPrediction(BaseModel):
    image_id: int | str
    instances: list[InstanceDetectionPredictionInstance]
