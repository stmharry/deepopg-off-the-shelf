from pathlib import Path
from typing import List, Optional, Union

from detectron2.structures import BoxMode
from pydantic import BaseModel


class CocoRLE(BaseModel):
    size: List[int]
    counts: str


class CocoCategory(BaseModel):
    id: Optional[int] = None
    name: str


class CocoImage(BaseModel):
    id: Union[str, int]
    file_name: str
    width: int
    height: int


class CocoAnnotation(BaseModel):
    id: Union[str, int]
    image_id: Union[str, int]
    category_id: Union[int, str]
    bbox: List[int]
    segmentation: CocoRLE
    area: int
    iscrowd: int = 0


class Coco(BaseModel):
    categories: List[CocoCategory]
    images: List[CocoImage]
    annotations: List[CocoAnnotation]


class InstanceDetectionAnnotation(BaseModel):
    bbox: List[int]
    bbox_mode: BoxMode
    category_id: int
    segmentation: CocoRLE
    iscrowd: int


class InstanceDetectionData(BaseModel):
    file_name: Path
    height: int
    width: int
    image_id: Union[str, int]
    annotations: List[InstanceDetectionAnnotation]


class InstanceDetectionPredictionInstance(BaseModel):
    image_id: Union[str, int]
    bbox: List[int]
    category_id: int
    segmentation: CocoRLE
    score: float


class InstanceDetectionPrediction(BaseModel):
    image_id: Union[str, int]
    instances: List[InstanceDetectionPredictionInstance]
