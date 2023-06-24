from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from detectron2.structures import BoxMode


class CocoRLE(BaseModel):
    size: List[int]
    counts: str


class CocoCategory(BaseModel):
    id: Optional[int] = None
    name: str


class CocoImage(BaseModel):
    id: Union[int, str]
    file_name: str
    width: int
    height: int


class CocoAnnotation(BaseModel):
    id: Union[int, str]
    image_id: Union[int, str]
    category_id: Union[int, str]
    bbox: List[int]
    segmentation: Union[CocoRLE, List[List[int]]]
    area: int
    iscrowd: int = 0
    metadata: Optional[Dict[str, Any]] = None


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
    image_id: Union[int, int]
    annotations: List[InstanceDetectionAnnotation]


class InstanceDetectionPredictionInstance(BaseModel):
    image_id: Union[int, str]
    bbox: List[int]
    category_id: int
    segmentation: CocoRLE
    score: float


class InstanceDetectionPrediction(BaseModel):
    image_id: Union[int, str]
    instances: List[InstanceDetectionPredictionInstance]
