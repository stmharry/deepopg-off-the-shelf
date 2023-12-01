from typing import Any, TypeAlias

from pydantic import BaseModel

ID: TypeAlias = int | str


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
    id: ID
    image_id: ID
    category_id: ID
    bbox: list[int]
    segmentation: CocoRLE | list[list[int]]
    area: int
    iscrowd: int = 0
    metadata: dict[str, Any] | None = None


class Coco(BaseModel):
    categories: list[CocoCategory]
    images: list[CocoImage]
    annotations: list[CocoAnnotation]
