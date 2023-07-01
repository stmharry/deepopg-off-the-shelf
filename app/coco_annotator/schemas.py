from pathlib import Path

from pydantic import BaseModel


class CocoAnnotatorDataset(BaseModel):
    id: int
    name: str
    directory: Path
    categories: list[str]
    owner: str
    users: list
    annotate_url: str
    default_annotation_metadata: dict[str, str]
    deleted: bool


class CocoAnnotatorCategory(BaseModel):
    id: int
    name: str
    supercategory: str
    metadata: dict[str, str]
    creator: str
    keypoint_colors: list[str]


class CocoAnnotatorImage(BaseModel):
    id: int
    dataset_id: int
    category_ids: list[int]
    path: Path
    width: int
    height: int
    file_name: str
    annotated: bool
    annotating: list
    num_annotations: int
    metadata: dict
    deleted: bool
