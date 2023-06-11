from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel


class CocoAnnotatorDataset(BaseModel):
    id: int
    name: str
    directory: Path
    categories: List[str]
    owner: str
    users: List
    annotate_url: str
    default_annotation_metadata: Dict[str, str]
    deleted: bool


class CocoAnnotatorCategory(BaseModel):
    id: int
    name: str
    supercategory: str
    metadata: Dict[str, str]
    creator: str
    keypoint_colors: List[str]


class CocoAnnotatorImage(BaseModel):
    id: int
    dataset_id: int
    category_ids: List[int]
    path: Path
    width: int
    height: int
    file_name: str
    annotated: bool
    annotating: List
    num_annotations: int
    metadata: Dict
    deleted: bool
