from pathlib import Path

from pydantic import BaseModel

from app.schemas import ID, CocoRLE


class SemanticSegmentationData(BaseModel):
    file_name: Path
    height: int
    width: int
    image_id: ID
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
