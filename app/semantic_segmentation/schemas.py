from pathlib import Path

from pydantic import BaseModel

from app.schemas import ID


class SemanticSegmentationData(BaseModel):
    file_name: Path
    height: int
    width: int
    image_id: ID
    sem_seg_file_name: Path
