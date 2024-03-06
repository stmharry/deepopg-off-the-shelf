from pathlib import Path

import pandas as pd
from pydantic import BaseModel, RootModel


class FindingLabel(BaseModel):
    file_name: str
    fdi: int
    finding: str


class FindingLabelList(RootModel[list[FindingLabel]]):
    def to_csv(self, path: Path) -> None:
        (
            pd.DataFrame(self.model_dump())
            .sort_values(["file_name", "fdi", "finding"])
            .to_csv(path, index=False)
        )


class FindingPrediction(BaseModel):
    file_name: str
    fdi: int
    finding: str
    score: float


class FindingPredictionList(RootModel[list[FindingPrediction]]): ...
