from pathlib import Path
from typing import Any

from pydantic import parse_file_as

from app.schemas import ID
from app.semantic_segmentation.schemas import (
    SemanticSegmentationPrediction,
    SemanticSegmentationPredictionInstance,
)


def parse_predictions_json(
    path: Path, file_name_to_image_id: dict[Any, ID]
) -> list[SemanticSegmentationPrediction]:
    instances: list[SemanticSegmentationPredictionInstance] = parse_file_as(
        list[SemanticSegmentationPredictionInstance], path
    )

    file_name: Path | None = None
    image_id: ID | None = None
    _instances: list[SemanticSegmentationPredictionInstance] = []

    predictions: list[SemanticSegmentationPrediction] = []
    for instance in instances:
        if instance.file_name != file_name:
            image_id = file_name_to_image_id.get(instance.file_name, None)

            if file_name is not None:
                predictions.append(
                    SemanticSegmentationPrediction(
                        image_id=image_id, file_name=file_name, instances=_instances
                    )
                )

            file_name = instance.file_name
            _instances = []

        _instances.append(instance.copy(update={"image_id": image_id}))

    if file_name is not None:
        predictions.append(
            SemanticSegmentationPrediction(
                image_id=image_id, file_name=file_name, instances=_instances
            )
        )

    return predictions
