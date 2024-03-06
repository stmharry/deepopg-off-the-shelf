import itertools
from collections.abc import Iterator
from pathlib import Path

from absl import app, flags, logging

from app.instance_detection import (
    InstanceDetection,
    InstanceDetectionData,
    InstanceDetectionFactory,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
    InstanceDetectionPredictionList,
)
from app.masks import Mask
from app.tasks import Pool, filter_none, track_progress

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_enum(
    "dataset_name",
    "pano",
    InstanceDetectionFactory.available_dataset_names(),
    "Dataset name.",
)
flags.DEFINE_string(
    "prediction", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
FLAGS = flags.FLAGS


def process_data(
    data: InstanceDetectionData,
) -> InstanceDetectionPrediction:
    image_name: str = data.file_name.stem
    label_file_path: Path = Path(FLAGS.result_dir, "labels", f"{image_name}.txt")

    instances: list[InstanceDetectionPredictionInstance] = []
    with open(label_file_path, "r") as f:
        for line in f:
            iterator: Iterator[str] = iter(line.strip().split())

            category_id: int = int(next(iterator))

            polygon: list[int] = []
            score: float | None = None
            while batch := tuple(itertools.islice(iterator, 2)):
                match batch:
                    case (x_str, y_str):
                        polygon.append(int(float(x_str) * data.width))
                        polygon.append(int(float(y_str) * data.height))

                    case (score_str,):
                        score = float(score_str)
                        break

                    case _:
                        raise ValueError(
                            f"Invalid line from {label_file_path!s}: {line.strip()}"
                        )

            # three points are needed to form a polygon
            if len(polygon) < 6:
                logging.info(
                    f"No polygon found in line from {label_file_path!s}:"
                    f" {line.strip()}"
                )
                continue

            if score is None:
                logging.warning(
                    f"Score not found in line from {label_file_path!s}:"
                    f" {line.strip()}, setting to 1.0"
                )
                score = 1.0

            mask: Mask = Mask.from_obj([polygon], height=data.height, width=data.width)

            bbox: list[int] = mask.bbox_xywh
            if bbox[0] >= data.width or bbox[1] >= data.height:
                logging.info(
                    f"bbox out of bound in line from {label_file_path!s}:"
                    f" {line.strip()}, skipping instance"
                )
                continue

            if bbox[2] <= 0 or bbox[3] <= 0:
                logging.info(
                    f"bbox has zero or negative size in line from {label_file_path!s}:"
                    f" {line.strip()}, skipping instance"
                )
                continue

            instances.append(
                InstanceDetectionPredictionInstance(
                    image_id=data.image_id,
                    bbox=bbox,
                    category_id=category_id,
                    segmentation=mask.polygons,
                    score=score,
                )
            )

    return InstanceDetectionPrediction(image_id=data.image_id, instances=instances)


def main(_):
    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = data_driver.get_coco_dataset(
        dataset_name=FLAGS.dataset_name
    )

    with Pool(num_workers=FLAGS.num_workers) as pool:
        predictions: list[InstanceDetectionPrediction] = list(
            dataset
            | track_progress
            | pool.parallel_pipe(process_data, allow_unordered=True)
            | filter_none
        )

    predictions_path: Path = Path(FLAGS.result_dir, FLAGS.prediction)
    logging.info(f"Saving predictions to {predictions_path}")

    InstanceDetectionPredictionList.model_validate(
        predictions
    ).to_detectron2_detection_pth(path=predictions_path)


if __name__ == "__main__":
    app.run(main)
