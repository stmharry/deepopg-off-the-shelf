from pathlib import Path

import numpy as np
from absl import app, flags, logging

from app.semantic_segmentation.datasets import SemanticSegmentation
from app.semantic_segmentation.schemas import (
    SemanticSegmentationPrediction,
    SemanticSegmentationPredictionList,
)
from app.utils import read_image
from detectron2.data import Metadata, MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import VisImage, Visualizer

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("visualize_dir", "visualize", "Visualizer directory.")
flags.DEFINE_string(
    "prediction_name", "sem_seg_predictions.json", "Input prediction file name."
)
FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.INFO)

    data_driver: SemanticSegmentation | None = SemanticSegmentation.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    if data_driver is None:
        raise ValueError(f"Dataset {FLAGS.dataset_name} not found.")

    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    file_name_to_data: dict[Path, dict] = {
        Path(data["file_name"]): data for data in data_driver.dataset
    }
    predictions: list[
        SemanticSegmentationPrediction
    ] = SemanticSegmentationPredictionList.from_detectron2_semseg_output_json(
        Path(FLAGS.result_dir, FLAGS.prediction_name),
        file_name_to_image_id={
            file_name: data["image_id"] for file_name, data in file_name_to_data.items()
        },
    )

    visualize_dir: Path = Path(FLAGS.result_dir, FLAGS.visualize_dir)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    for prediction in predictions:
        file_name: Path = prediction.file_name

        data: dict = file_name_to_data[file_name]
        image_rgb: np.ndarray = read_image(file_name)

        visualizer = Visualizer(image_rgb, metadata=metadata)
        detectron2_instances: Instances = prediction.to_detectron2_instances(
            height=data["height"], width=data["width"]
        )
        image_vis: VisImage = visualizer.draw_instance_predictions(detectron2_instances)

        image_path: Path = Path(visualize_dir, f"{file_name.stem}.png")

        logging.info(f"Saving to {image_path!s}.")
        image_vis.save(image_path)


if __name__ == "__main__":
    app.run(main)
