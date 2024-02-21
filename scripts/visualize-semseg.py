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
flags.DEFINE_string(
    "prediction", "inference/sem_seg_predictions.json", "Input prediction file name."
)
flags.DEFINE_string("visualize_dir", "visualize", "Visualizer directory.")
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
        Path(FLAGS.result_dir, FLAGS.prediction),
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

        semseg_mask: np.ndarray
        if len(detectron2_instances.pred_masks) == 0:
            semseg_mask = np.zeros((data["height"], data["width"]), dtype=np.uint8)
        else:
            pred_masks: np.ndarray = np.r_[
                "0, 3",
                np.zeros((data["height"], data["width"]), dtype=np.uint8),
                detectron2_instances.pred_masks,
            ]
            pred_classes: np.ndarray = np.r_[0, detectron2_instances.pred_classes]

            semseg_mask = pred_classes[np.argmax(pred_masks, axis=0)]

        image_vis: VisImage = visualizer.draw_sem_seg(semseg_mask, alpha=0.5)

        image_path: Path = Path(visualize_dir, f"{file_name.stem}.png")

        logging.info(f"Saving to {image_path!s}.")
        image_vis.save(image_path)


if __name__ == "__main__":
    app.run(main)
