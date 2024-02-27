import contextlib
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.schemas import ID
from app.semantic_segmentation.datasets import SemanticSegmentation
from app.semantic_segmentation.schemas import (
    SemanticSegmentationData,
    SemanticSegmentationPrediction,
    SemanticSegmentationPredictionList,
)
from app.tasks import map_fn
from app.utils import read_image
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import VisImage, Visualizer

flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_enum(
    "dataset_name",
    "pano_semseg_v4",
    SemanticSegmentation.available_dataset_names(),
    "Dataset name.",
)
flags.DEFINE_string(
    "prediction", "inference/sem_seg_predictions.json", "Input prediction file name."
)
flags.DEFINE_string("visualize_dir", "visualize", "Visualizer directory.")
flags.DEFINE_boolean("force", False, "Overwrite existing files.")
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
FLAGS = flags.FLAGS


def _visualize_binarized_semseg(
    detectron2_instances: Instances,
    visualizer: Visualizer,
    output_image_path: Path,
) -> None:
    height: int = visualizer.output.height
    width: int = visualizer.output.width

    semseg_mask: np.ndarray
    if len(detectron2_instances.pred_masks) == 0:
        semseg_mask = np.zeros((height, width), dtype=np.uint8)

    else:
        pred_masks: np.ndarray = np.r_[
            "0,3",
            np.zeros((height, width), dtype=np.uint8),
            detectron2_instances.pred_masks,
        ]
        pred_classes: np.ndarray = np.r_[0, detectron2_instances.pred_classes]

        semseg_mask = pred_classes[np.argmax(pred_masks, axis=0)]

    logging.info(f"Saving to {output_image_path!s}.")
    image_vis: VisImage = visualizer.draw_sem_seg(semseg_mask, alpha=0.5)
    image_vis.save(output_image_path)


def _visualize_heatmap_semseg(
    npz_path: Path,
    visualizer: Visualizer,
    output_image_path: Path,
) -> None:
    with np.load(npz_path) as npz:
        prob_uint16: np.ndarray = npz["prob"]  # (C, H, W)

    prob: np.ndarray = prob_uint16.astype(np.float32) / 65535.0

    for num, (class_prob, stuff_class) in enumerate(
        zip(prob, visualizer.metadata.stuff_classes)
    ):
        if num == 0:
            continue

        visualizer.draw_soft_mask(
            class_prob,
            # we don't use metadata's color as it might be too continuous
            color=cm.hsv(((num - 1) / 8) % 1),  # type: ignore
            text=stuff_class.split("_")[1],
            alpha=0.75,
        )

    logging.info(f"Saving to {output_image_path!s}.")
    image_vis: VisImage = visualizer.output
    image_vis.save(output_image_path)


def visualize_data(
    data: SemanticSegmentationData,
    prediction: SemanticSegmentationPrediction,
    metadata: Metadata,
    visualize_dir: Path,
) -> None:
    output_image_path: Path
    image_rgb: np.ndarray = read_image(prediction.file_name)

    output_image_path = Path(
        visualize_dir, f"{prediction.file_name.stem}.binarized.jpg"
    )
    if (not FLAGS.force) and output_image_path.exists():
        logging.info(f"Skipping {output_image_path!s}.")
    else:
        _visualize_binarized_semseg(
            detectron2_instances=prediction.to_detectron2_instances(
                height=data.height, width=data.width
            ),
            visualizer=Visualizer(image_rgb, metadata=metadata),
            output_image_path=output_image_path,
        )

    output_image_path = Path(visualize_dir, f"{prediction.file_name.stem}.heatmap.jpg")
    if (not FLAGS.force) and output_image_path.exists():
        logging.info(f"Skipping {output_image_path!s}.")
    else:
        _visualize_heatmap_semseg(
            npz_path=Path(FLAGS.result_dir, "inference", f"{data.file_name.stem}.npz"),
            visualizer=Visualizer(image_rgb, metadata=metadata),
            output_image_path=output_image_path,
        )


def main(_):
    data_driver: SemanticSegmentation | None = SemanticSegmentation.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    if data_driver is None:
        raise ValueError(f"Dataset {FLAGS.dataset_name} not found.")

    dataset: list[SemanticSegmentationData] = parse_obj_as(
        list[SemanticSegmentationData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    predictions: list[SemanticSegmentationPrediction] = (
        SemanticSegmentationPredictionList.from_detectron2_semseg_output_json(
            Path(FLAGS.result_dir, FLAGS.prediction),
            file_name_to_image_id={data.file_name: data.image_id for data in dataset},
        )
    )

    name_to_prediction: dict[ID, SemanticSegmentationPrediction] = {  # type: ignore
        prediction.file_name.stem: prediction for prediction in predictions
    }

    visualize_dir: Path = Path(FLAGS.result_dir, FLAGS.visualize_dir)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        tasks: list[tuple] = [
            (
                data,
                name_to_prediction[data.file_name.stem],
                metadata,
                visualize_dir,
            )
            for data in dataset
            if data.file_name.stem in name_to_prediction
        ]
        for _ in map_fn(
            visualize_data, tasks=tasks, stack=stack, num_workers=FLAGS.num_workers
        ):
            ...


if __name__ == "__main__":
    app.run(main)
