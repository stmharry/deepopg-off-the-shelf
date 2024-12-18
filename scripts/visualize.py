import re
from pathlib import Path

import numpy as np
import pipe
from absl import app, flags, logging

from app.instance_detection import (
    InstanceDetection,
    InstanceDetectionData,
    InstanceDetectionFactory,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionList,
)
from app.tasks import Pool, track_progress
from app.utils import read_image
from detectron2.data import Metadata, MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer

flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_enum(
    "dataset_name",
    "pano",
    InstanceDetectionFactory.available_dataset_names(),
    "Dataset name.",
)
flags.DEFINE_string("visualize_dir", "visualize", "Visualizer directory.")
flags.DEFINE_string(
    "prediction", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_bool(
    "use_gt_as_prediction",
    False,
    "Set to true to perform command on ground truth. Useful when we do not have ground"
    " truth finding summary but only ground truth segmentation.",
)
flags.DEFINE_bool(
    "visualize_subset",
    False,
    "Set to true to visualize tooth-only, m3-only, and findings-only objects.",
)
flags.DEFINE_float("min_score", 0.0, "Minimum score to visualize.")
flags.DEFINE_boolean("force", False, "Overwrite existing files.")
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
FLAGS = flags.FLAGS


def visualize_data(
    data: InstanceDetectionData,
    prediction: InstanceDetectionPrediction,
    *,
    metadata: Metadata,
    category_re_groups: dict[str | None, str],
    visualize_dir: Path,
    ignore_scores: bool = False,
    extension: str = "jpg",
) -> None:
    for group_name, re_pattern in category_re_groups.items():
        image_path: Path
        if group_name is None:
            image_path = Path(visualize_dir, f"{data.file_name.stem}.{extension}")

        else:
            image_path = Path(
                visualize_dir, f"{data.file_name.stem}.{group_name}.{extension}"
            )

        if (not FLAGS.force) and image_path.exists():
            logging.info(f"Skipping {data.image_id} as it already exists.")
            continue

        visualize_category_ids: list[int] = []
        thing_classes: list[str] = []
        for category_id, thing_class in enumerate(metadata.thing_classes):
            match_obj: re.Match | None = re.match(re_pattern, thing_class)

            # uninterested category as far as this group is concerned
            if match_obj is None:
                thing_classes.append(thing_class)

            else:
                matched_name: str | None = match_obj.groupdict().get("name")
                if matched_name is None:
                    matched_name = thing_class

                assert isinstance(matched_name, str)

                visualize_category_ids.append(category_id)
                thing_classes.append(matched_name)

        image_rgb: np.ndarray = read_image(data.file_name)
        group_metadata: Metadata = Metadata(
            thing_classes=thing_classes,
            thing_colors=metadata.thing_colors,
        )
        visualizer = Visualizer(
            image_rgb,
            metadata=group_metadata,
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION,
        )

        instances: Instances = prediction.to_detectron2_instances(
            height=data.height,
            width=data.width,
            category_ids=visualize_category_ids,
            min_score=FLAGS.min_score,
        )
        logging.debug(f"Visualizing {len(instances)} instances.")
        if ignore_scores:
            instances.remove("scores")

        visualizer.draw_instance_predictions(instances)

        logging.info(f"Saving to {image_path}.")
        image_vis: VisImage = visualizer.output
        image_vis.save(image_path)


def main(_):
    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = data_driver.get_coco_dataset(
        dataset_name=FLAGS.dataset_name
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    category_re_groups: dict[str | None, str]
    if FLAGS.visualize_subset:
        category_re_groups = {
            # None: r"(?P<name>.*)",
            "tooth": r"TOOTH(_(?P<name>\d+))?",
            # "m3": r"TOOTH_(?P<name>18|28|38|48)",
            "findings": r"(?!TOOTH)",
        }

    else:
        category_re_groups = {
            None: r"(?P<name>.*)",
        }

    predictions: list[InstanceDetectionPrediction]
    if FLAGS.use_gt_as_prediction:
        predictions = [
            InstanceDetectionPrediction.from_instance_detection_data(data)
            for data in dataset
        ]
    else:
        predictions = InstanceDetectionPredictionList.from_detectron2_detection_pth(
            Path(FLAGS.result_dir, FLAGS.prediction)
        ).root

    id_to_prediction: dict[str | int, InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    visualize_dir: Path = Path(FLAGS.result_dir, FLAGS.visualize_dir)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    with Pool(num_workers=FLAGS.num_workers) as pool:
        list(
            dataset
            | track_progress
            | pipe.filter(lambda data: data.image_id in id_to_prediction)
            | pipe.map(lambda data: (data, id_to_prediction[data.image_id]))
            | pool.parallel_pipe(
                visualize_data, unpack_input=True, allow_unordered=True
            )(
                metadata=metadata,
                category_re_groups=category_re_groups,
                visualize_dir=visualize_dir,
                ignore_scores=FLAGS.use_gt_as_prediction,
                extension="jpg",
            )
        )


if __name__ == "__main__":
    app.run(main)
