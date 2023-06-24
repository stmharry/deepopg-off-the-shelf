import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import ijson
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import pycocotools.mask
import rich.progress
import torch
from absl import app, flags, logging
from pydantic import parse_file_as, parse_obj_as

from app.coco_annotator.clients import CocoAnnotatorClient
from app.coco_annotator.schemas import CocoAnnotatorDataset, CocoAnnotatorImage
from app.instance_detection.datasets import InstanceDetectionV1
from app.instance_detection.schemas import (
    Coco,
    CocoAnnotation,
    CocoCategory,
    CocoImage,
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
)
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.structures import BoxMode, Instances
from detectron2.utils.visualizer import VisImage, Visualizer

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string(
    "prediction_name", "instances_predictions.pth", "Prediction file name."
)
flags.DEFINE_boolean("show_visualizer", False, "Visualize the results to images.")
flags.DEFINE_string("visualizer_dir", "visualize", "Visualizer directory.")
flags.DEFINE_float("visualizer_min_score", 0.05, "Minimum score to visualize.")
flags.DEFINE_boolean(
    "show_coco_annotator", False, "Visualize the results in coco annotator."
)
flags.DEFINE_string("coco_annotator_url", None, "Coco annotator API url.")
FLAGS = flags.FLAGS


def as_detectron2_instances(
    prediction: InstanceDetectionPrediction,
    image_size: Tuple[int, int],
    min_score: float,
    category_ids: Optional[List[int]] = None,
) -> Instances:
    scores: List[float] = []
    pred_boxes: List[List[int]] = []
    pred_classes: List[int] = []
    pred_masks: List[npt.NDArray[np.uint8]] = []

    for instance in prediction.instances:
        score: float = instance.score

        if score < min_score:
            continue

        if (category_ids is not None) and (instance.category_id not in category_ids):
            continue

        scores.append(score)
        pred_boxes.append(
            BoxMode.convert(instance.bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        )
        pred_classes.append(instance.category_id)
        pred_masks.append(pycocotools.mask.decode(instance.segmentation.dict()))

    return Instances(
        image_size=image_size,
        scores=np.array(scores),
        pred_boxes=np.array(pred_boxes),
        pred_classes=np.array(pred_classes),
        pred_masks=np.array(pred_masks),
    )


def show_visualizer(
    dataset: List[InstanceDetectionData],
    metadata: Metadata,
    min_score: float,
    category_re_groups: Dict[str, str] = {
        "all": ".*",
        "tooth": r"TOOTH_\d+",
        "m3": r"TOOTH_(18|28|38|48)",
        "findings": r"(?!TOOTH_\d+)",
    },
) -> None:
    prediction_path: Path = Path(FLAGS.result_dir, FLAGS.prediction_name)

    predictions_obj = torch.load(prediction_path)
    predictions = parse_obj_as(List[InstanceDetectionPrediction], predictions_obj)
    id_to_prediction: Dict[Union[str, int], InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    visualize_dir: Path = Path(FLAGS.result_dir, FLAGS.visualizer_dir)
    Path(visualize_dir).mkdir(exist_ok=True)

    for data in dataset:
        if data.image_id not in id_to_prediction:
            logging.warning(f"Image id {data.image_id} not found in predictions.")
            continue

        logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

        prediction: InstanceDetectionPrediction = id_to_prediction[data.image_id]

        for group_name, re_pattern in category_re_groups.items():
            image_path: Path
            if group_name == "all":
                image_path = Path(
                    visualize_dir, f"{data.file_name.stem}{data.file_name.suffix}"
                )
            else:
                image_path = Path(
                    visualize_dir,
                    f"{data.file_name.stem}_{group_name}{data.file_name.suffix}",
                )

            if image_path.exists():
                logging.info(f"Skipping {data.image_id} as it already exists.")
                continue

            category_ids: List[int] = [
                category_id
                for (category_id, category) in enumerate(metadata.thing_classes)
                if re.match(re_pattern, category)
            ]

            instances: Instances = as_detectron2_instances(
                prediction,
                image_size=(data.height, data.width),
                min_score=min_score,
                category_ids=category_ids,
            )

            image_bw: np.ndarray = iio.imread(data.file_name)
            image_rgb: np.ndarray = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)

            visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0)
            image_vis: VisImage = visualizer.draw_instance_predictions(instances)

            logging.info(f"Saving to {image_path}.")
            image_vis.save(image_path)


def show_coco_annotator(
    data_driver: InstanceDetectionV1,
    dataset: List[InstanceDetectionData],
    metadata: Metadata,
    url: str,
) -> None:
    random_suffix: str = "".join(random.sample(string.ascii_lowercase, 4))
    name: str = f"{Path(FLAGS.result_dir).name}-{random_suffix}"
    image_ids: Set[int] = set([int(data.image_id) for data in dataset])

    result_catogory_id_to_coco_category_id: Dict[int, int] = {
        value: key
        for (key, value) in metadata.thing_dataset_id_to_contiguous_id.items()
    }

    ###

    client: CocoAnnotatorClient = CocoAnnotatorClient(url=url)
    client.login()

    ### dataset

    ca_dataset: Optional[CocoAnnotatorDataset]
    ca_dataset = client.get_dataset_by_name(name=name)
    if ca_dataset is None:
        ca_dataset = client.create_dataset(name=name)

    client.update_dataset(ca_dataset.id, categories=metadata.thing_classes)

    assert ca_dataset is not None

    ### categories

    with open(data_driver.coco_path) as f:
        coco_categories: List[CocoCategory] = parse_obj_as(
            List[CocoCategory], list(ijson.items(f, "categories.item"))
        )

    ### annotations

    coco_instances: List[InstanceDetectionPredictionInstance] = parse_file_as(
        List[InstanceDetectionPredictionInstance],
        Path(FLAGS.result_dir, "coco_instances_results.json"),
    )

    coco_annotations: List[CocoAnnotation] = []
    for num, coco_instance in rich.progress.track(
        enumerate(coco_instances),
        total=len(coco_instances),
        description="Converting annotations...",
    ):
        if int(coco_instance.image_id) not in image_ids:
            continue

        category_id: int = result_catogory_id_to_coco_category_id[
            coco_instance.category_id
        ]

        rle_obj: Dict[str, Any] = coco_instance.segmentation.dict()
        segmentation: npt.NDArray[np.uint8] = pycocotools.mask.decode(rle_obj)
        contours: List[npt.NDArray[np.int32]]
        contours, _ = cv2.findContours(
            segmentation[..., None], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        coco_annotations.append(
            CocoAnnotation(
                id=num,
                image_id=coco_instance.image_id,
                category_id=category_id,
                bbox=coco_instance.bbox,
                segmentation=[contour.flatten().tolist() for contour in contours],
                area=pycocotools.mask.area(rle_obj),
                metadata={"score": coco_instance.score},
            )
        )

    ### images

    ca_images: List[CocoAnnotatorImage] = client.get_images()
    ca_image_names: Set[str] = {ca_image.file_name for ca_image in ca_images}

    with open(data_driver.coco_path) as f:
        all_coco_images: List[CocoImage] = parse_obj_as(
            List[CocoImage], list(ijson.items(f, "images.item"))
        )

    coco_images: List[CocoImage] = []
    for coco_image in all_coco_images:
        # not in this dataset
        if coco_image.id not in image_ids:
            continue

        file_name: Path = Path(coco_image.file_name)
        ca_image_name: str = f"{file_name.stem}-{random_suffix}{file_name.suffix}"

        # already uploaded
        if ca_image_name in ca_image_names:
            continue

        try:
            client.create_image(
                file_path=Path(data_driver.image_dir, coco_image.file_name),
                file_name=ca_image_name,
                dataset_id=ca_dataset.id,
            )

        except ValueError:
            logging.warning(f"Failed to create image {ca_image_name}")
            continue

        coco_images.append(
            CocoImage(
                id=coco_image.id,
                file_name=ca_image_name,
                width=coco_image.width,
                height=coco_image.height,
            )
        )

    coco: Coco = Coco(
        images=coco_images,
        categories=coco_categories,
        annotations=coco_annotations,
    )

    client.upload_coco(coco=coco, dataset_id=ca_dataset.id)


def main(_):
    data_driver = InstanceDetectionV1.register(root_dir=FLAGS.data_dir)

    dataset: List[InstanceDetectionData] = parse_obj_as(
        List[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get("pano")

    if FLAGS.show_visualizer:
        show_visualizer(
            dataset=dataset, metadata=metadata, min_score=FLAGS.visualizer_min_score
        )

    if FLAGS.show_coco_annotator:
        show_coco_annotator(
            data_driver=data_driver,
            dataset=dataset,
            metadata=metadata,
            url=FLAGS.coco_annotator_url,
        )


if __name__ == "__main__":
    app.run(main)
