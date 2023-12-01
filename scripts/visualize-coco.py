import random
import string
from pathlib import Path

import rich.progress
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.coco_annotator.clients import CocoAnnotatorClient
from app.coco_annotator.schemas import CocoAnnotatorDataset, CocoAnnotatorImage
from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
)
from app.schemas import Coco, CocoAnnotation, CocoCategory, CocoImage
from app.utils import (
    instance_detection_data_to_prediction,
    load_predictions,
    prediction_to_coco_annotations,
)
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string(
    "prediction_name", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_bool(
    "use_gt_as_prediction",
    False,
    "Set to true to perform command on ground truth. Useful when we do not have ground truth "
    "finding summary but only ground truth segmentation.",
)
flags.DEFINE_string("coco_annotator_url", None, "Coco annotator API url.")
FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.INFO)

    data_driver: InstanceDetection = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    url: str = FLAGS.coco_annotator_url

    random_suffix: str = "".join(random.sample(string.ascii_lowercase, 4))
    name: str = f"{Path(FLAGS.result_dir).name}-{random_suffix}"
    image_ids: set[int] = set([int(data.image_id) for data in dataset])

    ###

    client: CocoAnnotatorClient = CocoAnnotatorClient(url=url)
    client.login()

    ### dataset

    ca_dataset: CocoAnnotatorDataset | None
    ca_dataset = client.get_dataset_by_name(name=name)
    if ca_dataset is None:
        ca_dataset = client.create_dataset(name=name)

    client.update_dataset(ca_dataset.id, categories=metadata.thing_classes)

    assert ca_dataset is not None

    ### categories

    coco_categories: list[CocoCategory] = InstanceDetection.get_coco_categories(
        data_driver.coco_paths[0]
    )

    ### annotations

    predictions: list[InstanceDetectionPrediction]
    if FLAGS.use_gt_as_prediction:
        predictions = [
            instance_detection_data_to_prediction(instance_detection_data=data)
            for data in dataset
        ]
    else:
        predictions = load_predictions(
            prediction_path=Path(FLAGS.result_dir, FLAGS.prediction_name)
        )

    predictions = [
        prediction
        for prediction in predictions
        if int(prediction.image_id) in image_ids
    ]

    coco_annotations: list[CocoAnnotation] = []
    for prediction in rich.progress.track(
        predictions, total=len(predictions), description="Converting predictions..."
    ):
        _coco_annotations: list[CocoAnnotation] = prediction_to_coco_annotations(
            prediction=prediction,
            coco_categories=coco_categories,
            start_id=len(coco_annotations),
        )
        coco_annotations.extend(_coco_annotations)

    ### images

    ca_images: list[CocoAnnotatorImage] = client.get_images()
    ca_image_names: set[str] = {ca_image.file_name for ca_image in ca_images}

    all_coco_images: list[CocoImage] = InstanceDetection.get_coco_images(
        data_driver.coco_paths[0]
    )

    coco_images: list[CocoImage] = []
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


if __name__ == "__main__":
    app.run(main)
