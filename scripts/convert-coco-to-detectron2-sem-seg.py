from pathlib import Path

import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
import scipy.ndimage
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import InstanceDetectionData
from app.utils import Mask
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("mask_dir", "masks", "Mask directory (relative to `data_dir`).")

FLAGS = flags.FLAGS


def main(_):
    data_driver: InstanceDetection = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )

    if FLAGS.dataset_name == "pano_all":
        directory_name = "PROMATON"

    elif FLAGS.dataset_name == "pano_ntuh":
        directory_name = "NTUH"

    else:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_name}")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    output_dir: Path = Path(FLAGS.data_dir, FLAGS.mask_dir, directory_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for data in dataset:
        logging.info(f"Converting {data.file_name!s}...")

        mask_path: Path = Path(output_dir, f"{data.file_name.stem}.png")
        if mask_path.exists():
            logging.info(f"Skipping {mask_path!s} (already exists).")
            continue

        category_ids: list[int] = []
        bitmasks: list[np.ndarray] = []
        for annotation in data.annotations:
            if not metadata.thing_classes[annotation.category_id].startswith("TOOTH"):
                continue

            mask: Mask = Mask.from_obj(
                annotation.segmentation, height=data.height, width=data.width
            )

            category_ids.append(annotation.category_id)
            bitmasks.append(mask.bitmask)

        if len(bitmasks) == 0:
            category_id_map = np.zeros((data.height, data.width), dtype=np.uint8)

        else:
            all_instances_mask: np.ndarray = np.logical_or.reduce(bitmasks, axis=0)
            all_instances_slice: tuple[slice, slice] = scipy.ndimage.find_objects(
                all_instances_mask, max_label=1
            )[0]

            objectness_maps: list[np.ndarray] = []
            for bitmask in bitmasks:
                objectness_map: np.ndarray = scipy.ndimage.distance_transform_cdt(  # type: ignore
                    bitmask[all_instances_slice]
                )
                objectness_maps.append(objectness_map)

            index_map = np.argmax(objectness_maps, axis=0)
            category_id_map = np.take(category_ids, indices=index_map)

            category_id_map = np.pad(
                category_id_map,
                pad_width=[
                    (
                        all_instances_slice[0].start,
                        all_instances_mask.shape[0] - all_instances_slice[0].stop,
                    ),
                    (
                        all_instances_slice[1].start,
                        all_instances_mask.shape[1] - all_instances_slice[1].stop,
                    ),
                ],
            )

            category_id_map = np.where(all_instances_mask, category_id_map, 0)

        category_color_map = (
            cm.get_cmap("viridis")(category_id_map / np.max(category_id_map)) * 255
        )

        # Save mask

        logging.info(f"Saving to {mask_path!s}.")
        iio.imwrite(mask_path, category_id_map.astype(np.uint8))

        mask_vis_path: Path = Path(output_dir, f"{data.file_name.stem}_vis.png")
        logging.info(f"Saving to {mask_vis_path!s}.")
        iio.imwrite(mask_vis_path, category_color_map.astype(np.uint8))


if __name__ == "__main__":
    app.run(main)
