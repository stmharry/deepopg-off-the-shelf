import multiprocessing
from pathlib import Path

import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
import scipy.ndimage
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import InstanceDetectionData
from app.masks import Mask
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("mask_dir", "masks", "Mask directory (relative to `data_dir`).")
flags.DEFINE_integer("num_workers", 1, "Number of processes to use.")

FLAGS = flags.FLAGS

CATEGORY_NAME_TO_SEMSEG_CLASS_ID: dict[str, int] = {
    "TOOTH_11": 1,
    "TOOTH_12": 2,
    "TOOTH_13": 3,
    "TOOTH_14": 4,
    "TOOTH_15": 5,
    "TOOTH_16": 6,
    "TOOTH_17": 7,
    "TOOTH_18": 8,
    "TOOTH_21": 9,
    "TOOTH_22": 10,
    "TOOTH_23": 11,
    "TOOTH_24": 12,
    "TOOTH_25": 13,
    "TOOTH_26": 14,
    "TOOTH_27": 15,
    "TOOTH_28": 16,
    "TOOTH_31": 17,
    "TOOTH_32": 18,
    "TOOTH_33": 19,
    "TOOTH_34": 20,
    "TOOTH_35": 21,
    "TOOTH_36": 22,
    "TOOTH_37": 23,
    "TOOTH_38": 24,
    "TOOTH_41": 25,
    "TOOTH_42": 26,
    "TOOTH_43": 27,
    "TOOTH_44": 28,
    "TOOTH_45": 29,
    "TOOTH_46": 30,
    "TOOTH_47": 31,
    "TOOTH_48": 32,
}


def _process(
    data: InstanceDetectionData, metadata: Metadata, output_dir: Path
) -> InstanceDetectionData | None:
    logging.info(f"Converting {data.file_name!s}...")

    mask_path: Path = Path(output_dir, f"{data.file_name.stem}.png")
    if mask_path.exists():
        logging.info(f"Skipping {mask_path!s} (already exists).")
        return data

    category_ids: list[int] = []
    bitmasks: list[np.ndarray] = []
    for annotation in data.annotations:
        category_name: str = metadata.thing_classes[annotation.category_id]

        semseg_class_id: int | None = CATEGORY_NAME_TO_SEMSEG_CLASS_ID.get(
            category_name
        )
        if semseg_class_id is None:
            continue

        mask: Mask = Mask.from_obj(
            annotation.segmentation, height=data.height, width=data.width
        )

        category_ids.append(semseg_class_id)
        bitmasks.append(mask.bitmask)

    if len(bitmasks) == 0:
        category_id_map = np.zeros((data.height, data.width), dtype=np.uint8)

    else:
        all_instances_mask: np.ndarray = np.logical_or.reduce(bitmasks, axis=0)
        if all_instances_mask.shape != (data.height, data.width):
            logging.error(
                f"Mask shape {all_instances_mask.shape} does not match image shape "
                f"{(data.height, data.width)}."
            )
            return None

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

    return data


def process(
    data: InstanceDetectionData, metadata: Metadata, output_dir: Path
) -> InstanceDetectionData | None:
    try:
        return _process(data, metadata=metadata, output_dir=output_dir)

    except ValueError as e:
        logging.error(e)
        return None


def main(_):
    if FLAGS.dataset_name == "pano_all":
        directory_name = "PROMATON"

    elif FLAGS.dataset_name == "pano_ntuh":
        directory_name = "NTUH"

    else:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_name}")

    driver: InstanceDetection | None = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    if driver is None:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_name}")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    output_dir: Path = Path(FLAGS.data_dir, FLAGS.mask_dir, directory_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(FLAGS.num_workers) as pool:
        _: list[InstanceDetectionData | None] = pool.starmap(
            process,
            [(data, metadata, output_dir) for data in dataset],
        )


if __name__ == "__main__":
    app.run(main)
