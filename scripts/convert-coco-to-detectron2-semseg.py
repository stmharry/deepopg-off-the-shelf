from pathlib import Path

import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
import scipy.ndimage
from absl import app, flags, logging

from app.instance_detection import (
    InstanceDetection,
    InstanceDetectionData,
    InstanceDetectionFactory,
)
from app.masks import Mask
from app.tasks import Pool, track_progress
from detectron2.data import Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", "./data", "Data directory.")
# we don't use enum here because we have a match later to check for the value
flags.DEFINE_string("dataset_prefix", "pano", "Dataset prefix.")
flags.DEFINE_string(
    "mask_dir", "masks", "Mask directory to save masks to (relative to `data_dir`)."
)
flags.DEFINE_integer("num_workers", 0, "Number of processes to use.")
FLAGS = flags.FLAGS

CATEGORY_NAME_POSTFIX_TO_SEMSEG_CLASS_ID: dict[str, int] = {
    f"_{quadrant}{index}": (quadrant - 1) * 8 + index
    for quadrant in range(1, 5)
    for index in range(1, 9)
}


def build_category_name_to_semseg_class_id(
    metadata: Metadata,
) -> dict[str, int]:

    category_name_to_semseg_class_id: dict[str, int] = {}
    for _, category_name in enumerate(metadata.thing_classes):
        for (
            postfix,
            semseg_class_id,
        ) in CATEGORY_NAME_POSTFIX_TO_SEMSEG_CLASS_ID.items():

            if category_name.endswith(postfix):
                category_name_to_semseg_class_id[category_name] = semseg_class_id
                break

    return category_name_to_semseg_class_id


def process_data(
    data: InstanceDetectionData,
    *,
    metadata: Metadata,
    category_name_to_semseg_class_id: dict[str, int],
    output_dir: Path,
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

        semseg_class_id: int | None = category_name_to_semseg_class_id.get(
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


def main(_):
    directory_name: str
    match FLAGS.dataset_prefix:
        case "pano" | "pano_raw":
            directory_name = "PROMATON"

        case "pano_ntuh":
            directory_name = "NTUH"

        case _:
            raise ValueError(f"Unknown dataset name: {FLAGS.dataset_prefix}")

    data_driver: InstanceDetection = InstanceDetectionFactory.register_by_name(
        dataset_name=FLAGS.dataset_prefix, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = data_driver.get_coco_dataset(
        dataset_name=FLAGS.dataset_prefix
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_prefix)

    category_name_to_semseg_class_id: dict[str, int] = (
        build_category_name_to_semseg_class_id(metadata)
    )
    logging.info(
        f"Using class name to semseg ID map: {category_name_to_semseg_class_id}"
    )

    output_dir: Path = Path(FLAGS.data_dir, FLAGS.mask_dir, directory_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Pool(num_workers=FLAGS.num_workers) as pool:
        list(
            dataset
            | track_progress
            | pool.parallel_pipe(process_data, allow_unordered=True)(
                metadata=metadata,
                category_name_to_semseg_class_id=category_name_to_semseg_class_id,
                output_dir=output_dir,
            )
        )


if __name__ == "__main__":
    app.run(main)
